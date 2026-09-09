"""Single-device PPO self-play: GPU environment, rollout, vector GAE and updates.

Each critic output belongs to a seat, so opponent turns and same-player
microdecisions are handled without the common incorrect alternating-sign GAE.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env, network


@dataclass(frozen=True)
class Config:
    envs: int = 2048
    horizon: int = 128
    updates: int = 500
    epochs: int = 3
    minibatches: int = 32
    width: int = 256
    residual_blocks: int = 0
    residual_taper: bool = False
    value_head_width: int = 0
    value_head_layers: int = 0
    policy_head_width: int = 0
    policy_head_layers: int = 0
    residual_stage_widths: str = ''
    players: int = 2
    mixed_players: bool = False
    seed: int = 41
    lr: float = 3e-4
    gamma: float = .997
    gae_lambda: float = .95
    entropy: float = .01
    value_loss_coef: float = .25
    shaping: float = .25
    max_turns: int = 400
    bf16: bool = False
    save_every: int = 50
    log_every: int = 10


def advantages(reward, value, last_value, discount, continuation, trace_lambda):
    """All tensors [time,env,seat], discounts [time,env]; terminal breaks trace."""
    next_values = jnp.concatenate((value[1:], last_value[None]), 0)
    delta = reward + discount[..., None] * next_values - value
    def body(carry, xs):
        d, gamma, cont, lam = xs
        adv = d + (gamma * cont * lam)[..., None] * carry
        return adv, adv
    _, advantage = jax.lax.scan(body, jnp.zeros_like(last_value),
        (delta, discount, continuation, trace_lambda), reverse=True)
    return advantage, advantage + value


def make_update(cfg, optimizer):
    batch_size = cfg.envs * cfg.horizon
    mb_size = batch_size // cfg.minibatches

    def update(params, opt_state, states, key):
        def collect(carry, _):
            s, rng = carry
            rng, ka, kr = jax.random.split(rng, 3)
            obs, mask = env.batch_observe(s), env.batch_mask(s)
            logits, relative = network.apply(params, obs, mask, cfg.bf16)
            values = network.absolute_values(relative, s.player, s.nplayers)
            actions = jax.random.categorical(ka, logits).astype(jnp.int32)
            logprob = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], -1)[:, 0]
            ns = env.batch_step(s, actions)
            advanced = ns.turns != s.turns
            timeout = (ns.turns >= cfg.max_turns) & ~ns.done
            ended = ns.done | timeout
            gamma = jnp.where(advanced, cfg.gamma, 1.)
            phi = jax.vmap(env.potential)(s)
            next_phi = jnp.where(ended[:, None], 0., jax.vmap(env.potential)(ns))
            reward = jax.vmap(env.outcome)(ns) + cfg.shaping * (gamma[:, None] * next_phi - phi)
            # Training time limit is an explicit artificial draw, recorded separately.
            discount = gamma * ~ended
            fresh = jax.vmap(env.reset)(jax.random.split(kr, cfg.envs), ns.nplayers)
            reset_states = jax.tree.map(lambda a, b: jnp.where(ended.reshape((cfg.envs,) + (1,) * (a.ndim - 1)), b, a), ns, fresh)
            transition = dict(obs=obs, mask=mask, actions=actions, logprob=logprob, value=values,
                reward=reward, discount=discount, continuation=(~ended).astype(jnp.float32),
                trace_lambda=jnp.where(advanced, cfg.gae_lambda, 1.), player=s.player,
                games=ns.done, timeout=timeout, scores=ns.scores * ns.done[:, None],
                turns=ns.turns * ns.done, advanced=advanced, nplayers=ns.nplayers)
            return (reset_states, rng), transition

        (states, key), roll = jax.lax.scan(collect, (states, key), None, length=cfg.horizon)
        _, last_relative = network.apply(params, env.batch_observe(states), env.batch_mask(states), cfg.bf16)
        last_value = network.absolute_values(last_relative, states.player, cfg.players)
        adv, targets = advantages(roll['reward'], roll['value'], last_value, roll['discount'],
                                  roll['continuation'], roll['trace_lambda'])
        own_adv = jnp.take_along_axis(adv, roll['player'][..., None], -1)[..., 0]
        # Normalize across rollout, rather than separately for each minibatch.
        own_adv = (own_adv - own_adv.mean()) / (own_adv.std() + 1e-8)
        train_data = {k: roll[k] for k in ('obs', 'mask', 'actions', 'logprob', 'value', 'player', 'nplayers')}
        train_data.update(adv=own_adv, target=targets)
        train_data = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), train_data)

        def loss_fn(p, batch):
            logits, relative = network.apply(p, batch['obs'], batch['mask'], cfg.bf16)
            values = network.absolute_values(relative, batch['player'], batch['nplayers'])
            logprobs = jax.nn.log_softmax(logits)
            logprob = jnp.take_along_axis(logprobs, batch['actions'][:, None], -1)[:, 0]
            ratio = jnp.exp(logprob - batch['logprob'])
            policy_loss = -jnp.minimum(ratio * batch['adv'], jnp.clip(ratio, .8, 1.2) * batch['adv']).mean()
            # Value learning deliberately uses the plain per-seat MSE, without
            # PPO value clipping, so the dedicated value head receives the full
            # regression signal from terminal zero-sum outcomes.
            value_mse = (values - batch['target']) ** 2
            active = jnp.arange(4) < batch['nplayers'][:, None]
            value_mse = (value_mse * active).sum() / active.sum()
            entropy = -(jax.nn.softmax(logits) * logprobs).sum(-1).mean()
            kl = ((ratio - 1.) - (logprob - batch['logprob'])).mean()
            clipfrac = (jnp.abs(ratio - 1.) > .2).mean()
            return policy_loss + cfg.value_loss_coef * value_mse - cfg.entropy * entropy, jnp.array([policy_loss, value_mse, entropy, kl, clipfrac])

        def epoch(carry, _):
            p, opt, rng = carry
            rng, kp = jax.random.split(rng)
            indices = jax.random.permutation(kp, batch_size).reshape((cfg.minibatches, mb_size))
            def minibatch(carry, index):
                p, opt = carry
                batch = jax.tree.map(lambda x: x[index], train_data)
                (_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(p, batch)
                updates, opt = optimizer.update(grad, opt, p)
                return (optax.apply_updates(p, updates), opt), metrics
            (p, opt), metrics = jax.lax.scan(minibatch, (p, opt), indices)
            return (p, opt, rng), metrics.mean(0)
        (params, opt_state, key), metrics = jax.lax.scan(epoch, (params, opt_state, key), None, length=cfg.epochs)
        count = roll['games'].sum()
        player_counts = jnp.arange(2, 5)
        games_by_players = jnp.stack(
            [jnp.sum(roll['games'] & (roll['nplayers'] == n)) for n in player_counts])
        timeouts_by_players = jnp.stack(
            [jnp.sum(roll['timeout'] & (roll['nplayers'] == n)) for n in player_counts])
        turn_sums_by_players = jnp.stack(
            [jnp.sum(roll['turns'] * (roll['nplayers'] == n)) for n in player_counts])
        stats = dict(loss=metrics.mean(0), games=count, timeouts=roll['timeout'].sum(),
                     mean_score=roll['scores'].sum() /
                                jnp.maximum(jnp.sum(roll['games'] * roll['nplayers']), 1),
                     mean_turns=roll['turns'].sum() / jnp.maximum(count, 1), turns=roll['advanced'].sum(),
                     games_by_players=games_by_players, timeouts_by_players=timeouts_by_players,
                     mean_turns_by_players=turn_sums_by_players / jnp.maximum(games_by_players, 1),
                     reward=roll['reward'].mean())
        return params, opt_state, states, key, stats
    return jax.jit(update, donate_argnums=(0, 1, 2, 3))


def save(path, params, cfg, update, opt_state=None, states=None, key=None):
    """Atomic NPZ with no pickle. Full state supports exact uninterrupted resume."""
    if isinstance(params, list):
        data = {f'layer_{i}_{name}': np.asarray(value)
                for i, layer in enumerate(params) for name, value in layer.items()}
    else:
        data = {f'param_{i}': np.asarray(value) for i, value in enumerate(jax.tree.leaves(params))}
        data['param_format'] = np.array('pytree-v2')
    data['config'] = np.array(json.dumps(asdict(cfg)))
    data['update'] = np.array(update)
    if opt_state is not None:
        for i, leaf in enumerate(jax.tree.leaves(opt_state)):
            data[f'opt_{i}'] = np.asarray(leaf)
        for field, leaf in zip(states._fields, states):
            data[f'state_{field}'] = np.asarray(leaf)
        data['key'] = np.asarray(key)
    path = Path(path)
    temp = path.with_suffix('.tmp.npz')
    np.savez(temp, **data)
    os.replace(temp, path)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        cfg = Config(**json.loads(str(data['config'])))
        if 'param_format' in data:
            obs_dim = env.observe(env.reset(jax.random.PRNGKey(0), cfg.players)).shape[0]
            template = network.init(jax.random.PRNGKey(0), obs_dim, cfg.width, cfg.residual_blocks,
                                    cfg.residual_taper, cfg.value_head_width, cfg.value_head_layers,
                                    cfg.policy_head_width, cfg.policy_head_layers, cfg.residual_stage_widths)
            leaves, tree = jax.tree.flatten(template)
            params = jax.tree.unflatten(tree, [jnp.asarray(data[f'param_{i}']) for i in range(len(leaves))])
        else:
            params = [{name: jnp.asarray(data[f'layer_{i}_{name}']) for name in ('w', 'b')} for i in range(3)]
    return params, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in Config.__dataclass_fields__.items():
        arg = '--' + name.replace('_', '-')
        parser.add_argument(arg, **(dict(action='store_true') if field.type is bool else dict(type=field.type)), default=field.default)
    parser.add_argument('--out', default='runs/a100')
    parser.add_argument('--resume')
    parser.add_argument('--warm-start', help='Load policy weights, start fresh optimizer / environments (permits new batch and precision)')
    parser.add_argument('--allow-cpu', action='store_true', help='Explicitly allow diagnostic CPU runs')
    args = parser.parse_args()
    if args.resume and args.warm_start:
        parser.error('--resume and --warm-start are mutually exclusive')
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    if (any(getattr(cfg, name) <= 0 for name in ('envs', 'horizon', 'updates', 'epochs', 'minibatches', 'width', 'max_turns', 'save_every', 'log_every'))
            or cfg.residual_blocks < 0 or cfg.value_head_width < 0 or cfg.value_head_layers < 0
            or cfg.policy_head_width < 0 or cfg.policy_head_layers < 0 or cfg.value_loss_coef < 0
            or bool(cfg.value_head_width) != bool(cfg.value_head_layers)
            or bool(cfg.policy_head_width) != bool(cfg.policy_head_layers)):
        parser.error('Batch, iteration, model, and interval sizes must be positive')
    if cfg.players not in (2, 3, 4) or cfg.envs * cfg.horizon % cfg.minibatches:
        parser.error('players must be 2..4; envs*horizon must divide by minibatches')
    devices = jax.devices()
    if not args.allow_cpu and (len(devices) != 1 or devices[0].platform != 'gpu' or 'A100' not in devices[0].device_kind):
        raise RuntimeError(f'Expected one local A100, found {devices}; use --allow-cpu only for diagnostics')
    jax.config.update('jax_default_matmul_precision', 'tensorfloat32')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if ((out / 'latest.npz').exists() or (out / 'metrics.jsonl').exists()) and not args.resume:
        raise FileExistsError(f'{out} contains a run; choose a new --out or --resume')
    jax.config.update('jax_compilation_cache_dir', str(Path('.jax_cache').resolve()))
    device_info = dict(jax=jax.__version__, devices=[str(d) for d in devices], device_kind=devices[0].device_kind,
                       card_source=env.DATA['source'], config=asdict(cfg),
                       parent_checkpoint=args.resume or args.warm_start)
    key, kp, ke = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
    player_counts = jnp.where(cfg.mixed_players, 2 + jnp.arange(cfg.envs) % 3,
                              jnp.full(cfg.envs, cfg.players))
    states = jax.vmap(env.reset)(jax.random.split(ke, cfg.envs), player_counts)
    params = network.init(kp, env.observe(jax.tree.map(lambda x: x[0], states)).shape[0], cfg.width,
                          cfg.residual_blocks, cfg.residual_taper, cfg.value_head_width,
                          cfg.value_head_layers, cfg.policy_head_width, cfg.policy_head_layers,
                          cfg.residual_stage_widths)
    if args.warm_start:
        params, parent_cfg = load(args.warm_start)
        player_compatible = (parent_cfg.players == cfg.players or cfg.mixed_players)
        if (not player_compatible or parent_cfg.width > cfg.width
                or parent_cfg.residual_blocks != cfg.residual_blocks
                or parent_cfg.residual_taper != cfg.residual_taper
                or parent_cfg.value_head_width != cfg.value_head_width
                or parent_cfg.value_head_layers != cfg.value_head_layers
                or parent_cfg.policy_head_width != cfg.policy_head_width
                or parent_cfg.policy_head_layers != cfg.policy_head_layers
                or parent_cfg.residual_stage_widths != cfg.residual_stage_widths):
            raise ValueError('Warm-start requires compatible players, architecture, and non-shrinking width')
        if parent_cfg.width < cfg.width:
            params = network.widen(params, cfg.width, jax.random.fold_in(kp, 800))
    optimizer = optax.chain(optax.clip_by_global_norm(.5), optax.adam(cfg.lr, eps=1e-5))
    opt_state = optimizer.init(params)
    start_update = 0
    if args.resume:
        params, old_cfg = load(args.resume)
        if any(getattr(old_cfg, k) != getattr(cfg, k) for k in asdict(cfg) if k not in ('updates', 'log_every', 'save_every')):
            raise ValueError('Exact resume requires matching training configuration (except updates/logging/saving)')
        with np.load(args.resume, allow_pickle=False) as data:
            leaves, tree = jax.tree.flatten(opt_state)
            opt_state = jax.tree.unflatten(tree, [jnp.asarray(data[f'opt_{i}']) for i in range(len(leaves))])
            states = env.State(**{f: jnp.asarray(data[f'state_{f}']) for f in env.State._fields})
            key = jnp.asarray(data['key'])
            start_update = int(data['update'])
    if start_update >= cfg.updates:
        parser.error('--updates must exceed the restored update number')
    # Validate checkpoint/config before modifying an existing run's metadata.
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2) + '\n')
    (out / 'provenance.json').write_text(json.dumps(device_info, indent=2) + '\n')
    print(json.dumps(device_info), flush=True)
    update = make_update(cfg, optimizer)
    started = time.perf_counter()
    total_games = 0
    with (out / 'metrics.jsonl').open('a') as log:
        for i in range(start_update + 1, cfg.updates + 1):
            before = time.perf_counter()
            params, opt_state, states, key, stats = update(params, opt_state, states, key)
            stats = jax.device_get(stats)
            elapsed = time.perf_counter() - before
            total_games += int(stats['games'])
            row = dict(update=i, decisions=i * cfg.envs * cfg.horizon, seconds=elapsed,
                decisions_per_second=cfg.envs * cfg.horizon / elapsed,
                turns_per_second=float(stats['turns']) / elapsed,
                games=int(stats['games']), timeouts=int(stats['timeouts']), mean_score=float(stats['mean_score']),
                mean_turns=float(stats['mean_turns']), policy_loss=float(stats['loss'][0]), value_mse=float(stats['loss'][1]),
                entropy=float(stats['loss'][2]), approx_kl=float(stats['loss'][3]), clip_fraction=float(stats['loss'][4]),
                games_by_players={str(n): int(stats['games_by_players'][n - 2]) for n in range(2, 5)},
                timeouts_by_players={str(n): int(stats['timeouts_by_players'][n - 2]) for n in range(2, 5)},
                mean_turns_by_players={str(n): float(stats['mean_turns_by_players'][n - 2]) for n in range(2, 5)})
            if not all(np.isfinite(x) for x in row.values() if isinstance(x, (int, float))):
                raise FloatingPointError(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            if i == start_update + 1 or i % cfg.log_every == 0:
                print(json.dumps(row), flush=True)
            if i % cfg.save_every == 0 or i == cfg.updates:
                save(out / 'latest.npz', params, cfg, i, opt_state, states, key)
                save(out / f'policy_{i:06d}.npz', params, cfg, i)
    print(json.dumps(dict(finished=True, updates=cfg.updates, games_this_session=total_games,
                          wall_seconds=time.perf_counter() - started, checkpoint=str(out / 'latest.npz'))), flush=True)


if __name__ == '__main__':
    main()
