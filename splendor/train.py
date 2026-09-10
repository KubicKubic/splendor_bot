"""Single-device PPO self-play: GPU environment, rollout, vector GAE and updates.

Each critic output belongs to a seat, so opponent turns and same-player
microdecisions are handled without the common incorrect alternating-sign GAE.
"""
import argparse
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env, network

# A game needs at least three scoring-card purchases plus the resource turns
# that fund them, so eight fresh states per environment safely cover a
# 128-step rollout.  The overflow counter makes any future horizon/rule
# violation visible rather than silently changing reset semantics.
RESET_POOL_SIZE = 8


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
    gae_lambda: float = 1.0
    entropy: float = .01
    value_loss_coef: float = .25
    shaping: float = .25
    max_turns: int = 400
    bf16: bool = False
    save_every: int = 50
    log_every: int = 10
    observation_version: int = env.OBSERVATION_VERSION


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


def value_statistics(target, prediction, active):
    """Masked MSE and standard explained variance for matching predictions."""
    count = jnp.maximum(active.sum(), 1)
    target_mean = (target * active).sum() / count
    target_variance = (((target - target_mean) ** 2) * active).sum() / count
    error = target - prediction
    error_mean = (error * active).sum() / count
    error_variance = (((error - error_mean) ** 2) * active).sum() / count
    mse = ((error ** 2) * active).sum() / count
    explained_variance = 1. - error_variance / jnp.maximum(target_variance, 1e-8)
    return mse, explained_variance, target_mean, jnp.sqrt(target_variance), error_mean


def make_update(cfg, optimizer):
    batch_size = cfg.envs * cfg.horizon
    mb_size = batch_size // cfg.minibatches

    def update(params, opt_state, states, key):
        # Shuffling three decks is much more expensive than selecting a
        # precomputed reset state.  Previously all envs generated a fresh game
        # at every micro-step although only ~1% actually ended.  Generate a
        # bounded independent pool once per rollout and consume it per env.
        key, rollout_key, reset_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, RESET_POOL_SIZE * cfg.envs).reshape(
            RESET_POOL_SIZE, cfg.envs, 2)
        reset_pool = jax.vmap(lambda keys: jax.vmap(env.reset)(keys, states.nplayers))(reset_keys)
        env_indices = jnp.arange(cfg.envs)

        def collect(carry, _):
            s, rng, reset_index = carry
            rng, ka = jax.random.split(rng)
            obs = env.batch_observe_for_version(s, cfg.observation_version)
            mask = env.batch_mask(s)
            # This cast is mathematically identical to network.apply's first
            # cast, but stores half as many bytes in the rollout used by every
            # PPO epoch/minibatch.
            obs = obs.astype(jnp.bfloat16) if cfg.bf16 else obs
            logits, relative = network.apply(params, obs, mask, cfg.bf16)
            values = network.absolute_values(relative, s.player, s.nplayers)
            actions = jax.random.categorical(ka, logits).astype(jnp.int32)
            logprob = jnp.take_along_axis(jax.nn.log_softmax(logits), actions[:, None], -1)[:, 0]
            ns = env.batch_step(s, actions)
            advanced = ns.turns != s.turns
            # max_turns is diagnostic only.  Treating a time limit as a zero-
            # reward terminal changes the actual game and lets agents seek a
            # draw by stalling.  Rollouts are already finite and long games
            # remain live across update boundaries with a normal bootstrap.
            long_game = advanced & (ns.turns == cfg.max_turns) & ~ns.done
            ended = ns.done
            gamma = jnp.where(advanced, cfg.gamma, 1.)
            phi = jax.vmap(env.potential)(s)
            next_phi = jnp.where(ended[:, None], 0., jax.vmap(env.potential)(ns))
            reward = jax.vmap(env.outcome)(ns) + cfg.shaping * (gamma[:, None] * next_phi - phi)
            discount = gamma * ~ended
            reset_slot = jnp.minimum(reset_index, RESET_POOL_SIZE - 1)
            fresh = jax.tree.map(lambda leaf: leaf[reset_slot, env_indices], reset_pool)
            reset_overflow = ended & (reset_index >= RESET_POOL_SIZE)
            reset_states = jax.tree.map(lambda a, b: jnp.where(ended.reshape((cfg.envs,) + (1,) * (a.ndim - 1)), b, a), ns, fresh)
            transition = dict(obs=obs, mask=mask, actions=actions, logprob=logprob, value=values,
                reward=reward, discount=discount, continuation=(~ended).astype(jnp.float32),
                trace_lambda=jnp.where(advanced, cfg.gae_lambda, 1.), player=s.player,
                games=ns.done, long_game=long_game, scores=ns.scores * ns.done[:, None],
                turns=ns.turns * ns.done, advanced=advanced, nplayers=ns.nplayers,
                ended=ended, reset_overflow=reset_overflow)
            return (reset_states, rng, reset_index + ended.astype(jnp.int32)), transition

        initial_reset_index = jnp.zeros(cfg.envs, jnp.int32)
        (states, _, _), roll = jax.lax.scan(
            collect, (states, rollout_key, initial_reset_index), None, length=cfg.horizon)
        last_observations = env.batch_observe_for_version(states, cfg.observation_version)
        _, last_relative = network.apply(params, last_observations, env.batch_mask(states), cfg.bf16)
        # Mixed batches contain 2P, 3P, and 4P environments.  Bootstrap each
        # rollout with its actual player count; cfg.players is merely the
        # maximum/default and would rotate 2P/3P seats into nonexistent slots.
        last_value = network.absolute_values(last_relative, states.player, states.nplayers)
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
            # Reuse log-softmax instead of asking XLA for a second softmax.
            entropy = -(jnp.exp(logprobs) * logprobs).sum(-1).mean()
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
        long_games_by_players = jnp.stack(
            [jnp.sum(roll['long_game'] & (roll['nplayers'] == n)) for n in player_counts])
        turn_sums_by_players = jnp.stack(
            [jnp.sum(roll['turns'] * (roll['nplayers'] == n)) for n in player_counts])
        active_values = jnp.arange(4) < roll['nplayers'][..., None]
        prediction_mse, explained_variance, target_mean, target_std, value_bias = value_statistics(
            targets, roll['value'], active_values)
        stats = dict(loss=metrics.mean(0), games=count, long_games=roll['long_game'].sum(),
                     resets=roll['ended'].sum(), reset_overflows=roll['reset_overflow'].sum(),
                     mean_score=roll['scores'].sum() /
                                jnp.maximum(jnp.sum(roll['games'] * roll['nplayers']), 1),
                     mean_turns=roll['turns'].sum() / jnp.maximum(count, 1), turns=roll['advanced'].sum(),
                     games_by_players=games_by_players, long_games_by_players=long_games_by_players,
                     mean_turns_by_players=turn_sums_by_players / jnp.maximum(games_by_players, 1),
                     value_explained_variance=explained_variance,
                     value_prediction_mse=prediction_mse, value_bias=value_bias,
                     value_target_mean=target_mean, value_target_std=target_std,
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
        config_data = json.loads(str(data['config']))
        # Checkpoints predating public opponent reserves used the 363-feature
        # actor-private schema.  Preserve inference/evaluation compatibility.
        config_data.setdefault('observation_version', 1)
        cfg = Config(**config_data)
        if 'param_format' in data:
            obs_dim = env.observe(env.reset(jax.random.PRNGKey(0), cfg.players), cfg.observation_version).shape[0]
            template = network.init(jax.random.PRNGKey(0), obs_dim, cfg.width, cfg.residual_blocks,
                                    cfg.residual_taper, cfg.value_head_width, cfg.value_head_layers,
                                    cfg.policy_head_width, cfg.policy_head_layers, cfg.residual_stage_widths)
            leaves, tree = jax.tree.flatten(template)
            params = jax.tree.unflatten(tree, [jnp.asarray(data[f'param_{i}']) for i in range(len(leaves))])
        else:
            params = [{name: jnp.asarray(data[f'layer_{i}_{name}']) for name in ('w', 'b')} for i in range(3)]
    return params, cfg


def reconcile_metrics(path, restored_update):
    """Make a resumed run's metrics agree with its recoverable checkpoint.

    A crash can leave metrics newer than latest.npz, and repeated resumes can
    create duplicate update numbers.  Retain the last record for each update
    through the restored checkpoint and atomically discard unrecoverable rows.
    """
    path = Path(path)
    if not path.exists():
        return
    retained = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            update = int(row['update'])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid metrics row {number} in {path}') from exc
        if update <= restored_update:
            retained[update] = row
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(retained[i]) + '\n' for i in sorted(retained)))
    os.replace(temporary, path)


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
    # Exact resume is governed by the immutable schema stored in the
    # checkpoint.  This also lets pre-v3 jobs resume without requiring callers
    # to know or repeat their historical observation-version flag.
    if args.resume:
        with np.load(args.resume, allow_pickle=False) as data:
            resume_config = json.loads(str(data['config']))
        cfg = replace(cfg, observation_version=int(resume_config.get('observation_version', 1)))
    if (any(getattr(cfg, name) <= 0 for name in ('envs', 'horizon', 'updates', 'epochs', 'minibatches', 'width', 'max_turns', 'save_every', 'log_every'))
            or cfg.residual_blocks < 0 or cfg.value_head_width < 0 or cfg.value_head_layers < 0
            or cfg.policy_head_width < 0 or cfg.policy_head_layers < 0 or cfg.value_loss_coef < 0
            or not 0. <= cfg.gamma <= 1. or not 0. <= cfg.gae_lambda <= 1.
            or bool(cfg.value_head_width) != bool(cfg.value_head_layers)
            or bool(cfg.policy_head_width) != bool(cfg.policy_head_layers)):
        parser.error('Invalid batch/model sizes, loss weights, gamma, or GAE lambda')
    if cfg.players not in (2, 3, 4) or cfg.envs * cfg.horizon % cfg.minibatches:
        parser.error('players must be 2..4; envs*horizon must divide by minibatches')
    if not args.resume and cfg.observation_version != env.OBSERVATION_VERSION:
        parser.error(f'New training must use observation version {env.OBSERVATION_VERSION}')
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
                or parent_cfg.residual_stage_widths != cfg.residual_stage_widths
                or parent_cfg.observation_version != cfg.observation_version):
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
    if args.resume:
        reconcile_metrics(out / 'metrics.jsonl', start_update)
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
                games=int(stats['games']), long_games=int(stats['long_games']), resets=int(stats['resets']),
                reset_overflows=int(stats['reset_overflows']), mean_score=float(stats['mean_score']),
                mean_turns=float(stats['mean_turns']), policy_loss=float(stats['loss'][0]), value_mse=float(stats['loss'][1]),
                entropy=float(stats['loss'][2]), approx_kl=float(stats['loss'][3]), clip_fraction=float(stats['loss'][4]),
                value_explained_variance=float(stats['value_explained_variance']),
                value_prediction_mse=float(stats['value_prediction_mse']), value_bias=float(stats['value_bias']),
                value_target_mean=float(stats['value_target_mean']), value_target_std=float(stats['value_target_std']),
                games_by_players={str(n): int(stats['games_by_players'][n - 2]) for n in range(2, 5)},
                long_games_by_players={str(n): int(stats['long_games_by_players'][n - 2]) for n in range(2, 5)},
                mean_turns_by_players={str(n): float(stats['mean_turns_by_players'][n - 2]) for n in range(2, 5)})
            if not all(np.isfinite(x) for x in row.values() if isinstance(x, (int, float))):
                raise FloatingPointError(row)
            if row['reset_overflows']:
                raise RuntimeError(f'Reset pool exhausted: {row}')
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
