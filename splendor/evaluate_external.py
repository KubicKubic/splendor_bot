"""Pure-JAX paired-seat evaluation against the ported public ValueBuyBot.

The vendored Python implementation is used only by differential tests. The
formal match loop, both agents, and HullQin transitions are one compiled JAX
program, so no Python callback or host state appears between decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import env, network
from .elo import atomic_json, fit_elo, match_summary
from .external_bot import VENDOR, COMMIT
from .train import load
from .valuebot_jax import act as valuebot_act


def make_match(games=2048, max_decisions=4000, bf16=True,
               observation_version=env.OBSERVATION_VERSION):
    """Build one fully compiled model-vs-ValueBuyBot match function."""
    if games < 4 or games % 2:
        raise ValueError('games must be even and >= 4 for paired deals')

    def match(params, seed):
        key, deal_key = jax.random.split(jax.random.PRNGKey(seed))
        deal_keys = jax.random.split(deal_key, games // 2)
        states = env.batch_reset(jnp.concatenate((deal_keys, deal_keys)), 2)
        model_seat = jnp.repeat(jnp.arange(2), games // 2)
        returns = jnp.zeros((games, 6), jnp.int32)

        def body(carry):
            s, returns, rng, count, invalid = carry
            rng, action_key = jax.random.split(rng)
            mask = env.batch_mask(s)
            observations = env.batch_observe_for_version(s, observation_version)
            logits, _ = network.apply(params, observations, mask, bf16)
            model_action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            bot_action, next_returns, valid = jax.vmap(valuebot_act)(s, returns)
            bot_turn = (~s.done) & (s.player != model_seat)
            action = jnp.where(bot_turn, bot_action, model_action)
            returns = jnp.where(bot_turn[:, None], next_returns, returns)
            invalid = invalid + jnp.sum(bot_turn & ~valid)
            return env.batch_step(s, action), returns, rng, count + 1, invalid

        states, returns, _, steps, invalid = jax.lax.while_loop(
            lambda carry: (~jnp.all(carry[0].done)) & (carry[3] < max_decisions),
            body, (states, returns, key, jnp.int32(0), jnp.int32(0)))
        winners = jax.vmap(env.winners)(states)
        own = jnp.take_along_axis(winners, model_seat[:, None], -1)[:, 0]
        score = own / jnp.maximum(winners.sum(-1), 1)
        return dict(score=score, done=states.done, seat_a=model_seat,
                    turns=states.turns, decisions_executed=steps,
                    final_scores=states.scores[:, :2], bot_invalid_actions=invalid,
                    bot_return_tokens=returns.sum())
    return jax.jit(match)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', help='Optional labels corresponding to checkpoints')
    parser.add_argument('--games', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--max-decisions', type=int, default=4000)
    parser.add_argument('--fp32', action='store_true', help='Use FP32 inference (default: common BF16)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.games < 4 or args.games % 2:
        parser.error('games must be even and >= 4')
    if args.names and len(args.names) != len(args.checkpoints):
        parser.error('--names must have exactly one label per checkpoint')
    names = args.names or [Path(p).parent.name + '/' + Path(p).stem for p in args.checkpoints]
    if len(names) != len(set(names)):
        parser.error('checkpoint labels must be unique')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ['git', '-C', str(VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(
        ['git', '-C', str(VENDOR), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if revision != COMMIT or dirty:
        raise RuntimeError('External bot must be the fixed, unmodified upstream commit')

    loaded = []
    cfg0 = None
    for path in args.checkpoints:
        params, cfg = load(path)
        if cfg.players != 2 and not cfg.mixed_players:
            raise ValueError('Only 2-player checkpoints are supported')
        if cfg0 is not None and cfg.width != cfg0.width:
            raise ValueError('All checkpoints must use the same network shape')
        if cfg0 is not None and cfg.observation_version != cfg0.observation_version:
            raise ValueError('External-bot batches must use one observation schema')
        cfg0 = cfg if cfg0 is None else cfg0
        loaded.append(params)

    sources = ('env.py', 'network.py', 'valuebot_jax.py', 'evaluate_external.py')
    provenance = dict(
        bot='edwadli/splendor-ai ValueBuyBot (faithful JAX port)',
        repository='https://github.com/edwadli/splendor-ai', commit=revision,
        license='Apache-2.0', seed=args.seed, games=args.games,
        max_decisions=args.max_decisions, paired_seat_deals=True,
        policy_sampling='categorical', policy_inference_precision='fp32' if args.fp32 else 'bf16',
        external_bot_deterministic=True,
        models={name: dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                for name, path in zip(names, args.checkpoints)},
        card_source=env.DATA['source'], jax=jax.__version__,
        devices=[d.device_kind for d in jax.devices()],
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in sources},
        notes=[
            'All decisions and state transitions execute inside one JIT-compiled JAX loop.',
            'The original Python bot is used only for differential conformance tests.',
            'Public reserved-card identities are passed to both agents; hidden deck order is not.',
            'Normal-colors-first payment and first eligible noble fill choices absent upstream.',
            'Any JAX bot illegal action invalidates the formal result; no fallback is used.',
            'Elo is a direct external-anchor transformation, not a human or global rating.'])
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != provenance:
        raise ValueError('Protocol changed; use a new output directory')
    atomic_json(protocol_path, provenance)

    run = make_match(args.games, args.max_decisions, not args.fp32, cfg0.observation_version)
    compile_start = time.perf_counter()
    executable = run.lower(loaded[0], args.seed).compile()
    compile_seconds = time.perf_counter() - compile_start
    summary = []
    for index, (name, path, params) in enumerate(zip(names, args.checkpoints, loaded)):
        start = time.perf_counter()
        # Reuse identical paired deals across checkpoints for a lower-variance
        # learning-curve comparison. Policy randomness is also held fixed.
        result = jax.device_get(executable(params, args.seed))
        seconds = time.perf_counter() - start
        invalid = int(result['bot_invalid_actions'])
        if invalid:
            np.savez(out / f'invalid_match_{index:02d}.npz', **result)
            raise RuntimeError(f'{invalid} illegal JAX ValueBuyBot actions; result rejected')
        np.savez(out / f'match_{index:02d}.npz', **result)
        row = dict(name=name, checkpoint=path, seed=args.seed,
                   compile_seconds=compile_seconds if index == 0 else 0., seconds=seconds,
                   games_per_second=args.games / seconds, bot_invalid_actions=invalid,
                   **match_summary(result))
        row['external_anchor_elo'] = float(fit_elo(
            2, [(0, 1)], [result['score'][result['done']].sum()],
            [result['done'].sum()], anchor=1)[0])
        summary.append(row)
        atomic_json(out / 'results.json', summary)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
