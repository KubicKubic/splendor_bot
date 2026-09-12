"""Seat-balanced multiplayer matches for two checkpoints with different schemas."""
import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

# Permit direct ``python tools/evaluate_mixed_rosters.py`` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from splendor import env, network
from splendor.train import load


def make_match(old_params, old_cfg, new_params, new_cfg, players, new_seats, games, seed):
    base = jnp.array([1] * new_seats + [0] * (players - new_seats), jnp.int32)
    roster = jnp.stack([jnp.roll(base, i % players) for i in range(games)])

    @jax.jit
    def run():
        key, deal_key = jax.random.split(jax.random.PRNGKey(seed))
        states = env.batch_reset(jax.random.split(deal_key, games), players)

        def body(carry):
            state, rng, steps = carry
            rng, action_key = jax.random.split(rng)
            mask = env.batch_mask(state)
            old_logits, _ = network.apply(old_params,
                env.batch_observe_for_version(state, old_cfg.observation_version), mask, True)
            new_logits, _ = network.apply(new_params,
                env.batch_observe_for_version(state, new_cfg.observation_version), mask, True)
            is_new = jnp.take_along_axis(roster, state.player[:, None], 1)[:, 0].astype(bool)
            logits = jnp.where(is_new[:, None], new_logits, old_logits)
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            return env.batch_step(state, action), rng, steps + 1

        states, _, steps = jax.lax.while_loop(
            lambda x: (~jnp.all(x[0].done)) & (x[2] < 4000), body,
            (states, key, jnp.int32(0)))
        winners = jax.vmap(env.winners)(states)[:, :players]
        shares = jnp.where(states.truncated[:, None], 1. / players,
                           winners / jnp.maximum(winners.sum(-1, keepdims=True), 1))
        return states.done, states.truncated, states.turns, shares, steps

    done, truncated, turns, shares, steps = map(np.asarray, jax.device_get(run()))
    is_new = np.asarray(roster, bool)
    return dict(players=players, new_seats=new_seats, old_seats=players - new_seats,
                games=games, completed=int(done.sum()),
                environment_truncations=int(truncated.sum()), mean_turns=float(turns.mean()),
                decisions_executed=int(steps),
                new_per_seat_win_share=float(shares[is_new].mean()),
                old_per_seat_win_share=float(shares[~is_new].mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', required=True); p.add_argument('--new', required=True)
    p.add_argument('--games', type=int, default=4096); p.add_argument('--seed', type=int, default=20260913)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    old, old_cfg = load(args.old); new, new_cfg = load(args.new)
    rows = []
    for players in (2, 3, 4):
        games = args.games - args.games % players
        for new_seats in range(1, players):
            row = make_match(old, old_cfg, new, new_cfg, players, new_seats, games,
                             args.seed + 100 * players + new_seats)
            row['per_seat_advantage'] = row['new_per_seat_win_share'] - row['old_per_seat_win_share']
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(old=args.old, new=args.new, seed=args.seed, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
