"""Seat-balanced held-out evaluation against random and a public-state heuristic."""
import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import env, network
from .train import load


def heuristic(s):
    """Buy valuable cards; otherwise take gems toward a cheap useful card."""
    ids = env.targets(s)
    safe = jnp.maximum(ids, 0)
    p = s.player
    cost = jnp.maximum(env.COST[safe] - s.bonuses[p], 0)
    missing = jnp.maximum(cost - s.gems[p, :5], 0).sum(-1) - s.gems[p, 5]
    missing = jnp.maximum(missing, 0)
    utility = env.POINT[safe] * 1.6 + 1.8 / (1 + s.bonuses[p, env.BONUS[safe]])
    target_rank = jnp.where(ids >= 0, utility - 1.1 * missing - .08 * cost.sum(-1), -1e6)
    target = jnp.argmax(target_rank)
    need = jnp.maximum(cost[target] - s.gems[p, :5], 0)
    take = jnp.minimum(env.TAKES[:, :5], need).sum(-1) + .12 * env.TAKES.sum(-1)
    take -= .5 * jnp.maximum(s.gems[p].sum() + env.TAKES.sum(-1) - 10, 0)
    score = jnp.full(env.N_ACTIONS, -100.)
    score = score.at[1:31].set(take)
    score = score.at[31:46].set(10. + utility - .05 * cost.sum(-1))
    score = score.at[46:58].set(.1 + .02 * env.POINT[jnp.maximum(s.market.ravel(), 0)])
    score = score.at[58:61].set(.01)
    # Prefer discarding resources not needed for the current target, preserve gold.
    score = score.at[61:66].set(s.gems[p, :5] - cost[target])
    score = score.at[66].set(-10.)
    score = score.at[67:72].set(1.)
    score = score.at[72:78].set(-jnp.arange(6, dtype=jnp.float32))
    return jnp.where(env.legal_mask(s), score, -1e9)


def make_evaluate(cfg, games, max_decisions, opponent, deterministic=False, players=None):
    players = cfg.players if players is None else players
    def evaluate(params, seed):
        key, kr = jax.random.split(jax.random.PRNGKey(seed))
        states = env.batch_reset(jax.random.split(kr, games), players)
        seats = jnp.arange(games) % players
        def body(_, carry):
            s, key = carry
            key, ka = jax.random.split(key)
            mask = env.batch_mask(s)
            logits, _ = network.apply(params, env.batch_observe(s), mask, cfg.bf16)
            other = jnp.where(mask, 0., -1e9) if opponent == 'random' else jax.vmap(heuristic)(s)
            own_actions = jnp.argmax(logits, -1) if deterministic else jax.random.categorical(ka, logits)
            # Random remains random even in deterministic-policy evaluations.
            other_actions = jax.random.categorical(jax.random.fold_in(ka, 1), other) if opponent == 'random' else jnp.argmax(other, -1)
            action = jnp.where(s.player == seats, own_actions, other_actions).astype(jnp.int32)
            return env.batch_step(s, action), key
        states, _ = jax.lax.fori_loop(0, max_decisions, body, (states, key))
        wins = jax.vmap(env.winners)(states)
        own = jnp.take_along_axis(wins, seats[:, None], -1)[:, 0]
        credit = own / jnp.maximum(wins.sum(-1), 1)
        return dict(done=states.done, credit=credit, sole_win=own & (wins.sum(-1) == 1),
                    tie=own & (wins.sum(-1) > 1), scores=states.scores, turns=states.turns, seats=seats)
    return jax.jit(evaluate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--games', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260909)
    parser.add_argument('--max-decisions', type=int, default=2000)
    parser.add_argument('--opponent', choices=['random', 'heuristic'], default='heuristic')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--untrained', action='store_true', help='Evaluate the identical seeded network before learning')
    parser.add_argument('--out')
    parser.add_argument('--players', type=int, choices=[2, 3, 4], help='Evaluation player count')
    args = parser.parse_args()
    params, cfg = load(args.checkpoint)
    if args.untrained:
        _, kp, _ = jax.random.split(jax.random.PRNGKey(cfg.seed), 3)
        params = network.init(kp, env.observe(env.reset(jax.random.PRNGKey(0), cfg.players)).shape[0], cfg.width)
    players = args.players or cfg.players
    if args.games % players:
        parser.error('games must divide evenly among player seats')
    run = make_evaluate(cfg, args.games, args.max_decisions, args.opponent, args.deterministic, players)
    start = time.perf_counter()
    result = jax.device_get(run(params, args.seed))
    credit = result['credit']
    done = result['done']
    report = dict(checkpoint=args.checkpoint, opponent=args.opponent, games=args.games, seed=args.seed,
        deterministic=args.deterministic, untrained=args.untrained, completed=int(done.sum()), truncated=int((~done).sum()),
        wins=int(result['sole_win'].sum()), ties=int(result['tie'].sum()),
        win_credit_all_games=float(credit.mean()), standard_error=float(credit.std(ddof=1) / np.sqrt(args.games)),
        completed_win_credit=float(credit[done].mean()) if done.any() else None,
        players=players, by_seat=[float(credit[result['seats'] == s].mean()) for s in range(players)],
        mean_turns=float(result['turns'][done].mean()) if done.any() else None,
        seconds=time.perf_counter() - start)
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
