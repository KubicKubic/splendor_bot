"""Paired-seat checkpoint tournament and anchored batch Elo (Bradley–Terry).

Ratings belong only to the listed pool and protocol, not to any human scale.
Each deal is played twice, swapping seats. Draws score 1/2. Unfinished games
are reported and excluded from the likelihood; all-game bounds expose their
possible impact. Paired-deal bootstrap retains within-pair dependence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import env, network
from .train import load


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def make_match(games=2048, max_decisions=4000, bf16=True, observation_versions=None):
    if games < 4 or games % 2:
        raise ValueError('games must be even and >= 4 for paired deals')

    versions = observation_versions or (env.OBSERVATION_VERSION, env.OBSERVATION_VERSION)

    def match(params_a, params_b, seed):
        key, deal_key = jax.random.split(jax.random.PRNGKey(seed))
        # First half: A in seat 0; second half: same deals, A in seat 1.
        keys = jax.random.split(deal_key, games // 2)
        states = env.batch_reset(jnp.concatenate((keys, keys)), 2)
        seat_a = jnp.repeat(jnp.arange(2), games // 2)
        def body(carry):
            s, rng, count = carry
            rng, action_key = jax.random.split(rng)
            mask = env.batch_mask(s)
            obs_a = env.batch_observe_for_version(s, versions[0])
            obs_b = env.batch_observe_for_version(s, versions[1])
            logits_a, _ = network.apply(params_a, obs_a, mask, bf16)
            logits_b, _ = network.apply(params_b, obs_b, mask, bf16)
            logits = jnp.where((s.player == seat_a)[:, None], logits_a, logits_b)
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            return env.batch_step(s, action), rng, count + 1
        states, _, steps = jax.lax.while_loop(
            lambda carry: (~jnp.all(carry[0].done)) & (carry[2] < max_decisions),
            body, (states, key, jnp.int32(0)))
        winners = jax.vmap(env.winners)(states)
        own = jnp.take_along_axis(winners, seat_a[:, None], -1)[:, 0]
        # A public turn-cap result is defined by the game as a neutral draw.
        # `winners` is deliberately all false in that state, so it must not
        # fall through to the zero score used for a non-winning terminal seat.
        score = jnp.where(states.truncated, .5,
                          own / jnp.maximum(winners.sum(-1), 1))
        return dict(score=score, done=states.done, seat_a=seat_a, turns=states.turns,
                    truncated=states.truncated, decisions_executed=steps,
                    final_scores=states.scores[:, :2])
    return jax.jit(match)


def fit_elo(nmodels, pairs, wins, totals, anchor, anchor_elo=1000.):
    """Order-independent penalized binomial MLE; half a virtual win/loss per edge.

    The small, explicit smoothing prevents infinite ratings for swept matches.
    Newton solves in natural-logit units, then converts to conventional Elo.
    """
    pairs = np.asarray(pairs, int)
    wins, totals = np.asarray(wins, float), np.asarray(totals, float)
    free = [i for i in range(nmodels) if i != anchor]
    design = np.zeros((len(pairs), nmodels))
    design[np.arange(len(pairs)), pairs[:, 0]] = 1
    design[np.arange(len(pairs)), pairs[:, 1]] = -1
    design = design[:, free]
    if np.linalg.matrix_rank(design) < nmodels - 1:
        raise ValueError('Match graph must connect every model to the anchor')
    # Only apply smoothing to played edges.
    if np.any(totals <= 0):
        raise ValueError('Each match needs completed games')
    target, count = wins + .5, totals + 1.
    theta = np.zeros(nmodels - 1)
    for _ in range(60):
        z = design @ theta
        probability = 1. / (1. + np.exp(-np.clip(z, -30, 30)))
        grad = design.T @ (count * probability - target)
        hessian = design.T @ ((count * probability * (1 - probability))[:, None] * design)
        delta = np.linalg.solve(hessian + np.eye(len(free)) * 1e-9, grad)
        # Bound the step for highly separated pools.
        delta /= max(1., np.max(np.abs(delta)) / 2.)
        theta -= delta
        if np.max(np.abs(delta)) < 1e-9:
            break
    result = np.full(nmodels, anchor_elo)
    result[free] += theta * (400. / np.log(10.))
    return result


def match_summary(result):
    score = np.asarray(result['score'])
    done = np.asarray(result['done'], bool)
    truncated = np.asarray(result.get('truncated', np.zeros_like(done)), bool)
    n = len(score)
    completed = int(done.sum())
    if not completed:
        raise ValueError('No completed games; increase max_decisions')
    wins = int(((score == 1) & done).sum())
    draws = int(((score == .5) & done).sum())
    # Cluster ratio bootstrap accounts for two games sharing the initial deal.
    rng = np.random.default_rng(20260912)
    cluster_score = np.where(done, score, 0).reshape(2, n // 2).sum(0)
    cluster_count = done.reshape(2, n // 2).sum(0)
    indices = rng.integers(0, n // 2, (1000, n // 2))
    counts = cluster_count[indices].sum(1)
    boot = cluster_score[indices].sum(1) / np.maximum(counts, 1)
    ci = np.percentile(boot[counts > 0], [2.5, 97.5]).tolist()
    return dict(games=n, completed=completed, unfinished=n - completed,
        environment_truncations=int(truncated.sum()),
        a_wins=wins, draws=draws, b_wins=completed - wins - draws,
        a_score_completed=float(score[done].mean()),
        a_score_ci95=ci,
        a_score_all_bounds=[float(score[done].sum() / n), float((score[done].sum() + n - completed) / n)],
        a_score_by_seat=[float(score[done & (result['seat_a'] == seat)].mean())
                         if np.any(done & (result['seat_a'] == seat)) else None for seat in range(2)],
        mean_turns=float(np.asarray(result['turns'])[done].mean()),
        decisions_executed=int(result['decisions_executed']))


def bootstrap_ratings(nmodels, pairs, results, anchor, samples=400, seed=20260912):
    rng = np.random.default_rng(seed)
    # Row corresponds to one initial deal, containing both seat-swapped games.
    clusters = []
    for result in results:
        score, done = np.asarray(result['score']), np.asarray(result['done'])
        n = len(score) // 2
        clusters.append((np.where(done, score, 0).reshape(2, n).sum(0), done.reshape(2, n).sum(0)))
    ratings = []
    for _ in range(samples):
        wins, totals = [], []
        for cluster_score, cluster_done in clusters:
            index = rng.integers(0, len(cluster_score), len(cluster_score))
            wins.append(cluster_score[index].sum())
            totals.append(cluster_done[index].sum())
        ratings.append(fit_elo(nmodels, pairs, wins, totals, anchor))
    return np.asarray(ratings)


def analyze(models, pairs, results, anchor, bootstrap=400):
    totals = [int(r['done'].sum()) for r in results]
    wins = [float(r['score'][r['done']].sum()) for r in results]
    elo = fit_elo(len(models), pairs, wins, totals, anchor)
    boot = bootstrap_ratings(len(models), pairs, results, anchor, bootstrap)
    intervals = np.percentile(boot, [2.5, 97.5], axis=0)
    table = [dict(name=m['name'], elo=float(elo[i]), ci95=[float(intervals[0, i]), float(intervals[1, i])],
                  checkpoint=m['checkpoint'], training_decisions=m.get('training_decisions')) for i, m in enumerate(models)]
    changes = []
    for i in range(1, len(models)):
        delta = boot[:, i] - boot[:, i - 1]
        changes.append(dict(older=models[i - 1]['name'], newer=models[i]['name'],
            elo_change=float(elo[i] - elo[i - 1]), ci95=np.percentile(delta, [2.5, 97.5]).tolist()))
    residuals = []
    for (a, b), win, total in zip(pairs, wins, totals):
        expected = 1 / (1 + 10 ** ((elo[b] - elo[a]) / 400))
        residuals.append(dict(a=models[a]['name'], b=models[b]['name'], observed=win / total,
                              elo_predicted=float(expected), residual=float(win / total - expected)))
    return dict(anchor=models[anchor]['name'], anchor_elo=1000., ratings=table, adjacent_changes=changes,
        pairwise_fit=residuals, bootstrap_samples=bootstrap,
        notes=['Pool-relative batch Elo, not a human rating.',
               '95% percentile intervals bootstrap paired initial deals, not independent seat games.',
               'Unfinished games excluded; per-match all-game bounds reported.',
               '0.5 virtual win and 0.5 virtual loss per match for finite estimates.',
               'Elo compresses matchups; inspect pairwise residuals for nontransitivity.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON: models in chronological order, anchor name, seed, games')
    parser.add_argument('--out', required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--available-prefix', action='store_true', help='Evaluate the chronological prefix whose immutable checkpoints exist')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if args.available_prefix:
        available = []
        for model in manifest['models']:
            if not Path(model['checkpoint']).is_file():
                break
            available.append(model)
        manifest['models'] = available
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = manifest['models']
    names = [m['name'] for m in models]
    if len(names) != len(set(names)) or len(names) < 2:
        parser.error('Need at least 2 uniquely named models')
    anchor = names.index(manifest['anchor'])
    loaded, configs, hashes = [], [], []
    for m in models:
        p, cfg = load(m['checkpoint'])
        if cfg.players != 2 and not cfg.mixed_players:
            parser.error('This Elo protocol is for two-player games')
        loaded.append(p)
        configs.append(cfg)
        hashes.append(hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest())
    pairs, results, summaries = [], [], []
    # Fixed manifest order determines fixed match seeds; save it for reproducibility.
    protocol = dict(manifest=manifest, sha256=dict(zip(names, hashes)), card_source=env.DATA['source'],
                    jax=jax.__version__, devices=[d.device_kind for d in jax.devices()],
                    code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('env.py', 'network.py', 'elo.py')})
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing tournament protocol differs; select a new output directory')
    atomic_json(protocol_path, protocol)
    for a in range(len(models)):
        for b in range(a + 1, len(models)):
            path = out / f'match_{a:02d}_{b:02d}.npz'
            start = time.perf_counter()
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    result = {k: data[k] for k in data.files}
            else:
                seed = manifest['seed'] + 1009 * a + 9176 * b
                run = make_match(manifest.get('games', 2048), manifest.get('max_decisions', 4000),
                                 manifest.get('bf16', True),
                                 (configs[a].observation_version, configs[b].observation_version))
                result = jax.device_get(run(loaded[a], loaded[b], seed))
                temp = path.with_suffix('.tmp.npz')
                np.savez(temp, **result)
                os.replace(temp, path)
            summary = dict(a=names[a], b=names[b], seed=manifest['seed'] + 1009 * a + 9176 * b,
                           seconds=time.perf_counter() - start, **match_summary(result))
            pairs.append((a, b))
            results.append(result)
            summaries.append(summary)
            atomic_json(out / 'matches.json', summaries)
            print(json.dumps(summary), flush=True)
    report = analyze(models, pairs, results, anchor, args.bootstrap)
    report['matches'] = summaries
    atomic_json(out / 'ratings.json', report)
    print(json.dumps(dict(ratings=report['ratings'], adjacent_changes=report['adjacent_changes']), indent=2), flush=True)


if __name__ == '__main__':
    main()
