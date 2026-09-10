"""Persistent O(N) checkpoint Elo ladder plus 2/3/4-player diagnostics."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from splendor import env
from splendor.elo import analyze, atomic_json, make_match, match_summary
from splendor.evaluate import make_evaluate
from splendor.train import load


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eval_summary(result, checkpoint, players, seed):
    result = jax.device_get(result)
    credit, done, seats = result['credit'], result['done'], result['seats']
    return dict(checkpoint=str(checkpoint), players=players, games=len(done), seed=seed,
                completed=int(done.sum()), unfinished=int((~done).sum()),
                wins=int(result['sole_win'].sum()), ties=int(result['tie'].sum()),
                score_all=float(credit.mean()),
                score_completed=float(credit[done].mean()) if done.any() else None,
                by_seat=[float(credit[seats == s].mean()) for s in range(players)],
                mean_turns=float(result['turns'][done].mean()) if done.any() else None)


def automatic_config(run_dir, training):
    """Build a reproducible ladder protocol from the run's saved config."""
    if not training.get('mixed_players') and training.get('players') != 2:
        raise ValueError('Automatic Elo supports mixed-player or fixed 2P training')
    immutable_training = {key: value for key, value in training.items()
                          if key not in ('updates', 'save_every', 'log_every')}
    return dict(
        run_dir=str(run_dir),
        model_prefix=training.get('architecture', 'mlp'),
        first_update=training['save_every'],
        last_update=training['updates'],
        interval=training['save_every'],
        lag=4,
        games=1024,
        heuristic_games=384,
        max_decisions=4000,
        seed=20260911,
        bf16=training['bf16'],
        bootstrap=300,
        poll_seconds=10,
        expected_training=immutable_training,
        automatic=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config')
    source.add_argument('--run-dir', help='derive protocol from RUN_DIR/config.json when it appears')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
    else:
        config_path = Path(args.run_dir) / 'config.json'
        while not config_path.is_file():
            time.sleep(1)
        cfg = automatic_config(args.run_dir, json.loads(config_path.read_text()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    updates = list(range(cfg['first_update'], cfg['last_update'] + 1, cfg['interval']))
    if not updates:
        updates = [cfg['last_update']]
    elif updates[-1] != cfg['last_update']:
        updates.append(cfg['last_update'])
    protocol_config = ({key: value for key, value in cfg.items() if key != 'last_update'}
                       if cfg.get('automatic') else cfg)
    fixed_protocol = dict(config=protocol_config, algorithm='sparse Bradley-Terry checkpoint ladder',
        edges='previous checkpoint plus lag checkpoints back', anchor_update=updates[0],
        notes=['Match count grows linearly rather than as a full round robin.',
               'Each 2P match uses identical deals with swapped seats.',
               '2P/3P/4P heuristic diagnostics reuse fixed seeds across checkpoints.',
               'Unfinished games are excluded from Elo and separately reported.'],
        code_sha256={name: sha256(Path(__file__).parents[1] / 'splendor' / name)
                     for name in ('env.py', 'network.py', 'elo.py', 'evaluate.py')},
        jax=jax.__version__, devices=[d.device_kind for d in jax.devices()],
        card_source=env.DATA['source'])
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != fixed_protocol:
        raise ValueError('Ladder protocol changed; choose a new output directory')
    atomic_json(protocol_path, fixed_protocol)

    match_fn = make_match(cfg['games'], cfg['max_decisions'], cfg['bf16'])
    diagnostic_fns = {}
    params_by_update = {}
    models, pairs, results, summaries, diagnostics = [], [], [], [], []
    seen_edges = set()
    cursor = 0
    while cursor < len(updates):
        update = updates[cursor]
        checkpoint = Path(cfg['run_dir']) / f'policy_{update:06d}.npz'
        if not checkpoint.is_file():
            time.sleep(cfg['poll_seconds'])
            continue
        params, train_cfg = load(checkpoint)
        expected = cfg.get('expected_training', {})
        actual = asdict(train_cfg)
        mismatches = {key: (actual.get(key), value) for key, value in expected.items()
                      if actual.get(key) != value}
        if (not train_cfg.mixed_players and train_cfg.players != 2) or train_cfg.gamma != 1. or mismatches:
            raise ValueError(f'Unexpected mixed-training configuration in {checkpoint}: {mismatches}')
        params_by_update[update] = params
        models.append(dict(name=f'{cfg.get("model_prefix", "mixed")}_{update:06d}', checkpoint=str(checkpoint),
                           training_decisions=cfg.get('decision_offset', 0) +
                                              update * train_cfg.envs * train_cfg.horizon,
                           sha256=sha256(checkpoint)))

        diagnostic_players = (2, 3, 4) if train_cfg.mixed_players else (2,)
        for players in diagnostic_players:
            if players not in diagnostic_fns:
                diagnostic_fns[players] = make_evaluate(
                    train_cfg, cfg['heuristic_games'], cfg['max_decisions'],
                    'heuristic', False, players)
            seed = cfg['seed'] + players * 100003
            result = diagnostic_fns[players](params, seed)
            row = eval_summary(result, checkpoint, players, seed)
            row['update'] = update
            diagnostics.append(row)
        atomic_json(out / 'diagnostics.json', diagnostics)

        candidate_edges = []
        if cursor > 0:
            candidate_edges.append((cursor - 1, cursor))
        if cursor >= cfg['lag']:
            candidate_edges.append((cursor - cfg['lag'], cursor))
        for a, b in candidate_edges:
            edge = (a, b)
            if edge in seen_edges:
                continue
            ua, ub = updates[a], updates[b]
            path = out / f'match_{ua:06d}_{ub:06d}.npz'
            seed = cfg['seed'] + 1009 * a + 9176 * b
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    result = {k: data[k] for k in data.files}
            else:
                result = jax.device_get(match_fn(params_by_update[ua], params_by_update[ub], seed))
                temp = path.with_suffix('.tmp.npz')
                np.savez(temp, **result)
                temp.replace(path)
            pairs.append(edge); results.append(result); seen_edges.add(edge)
            summaries.append(dict(a=models[a]['name'], b=models[b]['name'], seed=seed,
                                  **match_summary(result)))
            atomic_json(out / 'matches.json', summaries)

        if len(models) >= 2:
            report = analyze(models, pairs, results, anchor=0, bootstrap=cfg['bootstrap'])
            report['checkpoint_sha256'] = {m['name']: m['sha256'] for m in models}
            report['diagnostics'] = diagnostics
            report['sparse_edges'] = summaries
            atomic_json(out / 'ratings.json', report)
            newest = report['ratings'][-1]
            print(json.dumps(dict(update=update, elo=newest['elo'], ci95=newest['ci95'],
                                  edges=len(pairs),
                                  diagnostics=diagnostics[-len(diagnostic_players):])), flush=True)
        else:
            print(json.dumps(dict(update=update, anchor=True,
                                  diagnostics=diagnostics[-len(diagnostic_players):])), flush=True)
        cursor += 1
    atomic_json(out / 'complete.json', dict(completed=True, last_update=updates[-1],
                                             checkpoints=len(updates), edges=len(pairs)))


if __name__ == '__main__':
    main()
