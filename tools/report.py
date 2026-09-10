"""Summarize durable logs and evaluations, without inventing missing results."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = {}
    for path in sorted((ROOT / 'runs').glob('a100_*/metrics.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if not rows:
            continue
        run = path.parent
        completed = sum(r['games'] for r in rows)
        report[run.name] = dict(config=json.loads((run / 'config.json').read_text()),
            updates=rows[-1]['update'], decisions=rows[-1]['decisions'],
            first_logged_update=rows[0]['update'],
            decisions_in_log=sum(r['decisions'] // r['update'] for r in rows),
            completed_games=completed,
            long_games=sum(r.get('long_games', r.get('timeouts', 0)) for r in rows),
            completed_game_mean_turns=(sum(r['mean_turns'] * r['games'] for r in rows) / completed
                                       if completed else None),
            latest_batch_mean_turns=rows[-1]['mean_turns'],
            measured_update_seconds=sum(r['seconds'] for r in rows),
            median_decisions_per_second=statistics.median(r['decisions_per_second'] for r in rows[5:] or rows),
            median_turns_per_second=statistics.median(r['turns_per_second'] for r in rows[5:] or rows),
            evaluations={p.stem: json.loads(p.read_text()) for p in sorted(run.glob('eval_*.json'))})
    (ROOT / 'runs/summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
