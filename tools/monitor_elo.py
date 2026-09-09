"""Evaluate immutable checkpoints as they appear, finish with the full pool.

Launch beside training. Every prefix has its own immutable protocol directory.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--timeout-seconds', type=float, default=3600.)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error('poll-seconds must be 1..60')
    manifest = json.loads(Path(args.manifest).read_text())
    total = len(manifest['models'])
    out = Path(args.out)
    started, previous = time.monotonic(), 0
    while time.monotonic() - started < args.timeout_seconds:
        count = 0
        for model in manifest['models']:
            if not Path(model['checkpoint']).is_file():
                break
            count += 1
        anchor_ready = manifest['anchor'] in [m['name'] for m in manifest['models'][:count]]
        if count >= 2 and count != previous and anchor_ready:
            target = out if count == total else out.with_name(out.name + f'_prefix_{count:02d}')
            print(json.dumps(dict(available=count, total=total, evaluating=str(target))), flush=True)
            subprocess.run([sys.executable, '-m', 'splendor.elo', '--manifest', args.manifest,
                '--available-prefix', '--out', str(target)], check=True)
            subprocess.run([sys.executable, 'tools/plot_elo.py', str(target)], check=True)
            previous = count
        if count == total:
            print(json.dumps(dict(finished=True, tournament=str(out))), flush=True)
            return
        time.sleep(args.poll_seconds)
    raise TimeoutError('Timed out waiting for checkpoints; existing results are preserved')


if __name__ == '__main__':
    main()
