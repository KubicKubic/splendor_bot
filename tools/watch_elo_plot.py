"""Refresh one fixed Elo PNG only when a newly rated model joins the ladder."""
import argparse
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def model_signature(rows):
    """Identity of the rated checkpoints, deliberately excluding mutable Elo values."""
    return [[str(row['name']), int(row['training_decisions'])] for row in rows]


def state_path(output):
    return output.with_name('.' + output.stem + '.state.json')


def already_rendered(output, signature):
    try:
        return output.is_file() and json.loads(state_path(output).read_text()).get('models') == signature
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


def render_waiting(output):
    """Create a truthful initial image before any checkpoint has been rated."""
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor('#081b21'); ax.set_facecolor('#102b34')
    ax.set_title('Splendor JAX Mixed 2P/3P/4P — Live Elo', color='#e8f5f3', pad=12)
    ax.text(.5, .54, 'Waiting for the first evaluated checkpoint', transform=ax.transAxes,
            ha='center', va='center', color='#e8f5f3', fontsize=15)
    ax.text(.5, .46, 'The chart updates only when a new model finishes Elo evaluation.',
            transform=ax.transAxes, ha='center', va='center', color='#8eafb2', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color('#ffffff22')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name('.' + output.name + '.tmp')
    fig.savefig(temp, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    os.replace(temp, output)


def render(source, output):
    report = json.loads(source.read_text())
    ratings = report.get('ratings', [])
    if not ratings:
        return
    # `ratings.json` is atomically replaced by the ladder.  Still validate the
    # data here: a partially written or otherwise malformed external report
    # must not replace the last known-good image with a misleading plot.
    try:
        rows = sorted(ratings, key=lambda r: float(r['training_decisions']))
        x = np.array([float(r['training_decisions']) / 1e9 for r in rows])
        y = np.array([float(r['elo']) for r in rows])
        lo = np.array([float(r['ci95'][0]) for r in rows])
        hi = np.array([float(r['ci95'][1]) for r in rows])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f'invalid ratings report: {exc}') from exc
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(lo).all()
            and np.isfinite(hi).all() and np.all(np.diff(x) >= 0) and np.all(lo <= y)
            and np.all(y <= hi)):
        raise ValueError('ratings must have ordered finite decisions and enclosing confidence intervals')
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor('#081b21'); ax.set_facecolor('#102b34')
    ax.fill_between(x, lo, hi, color='#55d7d2', alpha=.18, label='95% paired-deal bootstrap CI')
    ax.plot(x, y, color='#55d7d2', marker='o', markersize=4, linewidth=2.2,
            label='Sparse ladder Elo')
    ax.axhline(1000, color='#f1c75b', linewidth=1, alpha=.7,
               label='First checkpoint anchor = 1000')
    latest = rows[-1]
    # Give both the CI band and the endpoint label room.  The prior fixed
    # margins cut off the label once a long run approached the upper-right
    # corner of the chart.
    x_span = max(float(x[-1] - x[0]), 1.)
    y_min, y_max = min(float(lo.min()), 1000.), max(float(hi.max()), 1000.)
    y_span = max(y_max - y_min, 1.)
    ax.set_xlim(float(x[0]) - .05 * x_span, float(x[-1]) + .12 * x_span)
    ax.set_ylim(y_min - .06 * y_span, y_max + .16 * y_span)
    ax.annotate(f"{latest['elo']:.1f}\nupdate {latest['name'].split('_')[-1]}",
                (x[-1], y[-1]), xytext=(-10, -12), textcoords='offset points',
                ha='right', va='top', color='#e8f5f3', fontsize=9,
                bbox=dict(boxstyle='round,pad=.2', fc='#102b34', ec='none', alpha=.88))
    ax.set_title('Splendor JAX Mixed 2P/3P/4P — Live Elo', color='#e8f5f3', pad=12)
    ax.set_xlabel('Training decisions (billions)', color='#8eafb2')
    ax.set_ylabel('Pool-relative Elo', color='#8eafb2')
    ax.grid(color='white', alpha=.08)
    ax.tick_params(colors='#8eafb2')
    for spine in ax.spines.values(): spine.set_color('#ffffff22')
    legend = ax.legend(frameon=False, loc='best')
    for text in legend.get_texts(): text.set_color('#b9ced0')
    fig.tight_layout(rect=(0, .025, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name('.' + output.name + '.tmp')
    fig.savefig(temp, format='png', facecolor=fig.get_facecolor())
    plt.close(fig)
    os.replace(temp, output)
    state = state_path(output)
    state_temp = state.with_name('.' + state.name + '.tmp')
    state_temp.write_text(json.dumps({'models': model_signature(rows)}, indent=2) + '\n')
    os.replace(state_temp, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/mixed_longrun_ladder/ratings.json')
    parser.add_argument('--output', default='runs/mixed_longrun_ladder/elo_live.png')
    parser.add_argument('--interval', type=float, default=30.,
                        help='poll period in seconds; no redraw occurs without a new rated model')
    args = parser.parse_args(); source, output = Path(args.source), Path(args.output)
    if not output.is_file():
        render_waiting(output)
        print(f'created waiting image {output}', flush=True)
    previous = None
    while True:
        try:
            stamp = (source.stat().st_mtime_ns, source.stat().st_size)
            if stamp != previous:
                report = json.loads(source.read_text())
                rows = report.get('ratings', [])
                signature = model_signature(rows) if rows else []
                if rows and not already_rendered(output, signature):
                    render(source, output)
                    print(f'updated {output} for {rows[-1]["name"]}', flush=True)
                previous = stamp
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
