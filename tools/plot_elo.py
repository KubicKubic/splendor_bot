"""Standalone Elo curve + matchup heatmap from durable tournament artifacts."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    path = Path(args.directory)
    report = json.loads((path / 'ratings.json').read_text())
    ratings = report['ratings']
    labels = [r['name'] for r in ratings]
    elo = np.array([r['elo'] for r in ratings])
    ci = np.array([r['ci95'] for r in ratings])
    # Small bootstrap Monte Carlo errors can put a percentile just beyond the point estimate.
    errors = np.maximum(np.vstack((elo - ci[:, 0], ci[:, 1] - elo)), 0)
    x = np.array([r['training_decisions'] / 1e9 for r in ratings])
    matrix = np.full((len(ratings), len(ratings)), np.nan)
    np.fill_diagonal(matrix, .5)
    for m in report['matches']:
        a, b = labels.index(m['a']), labels.index(m['b'])
        matrix[a, b] = m['a_score_completed']
        matrix[b, a] = 1 - m['a_score_completed']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(x, elo, yerr=errors, fmt='o-', color='#1769aa', capsize=4, linewidth=1.8)
    ax.axhline(1000, color='gray', linestyle='--', linewidth=1)
    ax.set(xlabel='Cumulative training decisions (billions)', ylabel='Pool-relative Elo',
           title='Checkpoint strength (paired-deal bootstrap 95% CI)')
    ax.grid(alpha=.2)
    for xx, yy in zip(x, elo):
        ax.annotate(f'{yy:.0f}', (xx, yy), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
    im = axes[1].imshow(matrix, vmin=0, vmax=1, cmap='RdBu')
    short = ['initial' if s == 'initial_ppo_500' else 'fast500' if s == 'fast_500' else s.split('_')[-1] for s in labels]
    axes[1].set_xticks(range(len(labels)), short, rotation=45, ha='right')
    axes[1].set_yticks(range(len(labels)), short)
    axes[1].set(title='Row model score vs column model', xlabel='Opponent', ylabel='Model')
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, f'{100 * matrix[i, j]:.0f}%', ha='center', va='center', fontsize=8,
                         color='white' if abs(matrix[i, j] - .5) > .3 else 'black')
    fig.colorbar(im, ax=axes[1], shrink=.8, label='Win + 0.5 × draw, completed games')
    fig.savefig(path / 'elo_curve.png', dpi=180)
    fig.savefig(path / 'elo_curve.pdf')
    print(path / 'elo_curve.png')


if __name__ == '__main__':
    main()
