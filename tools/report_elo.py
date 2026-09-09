"""Generate the final evidence summary after fixed tournament and confirmation."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    tournament = json.loads((ROOT / 'runs/elo_long/ratings.json').read_text())
    confirmation = json.loads((ROOT / 'runs/elo_confirm/ratings.json').read_text())
    rows = [json.loads(line) for line in (ROOT / 'runs/a100_2p_long/metrics.jsonl').read_text().splitlines() if line]
    cfg = json.loads((ROOT / 'runs/a100_2p_long/config.json').read_text())
    games = sum(row['games'] for row in rows)
    steps = len(rows) * cfg['envs'] * cfg['horizon']
    seconds = sum(row['seconds'] for row in rows)
    matches = tournament['matches']
    total_matches = sum(m['games'] for m in matches)
    unfinished = sum(m['unfinished'] for m in matches)
    last = tournament['ratings'][-1]
    c = confirmation['matches'][0]
    new_score = 1 - c['a_score_completed']
    score_ci = [1 - c['a_score_ci95'][1], 1 - c['a_score_ci95'][0]]
    changes = tournament['adjacent_changes'][2:]
    positive = sum(change['ci95'][0] > 0 for change in changes)
    monotonic = all(change['elo_change'] > 0 for change in changes)
    residual = max(abs(row['residual']) for row in tournament['pairwise_fit'])
    lines = ['# 继续训练与模型 Elo 评估结果', '',
        f'本轮起点固定为 Elo 1000；最终模型为 **{last["elo"]:.1f}**，95% 区间 '
        f'**[{last["ci95"][0]:.1f}, {last["ci95"][1]:.1f}]**。', '',
        f'按预先固定方案，从第 {rows[0]["update"] - 1} 次更新继续到第 {rows[-1]["update"]} 次更新，'
        f'新增 {steps:,} 个决策，完成 {games:,} 局；训练循环计时约 {seconds / 60:.2f} 分钟。'
        '计时包含编译及并行评估带来的资源竞争，不包含所有文件保存开销。',
        f'稳态完整 PPO 吞吐中位数为 {statistics.median(row["decisions_per_second"] for row in rows[5:]):,.0f} 决策/秒；'
        f'400 回合人工截断共 {sum(row["timeouts"] for row in rows):,} 局。', '',
        '## 同一对手池的 Elo 曲线', '',
        f'{len(tournament["ratings"])} 个模型完整循环赛，{len(matches)} 组比赛，每组 2048 局，'
        f'共 {total_matches:,} 局，未完成 {unfinished} 局。所有模型以 BF16、温度 1 采样，'
        '同一初始牌局交换先后手，平局折半。', '',
        '| 检查点 | 累计训练决策（十亿） | Elo | 95% 区间 |',
        '|---|---:|---:|---|']
    for r in tournament['ratings']:
        lines.append(f'| {r["name"]} | {r["training_decisions"] / 1e9:.3f} | {r["elo"]:.1f} | '
                     f'[{r["ci95"][0]:.1f}, {r["ci95"][1]:.1f}] |')
    lines += ['', '![Elo 曲线及互战矩阵](runs/elo_long/elo_curve.png)', '',
        '## 是否逐渐变强', '',
        f'本轮各保存阶段的 Elo 点估计{"持续上升" if monotonic else "并非严格单调"}；'
        f'{len(changes)} 次相邻比较中，{positive} 次提升的成对 bootstrap 95% 区间完全高于 0。', '',
        '| 比较 | Elo 变化 | 95% 区间 |', '|---|---:|---|']
    for change in changes:
        lines.append(f'| {change["older"]} → {change["newer"]} | {change["elo_change"]:+.1f} | '
                     f'[{change["ci95"][0]:+.1f}, {change["ci95"][1]:+.1f}] |')
    lines += ['', '这些区间是逐项 95% 区间，未做多重比较校正；是否整体变强还用以下独立对战确认。', '',
        '## 新种子独立确认', '',
        f'最终第 6000 次更新与本轮起点直接对战 {c["games"]:,} 局（配置 seed 20261001，'
        f'实际派生对战 seed {c["seed"]}）：', '',
        f'- 最终模型独赢 {c["b_wins"]:,}、输 {c["a_wins"]:,}、并列 {c["draws"]:,}、未完成 {c["unfinished"]:,}。',
        f'- 最终模型得分率（平局折半、仅已完成局）**{100 * new_score:.2f}%**，'
        f'成对 bootstrap 95% 区间 **[{100 * score_ci[0]:.2f}%, {100 * score_ci[1]:.2f}%]**。',
        f'- 若将未完成局全部判负/判胜，全部局数得分率界限为 '
        f'[{100 * (1 - c["a_score_all_bounds"][1]):.2f}%, {100 * (1 - c["a_score_all_bounds"][0]):.2f}%]。', '',
        '最终检查点与确认协议在结果出来前固定，未从评估结果中挑选最强中间版本。', '',
        '## 解释边界', '',
        f'单一 Elo 对各组实测得分率的最大偏差为 {100 * residual:.2f} 个百分点。'
        'Elo 是这组模型和此推理协议下的相对强度，不是人类 Elo，也不说明对所有外部对手都更强。'
        '区间只反映对战抽样误差，本次只有一个训练 seed；长期继续训练不保证永远上升。', '',
        '完整方法见 [ELO_PROTOCOL.md](ELO_PROTOCOL.md)。17 项规则、训练恢复和 Elo 测试通过。', '',
        '## 文件与复现', '',
        '- 最终推理权重：[policy_006000.npz](runs/a100_2p_long/policy_006000.npz)。',
        '- 可完整恢复的检查点：[latest.npz](runs/a100_2p_long/latest.npz)。',
        '- 循环赛评分：[ratings.json](runs/elo_long/ratings.json)；逐组及逐局结果在同目录。',
        '- 独立确认：[ratings.json](runs/elo_confirm/ratings.json)。',
        '- 图表：[PNG](runs/elo_long/elo_curve.png)、[PDF](runs/elo_long/elo_curve.pdf)。', '',
        '```bash', 'cd /mnt/pfs/guoyuchong/guoyuchong/spld',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_long.json --out runs/elo_long',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_confirm.json --out runs/elo_confirm',
        '# 继续同一训练轨迹至第 7000 次更新',
        'bash train_a100_fast.sh --resume runs/a100_2p_long/latest.npz --out runs/a100_2p_long --updates 7000 --lr 0.0001 --save-every 1000 --log-every 100', '```', '']
    (ROOT / 'ELO_RESULTS.md').write_text('\n'.join(lines))
    print(ROOT / 'ELO_RESULTS.md')


if __name__ == '__main__':
    main()
