# 继续训练与模型 Elo 评估结果

> 此后又继续到第 8000 次更新（累计 8.520B 决策）。新模型相对本页第 6000 模型提升
> +22.9 Elo，且新增阶段自然结束局平均 56.476 个玩家回合；详见
> [TRAINING_LENGTH.md](TRAINING_LENGTH.md)。当前推荐权重为
> [policy_008000.npz](runs/a100_2p_long/policy_008000.npz)。

本轮起点固定为 Elo 1000；最终模型为 **1168.9**，95% 区间 **[1161.1, 1178.7]**。

按预先固定方案，从第 1000 次更新继续到第 6000 次更新，新增 5,242,880,000 个决策，完成 75,729,241 局；训练循环计时约 18.85 分钟。计时包含编译及并行评估带来的资源竞争，不包含所有文件保存开销。
稳态完整 PPO 吞吐中位数为 4,933,240 决策/秒；400 回合人工截断共 8,504 局。

## 同一对手池的 Elo 曲线

8 个模型完整循环赛，28 组比赛，每组 2048 局，共 57,344 局，未完成 7 局。所有模型以 BF16、温度 1 采样，同一初始牌局交换先后手，平局折半。

| 检查点 | 累计训练决策（十亿） | Elo | 95% 区间 |
|---|---:|---:|---|
| initial_ppo_500 | 0.131 | 832.4 | [823.7, 841.6] |
| fast_500 | 0.655 | 958.2 | [950.1, 966.1] |
| before_long_1000 | 1.180 | 1000.0 | [1000.0, 1000.0] |
| long_2000 | 2.228 | 1066.6 | [1058.6, 1074.2] |
| long_3000 | 3.277 | 1106.3 | [1098.2, 1114.5] |
| long_4000 | 4.325 | 1134.3 | [1127.0, 1143.0] |
| long_5000 | 5.374 | 1153.2 | [1143.9, 1160.3] |
| long_6000 | 6.423 | 1168.9 | [1161.1, 1178.7] |

![Elo 曲线及互战矩阵](runs/elo_long/elo_curve.png)

## 是否逐渐变强

本轮各保存阶段的 Elo 点估计持续上升；5 次相邻比较中，5 次提升的成对 bootstrap 95% 区间完全高于 0。

| 比较 | Elo 变化 | 95% 区间 |
|---|---:|---|
| before_long_1000 → long_2000 | +66.6 | [+58.6, +74.2] |
| long_2000 → long_3000 | +39.6 | [+32.1, +48.6] |
| long_3000 → long_4000 | +28.0 | [+20.0, +35.4] |
| long_4000 → long_5000 | +18.9 | [+10.7, +26.6] |
| long_5000 → long_6000 | +15.7 | [+7.2, +24.6] |

这些区间是逐项 95% 区间，未做多重比较校正；是否整体变强还用以下独立对战确认。

## 新种子独立确认

最终第 6000 次更新与本轮起点直接对战 8,192 局（配置 seed 20261001，实际派生对战 seed 20270177）：

- 最终模型独赢 5,926、输 2,222、并列 43、未完成 1。
- 最终模型得分率（平局折半、仅已完成局）**72.61%**，成对 bootstrap 95% 区间 **[71.59%, 73.59%]**。
- 若将未完成局全部判负/判胜，全部局数得分率界限为 [72.60%, 72.61%]。

最终检查点与确认协议在结果出来前固定，未从评估结果中挑选最强中间版本。

## 解释边界

单一 Elo 对各组实测得分率的最大偏差为 2.28 个百分点。Elo 是这组模型和此推理协议下的相对强度，不是人类 Elo，也不说明对所有外部对手都更强。区间只反映对战抽样误差，本次只有一个训练 seed；长期继续训练不保证永远上升。

完整方法见 [ELO_PROTOCOL.md](ELO_PROTOCOL.md)。加入外部 bot 差分与批量对战后，全套 **20 项测试**通过。

## 文件与复现

- 最终推理权重：[policy_006000.npz](runs/a100_2p_long/policy_006000.npz)。
- 可完整恢复的检查点：[latest.npz](runs/a100_2p_long/latest.npz)。
- 循环赛评分：[ratings.json](runs/elo_long/ratings.json)；逐组及逐局结果在同目录。
- 独立确认：[ratings.json](runs/elo_confirm/ratings.json)。
- 图表：[PNG](runs/elo_long/elo_curve.png)、[PDF](runs/elo_long/elo_curve.pdf)。

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/spld
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_long.json --out runs/elo_long
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_confirm.json --out runs/elo_confirm
# 继续同一训练轨迹至第 7000 次更新
bash train_a100_fast.sh --resume runs/a100_2p_long/latest.npz --out runs/a100_2p_long --updates 7000 --lr 0.0001 --save-every 1000 --log-every 100
```
