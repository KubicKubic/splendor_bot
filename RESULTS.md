# 本地 A100 训练结果（2026-09-09）

已完成双人版训练。这里保留最初两阶段结果；继续训练和最终评估见
[`ELO_RESULTS.md`](ELO_RESULTS.md) 与 [`EXTERNAL_BOT.md`](EXTERNAL_BOT.md)。推荐推理权重：
[`runs/a100_2p_long/policy_006000.npz`](runs/a100_2p_long/policy_006000.npz)（约 707 KiB，180050 个参数）。
完整恢复检查点：[`runs/a100_2p_long/latest.npz`](runs/a100_2p_long/latest.npz)（约 12 MiB）。
环境支持 2–4 人，三人/四人规则已测试；当前交付权重训练人数为 2。

## 实际训练

设备为一张本地 NVIDIA A100-SXM4-80GB，JAX 0.4.35 / Optax 0.2.4。
共享两层 256 宽 MLP，78 个动作，4 个价值输出，363 维观测。所有环境和 PPO 运算在 GPU 上。

| 阶段 | 并行局数 / 精度 | 决策数 | 自然结束局数 | 训练墙钟时间 | 稳态决策/秒中位数 |
|---|---|---:|---:|---:|---:|
| 从随机权重自博弈 | 2048 / FP32 + TF32 | 131,072,000 | 2,176,837 | 76.43 秒 | 1,920,238 |
| 加载首阶段权重继续自博弈 | 8192 / BF16 | 1,048,576,000 | 16,483,636 | 222.50 秒 | 4,915,971 |

合计 **1,179,648,000 个决策、18,660,473 局**，训练墙钟约 299 秒，包含两次 JIT 编译。
吞吐包含环境步进、策略采样、GAE 以及 3 个 epoch 的 PPO 更新；去掉前 5 次更新后取中位数。
决策包含付款/弃币等小决策，第二阶段真实回合吞吐中位数为 4,263,337 回合/秒。
运行中一次显存采样为 2529 MiB，GPU 利用率 99%；这不是峰值显存测量。

首阶段训练 seed 41，学习率 3e-4；第二阶段 seed 41，学习率 1e-4，重新初始化 Adam 和环境。
两阶段均训练双人版、horizon 128、PPO epochs 3；完整参数和逐次日志见各阶段 `config.json` 和 `metrics.jsonl`。
达到 400 个真实回合仍未结束时训练按人工和局截断，两个阶段分别 164 / 2068 局，已单独计数。

## 独立评估

最终评估使用未用于训练或首轮评估的 seed **20260910**，先后手各 1024 局；策略按分布采样。
每局最多 2000 个决策，不把超时当作自然完赛。

| 对手 | 局数 | 独赢 | 并列 | 超时 | 全部局数中的胜率（并列折半） |
|---|---:|---:|---:|---:|---:|
| 买牌/拿币启发式 | 2048 | 1727 | 1 | 0 | **84.3506%** |
| 均匀随机合法动作 | 2048 | 2047 | 0 | 1 | **99.9512%** |

启发式对手评估标准误差约 0.80 个百分点，粗略 95% 区间为 82.8%–85.9%。
先手/后手胜率分别 82.91% / 85.79%。启发式实现见 `splendor/evaluate.py`，不代表人类高手。
对随机对手有 1 局未能在预算内结束，说明仍存在罕见拖延/循环，未将该局算作胜利。

首轮评估（seed 20260909，1024 局）中，同种子未训练网络对启发式为 0 胜，
首阶段模型胜率 68.9941%。最后一轮使用新种子，未据此搜索或挑选中途检查点。

## 验证

12 项测试通过：完整卡表与站点原模块一致；2/3/4 人合计 960 次操作逐项与站点 JavaScript 对照；
所有可选黄金支付组合一致；贵族/弃币/末轮/并列规则；隐藏信息隔离；多玩家 GAE；
32768 次 GPU 批量转移守恒；完整检查点恢复后下一次 PPO 更新逐数组一致；推理动作适配。
另实际通过 CLI 从首阶段检查点恢复到第 501 次更新，结果保存在 `runs/resume_check/`。

站点参考资源为 `ccbs.73a7734d.chunk.js`，SHA-256：
`9f284b04fd638f4e4bc800b2d074b86a6045806c73a84445a7caef9031fe3f6b`。

## 复查与继续训练

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/spld
# 所有测试
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m pytest tests -q
# 重现最终启发式评估
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.evaluate runs/a100_2p_fast/latest.npz --games 2048 --seed 20260910 --opponent heuristic
# 精确恢复最终训练，目标 1500 次更新
bash train_a100_fast.sh --resume runs/a100_2p_fast/latest.npz --out runs/a100_2p_fast --updates 1500 --lr 0.0001 --save-every 100 --log-every 100
```

机器可读汇总：[`runs/summary.json`](runs/summary.json)。
吞吐配置对比：[`runs/benchmark.json`](runs/benchmark.json)。
原始最终评估：[`eval_heuristic.json`](runs/a100_2p_fast/eval_heuristic.json)、
[`eval_random.json`](runs/a100_2p_fast/eval_random.json)。
