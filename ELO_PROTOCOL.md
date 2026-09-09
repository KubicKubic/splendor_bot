# 模型强度验证协议

本次在原 `a100_2p_fast` 第 1000 次更新的完整状态上继续训练至第 6000 次更新。
学习率、优化器状态、规则、奖励、网络结构保持相同，以检验继续自博弈本身的作用。
新增 5000 次更新，每次 8192 × 128 个决策，总共新增 5,242,880,000 个决策。
训练中的 400 个玩家回合截断仍保留；不把它引入正常 Elo 比赛的胜负判定。

## 固定协议

- 在结果出来前固定 `configs/elo_long.json`：8 个按时间排列的模型，28 组循环赛，每组 2048 局。
- 本轮训练前的 `before_long_1000` 固定为 Elo 1000，其余值均为相对等级，不与人类 Elo 对应。
- 同一初始牌局打两次，交换两个模型的先后手。每组 1024 个初始局面，配对局使用独立策略抽样。
- 所有模型使用相同 BF16 推理精度、相同合法动作掩码；按照策略分布采样，温度 1，无额外搜索。
- 自然并列记 0.5 分。最多 4000 个小决策仍未结束的局单独报告并从 Elo 拟合中剔除；
  同时给出将未完成局全部判胜/全部判负时的得分界限，避免隐藏截断的影响。
- 检查点文件和评估代码 SHA-256、卡表来源、设备、种子及逐局结果均保存到评估目录。
- 使用 Bradley–Terry 逻辑胜率模型批量拟合 Elo（400 分差对应 10:1），避免在线 Elo 的比赛顺序依赖。
  每组比赛显式加 0.5 个虚拟胜局和 0.5 个虚拟负局，防止全胜/全败导致无限 Elo。
- Elo 的 95% 区间用 400 次初始局面成对 bootstrap：同局面换位的两场一起抽样。
  各组直接对战得分率的区间使用 1000 次同样的成对重采样。
- 同时输出相邻检查点的 Elo 差及区间、每组实测胜率和 Elo 预测残差。
  若存在互相克制或某一阶段退步，照实报告，不用单一排名掩盖。

另预先固定 `configs/elo_confirm.json`：最终第 6000 次更新与本轮开始前模型，
使用不同种子再对战 8192 局，作为额外确认；最终模型不从循环赛结果中择优挑选。

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/spld
# 继续训练（本次已实际启动，重复运行前检查现有进度）
bash train_a100_fast.sh --resume runs/a100_2p_fast/latest.npz --out runs/a100_2p_long --updates 6000 --lr 0.0001 --save-every 1000 --log-every 100
# 全部检查点生成后执行循环赛
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_long.json --out runs/elo_long
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.elo --manifest configs/elo_confirm.json --out runs/elo_confirm
# 图表
../generals_bot/.conda_envs/generals_bot/bin/python tools/plot_elo.py runs/elo_long
# 或在训练同时，自动等待并评估每个新检查点；最终写入完整循环赛目录
XLA_PYTHON_CLIENT_PREALLOCATE=false ../generals_bot/.conda_envs/generals_bot/bin/python tools/monitor_elo.py --manifest configs/elo_long.json --out runs/elo_long
```

同一协议重新执行会读取已经完成的逐局缓存；协议、模型文件哈希不同则拒绝混用缓存。
模型文件路径在 `spld` 目录下解析。最终 JSON 含 `ratings`、`adjacent_changes`、`pairwise_fit` 和 `matches`。
置信区间反映对战抽样误差，不包括训练 seed、对手池选择或数值精度的系统性差异。
