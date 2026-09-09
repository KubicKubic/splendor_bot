# Splendor JAX

纯 JAX 环境与 PPO 自博弈，模型、采样、环境步进、GAE、优化全部在单张 GPU 上运行。
本地训练已完成；最终权重和对局长度跟踪见 [TRAINING_LENGTH.md](TRAINING_LENGTH.md)，
此前的训练吞吐与完整 Elo 曲线见 [ELO_RESULTS.md](ELO_RESULTS.md)，
Elo 循环赛协议见 [ELO_PROTOCOL.md](ELO_PROTOCOL.md)。公开 `ValueBuyBot` 的纯 JAX 重写、
差分校验和外部强度对照见 [EXTERNAL_BOT.md](EXTERNAL_BOT.md)。
牌表和规则对齐 [game.hullqin.cn/ccbs](https://game.hullqin.cn/ccbs) 的
`ccbs.73a7734d.chunk.js`（2026-09-09 抓取；站点公告 2026-08-30 更新）。

## 使用

当前机器可直接使用已有 JAX 0.4.35 / Optax 0.2.4 环境：

```bash
cd /mnt/pfs/guoyuchong/guoyuchong/spld
bash train_a100.sh --out runs/new_run
# A100 80GB 实测更高吞吐：8192 并行局 + BF16，约 498 万决策/秒。
bash train_a100_fast.sh --out runs/fast_run
# 约 1M 参数的统一 2P/3P/4P 模型（人数近似等量，gamma=1）：
bash train_a100_mixed.sh --out runs/mixed_run --updates 2000
# 实时只读 Web 面板（训练进度、稳定性、分人数轮次、Elo 与固定对手胜率）：
../generals_bot/.conda_envs/generals_bot/bin/python tools/serve_training_monitor.py \
  --host 0.0.0.0 --port 8765
# 固定路径的 Elo PNG；仅在新模型完成 Elo 评分后原子更新：
../generals_bot/.conda_envs/generals_bot/bin/python tools/watch_elo_plot.py
# 2–4 人都可训练；默认 2 人，2048 并行局，256 隐层，500 次 PPO 更新。
bash train_a100.sh --players 4 --out runs/four_player
../generals_bot/.conda_envs/generals_bot/bin/python -m pytest tests -q
../generals_bot/.conda_envs/generals_bot/bin/python -m splendor.evaluate runs/new_run/latest.npz --opponent heuristic --out runs/new_run/eval_heuristic.json
```

新机器先在独立 Python 环境中 `pip install -r requirements.txt`，然后
`SPLD_PYTHON=/path/to/python bash train_a100.sh ...`。启动器要求识别到单张 A100；
`--allow-cpu` 只用于显式 CPU 调试。默认 FP32 参数和 TF32 矩阵乘；`--bf16` 可测混合精度。
不需要也不使用 Q 调度器。

`latest.npz` 保存模型、Adam、环境状态及 RNG，可以完整恢复；`policy_*.npz` 只保存策略。
恢复时需匹配原训练参数，除总更新数、日志及保存间隔：

```bash
bash train_a100.sh --resume runs/new_run/latest.npz --out runs/new_run --updates 1000
# 如果原模型来自 fast 启动器，恢复时仍使用 fast 启动器。
bash train_a100_fast.sh --resume runs/fast_run/latest.npz --out runs/fast_run --updates 1000
# 修改精度/批量时只加载权重，并重新初始化优化器、采样器：
bash train_a100_fast.sh --warm-start runs/new_run/latest.npz --out runs/finetune --lr 0.0001
```

原配置还修改过 `--lr`、`--players` 等参数时，恢复命令也需带上这些参数。
`tests/test_training.py` 已验证保存、恢复后的下一次 PPO 更新逐数组完全一致。

推理接口支持直接传 JAX `State`，也可接受站点前端的 `view` JSON 并给出完整操作：

```python
import jax
from splendor.agent import Agent

agent = Agent("runs/a100_2p_long/policy_008000.npz")
action_id = agent.act(state, jax.random.PRNGKey(0))
site_action = agent.plan_view(view_dict, jax.random.PRNGKey(1))
```

`python -m splendor.agent CHECKPOINT --view view.json` 会输出 JSON。
卡 ID 使用站点内部零基编号，公开卡在输入 `bankCard` 中仍按站点的一基编号。
推理适配器会把付款、弃币的小决策合成为站点一次操作，不执行联网落子。

## 规则与数据

- `data/cards.json`：站点原始顺序的 90 张发展卡（40/30/20）和 10 张贵族，附 URL 与 SHA-256。
  颜色顺序为白、蓝、绿、红、黑，黄金下标 5。卡 ID 为站点内部的零基编号。
- 2/3/4 人分别使用每色 4/5/7 枚，黄金 5 枚；每层公开 4 张，贵族数为人数加 1。
- 可拿 1–3 个不同色宝石，或在银行原有至少 4 个时拿 2 个同色；可自由 pass。
- 可预定公开牌或指定层随机暗牌；最多 3 张，有黄金则获得 1 枚，否则仍可预定。
- 完整保留额外黄金付款选择；每回合最多 1 个贵族，多候选时自选，先选贵族再弃币到 10。
- 有玩家达到 15 分后完成当前轮；高分胜，同分发展卡少者胜，仍相同则并列。
- 牌堆预洗牌后依次抽取与站点每次从剩余牌均匀抽取在分布上等价；并非虚构固定牌序。
- 对手预定牌仅公开层级数量，牌堆仅公开剩余张数；模型不读取隐藏牌身份或未来牌序。
- 站点显示计时不参与计分，未纳入策略输入。挂机 UI、房间网络协议不属于环境。

78 维离散动作：pass 0；取币 1–30；买牌 31–45；预定 46–60；弃币 61–66；
贵族 67–71；当前颜色的黄金支付数量 72–77。仅在付款有选择时进入付款阶段，
逐色选择 0–5 枚黄金；弃币逐枚选择。这些小决策保持同一个玩家，完成后才推进真实回合。
`env.step` 要求合法动作，调用者必须应用 `legal_mask`。

## 学习和评估

共享 MLP 策略，价值头预测各玩家回报；观测以当前玩家为第一位，绝对座位额外编码。
`train_a100_mixed.sh` 在同一 GPU batch 中近似等量覆盖 2/3/4 人，每局分别应用 HullQin
对应的 4/5/7 枚彩色宝石、3/4/5 个贵族、实际人数轮末与胜者规则，并分别记录终局长度。
GAE 保留每个座位独立的回报，回合内小决策折扣为 1，真正回合结束折扣为 0.997。
奖励为终局零和胜负（并列均分）加势函数差分塑形，势函数来自公开分数和已购卡数量。
塑形在终局归零，避免永久累积买卡奖励。模型为前馈网络，尚未加入历史记忆或搜索。

训练 400 个真实回合未结束的局按人工和局截断，并单独记录 `timeouts`；这是训练限制，
不是站点规则。评估默认最多 2000 个小决策，截断数单独报告，不冒充自然完赛。
对手包括均匀随机合法策略和可复查的买牌/拿币启发式，各座位等量，评估 seed 与训练不同。
与基线的胜率不能代表与人类高手的强度。
当前交付训练检查点为双人版；3/4 人环境已通过规则对照测试，可用 `--players` 另行训练。

`metrics.jsonl` 中 `decisions_per_second` 包含环境、网络和 PPO 优化，`turns_per_second`
按真实回合计数。首条包含编译时间，稳态测速应排除首条。

## 验证与来源审计

`tests/hullqin_oracle.cjs` 在离线、无 DOM 的隔离上下文中执行下载的原规则模块。
Python 测试对 2/3/4 人共 960 次实际操作逐项比较宝石、牌归属、贵族、等待阶段和胜者，
并检查牌/币守恒、隐藏信息、贵族与弃币顺序及多玩家 GAE。需要 Node.js。
额外覆盖黄金支付的全部合法组合、32768 次 GPU 批量转移的守恒、检查点精确恢复及推理适配器。
`tools/fetch_hullqin.py` 可重新获取固定版本并提取卡表，不执行远端 JavaScript。
参考文件是站点公开资源的审计快照，不对其额外授予许可证。

`PYTHONPATH=. python tools/benchmark.py` 比较完整 PPO 更新吞吐，排除编译和 5 次预热，
结果写入 `runs/benchmark.json`。测得更大批量吞吐更高，但每次更新使用更多样本，
不意味着相同样本数下策略一定更强。`--untrained` 评估同种子初始网络可作为学习前对照。
