#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# JAX 0.4.35 attempts to CUDA-graph-capture cuDNN flash attention inside the
# nested PPO scans; that combination invalidates stream capture on A100. The
# fused attention kernel remains enabled, only the incompatible graph wrapper
# is disabled for this launcher.
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_graph_level=0"

# 1,037,203-parameter typed-token actor/critic. Every public observation field is
# embedded by semantic type, then communicated through four attention blocks.
exec bash train_a100.sh \
  --envs 16384 --horizon 128 --epochs 1 --minibatches 64 \
  --architecture transformer --width 64 \
  --transformer-layers 4 --transformer-heads 4 --transformer-ff-dim 256 \
  --token-embed-width 64 \
  --policy-head-width 640 --policy-head-layers 2 \
  --value-head-width 448 --value-head-layers 2 \
  --value-loss-coef 10.0 --players 4 --mixed-players --bf16 \
  --gamma 1.0 --gae-lambda 0.9 --lr 0.00001 \
  --save-every 500 --log-every 10 "$@"
