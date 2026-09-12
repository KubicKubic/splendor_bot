#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# cuDNN Flash Attention remains enabled.  Only CUDA graph capture is disabled:
# JAX 0.4.35 cannot safely capture the nested PPO scan around the fused kernel.
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_graph_level=0"

# 1,613,811-parameter typed-token actor/critic.  Capacity is concentrated in
# six 128-wide attention/FFN blocks (rather than oversized terminal heads), so
# every public card, reserve, noble and player token can interact repeatedly.
# The 47 real tokens are padded to a 64-token cuDNN Flash Attention kernel.
exec bash train_a100.sh \
  --envs 16384 --horizon 128 --epochs 1 --minibatches 64 \
  --architecture transformer --width 128 \
  --observation-version 3 \
  --transformer-layers 6 --transformer-heads 4 --transformer-ff-dim 512 \
  --token-embed-width 128 \
  --policy-head-width 256 --policy-head-layers 2 \
  --value-head-width 224 --value-head-layers 2 \
  --value-loss-coef 1.0 --players 4 --mixed-players --bf16 \
  --gamma 1.0 --gae-lambda 0.9 --lr 0.0001 \
  --updates 200000 --save-every 500 --log-every 10 "$@"
