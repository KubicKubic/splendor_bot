#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Pooled Set Transformer: attention aggregates every semantic card/player set,
# then a wide residual trunk performs the costly reasoning in dense Tensor Core
# GEMMs.  This retains typed-token set interactions without per-token PPO cost.
exec bash train_a100.sh \
  --envs 16384 --horizon 128 --epochs 1 --minibatches 64 \
  --architecture pooled_transformer --width 416 \
  --observation-version 3 \
  --transformer-layers 4 --transformer-heads 4 --transformer-ff-dim 416 \
  --token-embed-width 64 \
  --policy-head-width 320 --policy-head-layers 2 \
  --value-head-width 256 --value-head-layers 2 \
  --value-loss-coef 1.0 --players 4 --mixed-players --bf16 \
  --gamma 1.0 --gae-lambda 0.9 --lr 0.0001 \
  --updates 200000 --save-every 500 --log-every 10 "$@"
