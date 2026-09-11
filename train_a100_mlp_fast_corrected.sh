#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# High-throughput 800-wide MLP, retaining the original 8,192-environment PPO
# geometry.  The current trainer supplies the corrected zero-sum four-seat
# critic, plain MSE value regression, and v3 public observation schema.
exec bash train_a100.sh \
  --envs 8192 --horizon 128 --epochs 3 --minibatches 32 \
  --architecture mlp --width 800 --mlp-hidden-layers 4 --mlp-activation gelu \
  --players 4 --mixed-players --bf16 \
  --gamma 1.0 --gae-lambda 0.9 --lr 0.0001 --value-loss-coef 1.0 \
  --entropy 0.01 --shaping 0.25 --max-turns 400 \
  --save-every 500 --log-every 50 --updates 100000 "$@"
