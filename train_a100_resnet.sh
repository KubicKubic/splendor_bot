#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Deep tapered residual policy/value trunk: six residual blocks / 12 main
# transforms, with widths 460 -> 316 -> 201.  A dedicated two-layer 384-wide
# value MLP provides four-seat MSE estimates; total size is ~2.0M parameters.
exec bash train_a100.sh \
  --envs 4096 --horizon 128 --epochs 3 --minibatches 32 \
  --width 460 --residual-blocks 6 --residual-taper --value-head-width 384 --value-head-layers 2 \
  --value-loss-coef 1.0 --players 4 --mixed-players --bf16 --gamma 1.0 --lr 0.0001 "$@"
