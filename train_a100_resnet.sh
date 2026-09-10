#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Deep tapered residual policy/value trunk: six residual blocks / 12 main
# transforms, with widths 675 -> 380 -> 212.  Dedicated two-layer policy and
# value heads have comparable capacity (500 vs 450) while policy is larger.
exec bash train_a100.sh \
  --envs 4096 --horizon 128 --epochs 3 --minibatches 32 \
  --width 675 --residual-blocks 6 --residual-taper --residual-stage-widths 675,380,212 \
  --policy-head-width 500 --policy-head-layers 2 --value-head-width 450 --value-head-layers 2 \
  --value-loss-coef 10.0 --players 4 --mixed-players --bf16 --gamma 1.0 --lr 0.0001 "$@"
