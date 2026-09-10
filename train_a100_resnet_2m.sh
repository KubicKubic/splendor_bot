#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Balanced ~2M residual actor/critic.  Observation v3 adds every public
# reserved card, masks padding seats, and retains aggregate face-down counts.
exec bash train_a100.sh \
  --envs 16384 --horizon 128 --epochs 3 --minibatches 32 \
  --width 480 --residual-blocks 6 --residual-taper \
  --residual-stage-widths 480,272,152 \
  --policy-head-width 352 --policy-head-layers 2 \
  --value-head-width 320 --value-head-layers 2 \
  --value-loss-coef 10.0 --players 4 --mixed-players --bf16 \
  --gamma 1.0 --gae-lambda 0.9 --lr 0.00001 "$@"
