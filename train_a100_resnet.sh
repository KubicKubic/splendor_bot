#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Deep tapered residual policy/value trunk: six residual blocks / 12 main
# transforms, with widths 480 -> 330 -> 210; ~2.02M parameters.
exec bash train_a100.sh \
  --envs 4096 --horizon 128 --epochs 3 --minibatches 32 \
  --width 480 --residual-blocks 6 --residual-taper --players 4 --mixed-players --bf16 \
  --gamma 1.0 --lr 0.0001 "$@"
