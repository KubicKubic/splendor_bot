#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# ~1M parameter shared policy, stratified 2/3/4-player HullQin games.
# Large 32K PPO minibatches measured fastest on the local A100 80GB.
exec bash train_a100.sh \
  --envs 8192 --horizon 128 --epochs 3 --minibatches 32 \
  --width 800 --players 4 --mixed-players --bf16 \
  --gamma 1.0 --lr 0.0001 "$@"
