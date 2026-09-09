#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Measured ~5M complete PPO decisions/s on the local A100 80GB.
# Caller-supplied flags come last and can override the batch defaults.
exec bash train_a100.sh --envs 8192 --minibatches 128 --bf16 "$@"
