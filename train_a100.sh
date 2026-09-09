#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
SPLD_PYTHON="${SPLD_PYTHON:-../generals_bot/.conda_envs/generals_bot/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$SPLD_PYTHON" -m splendor.train "$@"
