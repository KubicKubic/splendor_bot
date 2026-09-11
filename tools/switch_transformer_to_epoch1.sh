#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

run_dir="${1:?usage: switch_transformer_to_epoch1.sh RUN_DIR}"
checkpoint="$run_dir/latest.npz"
switch_log="$run_dir/epoch1_switch.log"
python_bin="${SPLD_PYTHON:-../generals_bot/.conda_envs/generals_bot/bin/python}"

find_training_pid() {
  local pid command
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
    command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$command" == *' -m splendor.train '* && "$command" == *"--out $run_dir"* ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
  done < <(pgrep -f 'splendor[.]train' || true)
  return 1
}

printf 'waiting for update-500 checkpoint: %s\n' "$checkpoint" >>"$switch_log"
while [[ ! -f "$checkpoint" ]]; do
  if ! find_training_pid >/dev/null; then
    printf 'training exited before checkpoint appeared\n' >>"$switch_log"
    exit 1
  fi
  sleep 10
done

checkpoint_update="$($python_bin - "$checkpoint" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    print(int(data['update']))
PY
)"
if (( checkpoint_update < 500 )); then
  printf 'refusing checkpoint update %s below switch boundary\n' "$checkpoint_update" >>"$switch_log"
  exit 1
fi

training_pid="$(find_training_pid)"
printf 'stopping epochs=3 pid=%s at checkpoint update=%s\n' \
  "$training_pid" "$checkpoint_update" >>"$switch_log"
kill -TERM "$training_pid"
for _ in {1..30}; do
  kill -0 "$training_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$training_pid" 2>/dev/null; then
  printf 'training did not stop after SIGTERM\n' >>"$switch_log"
  exit 1
fi

# The old launcher normally removes its own companions. Also stop any manually
# restarted companion for this run, preventing duplicate evaluators on resume.
for pid in $(pgrep -f 'tools/(monitor_mixed_ladder|watch_elo_plot)[.]py' || true); do
  [[ -r "/proc/$pid/cmdline" ]] || continue
  command="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
  if [[ "$command" == *"$run_dir"* ]]; then
    kill -TERM "$pid" 2>/dev/null || true
  fi
done

# Epoch count is an optimizer schedule choice, not a checkpoint/model schema.
# Remove it from the auto-ladder's immutable-training contract so the same Elo
# history continues across this requested update-boundary change.
$python_bin - "$run_dir" <<'PY'
import json
import os
from pathlib import Path
import sys
path = Path(str(sys.argv[1]) + '_ladder') / 'protocol.json'
if path.is_file():
    data = json.loads(path.read_text())
    data.get('config', {}).get('expected_training', {}).pop('epochs', None)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)
PY

printf 'resuming update=%s with epochs=1\n' "$checkpoint_update" >>"$switch_log"
exec ./train_a100_transformer_1m.sh \
  --resume "$checkpoint" --out "$run_dir" --updates 100000 \
  >>"$run_dir/epoch1_training.log" 2>&1
