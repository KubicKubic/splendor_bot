#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
SPLD_PYTHON="${SPLD_PYTHON:-../generals_bot/.conda_envs/generals_bot/bin/python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Every normal launch gets an isolated live ladder. Set SPLD_AUTO_ELO=0 only
# for short diagnostics/benchmarks where checkpoint evaluation is unwanted.
if [[ "${SPLD_AUTO_ELO:-1}" == 0 ]]; then
  exec "$SPLD_PYTHON" -m splendor.train "$@"
fi

run_dir="runs/a100"
expect_out=0
for argument in "$@"; do
  if (( expect_out )); then
    run_dir="$argument"
    expect_out=0
  elif [[ "$argument" == --out ]]; then
    expect_out=1
  elif [[ "$argument" == --out=* ]]; then
    run_dir="${argument#--out=}"
  fi
done
if (( expect_out )); then
  echo '--out requires a path' >&2
  exit 2
fi

ladder_dir="${SPLD_ELO_OUT:-${run_dir}_ladder}"
mkdir -p -- "$ladder_dir"
training_pid=''
ladder_pid=''
plot_pid=''
cleanup() {
  for child in "$ladder_pid" "$plot_pid" "$training_pid"; do
    if [[ -n "$child" ]]; then
      kill "$child" 2>/dev/null || true
    fi
  done
  wait "$ladder_pid" "$plot_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$SPLD_PYTHON" -m splendor.train "$@" &
training_pid=$!
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "$SPLD_PYTHON" \
  tools/monitor_mixed_ladder.py --run-dir "$run_dir" --out "$ladder_dir" \
  >>"$ladder_dir/monitor.log" 2>&1 &
ladder_pid=$!
"$SPLD_PYTHON" tools/watch_elo_plot.py \
  --source "$ladder_dir/ratings.json" --output "$ladder_dir/elo_live.png" \
  >>"$ladder_dir/plot.log" 2>&1 &
plot_pid=$!

set +e
wait "$training_pid"
status=$?
set -e
training_pid=''
if (( status == 0 )); then
  # Let the ladder consume the final checkpoint on a normally completed run.
  for _ in {1..120}; do
    [[ -f "$ladder_dir/complete.json" ]] && break
    kill -0 "$ladder_pid" 2>/dev/null || break
    sleep 1
  done
fi
exit "$status"
