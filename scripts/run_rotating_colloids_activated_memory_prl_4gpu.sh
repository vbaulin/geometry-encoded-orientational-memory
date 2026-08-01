#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python -B}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_activated_memory_prl_gpu}"
mkdir -p "$OUT_BASE"

COMMON=(
  --device cuda --n 24
  --lambdas 0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4
  --j0 4 --g0 5 --replicas 32
  --equilibration-steps 50000 --observation-steps 250000
  --stride 200 --write-steps 50000 --write-field 1.5 --dt 0.0025
)

pids=()
launch() {
  local gpu="$1" seeds="$2" label="$3"
  CUDA_VISIBLE_DEVICES="$gpu" $PYTHON_BIN scripts/rotating_colloids_activated_memory_test.py \
    --output-dir "$OUT_BASE/$label" --graph-seeds "$seeds" "${COMMON[@]}" \
    >"$OUT_BASE/${label}.log" 2>&1 &
  pids+=("$!")
}

launch 0 17,97 seeds_17_97
launch 1 29 seeds_29
launch 2 43 seeds_43
launch 3 71 seeds_71

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "At least one activated-memory shard failed; inspect $OUT_BASE/*.log" >&2
  exit "$status"
fi
FIGURE_DIR="${FIGURE_DIR:-tex/rotating_colloids/capillary_prl_figures}"
$PYTHON_BIN scripts/plot_rotating_colloids_activated_memory_prl.py \
  --input-dir "$OUT_BASE" --output-dir "$FIGURE_DIR"
echo "Activated-memory scan complete: $OUT_BASE"
echo "Activated-memory figure: $FIGURE_DIR/fig4_activated_memory_results.pdf"
