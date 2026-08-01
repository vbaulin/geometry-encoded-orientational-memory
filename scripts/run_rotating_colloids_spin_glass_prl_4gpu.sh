#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_spin_glass_prl_gpu}"
COMMON=(
  --device cuda
  --graph-seeds 17,29,43,71,97
  --lambdas 0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4
  --j0 4 --g0 5
  --replicas 48
  --max-replica-pairs 512
  --burn-in-steps 50000
  --sample-steps 100000
  --sample-stride 100
  --dt 0.0025
  --progress-every 1
)

pids=()
launch() {
  local gpu="$1"
  local sizes="$2"
  local label="$3"
  CUDA_VISIBLE_DEVICES="$gpu" $PYTHON_BIN scripts/rotating_colloids_spin_glass_test.py \
    --output-dir "$OUT_BASE/$label" --sizes "$sizes" "${COMMON[@]}" \
    >"$OUT_BASE/${label}.log" 2>&1 &
  pids+=("$!")
}

mkdir -p "$OUT_BASE"
launch 0 12,24 sizes_12_24
launch 1 16 sizes_16
launch 2 32 sizes_32
launch 3 48 sizes_48

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "At least one GPU shard failed; inspect $OUT_BASE/*.log" >&2
  exit "$status"
fi
echo "Four-GPU spin-glass diagnostics complete: $OUT_BASE"
