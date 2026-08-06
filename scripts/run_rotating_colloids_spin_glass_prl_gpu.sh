#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
DEVICE="${DEVICE:-cuda}"
GPU_INDEX="${GPU_INDEX:-0}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_spin_glass_prl_gpu}"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

$PYTHON_BIN scripts/rotating_colloids_spin_glass_test.py \
  --device "$DEVICE" \
  --output-dir "$OUT_BASE" \
  --sizes 12,16,24,32,48 \
  --graph-seeds 17,29,43,71,97 \
  --lambdas 0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4 \
  --j0 4 \
  --g0 5 \
  --replicas 48 \
  --max-replica-pairs 512 \
  --burn-in-steps 50000 \
  --sample-steps 100000 \
  --sample-stride 100 \
  --dt 0.0025 \
  --progress-every 1

echo "Spin-glass PRL diagnostics complete: $OUT_BASE"
