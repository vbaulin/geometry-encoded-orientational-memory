#!/usr/bin/env bash
set -euo pipefail

# Publication-scale validation of the autonomous capillary pair model.
# The default schedule is intended for one RTX 3090.  Every stage writes a
# resumable JSONL file, so an interrupted run can be restarted unchanged.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
DEVICE="${DEVICE:-cuda}"
BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl_gpu}"
SEEDS="${GRAPH_SEEDS:-17,29,43,71,97}"
PHASE_GRAPH_SEEDS="${PHASE_GRAPH_SEEDS:-17,29,43}"
PHASE_ONLY="${PHASE_ONLY:-0}"

mkdir -p "$BASE"

echo "[1/3] Dense capillary-pair state map"
$PYTHON_BIN scripts/rotating_colloids_capillary_pair.py \
  --device "$DEVICE" \
  --n 20 \
  --graph-seeds "$PHASE_GRAPH_SEEDS" \
  --output-dir "$BASE/dense_map_n20" \
  --skip-protocols \
  --j-values 0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.25,4.5,4.75,5 \
  --g-values 0,0.375,0.75,1.125,1.5,1.875,2.25,2.625,3,3.375,3.75,4.125,4.5,4.875,5.25,5.625,6,6.375,6.75,7.125,7.5 \
  --replicas 32 \
  --burn-in-steps 20000 \
  --sample-steps 60000 \
  --sample-stride 100 \
  --dt 0.0025 \
  --progress-every 10

echo "[1b/3] Data-resolved regime classification and phase plots"
$PYTHON_BIN scripts/classify_rotating_colloids_capillary_regimes.py \
  --input "$BASE/dense_map_n20/capillary_pair_scan.jsonl" \
  --output-dir "$BASE/phase_diagram"

if [[ "$PHASE_ONLY" == "1" ]]; then
  echo "Phase-map-only run complete: $BASE/phase_diagram"
  exit 0
fi

echo "[2/3] Selected-point finite-size and disorder scaling"
for N in 12 16 24 32 48; do
  $PYTHON_BIN scripts/rotating_colloids_capillary_pair.py \
    --device "$DEVICE" \
    --n "$N" \
    --graph-seeds "$SEEDS" \
    --output-dir "$BASE/finite_size_n${N}" \
    --skip-protocols \
    --j-values 4 \
    --g-values 0,5 \
    --replicas 48 \
    --burn-in-steps 30000 \
    --sample-steps 100000 \
    --sample-stride 100 \
    --dt 0.0025 \
    --progress-every 1
done

$PYTHON_BIN scripts/rotating_colloids_capillary_pair.py \
  --device "$DEVICE" \
  --n 32 \
  --graph-seeds "$SEEDS" \
  --output-dir "$BASE/matched_controls_n32" \
  --skip-protocols \
  --include-controls \
  --j-values 4 \
  --g-values 5 \
  --replicas 48 \
  --burn-in-steps 30000 \
  --sample-steps 100000 \
  --sample-stride 100 \
  --dt 0.0025 \
  --progress-every 1

echo "[3/3] Long-time overlap, aging, and write-release-read protocols"
for SEED in 17 29 43; do
  $PYTHON_BIN scripts/rotating_colloids_capillary_pair.py \
    --device "$DEVICE" \
    --n 32 \
    --graph-seeds "$SEED" \
    --output-dir "$BASE/dynamics_seed_${SEED}" \
    --skip-scan \
    --selected-j 4 \
    --selected-g 5 \
    --replicas 48 \
    --protocol-equilibration-steps 50000 \
    --protocol-steps 250000 \
    --protocol-stride 200 \
    --write-steps 50000 \
    --write-field 1.5 \
    --dt 0.0025
done

echo "Capillary-pair PRL validation complete: $BASE"
