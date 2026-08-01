#!/usr/bin/env bash
set -euo pipefail

# Publication validation for the independent groove-field model.  Each size is
# written to a separate directory so a failed job does not erase completed
# sizes.  On a single RTX 3090 the full set is expected to take several hours,
# depending on PyTorch and driver versions.

read -r -a PYTHON_CMD <<< "${PYTHON:-python -B}"
DEVICE="${DEVICE:-cuda}"
ROOT="${ROOT:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_hidden_memory_publication}"
SCRIPT="${SCRIPT:-scripts/rotating_colloids_groove_protocols.py}"
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"
SIZES="${SIZES:-16 24 32 48}"

for size in $SIZES; do
  "${PYTHON_CMD[@]}" "$SCRIPT" \
    --output-dir "$ROOT/n${size}" \
    --protocol all \
    --device "$DEVICE" \
    --sizes "$size" \
    --cluster-size "$CLUSTER_SIZE" \
    --replicas 48 \
    --Dr 0.45 \
    --dt 0.015 \
    --burn-steps 12000 \
    --sample-steps 36000 \
    --sample-stride 40 \
    --j-values 0,0.5,1,1.5,2,2.5,3 \
    --h-values 0.1,0.2,0.4,0.6,0.8,1.2,1.8 \
    --write-steps 24000 \
    --release-steps 80000 \
    --release-points 42 \
    --release-j-values 0,1,2,3 \
    --release-h-values 0.4,0.6,0.8 \
    --release-fractions 0,0.05,0.15 \
    --switch-steps 80000 \
    --switch-points 42 \
    --switch-j-values 0,1,2,3 \
    --switch-h-values 0.2,0.6
done
