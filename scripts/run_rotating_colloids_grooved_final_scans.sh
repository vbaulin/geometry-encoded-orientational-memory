#!/usr/bin/env bash
set -euo pipefail

# Publication-scale grooved-surface reruns of the recovered rotating-colloid
# PRL scans.  These commands mirror the final pair-bond runs and change only
# the geometry mechanism:
#
#   pair model:    -eps J cos 2(theta_i + theta_j - 2 phi_ij)
#   grooved model: -eps J cos 2(theta_i - alpha_i)
#
# alpha_i is the local surface-groove easy axis.  Square controls use uniform
# grooves, triangular controls use a deterministic three-sublattice groove
# pattern, mosaic controls use locally rotated groove domains, and long-range
# controls add quenched groove-axis disorder.

DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python -B}"
ROOT="${ROOT:-discoveries/theory_experiment_interface/rotating_colloids_hyperion}"
SCRIPT="${SCRIPT:-scripts/rotating_colloids_hyperion_case.py}"
AXIS_DISORDER="${AXIS_DISORDER:-0.8}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"

common=(
  --maximal
  --device "$DEVICE"
  --constraint-mode grooved
  --noise 0.45
  --dt 0.015
  --sample-stride 20
  --radial-bins 16
  --progress-every "$PROGRESS_EVERY"
)

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_uniform_scan_n16" \
  "${common[@]}" \
  --sizes 16 \
  --graph-mode square \
  --eps-min 0.0 --eps-max 1.2 --eps-count 31 \
  --j-min 0.05 --j-max 1.2 --j-count 31 \
  --burn-in-steps 10000 \
  --steps 50000 \
  --reps 16

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_uniform_memory_zoom_n16" \
  "${common[@]}" \
  --sizes 16 \
  --graph-mode square \
  --eps-min 1.05 --eps-max 1.25 --eps-count 21 \
  --j-min 0.75 --j-max 1.05 --j-count 21 \
  --burn-in-steps 15000 \
  --steps 70000 \
  --reps 64

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_uniform_finite_size" \
  "${common[@]}" \
  --sizes 16,24,32 \
  --graph-mode square \
  --eps-values 1.12,1.16,1.2 \
  --j-values 0.82,0.89,0.96 \
  --burn-in-steps 15000 \
  --steps 70000 \
  --reps 48

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_longrange_disorder" \
  "${common[@]}" \
  --sizes 16,24,32,48 \
  --graph-mode long-range \
  --graph-radius 2.25 \
  --coupling-decay 0.9 \
  --bond-disorder 0.25 \
  --easy-axis-disorder "$AXIS_DISORDER" \
  --eps-min 0.8 --eps-max 1.6 --eps-count 21 \
  --j-min 0.45 --j-max 1.15 --j-count 21 \
  --burn-in-steps 15000 \
  --steps 70000 \
  --reps 32

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_triangular_frustrated_n16" \
  "${common[@]}" \
  --sizes 16 \
  --graph-mode triangular \
  --eps-min 0.5 --eps-max 1.8 --eps-count 31 \
  --j-min 0.3 --j-max 1.5 --j-count 31 \
  --burn-in-steps 15000 \
  --steps 70000 \
  --reps 32

$PYTHON "$SCRIPT" \
  --output-dir "$ROOT/rotating_colloids_grooved_mosaic_hidden_search_n32" \
  "${common[@]}" \
  --sizes 32 \
  --graph-mode mosaic \
  --cluster-size 8 \
  --crosslink-k 2 \
  --crosslink-weight 0.18 \
  --domain-angle-step 0.7853981633974483 \
  --eps-min 0.75 --eps-max 1.85 --eps-count 29 \
  --j-min 0.35 --j-max 1.25 --j-count 29 \
  --burn-in-steps 20000 \
  --steps 90000 \
  --reps 48
