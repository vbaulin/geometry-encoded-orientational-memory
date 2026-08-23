#!/usr/bin/env bash
set -euo pipefail

# Submission-critical validations for four RTX 3090 GPUs. Every numerical job
# writes append-only rows and resumes when this script is restarted.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_submission_validations}"
PHASE_INPUT="${PHASE_INPUT:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl_gpu/dense_map_n20/capillary_pair_scan.jsonl}"
PHASE_OUTPUT="${PHASE_OUTPUT:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl_gpu/phase_diagram}"

mkdir -p "$BASE"

echo "[-1/4] Simulator protocol API preflight"
$PYTHON_BIN -c 'import inspect, torch; from scripts.rotating_colloids_capillary_pair import make_caged_graph, simulate_ensemble; assert "write_weight" in inspect.signature(simulate_ensemble).parameters; g=make_caged_graph(4, disorder=0.16, cutoff=2.6, alignment_range=1.35, alignment_decay=0.20, seed=17); r=simulate_ensemble(g, j_align=4.0, g_capillary=5.0, replicas=1, burn_in_steps=1, sample_steps=1, sample_stride=1, dt=0.0025, seed=23, device=torch.device("cpu")); assert "state_after_steps" in r'

echo "[0/4] Nonredundant regime-feature ablation"
MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PYTHON_BIN scripts/classify_rotating_colloids_capillary_regimes.py \
  --input "$PHASE_INPUT" --output-dir "$PHASE_OUTPUT"

echo "[1/4] Launching time-step weak-convergence shards"
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/validate_rotating_colloids_timestep.py \
  --device cuda --graph-seeds 17,29,43 --output-dir "$BASE/timestep_shard_0" \
  >"$BASE/timestep_shard_0.log" 2>&1 &
PID_T0=$!

(
  CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN scripts/validate_rotating_colloids_timestep.py \
    --device cuda --graph-seeds 71,97 --output-dir "$BASE/timestep_shard_1"

  echo "[3/4] Independent-noise sequence decoding"
  CUDA_VISIBLE_DEVICES=1 $PYTHON_BIN scripts/test_rotating_colloids_operation_order_memory.py \
    --device cuda --n 16 --graph-seeds 17,29,43,71,97 \
    --fields 1,2,3,5,8 --replicas 48 --noise-modes independent \
    --equilibration-steps 30000 --pulse-steps 2000 \
    --release-steps 8000 --stride 40 --dt 0.0025 \
    --output-dir "$BASE/order_independent_n16"
) >"$BASE/timestep_shard_1_and_order.log" 2>&1 &
PID_T1=$!

echo "[2/4] Launching mobile-centre cage-stability shards"
CUDA_VISIBLE_DEVICES=2 $PYTHON_BIN scripts/validate_rotating_colloids_mobile_cage.py \
  --device cuda --graph-seeds 17,29,43 --output-dir "$BASE/mobile_shard_0" \
  >"$BASE/mobile_shard_0.log" 2>&1 &
PID_M0=$!

CUDA_VISIBLE_DEVICES=3 $PYTHON_BIN scripts/validate_rotating_colloids_mobile_cage.py \
  --device cuda --graph-seeds 71,97 --output-dir "$BASE/mobile_shard_1" \
  >"$BASE/mobile_shard_1.log" 2>&1 &
PID_M1=$!

status=0
for pid in "$PID_T0" "$PID_T1" "$PID_M0" "$PID_M1"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  echo "A validation shard failed. Inspect $BASE/*.log; rerunning resumes." >&2
  exit "$status"
fi

echo "[4/4] Merging validation shards"
$PYTHON_BIN scripts/merge_rotating_colloids_submission_validations.py --root "$BASE"

$PYTHON_BIN scripts/plot_rotating_colloids_operation_order_memory.py \
  --input "$BASE/order_independent_n16/operation_order_memory.jsonl" \
  --noise-mode independent \
  --output "$BASE/order_independent_n16/independent_noise_order_memory"

echo "Submission validations complete: $BASE"
