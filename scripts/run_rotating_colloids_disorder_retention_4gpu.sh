#!/usr/bin/env bash
set -euo pipefail

# Reconstruct the raw write--release trajectories behind the positional-
# disorder result. Jobs are distributed round-robin over the visible GPUs.
# A complete, parseable protocol file is skipped; an interrupted cell reruns.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_disorder_retention_protocols}"
GPU_LIST="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS_ARRAY <<< "$GPU_LIST"
GPU_COUNT="${#GPUS_ARRAY[@]}"

if [[ "$GPU_COUNT" -lt 1 ]]; then
  echo "No GPUs supplied through GPUS." >&2
  exit 2
fi

mkdir -p "$BASE"

declare -a JOB_N=()
declare -a JOB_SIGMA=()
declare -a JOB_SEED=()

add_job() {
  JOB_N+=("$1")
  JOB_SIGMA+=("$2")
  JOB_SEED+=("$3")
}

# N=576: the low-disorder control used three graphs; every other reported
# disorder cell used five matched graph seeds.
for seed in 17 29 43; do
  add_job 24 0.05 "$seed"
done
for sigma in 0.08 0.11 0.16 0.28; do
  for seed in 17 29 43 71 97; do
    add_job 24 "$sigma" "$seed"
  done
done

# N=1024 replication of the retention maximum and its high-disorder control.
for sigma in 0.11 0.16; do
  for seed in 17 29 43 71 97; do
    add_job 32 "$sigma" "$seed"
  done
done

protocol_is_complete() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  $PYTHON_BIN -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert all(k in d for k in ("model","split_replica","write_release","no_capillary_split_replica","no_capillary_write_release"))' \
    "$path" >/dev/null 2>&1
}

run_worker() {
  local worker="$1"
  local gpu="${GPUS_ARRAY[$worker]}"
  local index n sigma seed output protocol started elapsed

  for index in "${!JOB_N[@]}"; do
    if (( index % GPU_COUNT != worker )); then
      continue
    fi
    n="${JOB_N[$index]}"
    sigma="${JOB_SIGMA[$index]}"
    seed="${JOB_SEED[$index]}"
    output="$BASE/N$((n*n))_sigma${sigma}_seed${seed}"
    protocol="$output/capillary_pair_protocols.json"

    if protocol_is_complete "$protocol"; then
      echo "[GPU $gpu] skip complete N=$((n*n)) sigma=$sigma seed=$seed"
      continue
    fi

    echo "[GPU $gpu] run N=$((n*n)) sigma=$sigma seed=$seed"
    started="$SECONDS"
    CUDA_VISIBLE_DEVICES="$gpu" $PYTHON_BIN scripts/rotating_colloids_capillary_pair.py \
      --device cuda \
      --n "$n" \
      --graph-seeds "$seed" \
      --seed 20260712 \
      --disorder "$sigma" \
      --output-dir "$output" \
      --skip-scan \
      --selected-j 4 \
      --selected-g 5 \
      --replicas 48 \
      --protocol-equilibration-steps 50000 \
      --protocol-steps 250000 \
      --protocol-stride 200 \
      --write-steps 50000 \
      --write-field 1.5 \
      --dt 0.0025 \
      >"$output.log" 2>&1
    elapsed=$((SECONDS - started))
    echo "[GPU $gpu] complete N=$((n*n)) sigma=$sigma seed=$seed elapsed=${elapsed}s"
  done
}

declare -a PIDS=()
for worker in "${!GPUS_ARRAY[@]}"; do
  run_worker "$worker" >"$BASE/worker_${worker}.log" 2>&1 &
  PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one worker failed. Inspect $BASE/worker_*.log; rerunning resumes." >&2
  exit 1
fi

$PYTHON_BIN scripts/analyze_rotating_colloids_disorder_protocols.py \
  --input-dir "$BASE" --node-count 576 --output-dir "$BASE/analysis_N576"
$PYTHON_BIN scripts/analyze_rotating_colloids_disorder_protocols.py \
  --input-dir "$BASE" --node-count 1024 --output-dir "$BASE/analysis_N1024"
$PYTHON_BIN scripts/build_rotating_colloids_disorder_retention_summary.py \
  --input-dir "$BASE" \
  --output discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_disorder_retention_summary.json

echo "Disorder-retention trajectories complete: $BASE"
echo "Publication summary rebuilt: discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_disorder_retention_summary.json"
echo "Rebuild the data deposit with:"
echo "  python -B scripts/build_rotating_colloids_release.py --disorder-retention-input '$BASE' --clean"
