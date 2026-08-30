#!/usr/bin/env bash
set -euo pipefail

# Dense matched-start release scan for four GPUs. Each shard is append-only and
# resumes by stable graph/coupling/target key.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/holonomy_matched_release_crossover}"
GPUS="${GPUS:-0,1,2,3}"
GRAPH_SEEDS="${GRAPH_SEEDS:-12345,12346,12347,12348,12349,12350,12351,12352}"
BETA_J_VALUES="${BETA_J_VALUES:-0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4}"
RELEASE_STEPS="${RELEASE_STEPS:-40000}"
REPLICAS="${REPLICAS:-48}"

mkdir -p "$OUT_BASE"
IFS=',' read -r -a devices <<<"$GPUS"
IFS=',' read -r -a seeds <<<"$GRAPH_SEEDS"
if [[ "${#devices[@]}" -lt 1 ]]; then
  echo "No GPU ids supplied" >&2
  exit 1
fi

shard_seeds=()
for ((i = 0; i < ${#devices[@]}; i++)); do shard_seeds[i]=""; done
for ((i = 0; i < ${#seeds[@]}; i++)); do
  slot=$((i % ${#devices[@]}))
  if [[ -z "${shard_seeds[slot]}" ]]; then
    shard_seeds[slot]="${seeds[i]}"
  else
    shard_seeds[slot]="${shard_seeds[slot]},${seeds[i]}"
  fi
done

pids=()
for ((i = 0; i < ${#devices[@]}; i++)); do
  selected="${shard_seeds[i]}"
  [[ -z "$selected" ]] && continue
  label="graphs_${selected//,/_}"
  echo "GPU ${devices[i]} <- ${selected}"
  CUDA_VISIBLE_DEVICES="${devices[i]}" $PYTHON_BIN \
    scripts/test_holonomy_matched_release_crossover.py \
    --device cuda \
    --output-dir "$OUT_BASE/$label" \
    --graph-seeds "$selected" \
    --beta-j-values "$BETA_J_VALUES" \
    --release-steps "$RELEASE_STEPS" \
    --replicas "$REPLICAS" \
    >"$OUT_BASE/$label.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "A matched-release shard failed; inspect $OUT_BASE/*.log. Rerunning resumes." >&2
  exit "$status"
fi

$PYTHON_BIN scripts/analyze_holonomy_matched_release_crossover.py \
  --input "$OUT_BASE" \
  --output-dir "$OUT_BASE/analysis"

echo "Matched-release crossover complete: $OUT_BASE"
echo "Report: $OUT_BASE/analysis/matched_release_crossover_report.md"
