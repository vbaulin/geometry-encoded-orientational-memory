#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_matched_preparation_prl}"
GPUS="${GPUS:-0,1,2,3}"
GRAPH_SEEDS="${GRAPH_SEEDS:-17,29,43,71,97}"
IFS=',' read -r -a devices <<< "$GPUS"
IFS=',' read -r -a seeds <<< "$GRAPH_SEEDS"
mkdir -p "$OUT_BASE"

"$PYTHON_BIN" -B -c 'import torch; from scripts.rotating_colloids_matched_preparation import run_case; assert torch.cuda.is_available(), "CUDA is unavailable"'
echo "Shared targets; independent noise; 3 matched-release and 2 own-write controls."
echo "Default: N=1024, 5 graphs, 2 disorders, 2 targets, 48 replicas, Dr*T=625."
echo "Both angular trajectories and RNG checkpoints are retained."

pids=()
for ((slot=0; slot<${#devices[@]}; slot++)); do
  selected=""
  for ((i=slot; i<${#seeds[@]}; i+=${#devices[@]})); do
    selected="${selected:+$selected,}${seeds[i]}"
  done
  [[ -n "$selected" ]] || continue
  label="graphs_${selected//,/_}"
  echo "GPU ${devices[slot]} <- $selected"
  CUDA_VISIBLE_DEVICES="${devices[slot]}" "$PYTHON_BIN" -B \
    scripts/rotating_colloids_matched_preparation.py \
    --device cuda --output-dir "$OUT_BASE/$label" \
    --graph-seeds "$selected" --sizes "${SIZES:-32}" \
    --disorders "${DISORDERS:-0.11,0.16}" --coupling-scales "${COUPLING_SCALES:-1}" \
    --targets "${TARGETS:-relaxed,quarter_turn}" --replicas "${REPLICAS:-48}" \
    --equilibration-steps "${EQUILIBRATION_STEPS:-50000}" \
    --write-steps "${WRITE_STEPS:-50000}" --release-steps "${RELEASE_STEPS:-250000}" \
    --dt "${DT:-0.0025}" --write-field "${WRITE_FIELD:-1.5}" \
    --checkpoint-steps "${CHECKPOINT_STEPS:-10000}" \
    > "$OUT_BASE/$label.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" != 0 ]]; then
  echo "A shard stopped; inspect $OUT_BASE/*.log. Rerunning the same command resumes its saved state." >&2
  exit "$status"
fi
"$PYTHON_BIN" -B scripts/rotating_colloids_matched_preparation.py \
  --analyze-only --output-dir "$OUT_BASE"
echo "Completed: $OUT_BASE/matched_preparation_report.json"
