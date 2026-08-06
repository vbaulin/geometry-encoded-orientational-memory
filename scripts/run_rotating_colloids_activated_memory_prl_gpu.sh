#!/usr/bin/env bash
set -euo pipefail

# Eight-coupling activated-retention scan used for Fig. 4 of the Letter.
#
# The scan is five independent quenched graphs at N=576.  Each graph is an
# independent job, so the work is sharded over however many CUDA devices are
# visible: four devices give the original four-way split, and a single
# RTX 3090 runs the five graphs sequentially in one process.  Every shard
# writes an append-only JSONL file, so an interrupted run resumes unchanged.
#
# Environment overrides:
#   GPUS                 comma-separated CUDA device ids (default: autodetect)
#   GRAPH_SEEDS          comma-separated graph seeds     (default: 17,29,43,71,97)
#   OUT_BASE             scan output directory
#   FIGURE_DIR           figure output directory
#   SKIP_FIGURE=1        run the scan only
#   PYTHON_BIN           python launcher (default: "python -B")

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python -B}"
OUT_BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_activated_memory_prl_gpu}"
GRAPH_SEEDS="${GRAPH_SEEDS:-17,29,43,71,97}"
mkdir -p "$OUT_BASE"

detect_gpus() {
  if [[ -n "${GPUS:-}" ]]; then
    echo "$GPUS"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    # Honour an externally imposed device mask; the child processes see the
    # masked devices renumbered from zero.
    local count
    count="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
    seq -s, 0 "$((count - 1))"
    return
  fi
  local n=0
  if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
  fi
  if [[ "$n" -lt 1 ]]; then
    n="$($PYTHON_BIN -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  fi
  if [[ "$n" -lt 1 ]]; then
    echo "No CUDA device found. Set GPUS explicitly or run on the GPU host." >&2
    exit 1
  fi
  seq -s, 0 "$((n - 1))"
}

IFS=',' read -r -a DEVICES <<<"$(detect_gpus)"
IFS=',' read -r -a SEEDS <<<"$GRAPH_SEEDS"
DEVICE_COUNT="${#DEVICES[@]}"
SEED_COUNT="${#SEEDS[@]}"
echo "Sharding ${SEED_COUNT} graph seeds over ${DEVICE_COUNT} CUDA device(s): ${DEVICES[*]}"

COMMON=(
  --device cuda --n 24
  --lambdas 0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4
  --j0 4 --g0 5 --replicas 32
  --equilibration-steps 50000 --observation-steps 250000
  --stride 200 --write-steps 50000 --write-field 1.5 --dt 0.0025
)

# Round-robin the graph seeds so every device gets a contiguous shard label.
shard_seeds=()
for ((i = 0; i < DEVICE_COUNT; i++)); do
  shard_seeds[i]=""
done
for ((i = 0; i < SEED_COUNT; i++)); do
  slot=$((i % DEVICE_COUNT))
  if [[ -z "${shard_seeds[slot]}" ]]; then
    shard_seeds[slot]="${SEEDS[i]}"
  else
    shard_seeds[slot]="${shard_seeds[slot]},${SEEDS[i]}"
  fi
done

pids=()
for ((i = 0; i < DEVICE_COUNT; i++)); do
  seeds="${shard_seeds[i]}"
  [[ -z "$seeds" ]] && continue
  label="seeds_${seeds//,/_}"
  echo "  device ${DEVICES[i]} <- graph seeds ${seeds} (${label})"
  CUDA_VISIBLE_DEVICES="${DEVICES[i]}" $PYTHON_BIN scripts/rotating_colloids_activated_memory_test.py \
    --output-dir "$OUT_BASE/$label" --graph-seeds "$seeds" "${COMMON[@]}" \
    >"$OUT_BASE/${label}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -ne 0 ]]; then
  echo "At least one activated-memory shard failed; inspect $OUT_BASE/*.log" >&2
  exit "$status"
fi

if [[ "${SKIP_FIGURE:-0}" == "1" ]]; then
  echo "Activated-memory scan complete (figure skipped): $OUT_BASE"
  exit 0
fi

FIGURE_DIR="${FIGURE_DIR:-tex/rotating_colloids/capillary_prl_figures}"
$PYTHON_BIN scripts/plot_rotating_colloids_activated_memory_prl.py \
  --input-dir "$OUT_BASE" --output-dir "$FIGURE_DIR"
echo "Activated-memory scan complete: $OUT_BASE"
echo "Activated-memory figure: $FIGURE_DIR/fig4_activated_memory_results.pdf"
