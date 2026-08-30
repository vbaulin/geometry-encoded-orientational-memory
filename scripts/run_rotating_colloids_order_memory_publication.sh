#!/usr/bin/env bash
set -euo pipefail

# Restore the four compact artifacts behind the operation-order result. The
# simulator is append-only and resumes by graph/field/mode/support key, so an
# interrupted invocation can be run again without duplicating completed rows.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python -B}"
DEVICE="${DEVICE:-cpu}"
BASE="${OUT_BASE:-discoveries/theory_experiment_interface/rotating_colloids_hyperion}"

common_args=(
  --device "$DEVICE"
  --graph-seeds 17,29,43
  --replicas 12
  --disorder 0.16
  --j-align 4
  --g-capillary 5
  --dt 0.0025
  --equilibration-steps 3000
  --pulse-steps 2000
  --release-steps 8000
  --stride 40
  --seed 20260822
  --noise-modes common
)

echo "[1/4] N=144 field-amplitude scan"
$PYTHON_BIN scripts/test_rotating_colloids_operation_order_memory.py \
  "${common_args[@]}" \
  --n 12 \
  --fields 1,2,3,5,8 \
  --output-dir "$BASE/rotating_colloids_operation_order_memory_n12"

echo "[2/4] N=256 endpoint size check"
$PYTHON_BIN scripts/test_rotating_colloids_operation_order_memory.py \
  "${common_args[@]}" \
  --n 16 \
  --fields 8 \
  --output-dir "$BASE/rotating_colloids_operation_order_memory_n16"

echo "[3/4] N=144 driven-support control"
$PYTHON_BIN scripts/test_rotating_colloids_operation_order_memory.py \
  "${common_args[@]}" \
  --n 12 \
  --fields 8 \
  --contest-fractions 0.25,1 \
  --fraction-scan-only \
  --output-dir "$BASE/rotating_colloids_operation_order_memory_fraction_n12"

echo "[4/4] Relaxation-quotient minimal model"
$PYTHON_BIN scripts/test_relaxed_exchange_order.py \
  --output "$BASE/relaxed_exchange_order_minimal.json"

$PYTHON_BIN -c '
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
expected = {
    "rotating_colloids_operation_order_memory_n12/operation_order_memory.jsonl": 30,
    "rotating_colloids_operation_order_memory_n16/operation_order_memory.jsonl": 6,
    "rotating_colloids_operation_order_memory_fraction_n12/operation_order_memory.jsonl": 6,
}
for relative, count in expected.items():
    path = base / relative
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != count:
        raise SystemExit(f"{path}: found {len(rows)} rows, expected {count}")
minimal = base / "relaxed_exchange_order_minimal.json"
json.loads(minimal.read_text(encoding="utf-8"))
print(json.dumps({"complete": True, "row_counts": expected, "minimal_model": str(minimal)}))
' "$BASE"

echo "Operation-order publication artifacts complete: $BASE"
