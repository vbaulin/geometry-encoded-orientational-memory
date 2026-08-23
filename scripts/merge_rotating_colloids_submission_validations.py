#!/usr/bin/env python3
"""Merge sharded PRL validation records and build compact aggregate reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_rotating_colloids_timestep import build_report


def read_jsonl(paths: Sequence[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def deduplicate(
    rows: Sequence[Dict[str, object]], key_fields: Sequence[str]
) -> List[Dict[str, object]]:
    unique: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        previous = unique.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"conflicting validation rows for key {key}")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def mean_sem(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / math.sqrt(array.size))
        if array.size > 1
        else 0.0,
    }


def mobile_report(
    rows: Sequence[Dict[str, object]], *, core_diameter: float = 0.55
) -> Dict[str, object]:
    metrics = (
        "split_endpoint",
        "rms_displacement_endpoint",
        "edge_jaccard_mean",
        "initial_edges_retained_mean",
        "minimum_separation_mean",
        "position_force_clip_fraction",
    )
    stiffnesses = sorted({float(row["cage_stiffness"]) for row in rows})
    summaries = []
    for stiffness in stiffnesses:
        group = [row for row in rows if float(row["cage_stiffness"]) == stiffness]
        aggregate = {
            metric: mean_sem(float(row[metric]) for row in group)
            for metric in metrics
        }
        qualifies = (
            aggregate["split_endpoint"]["mean"] >= 0.30
            and aggregate["rms_displacement_endpoint"]["mean"] <= 0.10
            and aggregate["initial_edges_retained_mean"]["mean"] >= 0.95
            and aggregate["position_force_clip_fraction"]["mean"] <= 0.01
        )
        summaries.append(
            {
                "cage_stiffness": stiffness,
                "graph_count": len({int(row["graph_seed"]) for row in group}),
                "metrics": aggregate,
                "meets_predeclared_cage_memory_gate": bool(qualifies),
                "soft_core_penetration_observed": bool(
                    aggregate["minimum_separation_mean"]["mean"] < core_diameter
                ),
            }
        )
    passing = [
        item["cage_stiffness"]
        for item in summaries
        if item["meets_predeclared_cage_memory_gate"]
    ]
    return {
        "row_count": len(rows),
        "summaries": summaries,
        "minimum_tested_passing_stiffness": min(passing) if passing else None,
        "predeclared_gate": {
            "split_endpoint_min": 0.30,
            "rms_displacement_endpoint_max": 0.10,
            "initial_edges_retained_min": 0.95,
            "position_force_clip_fraction_max": 0.01,
        },
        "core_diameter": core_diameter,
        "soft_core_scope": (
            "A minimum-separation mean below the nominal core diameter records "
            "penetration of the deliberately soft core. Such a row tests retention "
            "under centre motion and topology change, but is not a hard-particle "
            "realization of finite-sized ellipsoids."
        ),
        "interpretation": (
            "Passing identifies an effective cage stiffness for which the mobile "
            "model preserves topology and orientational inheritance over the same "
            "finite observation window. It is not an equilibrium glass criterion."
        ),
    }


def independent_order_report(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    selected = [
        row for row in rows
        if str(row.get("noise_mode", "common")) == "independent"
        and (
            str(row["mode"]) == "partitioned"
            or math.isclose(float(row.get("contest_fraction_requested", 0.25)), 0.25)
        )
    ]
    if not selected:
        raise ValueError("no primary independent-noise order rows")
    fields = sorted({float(row["field"]) for row in selected})
    summaries: Dict[str, object] = {}
    for field in fields:
        for mode in ("partitioned", "contested"):
            group = [
                row for row in selected
                if math.isclose(float(row["field"]), field)
                and str(row["mode"]) == mode
            ]
            if not group:
                raise ValueError(f"missing independent-noise rows for field={field}, mode={mode}")
            summaries[f"{field:g}:{mode}"] = {
                "graph_count": len({int(row["graph_seed"]) for row in group}),
                "terminal_order_readout": mean_sem(
                    float(row["terminal_order_readout"]) for row in group
                ),
                "decode_accuracy_zero_threshold": mean_sem(
                    float(row["terminal_decode_accuracy_zero_threshold"])
                    for row in group
                ),
                "decode_d_prime": mean_sem(
                    float(row["terminal_decode_d_prime"]) for row in group
                ),
            }
    highest = max(fields)
    contested = summaries[f"{highest:g}:contested"]["terminal_order_readout"]
    partitioned = summaries[f"{highest:g}:partitioned"]["terminal_order_readout"]
    contrast_sem = math.hypot(float(contested["sem"]), float(partitioned["sem"]))
    contrast = float(contested["mean"]) - float(partitioned["mean"])
    return {
        "row_count": len(selected),
        "fields": fields,
        "summaries": summaries,
        "highest_field": highest,
        "highest_field_contested_minus_partitioned": {
            "mean": contrast,
            "sem": contrast_sem,
            "z": contrast / contrast_sem if contrast_sem > 0.0 else None,
        },
        "interpretation": (
            "Independent Brownian histories test whether pulse order remains "
            "decodable without common-random-number cancellation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "rotating_colloids_submission_validations"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    timestep_paths = sorted(root.glob("timestep_shard_*/timestep_validation.jsonl"))
    mobile_paths = sorted(root.glob("mobile_shard_*/mobile_cage_validation.jsonl"))
    order_path = root / "order_independent_n16" / "operation_order_memory.jsonl"
    if not timestep_paths:
        raise FileNotFoundError("no timestep validation shards found")
    if not mobile_paths:
        raise FileNotFoundError("no mobile-cage validation shards found")
    if not order_path.exists():
        raise FileNotFoundError("no independent-noise order validation rows found")

    timestep_rows = deduplicate(
        read_jsonl(timestep_paths), ("graph_seed", "dt")
    )
    mobile_rows = deduplicate(
        read_jsonl(mobile_paths), ("graph_seed", "cage_stiffness")
    )
    order_rows = deduplicate(
        read_jsonl([order_path]),
        (
            "graph_seed", "field", "mode", "contest_fraction_requested",
            "noise_mode",
        ),
    )
    timestep_dir = root / "timestep"
    mobile_dir = root / "mobile_cage"
    write_jsonl(timestep_dir / "timestep_validation.jsonl", timestep_rows)
    write_jsonl(mobile_dir / "mobile_cage_validation.jsonl", mobile_rows)
    (timestep_dir / "timestep_validation_report.json").write_text(
        json.dumps(build_report(timestep_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (mobile_dir / "mobile_cage_validation_report.json").write_text(
        json.dumps(mobile_report(mobile_rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    order_dir = root / "independent_order"
    order_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(order_dir / "operation_order_memory.jsonl", order_rows)
    (order_dir / "independent_noise_order_report.json").write_text(
        json.dumps(independent_order_report(order_rows), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "complete": True,
        "timestep_rows": len(timestep_rows),
        "mobile_cage_rows": len(mobile_rows),
        "independent_order_rows": len(order_rows),
        "timestep_shards": [str(path) for path in timestep_paths],
        "mobile_cage_shards": [str(path) for path in mobile_paths],
    }
    (root / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
