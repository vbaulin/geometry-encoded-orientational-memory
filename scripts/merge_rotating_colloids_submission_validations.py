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


def mobile_report(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
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
        "interpretation": (
            "Passing identifies an effective cage stiffness for which the mobile "
            "model preserves topology and orientational inheritance over the same "
            "finite observation window. It is not an equilibrium glass criterion."
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
    if not timestep_paths:
        raise FileNotFoundError("no timestep validation shards found")
    if not mobile_paths:
        raise FileNotFoundError("no mobile-cage validation shards found")

    timestep_rows = deduplicate(
        read_jsonl(timestep_paths), ("graph_seed", "dt")
    )
    mobile_rows = deduplicate(
        read_jsonl(mobile_paths), ("graph_seed", "cage_stiffness")
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
    manifest = {
        "complete": True,
        "timestep_rows": len(timestep_rows),
        "mobile_cage_rows": len(mobile_rows),
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
