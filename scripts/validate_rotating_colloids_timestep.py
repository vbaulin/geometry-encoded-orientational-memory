#!/usr/bin/env python3
"""Weak-convergence test for the capillary-rotor observables and memory readouts.

The physical burn-in, observation, write, and release times are held fixed as
the Euler--Maruyama step is reduced. Results are append-only and resumable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rotating_colloids_capillary_pair import (
    make_caged_graph,
    parse_float_list,
    parse_int_list,
    resolve_device,
    simulate_ensemble,
    split_replica_protocol,
    summarize_run,
    write_release_protocol,
)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def steps(duration: float, dt: float) -> int:
    return max(1, int(round(float(duration) / float(dt))))


def read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_report(rows: List[Dict[str, object]]) -> Dict[str, object]:
    observables = (
        "S_mean",
        "C2_mean",
        "G2_mean",
        "q_EA_mean",
        "window_autocorrelation",
        "split_endpoint",
        "written_endpoint",
    )
    by_dt: Dict[float, List[Dict[str, object]]] = {}
    for row in rows:
        by_dt.setdefault(float(row["dt"]), []).append(row)
    finest = min(by_dt)
    reference_rows = {int(row["graph_seed"]): row for row in by_dt[finest]}
    reference = {
        key: float(np.mean([float(row[key]) for row in reference_rows.values()]))
        for key in observables
    }
    summaries = []
    for dt in sorted(by_dt, reverse=True):
        group = by_dt[dt]
        values = {
            key: float(np.mean([float(row[key]) for row in group]))
            for key in observables
        }
        sems = {
            key: (
                float(np.std([float(row[key]) for row in group], ddof=1))
                / math.sqrt(len(group))
                if len(group) > 1 else 0.0
            )
            for key in observables
        }
        group_rows = {int(row["graph_seed"]): row for row in group}
        common_graphs = sorted(set(group_rows) & set(reference_rows))
        paired = {}
        for key in observables:
            differences = np.asarray(
                [
                    float(group_rows[seed][key])
                    - float(reference_rows[seed][key])
                    for seed in common_graphs
                ],
                dtype=float,
            )
            paired[key] = {
                "mean": float(differences.mean()) if differences.size else float("nan"),
                "sem": (
                    float(differences.std(ddof=1) / math.sqrt(differences.size))
                    if differences.size > 1 else 0.0
                ),
                "graph_count": int(differences.size),
            }
        summaries.append(
            {
                "dt": dt,
                "graph_count": len({int(row["graph_seed"]) for row in group}),
                "means": values,
                "sems": sems,
                "absolute_difference_from_finest": {
                    key: abs(values[key] - reference[key]) for key in observables
                },
                "paired_difference_from_finest": paired,
            }
        )
    return {
        "row_count": len(rows),
        "finest_dt": finest,
        "observables": list(observables),
        "summaries": summaries,
        "diagnostic": (
            "Weak convergence is reported as graph-averaged and matched-graph "
            "differences from the finest simulated step. No post-hoc pass threshold "
            "is applied."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "rotating_colloids_timestep_validation"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--graph-seeds", default="17,29,43,71,97")
    parser.add_argument("--dts", default="0.0025,0.00125,0.000625")
    parser.add_argument("--replicas", type=int, default=24)
    parser.add_argument("--j-align", type=float, default=4.0)
    parser.add_argument("--g-capillary", type=float, default=5.0)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--equilibration-time", type=float, default=75.0)
    parser.add_argument("--observation-time", type=float, default=125.0)
    parser.add_argument("--write-time", type=float, default=75.0)
    parser.add_argument("--release-time", type=float, default=125.0)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    parser.add_argument("--write-field", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.quick:
        args.device = "cpu"
        args.n = min(args.n, 6)
        args.graph_seeds = str(parse_int_list(args.graph_seeds)[0])
        args.dts = "0.01,0.005"
        args.replicas = min(args.replicas, 4)
        args.equilibration_time = 0.5
        args.observation_time = 1.0
        args.write_time = 0.5
        args.release_time = 1.0
        args.sample_interval = 0.1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "timestep_validation.jsonl"
    if args.no_resume and rows_path.exists():
        rows_path.unlink()
    existing = {
        (int(row["graph_seed"]), float(row["dt"])) for row in read_rows(rows_path)
    }
    device = resolve_device(args.device)
    dts = sorted(parse_float_list(args.dts), reverse=True)
    graph_seeds = parse_int_list(args.graph_seeds)

    total = len(dts) * len(graph_seeds)
    completed = 0
    for graph_seed in graph_seeds:
        graph = make_caged_graph(
            args.n,
            disorder=args.disorder,
            cutoff=2.6,
            alignment_range=1.35,
            alignment_decay=0.20,
            seed=graph_seed,
        )
        for dt in dts:
            completed += 1
            if (graph_seed, dt) in existing:
                continue
            stride = max(1, steps(args.sample_interval, dt))
            equilibrium = simulate_ensemble(
                graph,
                j_align=args.j_align,
                g_capillary=args.g_capillary,
                replicas=args.replicas,
                burn_in_steps=steps(args.equilibration_time, dt),
                sample_steps=steps(args.observation_time, dt),
                sample_stride=stride,
                dt=dt,
                seed=args.seed + graph_seed * 1009,
                device=device,
            )
            summary = summarize_run(equilibrium)
            split = split_replica_protocol(
                graph,
                j_align=args.j_align,
                g_capillary=args.g_capillary,
                parents=max(2, args.replicas // 2),
                equilibration_steps=steps(args.equilibration_time, dt),
                observation_steps=steps(args.observation_time, dt),
                stride=stride,
                dt=dt,
                seed=args.seed + graph_seed * 2003,
                device=device,
            )
            written = write_release_protocol(
                graph,
                j_align=args.j_align,
                g_capillary=args.g_capillary,
                replicas=args.replicas,
                equilibration_steps=steps(args.equilibration_time, dt),
                write_steps=steps(args.write_time, dt),
                release_steps=steps(args.release_time, dt),
                stride=stride,
                dt=dt,
                write_field=args.write_field,
                seed=args.seed + graph_seed * 3001,
                device=device,
            )
            row = {
                "graph_seed": graph_seed,
                "n": args.n,
                "node_count": args.n * args.n,
                "dt": dt,
                "replicas": args.replicas,
                "equilibration_time": args.equilibration_time,
                "observation_time": args.observation_time,
                "write_time": args.write_time,
                "release_time": args.release_time,
                **{key: float(summary[key]) for key in (
                    "S_mean", "C2_mean", "G2_mean", "q_EA_mean",
                    "window_autocorrelation",
                )},
                "split_endpoint": float(split["overlap_mean"][-1]),
                "written_endpoint": float(written["release_overlap"][-1]),
            }
            append_jsonl(rows_path, row)
            print(json.dumps({
                "event": "timestep_progress",
                "completed": completed,
                "total": total,
                "graph_seed": graph_seed,
                "dt": dt,
                "split_endpoint": row["split_endpoint"],
                "written_endpoint": row["written_endpoint"],
            }), flush=True)

    report = build_report(read_rows(rows_path))
    report["parameters"] = vars(args)
    (out_dir / "timestep_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
