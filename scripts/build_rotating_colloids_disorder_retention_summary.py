#!/usr/bin/env python3
"""Build the publication disorder-retention summary from analyzer reports.

The protocol analyzer deliberately writes one report per system size because
the director contribution subtracted from the overlap is size dependent. This
script performs only the final, publication-level merge: it keeps comparable
N=576 cells, preserves the N=1024 per-graph values, and computes the matched-
graph comparison used in the Letter and Supplemental Material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t


DATA_ROOT = Path(
    "discoveries/theory_experiment_interface/rotating_colloids_hyperion"
)
DEFAULT_INPUT = DATA_ROOT / "rotating_colloids_disorder_retention_protocols"
DEFAULT_OUTPUT = DATA_ROOT / "rotating_colloids_disorder_retention_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_summary(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("at least two graph realizations are required")
    graph_sd = float(array.std(ddof=1))
    return float(array.mean()), graph_sd, graph_sd / math.sqrt(array.size)


def table_by_disorder(report: dict[str, Any], node_count: int) -> dict[float, dict[str, Any]]:
    table: dict[float, dict[str, Any]] = {}
    for row in report["table"]:
        if int(row["node_count"]) != node_count:
            raise ValueError(
                f"report expected N={node_count} but contains N={row['node_count']}"
            )
        key = round(float(row["disorder"]), 8)
        if key in table:
            raise ValueError(f"duplicate disorder cell N={node_count}, sigma/a={key:g}")
        table[key] = row
    return table


def n576_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    table = table_by_disorder(report, 576)
    expected = {0.05, 0.08, 0.11, 0.16, 0.28}
    if set(table) != expected:
        raise ValueError(f"N=576 cells are {sorted(table)}; expected {sorted(expected)}")
    if not table[0.05].get("released_state_split", False):
        raise ValueError("N=576 sigma/a=0.05 must be flagged as a split released state")

    rows = []
    for sigma in (0.08, 0.11, 0.16, 0.28):
        source = table[sigma]
        block = source["connected_write_end"]
        rows.append(
            {
                "sigma_over_a": sigma,
                "graphs": int(source["graphs"]),
                "graph_seeds": source["graph_seeds"],
                "mean": float(block["mean"]),
                "graph_sd": float(block["graph_sd"]),
                "sem": float(block["sem"]),
            }
        )
    return rows


def n1024_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    table = table_by_disorder(report, 1024)
    expected = {0.11, 0.16}
    if set(table) != expected:
        raise ValueError(f"N=1024 cells are {sorted(table)}; expected {sorted(expected)}")

    rows = []
    for sigma in (0.11, 0.16):
        source = table[sigma]
        per_graph = source["per_graph"]
        seeds = [int(value) for value in per_graph["graph_seed"]]
        if seeds != [17, 29, 43, 71, 97]:
            raise ValueError(f"N=1024 sigma/a={sigma:g} graph seeds are {seeds}")
        connected = [float(value) for value in per_graph["connected_write_end"]]
        mean, graph_sd, sem = sample_summary(connected)
        rows.append(
            {
                "sigma_over_a": sigma,
                "graph_seeds": seeds,
                "S_end": [float(value) for value in per_graph["S_end"]],
                "Q_write": [float(value) for value in per_graph["write_end"]],
                "Q_target_conn": connected,
                "mean": mean,
                "graph_sd": graph_sd,
                "sem": sem,
                "connected_split_mean": float(source["connected_split_end"]["mean"]),
                "connected_split_graph_sd": float(source["connected_split_end"]["graph_sd"]),
                "angular_localization_bits_per_rotor_mean": float(
                    source["connected_angular_localization_bits_per_rotor"]["mean"]
                ),
                "angular_localization_bits_per_rotor_graph_sd": float(
                    source["connected_angular_localization_bits_per_rotor"]["graph_sd"]
                ),
            }
        )
    return rows


def comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    low, high = rows
    if low["graph_seeds"] != high["graph_seeds"]:
        raise ValueError("N=1024 comparison requires matched graph seeds")
    a = np.asarray(low["Q_target_conn"], dtype=float)
    b = np.asarray(high["Q_target_conn"], dtype=float)
    difference = a - b

    mean_difference = float(difference.mean())
    paired_sem = float(difference.std(ddof=1) / math.sqrt(difference.size))
    paired_t = mean_difference / paired_sem
    paired_df = int(difference.size - 1)
    paired_p = float(2.0 * student_t.sf(abs(paired_t), paired_df))

    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    welch_sem = math.sqrt(var_a / a.size + var_b / b.size)
    welch_t = mean_difference / welch_sem
    numerator = (var_a / a.size + var_b / b.size) ** 2
    denominator = (var_a / a.size) ** 2 / (a.size - 1) + (var_b / b.size) ** 2 / (b.size - 1)
    welch_df = numerator / denominator
    welch_p = float(2.0 * student_t.sf(abs(welch_t), welch_df))

    return {
        "mean_difference_0p11_minus_0p16": mean_difference,
        "welch_sem": welch_sem,
        "welch_t": welch_t,
        "welch_df": welch_df,
        "welch_p_two_sided": welch_p,
        "paired_sem": paired_sem,
        "paired_t": paired_t,
        "paired_df": paired_df,
        "paired_p_two_sided": paired_p,
        "analysis_note": (
            "Matching graph seeds scale the same Gaussian displacement field and use "
            "common random numbers. The paired estimate is the natural matched-realization "
            "comparison; the Welch estimate is retained as an independent-group comparison."
        ),
    }


def build_summary(input_dir: Path, n576_path: Path, n1024_path: Path) -> dict[str, Any]:
    report576 = read_json(n576_path)
    report1024 = read_json(n1024_path)
    if float(report576["tail_fraction"]) != float(report1024["tail_fraction"]):
        raise ValueError("N=576 and N=1024 reports use different tail fractions")

    rows576 = n576_rows(report576)
    rows1024 = n1024_rows(report1024)
    observation_times = {
        float(row["observation_time"])
        for report in (report576, report1024)
        for row in report["table"]
    }
    if len(observation_times) != 1:
        raise ValueError(f"protocol reports mix observation times {sorted(observation_times)}")

    return {
        "schema": "rotating_colloids_disorder_retention_summary_v3",
        "provenance": {
            "status": "derived_from_raw_protocol_trajectories",
            "description": (
                "Both system-size tables and the matched N=1024 comparison were regenerated "
                "from archived capillary_pair_protocols.json trajectories."
            ),
            "raw_protocol_root": str(input_dir),
            "source_reports": {
                "n576": {"path": str(n576_path), "sha256": sha256(n576_path)},
                "n1024": {"path": str(n1024_path), "sha256": sha256(n1024_path)},
            },
            "raw_n1024_available_locally": True,
            "tail_fraction": float(report576["tail_fraction"]),
            "observation_time_D_r_t": observation_times.pop(),
        },
        "metric": {
            "name": "connected_written_overlap",
            "symbol": "Q_target_conn",
            "definition": "Q_target - S_end^2",
        },
        "n576": rows576,
        "n1024": rows1024,
        "comparison": comparison(rows1024),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--n576-report", type=Path)
    parser.add_argument("--n1024-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    n576_path = args.n576_report or args.input_dir / "analysis_N576/disorder_protocol_report.json"
    n1024_path = args.n1024_report or args.input_dir / "analysis_N1024/disorder_protocol_report.json"
    for path in (n576_path, n1024_path):
        if not path.is_file():
            raise SystemExit(f"missing analyzer report: {path}")

    summary = build_summary(args.input_dir, n576_path, n1024_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "complete": True,
                "output": str(args.output),
                "n576_cells": len(summary["n576"]),
                "n1024_cells": len(summary["n1024"]),
                "paired_p_two_sided": summary["comparison"]["paired_p_two_sided"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
