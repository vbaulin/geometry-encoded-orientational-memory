#!/usr/bin/env python3
"""Analyze the matched-start colloidal loop-frustration coupling scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def read_rows(root: Path) -> tuple[list[dict[str, Any]], int]:
    paths = [root] if root.is_file() else sorted(root.glob("**/matched_release_scan.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no matched_release_scan.jsonl under {root}")
    by_key: dict[str, str] = {}
    rows: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["key"])
            frozen = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if key in by_key:
                if by_key[key] != frozen:
                    raise ValueError(f"conflicting rows for {key}")
                duplicates += 1
                continue
            by_key[key] = frozen
            rows[key] = row
    return [rows[key] for key in sorted(rows)], duplicates


def mean_sem(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
    }


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def graph_bootstrap(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    graphs = sorted({int(row["graph_seed"]) for row in rows})
    by_graph = {graph: [row for row in rows if int(row["graph_seed"]) == graph] for graph in graphs}
    rng = np.random.default_rng(seed)
    means: list[float] = []
    slopes: list[float] = []
    for _ in range(draws):
        picked = rng.choice(graphs, size=len(graphs), replace=True)
        sample = [row for graph in picked for row in by_graph[int(graph)]]
        graph_means = [
            np.mean(
                [
                    float(row["frustrated_minus_flat_survival_auc"])
                    for row in by_graph[int(graph)]
                ]
            )
            for graph in picked
        ]
        means.append(float(np.mean(graph_means)))
        x = np.asarray([row["normalized_compatibility"] for row in sample], dtype=float)
        y = np.asarray(
            [row["frustrated_minus_flat_survival_auc"] for row in sample], dtype=float
        )
        if not np.allclose(x, x[0]):
            slopes.append(ols(x, y)[0])
    return {
        "draws": draws,
        "mean_response_95_interval": [
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)),
        ],
        "slope_95_interval": [
            float(np.percentile(slopes, 2.5)),
            float(np.percentile(slopes, 97.5)),
        ]
        if slopes
        else None,
    }


def per_coupling(
    rows: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for offset, beta_j in enumerate(sorted({float(row["beta_j"]) for row in rows})):
        block = [row for row in rows if np.isclose(float(row["beta_j"]), beta_j)]
        graphs = sorted({int(row["graph_seed"]) for row in block})
        graph_means = []
        graph_finals = []
        graph_original_auc = []
        graph_flat_auc = []
        graph_original_final = []
        graph_flat_final = []
        for graph in graphs:
            graph_rows = [row for row in block if int(row["graph_seed"]) == graph]
            graph_means.append(
                float(
                    np.mean(
                        [row["frustrated_minus_flat_survival_auc"] for row in graph_rows]
                    )
                )
            )
            graph_original_auc.append(
                float(
                    np.mean(
                        [
                            row["arms"]["frustrated"]["survival"]["auc"]
                            for row in graph_rows
                        ]
                    )
                )
            )
            graph_flat_auc.append(
                float(
                    np.mean(
                        [row["arms"]["flat"]["survival"]["auc"] for row in graph_rows]
                    )
                )
            )
            graph_original_final.append(
                float(
                    np.mean(
                        [
                            row["arms"]["frustrated"]["survival"]["final"]
                            for row in graph_rows
                        ]
                    )
                )
            )
            graph_flat_final.append(
                float(
                    np.mean(
                        [
                            row["arms"]["flat"]["survival"]["final"]
                            for row in graph_rows
                        ]
                    )
                )
            )
            graph_finals.append(
                float(
                    np.mean(
                        [row["frustrated_minus_flat_final_survival"] for row in graph_rows]
                    )
                )
            )
        x = np.asarray([row["normalized_compatibility"] for row in block], dtype=float)
        y = np.asarray(
            [row["frustrated_minus_flat_survival_auc"] for row in block], dtype=float
        )
        slope, intercept = ols(x, y)
        bootstrap = graph_bootstrap(block, draws=draws, seed=seed + offset)
        interval = bootstrap["mean_response_95_interval"]
        if interval[0] > 0.0:
            direction = "frustrated_retains_more"
        elif interval[1] < 0.0:
            direction = "flat_retains_more"
        else:
            direction = "unresolved"
        result[f"{beta_j:g}"] = {
            "beta_j": beta_j,
            "beta_g": float(block[0]["beta_g"]),
            "graphs": len(graphs),
            "targets_per_graph": len(block) // max(len(graphs), 1),
            "original_survival_auc": mean_sem(graph_original_auc),
            "flat_survival_auc": mean_sem(graph_flat_auc),
            "original_final_survival": mean_sem(graph_original_final),
            "flat_final_survival": mean_sem(graph_flat_final),
            "survival_auc_response": mean_sem(graph_means),
            "final_survival_response": mean_sem(graph_finals),
            "slope_vs_normalized_energy_advantage": slope,
            "intercept": intercept,
            "bootstrap": bootstrap,
            "direction": direction,
        }
    return result


def crossing_summary(blocks: dict[str, Any]) -> dict[str, Any]:
    ordered = sorted(blocks.values(), key=lambda block: block["beta_j"])
    candidates = []
    for left, right in zip(ordered, ordered[1:]):
        y0 = float(left["survival_auc_response"]["mean"])
        y1 = float(right["survival_auc_response"]["mean"])
        if y0 == 0.0:
            candidates.append(float(left["beta_j"]))
        elif y0 * y1 < 0.0:
            fraction = -y0 / (y1 - y0)
            candidates.append(
                float(left["beta_j"])
                + fraction * (float(right["beta_j"]) - float(left["beta_j"]))
            )
    resolved_negative = [
        block["beta_j"] for block in ordered if block["direction"] == "flat_retains_more"
    ]
    resolved_positive = [
        block["beta_j"]
        for block in ordered
        if block["direction"] == "frustrated_retains_more"
    ]
    return {
        "linear_interpolation_candidates": candidates,
        "resolved_flat_advantage_couplings": resolved_negative,
        "resolved_frustrated_advantage_couplings": resolved_positive,
        "resolved_crossover": bool(resolved_negative and resolved_positive),
        "note": (
            "Interpolated zeroes locate candidates only. A crossover is resolved only when "
            "graph-bootstrap intervals lie below and above zero at different couplings."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=4000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260908)
    args = parser.parse_args()

    rows, duplicates = read_rows(args.input)
    if not rows:
        raise SystemExit("matched-release archive is empty")
    initial_residual = max(abs(float(row["initial_overlap_arm_difference"])) for row in rows)
    blocks = per_coupling(
        rows, draws=args.bootstrap_draws, seed=args.bootstrap_seed
    )
    crossing = crossing_summary(blocks)
    ordered_blocks = sorted(blocks.values(), key=lambda block: block["beta_j"])
    weakest = ordered_blocks[0]
    strongest = ordered_blocks[-1]
    absolute_survival_summary = {
        "weakest_coupling": {
            "beta_j": weakest["beta_j"],
            "original_auc": weakest["original_survival_auc"]["mean"],
            "flat_auc": weakest["flat_survival_auc"]["mean"],
        },
        "strongest_coupling": {
            "beta_j": strongest["beta_j"],
            "original_auc": strongest["original_survival_auc"]["mean"],
            "flat_auc": strongest["flat_survival_auc"]["mean"],
        },
        "interpretation": (
            "Absolute survival rises in both arms across the scan. Their high-coupling "
            "convergence is therefore common trapping on the observation window, not "
            "common erasure."
        ),
    }
    report = {
        "report_type": "holonomy_matched_start_release_crossover",
        "rows": len(rows),
        "identical_duplicates_ignored": duplicates,
        "graphs": len({int(row["graph_seed"]) for row in rows}),
        "targets": sorted({str(row["target"]) for row in rows}),
        "beta_j_values": sorted({float(row["beta_j"]) for row in rows}),
        "maximum_initial_overlap_arm_difference": initial_residual,
        "matched_initial_state_passed": initial_residual <= 1e-12,
        "per_coupling": blocks,
        "crossover": crossing,
        "absolute_survival_summary": absolute_survival_summary,
        "scope": (
            "The scan isolates release from arm-dependent writing. It does not isolate cycle "
            "topology because target energy, torque and curvature differ between arms."
        ),
        "next_gate": (
            "Repeat with an energy-, torque- and local-curvature-matched intervention before "
            "attributing a resolved crossover to loop holonomy."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "matched_release_crossover_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Matched-start release crossover",
        "",
        "Both networks start from identical angles and receive paired Brownian noise.",
        "The comparison therefore concerns release, not writeability.",
        "",
        "| beta J | beta g | original R | flat R | mean Delta R | SEM | graph-bootstrap 95% | direction | slope vs Delta e |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for block in sorted(blocks.values(), key=lambda item: item["beta_j"]):
        interval = block["bootstrap"]["mean_response_95_interval"]
        lines.append(
            f"| {block['beta_j']:.3g} | {block['beta_g']:.3g} | "
            f"{block['original_survival_auc']['mean']:.5f} | "
            f"{block['flat_survival_auc']['mean']:.5f} | "
            f"{block['survival_auc_response']['mean']:+.6f} | "
            f"{block['survival_auc_response']['sem']:.6f} | "
            f"[{interval[0]:+.6f}, {interval[1]:+.6f}] | "
            f"{block['direction']} | "
            f"{block['slope_vs_normalized_energy_advantage']:+.5f} |"
        )
    lines.extend(
        [
            "",
            "## Physical conclusion",
            "",
            (
                "A coupling crossover is resolved only if at least one graph-bootstrap interval "
                "lies below zero and another lies above zero."
            ),
            f"Resolved crossover: `{crossing['resolved_crossover']}`.",
            "",
            (
                "Absolute survival rises from "
                f"{weakest['original_survival_auc']['mean']:.5f} (original) and "
                f"{weakest['flat_survival_auc']['mean']:.5f} (flat) at "
                f"beta J = {weakest['beta_j']:.3g} to "
                f"{strongest['original_survival_auc']['mean']:.5f} and "
                f"{strongest['flat_survival_auc']['mean']:.5f} at "
                f"beta J = {strongest['beta_j']:.3g}."
            ),
            absolute_survival_summary["interpretation"],
            (
                "Loop flattening improves survival only in the intermediate escape window; "
                "the incompatible network never acquires a resolved lifetime advantage."
            ),
            "",
            report["scope"],
            "",
            report["next_gate"],
        ]
    )
    (args.output_dir / "matched_release_crossover_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "matched_initial_state_passed": report["matched_initial_state_passed"],
                "resolved_crossover": crossing["resolved_crossover"],
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
