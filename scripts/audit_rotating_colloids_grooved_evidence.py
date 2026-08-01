#!/usr/bin/env python3
"""Audit the evidential status of the grooved rotating-colloid scans.

The audit separates three questions that were previously mixed together:

1. Are the stored parameter grids complete and internally consistent?
2. Why do the observable-space plots collapse to narrow curves or clusters?
3. Do the existing observables establish a phase transition or remanent memory?

The output is deliberately conservative.  It reports what the completed scans
measure and identifies which additional protocol is required for a memory or
phase-transition claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")

RUNS = (
    ("uniform", "rotating_colloids_grooved_uniform_scan_n16"),
    ("uniform_memory", "rotating_colloids_grooved_uniform_memory_zoom_n16"),
    ("uniform_size", "rotating_colloids_grooved_uniform_finite_size"),
    ("long_range", "rotating_colloids_grooved_longrange_disorder"),
    ("triangular", "rotating_colloids_grooved_triangular_frustrated_n16"),
    ("mosaic", "rotating_colloids_grooved_mosaic_hidden_search_n32"),
)

OBSERVABLES = (
    "nematic_order_mean",
    "orientational_corr_nn_mean",
    "geometry_lock_mean",
    "q_EA_mean",
)

CONFIG_FIELDS = (
    "noise",
    "dt",
    "burn_in_steps",
    "sample_steps",
    "sample_stride",
    "replicates",
    "graph_mode",
    "constraint_mode",
    "cluster_size",
    "crosslink_k",
    "crosslink_weight",
    "easy_axis_disorder",
)


def finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_rows(folder: Path) -> list[dict[str, Any]]:
    paths = sorted(folder.glob("*_points.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one points JSONL in {folder}, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    with paths[0].open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {paths[0]}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def distinct(rows: list[dict[str, Any]], key: str) -> list[Any]:
    values = {json.dumps(row.get(key), sort_keys=True) for row in rows}
    return [json.loads(value) for value in sorted(values)]


def corr(x: np.ndarray, y: np.ndarray) -> float | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def observable_dimension(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = np.asarray([[finite(row.get(key)) for key in OBSERVABLES] for row in rows], dtype=float)
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.shape[0] < 4:
        return {"rows": int(matrix.shape[0]), "variance_ratio": [], "effective_dimension": None}
    std = np.std(matrix, axis=0)
    keep = std > 1e-12
    z = (matrix[:, keep] - np.mean(matrix[:, keep], axis=0)) / std[keep]
    singular = np.linalg.svd(z, full_matrices=False, compute_uv=False)
    variance = singular**2
    ratio = variance / np.sum(variance)
    effective = 1.0 / float(np.sum(ratio**2))
    return {
        "rows": int(matrix.shape[0]),
        "variance_ratio": [float(value) for value in ratio],
        "effective_dimension": effective,
        "first_two_variance": float(np.sum(ratio[:2])),
    }


def grid_completeness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_size: dict[str, Any] = {}
    for n in sorted({int(row["n"]) for row in rows}):
        subset = [row for row in rows if int(row["n"]) == n]
        eps = sorted({finite(row.get("eps_geom")) for row in subset})
        js = sorted({finite(row.get("j_align")) for row in subset})
        observed = {(finite(row.get("eps_geom")), finite(row.get("j_align"))) for row in subset}
        expected = {(e, j) for e in eps for j in js}
        by_size[str(n)] = {
            "rows": len(subset),
            "eps_count": len(eps),
            "j_count": len(js),
            "expected_cells": len(expected),
            "missing_cells": len(expected - observed),
        }
    keys = [str(row.get("point_key", "")) for row in rows]
    return {
        "by_size": by_size,
        "duplicate_point_keys": len(keys) - len(set(keys)),
        "empty_point_keys": sum(not key for key in keys),
    }


def transition_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("susceptibility_S_mean", "susceptibility_G_mean"):
        values = np.asarray([finite(row.get(key)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            median = float(np.median(values))
            maximum = float(np.max(values))
            out[key] = {
                "median": median,
                "maximum": maximum,
                "max_to_median": maximum / median if median > 1e-12 else None,
            }
    for key in ("binder_S_mean", "binder_G_mean"):
        values = np.asarray([finite(row.get(key)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            out[key] = {"minimum": float(np.min(values)), "maximum": float(np.max(values))}
    out["sizes"] = sorted({int(row["n"]) for row in rows})
    out["finite_size_transition_test_available"] = len(out["sizes"]) >= 3
    return out


def run_audit(label: str, folder: Path) -> dict[str, Any]:
    rows = load_rows(folder)
    g = np.asarray([finite(row.get("geometry_lock_mean")) for row in rows], dtype=float)
    q = np.asarray([finite(row.get("q_EA_mean")) for row in rows], dtype=float)
    lag20 = np.asarray([finite(row.get("temporal_corr_lag20_mean")) for row in rows], dtype=float)
    s = np.asarray([finite(row.get("nematic_order_mean")) for row in rows], dtype=float)
    c2 = np.asarray([finite(row.get("orientational_corr_nn_mean")) for row in rows], dtype=float)
    pinning_residual = q - g**2
    valid = np.isfinite(pinning_residual)
    hidden = (s <= 0.35) & (c2 >= 0.70) & (g >= 0.70) & (q >= 0.50)
    noise_values = np.asarray([finite(row.get("noise")) for row in rows], dtype=float)
    j_values = np.asarray([finite(row.get("j_align")) for row in rows], dtype=float)
    eps_values = np.asarray([finite(row.get("eps_geom")) for row in rows], dtype=float)
    j_over_dr = j_values / noise_values
    h_over_dr = eps_values * j_values / noise_values

    return {
        "label": label,
        "folder": str(folder),
        "row_count": len(rows),
        "grid": grid_completeness(rows),
        "config_variants": {key: distinct(rows, key) for key in CONFIG_FIELDS},
        "dimensionless_control_ranges": {
            "J_over_Dr": [float(np.nanmin(j_over_dr)), float(np.nanmax(j_over_dr))],
            "h_over_Dr": [float(np.nanmin(h_over_dr)), float(np.nanmax(h_over_dr))],
            "epsilon_equals_h_over_J": [float(np.nanmin(eps_values)), float(np.nanmax(eps_values))],
        },
        "observable_ranges": {
            "S": [float(np.nanmin(s)), float(np.nanmax(s))],
            "C2": [float(np.nanmin(c2)), float(np.nanmax(c2))],
            "G2": [float(np.nanmin(g)), float(np.nanmax(g))],
            "qEA": [float(np.nanmin(q)), float(np.nanmax(q))],
        },
        "hidden_threshold_cells": int(np.count_nonzero(hidden)),
        "observable_space": observable_dimension(rows),
        "pinning_identity": {
            "corr_qEA_G2_squared": corr(q, g**2),
            "mean_qEA_minus_G2_squared": float(np.mean(pinning_residual[valid])) if np.any(valid) else None,
            "rmse_qEA_minus_G2_squared": float(np.sqrt(np.mean(pinning_residual[valid] ** 2))) if np.any(valid) else None,
            "corr_qEA_lag20": corr(q, lag20),
        },
        "transition": transition_diagnostics(rows),
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Grooved rotating-colloid evidence audit",
        "",
        "The stored grain is one parameter cell at fixed graph construction and lattice size. "
        r"The dynamical equation uses `noise` as the rotational diffusion coefficient "
        r"\(D_r\), `j_align` as \(J\), and the local groove field "
        r"\(h=\epsilon J\). Consequently, the physical control coordinates are "
        r"\(J/D_r\) and \(h/D_r\), not the stored columns \(J\) and \(\epsilon\) themselves.",
        "",
        "## Data integrity",
        "",
        "| run | rows | duplicates | missing grid cells | config variants |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        missing = sum(item["missing_cells"] for item in run["grid"]["by_size"].values())
        variants = sum(len(values) > 1 for values in run["config_variants"].values())
        lines.append(
            f"| {run['label']} | {run['row_count']} | {run['grid']['duplicate_point_keys']} | "
            f"{missing} | {variants} |"
        )

    lines.extend(
        [
            "",
            "## Why the state-space traces are narrow",
            "",
            r"The table reports the fraction of standardized observable variance captured by the first principal component and the participation-ratio dimension of \((S,C_2,G_2,q_{\rm EA})\). A value near one for the first component means that a nominally two-parameter scan follows an almost one-dimensional response trajectory after projection into observable space.",
            "",
            "| run | PC1 variance | PC1+PC2 variance | effective dimension |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        dim = run["observable_space"]
        ratios = dim.get("variance_ratio", [])
        pc1 = ratios[0] if ratios else None
        lines.append(
            f"| {run['label']} | {fmt(pc1)} | {fmt(dim.get('first_two_variance'))} | "
            f"{fmt(dim.get('effective_dimension'))} |"
        )

    lines.extend(
        [
            "",
            "## Registration versus memory",
            "",
            r"For a stationary distribution symmetric about a fixed local groove axis \(\alpha_i\),",
            "",
            r"\[",
            r"\langle e^{2i\theta_i}\rangle_t = e^{2i\alpha_i}g_i,",
            r"\qquad",
            r"q_{\rm EA}=N^{-1}\sum_i g_i^2.",
            r"\]",
            "",
            r"If the local registration amplitudes \(g_i\) are similar, then \(q_{\rm EA}\simeq G_2^2\). Agreement with this identity means that the present \(q_{\rm EA}\) is largely a readout of static groove pinning. It does not demonstrate remanence after the groove field is removed.",
            "",
            r"| run | corr\((q_{\rm EA},G_2^2)\) | RMSE \(q_{\rm EA}-G_2^2\) | hidden threshold cells |",
            "|---|---:|---:|---:|",
        ]
    )
    for run in report["runs"]:
        identity = run["pinning_identity"]
        lines.append(
            f"| {run['label']} | {fmt(identity.get('corr_qEA_G2_squared'))} | "
            f"{fmt(identity.get('rmse_qEA_minus_G2_squared'))} | {run['hidden_threshold_cells']} |"
        )

    lines.extend(
        [
            "",
            "## Evidential conclusion",
            "",
            "The completed scans establish a geometry-registered orientational state in mosaic grooves: global nematic order is suppressed while local rod alignment and local groove registration remain high. They do not yet establish a thermodynamic glass phase or remanent memory. The sharp colored boundaries in previous regime maps came from diagnostic thresholds, and the mosaic scan has only one system size. A phase claim requires finite-size susceptibility or Binder evidence. A memory claim requires a write-release-read test in which the local groove field is reduced or removed and retention is compared with a noninteracting control.",
            "",
            "The direct application supported by the current model is a patterned orientational register: a substrate writes a spatially varying optical or dielectric anisotropy into a dense rod monolayer while keeping the sample globally isotropic. Persistent material memory is a separate prediction to test with the release protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "grooved_evidence_audit",
        help="Output path without extension",
    )
    args = parser.parse_args()

    runs = []
    for label, folder_name in RUNS:
        folder = args.root / folder_name
        if not folder.exists():
            raise FileNotFoundError(folder)
        runs.append(run_audit(label, folder))

    report = {
        "control_coordinate_definition": {
            "J_over_Dr": "j_align / noise",
            "h_over_Dr": "eps_geom * j_align / noise",
            "epsilon": "h / J",
        },
        "runs": runs,
        "evidence_status": {
            "geometry_registered_state": "supported",
            "remanent_memory": "not tested",
            "thermodynamic_phase_transition": "not established",
            "reason": "qEA is measured under the static pinning field and the mosaic scan has one system size",
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
