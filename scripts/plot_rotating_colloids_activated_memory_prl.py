#!/usr/bin/env python3
"""Build the coupling-dependent retention figure from the long GPU scan.

The report written next to the figure records every quantity the Letter quotes
from Fig. 4, including the panel (a) retention surface and the panel (c)
observation-window statistics.  When a previous report is present the script
also writes a delta so that a rerun cannot silently move a published number.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Reference times reported for the panel (a) retention surface.
SURFACE_REFERENCE_TIMES = (1.0, 5.0, 10.0, 50.0, 100.0, 250.0, 625.0)

# Couplings whose Fig. 4(b) values are quoted in the Letter.
QUOTED_LAMBDAS = (0.45, 0.6, 0.9)


def load_rows(input_dir: Path):
    if not input_dir.exists():
        raise SystemExit(
            f"input directory does not exist: {input_dir}\n"
            "The activated-memory scan is cluster-generated. Produce it with\n"
            "  bash scripts/run_rotating_colloids_activated_memory_prl_gpu.sh\n"
            "on the GPU host, or install the Zenodo deposit, before rebuilding Fig. 4."
        )
    paths = glob.glob(str(input_dir / "**" / "activated_memory_scan.jsonl"), recursive=True)
    if not paths:
        raise SystemExit(
            f"no activated_memory_scan.jsonl under {input_dir}\n"
            "Expected one shard file per GPU rank, for example\n"
            f"  {input_dir}/seeds_17_29_43_71_97/activated_memory_scan.jsonl"
        )
    return [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]


def interpolate_curve(curve, grid):
    return np.interp(grid, np.asarray(curve["time"], dtype=float), np.asarray(curve["overlap_mean"], dtype=float))


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def window_statistics(groups, lambdas):
    """Panel (c): finite-window q_EA against read window, per coupling."""
    statistics = {}
    for lam in lambdas:
        by_window = defaultdict(list)
        for row in groups[lam]:
            for item in row["observation_window"]["windows"]:
                by_window[float(item["window"])].append(float(item["q_EA_mean"]))
        windows = sorted(by_window)
        statistics[f"{lam:g}"] = {
            "window": windows,
            "q_EA_mean": [float(np.mean(by_window[window])) for window in windows],
            "q_EA_sem": [
                float(np.std(by_window[window], ddof=1) / np.sqrt(len(by_window[window]))) for window in windows
            ],
        }
    return statistics


def compare_reports(previous: dict, current: dict) -> dict:
    """Element-wise deviation between a frozen report and a fresh one."""
    deltas = {}
    worst = 0.0
    for key, block in current["integral_times"].items():
        old = previous.get("integral_times", {}).get(key)
        if old is None:
            deltas[key] = {"status": "absent_in_previous_report"}
            continue
        new_mean = np.asarray(block["mean"], dtype=float)
        old_mean = np.asarray(old["mean"], dtype=float)
        if new_mean.shape != old_mean.shape:
            deltas[key] = {"status": "shape_changed"}
            worst = float("inf")
            continue
        absolute = np.abs(new_mean - old_mean)
        relative = absolute / np.maximum(np.abs(old_mean), 1e-12)
        worst = max(worst, float(relative.max()))
        deltas[key] = {
            "status": "compared",
            "max_absolute_change": float(absolute.max()),
            "max_relative_change": float(relative.max()),
            "previous_mean": old_mean.tolist(),
            "current_mean": new_mean.tolist(),
        }
    return {"max_relative_change": worst, "per_series": deltas}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(Path(args.input_dir))
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["lambda"])].append(row)
    lambdas = sorted(groups)
    if any(len(groups[lam]) < 3 for lam in lambdas):
        short = {f"{lam:g}": len(groups[lam]) for lam in lambdas if len(groups[lam]) < 3}
        raise SystemExit(
            "at least three graph realizations are required per coupling; "
            f"incomplete couplings: {json.dumps(short)}\n"
            "Rerun the scan; it resumes completed graph/coupling points."
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.35), constrained_layout=True)

    # (a) Complete retention surface, averaged over quenched graphs.
    tmax = min(float(row["protocols"]["physical"]["split"]["time"][-1]) for row in rows)
    time_grid = np.concatenate(([0.0], np.geomspace(0.5, tmax, 180)))
    retention = []
    for lam in lambdas:
        curves = [interpolate_curve(row["protocols"]["physical"]["split"], time_grid) for row in groups[lam]]
        retention.append(np.mean(curves, axis=0))
    retention = np.asarray(retention)
    xplot = np.log10(1.0 + time_grid)
    ax = axes[0]
    im = ax.pcolormesh(xplot, lambdas, retention, shading="nearest", cmap="magma", vmin=0.0, vmax=1.0)
    contour = ax.contour(xplot, lambdas, retention, levels=(0.2, 0.4, 0.6, 0.8), colors="white", linewidths=0.55, alpha=0.8)
    ax.clabel(contour, fmt="%.1f", fontsize=5.8, inline_spacing=2)
    ticks = np.asarray([0, 1, 10, 100, 625], dtype=float)
    ticks = ticks[ticks <= tmax + 1e-9]
    ax.set_xticks(np.log10(1.0 + ticks), [f"{value:g}" for value in ticks])
    ax.set(xlabel=r"time $D_rt$", ylabel=r"coupling scale $\lambda$", title="retention landscape")
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.055, label=r"$Q_{\rm split}$")

    # (b) Finite-window integrated overlap avoids forcing a decay law.
    ax = axes[1]
    styles = (
        ("physical", "split_summary", "#2166ac", "o", "split replicas"),
        ("physical", "release_summary", "#1b9e77", "s", "written state"),
        ("no_capillary", "split_summary", "#b2182b", "o", r"$g=0$ split"),
        ("no_capillary", "release_summary", "#ef8a62", "s", r"$g=0$ written"),
    )
    summary = {}
    for protocol, key, color, marker, label in styles:
        means, errors = [], []
        for lam in lambdas:
            values = [float(row["protocols"][protocol][key]["positive_integral_time"]) for row in groups[lam]]
            mean, sem = mean_sem(values)
            means.append(float(mean)); errors.append(float(sem))
        ax.errorbar(lambdas, means, yerr=errors, color=color, marker=marker, ms=3.2, lw=1.2, capsize=1.5, label=label)
        summary[f"{protocol}_{key}"] = {"mean": means, "sem": errors}
    ax.set(
        xlabel=r"coupling scale $\lambda$",
        ylabel=r"integrated overlap $D_r\mathcal{A}_Q$",
        title="coupling extends retained overlap",
        yscale="log",
    )
    ax.legend(frameon=False, ncol=1, loc="center right", bbox_to_anchor=(0.99, 0.50))

    # (c) Observation-window dependence of the finite-window statistic.
    windows_by_lambda = window_statistics(groups, lambdas)
    ax = axes[2]
    selected = [lambdas[0], lambdas[len(lambdas) // 2], lambdas[-1]]
    colors = ("#4575b4", "#7b3294", "#d73027")
    for lam, color in zip(selected, colors):
        block = windows_by_lambda[f"{lam:g}"]
        ax.errorbar(
            block["window"], block["q_EA_mean"], yerr=block["q_EA_sem"],
            color=color, marker="o", ms=3.0, lw=1.2, capsize=1.5, label=rf"$\lambda={lam:g}$",
        )
    ax.set(xlabel=r"observation window $D_rT_{\rm obs}$", ylabel=r"finite-window $q_{\rm EA}$", title="memory depends on read time", xscale="log", ylim=(0, 1.02))
    ax.legend(frameon=False)

    for label, ax in zip("abc", axes):
        ax.text(0.02, 0.97, f"{label}", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8})
        ax.grid(alpha=0.15, lw=0.5)

    stem = out / "fig4_activated_memory_results"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)

    reference_times = [value for value in SURFACE_REFERENCE_TIMES if value <= tmax + 1e-9]
    surface = {
        f"{lam:g}": np.interp(reference_times, time_grid, retention[index]).tolist()
        for index, lam in enumerate(lambdas)
    }
    quoted = {}
    for lam in QUOTED_LAMBDAS:
        if lam not in groups:
            continue
        index = lambdas.index(lam)
        quoted[f"{lam:g}"] = {
            key: {"mean": summary[key]["mean"][index], "sem": summary[key]["sem"][index]}
            for key in summary
        }
        quoted[f"{lam:g}"]["split_ratio_to_g0"] = (
            summary["physical_split_summary"]["mean"][index] / summary["no_capillary_split_summary"]["mean"][index]
        )
        quoted[f"{lam:g}"]["release_ratio_to_g0"] = (
            summary["physical_release_summary"]["mean"][index] / summary["no_capillary_release_summary"]["mean"][index]
        )
    longest_window = {
        key: {"window": block["window"][-1], "q_EA_mean": block["q_EA_mean"][-1], "q_EA_sem": block["q_EA_sem"][-1]}
        for key, block in windows_by_lambda.items()
    }

    report = {
        "rows": len(rows),
        "lambdas": lambdas,
        "graphs_per_lambda": {str(lam): len(groups[lam]) for lam in lambdas},
        "metric_definition": {
            "name": "finite_window_positive_integrated_overlap",
            "symbol": "A_Q",
            "formula": "integral_0^Tobs max(Q(t), 0) dt / abs(Q(0))",
            "note": "This finite-window area is not an intrinsic relaxation lifetime.",
        },
        "integral_times": summary,
        "retention_surface": {
            "description": "Panel (a): graph-averaged Q_split at reference rotational times.",
            "reference_times": reference_times,
            "Q_split_mean": surface,
            "observed_time_span": tmax,
        },
        "window_statistics": {
            "description": "Panel (c): graph-averaged finite-window q_EA against read window.",
            "per_lambda": windows_by_lambda,
            "longest_window": longest_window,
        },
        "manuscript_values": {
            "description": "Quantities quoted from Fig. 4 in the Letter.",
            "integrated_overlap": quoted,
            "q_EA_at_longest_window": longest_window,
        },
    }
    report_path = out / "activated_memory_figure_report.json"
    previous = None
    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    result = {"output": str(stem), "rows": len(rows)}
    if previous is not None:
        delta = compare_reports(previous, report)
        (out / "activated_memory_report_delta.json").write_text(json.dumps(delta, indent=2) + "\n", encoding="utf-8")
        result["max_relative_change_vs_previous_report"] = delta["max_relative_change"]
        if delta["max_relative_change"] > 1e-9:
            result["warning"] = (
                "Fig. 4 values moved relative to the previous report. Update the numbers quoted "
                "in the Letter and the frozen expectations in "
                "scripts/audit_rotating_colloids_capillary_prl.py before submission."
            )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
