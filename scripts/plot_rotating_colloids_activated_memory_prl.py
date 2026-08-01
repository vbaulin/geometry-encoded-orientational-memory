#!/usr/bin/env python3
"""Build the coupling-dependent retention figure from the long GPU scan."""

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


def load_rows(input_dir: Path):
    paths = glob.glob(str(input_dir / "**" / "activated_memory_scan.jsonl"), recursive=True)
    return [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]


def interpolate_curve(curve, grid):
    return np.interp(grid, np.asarray(curve["time"], dtype=float), np.asarray(curve["overlap_mean"], dtype=float))


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(Path(args.input_dir))
    if not rows:
        raise SystemExit("no activated_memory_scan.jsonl rows found")
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["lambda"])].append(row)
    lambdas = sorted(groups)
    if any(len(groups[lam]) < 3 for lam in lambdas):
        raise SystemExit("at least three graph realizations are required per coupling")

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
    ax = axes[2]
    selected = [lambdas[0], lambdas[len(lambdas) // 2], lambdas[-1]]
    colors = ("#4575b4", "#7b3294", "#d73027")
    for lam, color in zip(selected, colors):
        by_window = defaultdict(list)
        for row in groups[lam]:
            for item in row["observation_window"]["windows"]:
                by_window[float(item["window"])].append(float(item["q_EA_mean"]))
        windows = sorted(by_window)
        mean = [np.mean(by_window[window]) for window in windows]
        sem = [np.std(by_window[window], ddof=1) / np.sqrt(len(by_window[window])) for window in windows]
        ax.errorbar(windows, mean, yerr=sem, color=color, marker="o", ms=3.0, lw=1.2, capsize=1.5, label=rf"$\lambda={lam:g}$")
    ax.set(xlabel=r"observation window $D_rT_{\rm obs}$", ylabel=r"finite-window $q_{\rm EA}$", title="memory depends on read time", xscale="log", ylim=(0, 1.02))
    ax.legend(frameon=False)

    for label, ax in zip("abc", axes):
        ax.text(0.02, 0.97, f"{label}", transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9.5, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8})
        ax.grid(alpha=0.15, lw=0.5)

    stem = out / "fig4_activated_memory_results"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)
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
    }
    (out / "activated_memory_figure_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(stem), "rows": len(rows)}))


if __name__ == "__main__":
    main()
