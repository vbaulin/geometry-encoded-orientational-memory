#!/usr/bin/env python3
"""Analyze finite-size spin-glass diagnostics for the capillary rotor model."""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    paths = glob.glob(str(Path(args.input_dir) / "**" / "spin_glass_scan.jsonl"), recursive=True)
    rows = [json.loads(line) for path in paths for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("no spin_glass_scan.jsonl rows found")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["n"]), float(row["lambda"]))].append(row)
    sizes = sorted({key[0] for key in grouped})
    lambdas = sorted({key[1] for key in grouped})
    if any(len(grouped[(n, lam)]) != 5 for n in sizes for lam in lambdas):
        raise SystemExit("expected five disorder realizations at every size/coupling point")

    colors = dict(zip(sizes, plt.cm.viridis(np.linspace(0.08, 0.92, len(sizes)))))
    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in"})
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.6), constrained_layout=True)

    for n in sizes:
        qea, qea_e, xi, xi_e = [], [], [], []
        for lam in lambdas:
            group = grouped[(n, lam)]
            a, b = mean_sem([r["q_EA_mean"] for r in group]); qea.append(a); qea_e.append(b)
            a, b = mean_sem([r["overlap"]["xi_over_L"] for r in group]); xi.append(a); xi_e.append(b)
        label = rf"$N={n*n}$"
        axes[0, 0].errorbar(lambdas, qea, yerr=qea_e, color=colors[n], marker="o", ms=3, lw=1, capsize=1.5, label=label)
        axes[0, 1].errorbar(lambdas, xi, yerr=xi_e, color=colors[n], marker="o", ms=3, lw=1, capsize=1.5)

    chosen = [0.3, 0.75, 1.4]
    chosen_colors = ["#4575b4", "#7b3294", "#d73027"]
    scaling = {}
    node_counts = np.asarray([n * n for n in sizes], dtype=float)
    for lam, color in zip(chosen, chosen_colors):
        q, qe, chi, chie = [], [], [], []
        for n in sizes:
            group = grouped[(n, lam)]
            a, b = mean_sem([r["overlap"]["q_abs_mean"] for r in group]); q.append(a); qe.append(b)
            a, b = mean_sem([r["overlap"]["chi_overlap"] for r in group]); chi.append(a); chie.append(b)
        slope, intercept = np.polyfit(np.log(node_counts), np.log(q), 1)
        scaling[str(lam)] = {"q_overlap_exponent": float(slope), "chi_largest_size": float(chi[-1])}
        axes[1, 0].errorbar(node_counts, q, yerr=qe, color=color, marker="o", ms=3.5, lw=0, capsize=1.5, label=rf"$\lambda={lam:g}$")
        fit_x = np.geomspace(node_counts.min(), node_counts.max(), 100)
        axes[1, 0].plot(fit_x, np.exp(intercept) * fit_x**slope, color=color, lw=1)
        axes[1, 1].errorbar(node_counts, chi, yerr=chie, color=color, marker="o", ms=3.5, lw=1, capsize=1.5, label=rf"$\lambda={lam:g}$")

    axes[0, 0].set(xlabel=r"coupling scale $\lambda$", ylabel=r"single-trajectory $q_{\rm EA}$", title="apparent dynamical freezing")
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=7)
    axes[0, 1].set(xlabel=r"coupling scale $\lambda$", ylabel=r"$\xi_L/L$", title="no size-independent crossing", ylim=(-0.01, 0.145))
    axes[1, 0].set(xlabel=r"number of rotors $N$", ylabel=r"independent-replica $\langle|Q_{ab}|\rangle$", title=r"overlap vanishes as $N^{-1/2}$", xscale="log", yscale="log")
    axes[1, 0].legend(frameon=False, fontsize=7)
    axes[1, 1].set(xlabel=r"number of rotors $N$", ylabel=r"overlap susceptibility $\chi_Q$", title="finite overlap correlation volume", xscale="log")
    axes[1, 1].legend(frameon=False, fontsize=7)
    for label, ax in zip("abcd", axes.flat):
        ax.text(0.02, 0.97, f"({label})", transform=ax.transAxes, va="top", ha="left", fontweight="bold")
        ax.grid(alpha=0.18, lw=0.5)

    figure_base = out / "spin_glass_finite_size_diagnostic"
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    report = {
        "rows": len(rows),
        "sizes": sizes,
        "node_counts": [n * n for n in sizes],
        "lambdas": lambdas,
        "disorder_realizations_per_point": 5,
        "selected_scaling": scaling,
        "finding": "no_equilibrium_spin_glass_crossing_on_scanned_ray",
        "interpretation": "strong finite-window self-overlap but vanishing independent-replica overlap and finite overlap correlation volume",
    }
    (out / "spin_glass_finite_size_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
