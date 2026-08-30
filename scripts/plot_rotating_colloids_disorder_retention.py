#!/usr/bin/env python3
"""Plot the positional-disorder dependence of released orientational memory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2C6DA4"
PURPLE = "#7B3294"
GREY = "#8A9099"
INK = "#202428"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    n576 = report["n576"]
    n1024 = report["n1024"]
    comparison = report["comparison"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.35), sharey=True)

    ax = axes[0]
    sigma = np.asarray([row["sigma_over_a"] for row in n576], dtype=float)
    means = np.asarray([row["mean"] for row in n576], dtype=float)
    sem = np.asarray([row["sem"] for row in n576], dtype=float)
    ax.axvspan(0.08, 0.11, color="#E7E8EA", alpha=0.75, lw=0)
    ax.errorbar(
        sigma,
        means,
        yerr=sem,
        color=BLUE,
        marker="o",
        markersize=4.5,
        lw=1.6,
        capsize=2.5,
        label=r"$N=576$",
    )
    ax.annotate(
        "director-suppression\ncrossover",
        xy=(0.095, 0.405),
        ha="center",
        va="bottom",
        color="#5C626A",
        fontsize=7.2,
    )
    ax.annotate(
        "retention maximum",
        xy=(0.11, 0.559),
        xytext=(0.155, 0.585),
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        ha="left",
        va="center",
        fontsize=7.4,
    )
    ax.set(
        xlabel=r"positional disorder $\sigma/a$",
        ylabel=r"connected written overlap $Q_{\rm target}^{\rm conn}$",
        title=r"full disorder scan ($N=576$)",
        xlim=(0.065, 0.295),
        ylim=(0.34, 0.61),
    )
    ax.text(0.02, 0.97, "a", transform=ax.transAxes, va="top", fontweight="bold", fontsize=11)

    ax = axes[1]
    x = np.asarray([0.11, 0.16], dtype=float)
    values = [np.asarray(row["Q_target_conn"], dtype=float) for row in n1024]
    for index in range(len(values[0])):
        ax.plot(x, [values[0][index], values[1][index]], color=GREY, lw=0.8, alpha=0.72, zorder=1)
        ax.scatter(x, [values[0][index], values[1][index]], s=16, facecolor="white", edgecolor=GREY, lw=0.8, zorder=2)
    means = np.asarray([row["mean"] for row in n1024], dtype=float)
    sem = np.asarray([row["sem"] for row in n1024], dtype=float)
    ax.errorbar(
        x,
        means,
        yerr=sem,
        color=PURPLE,
        marker="o",
        markersize=5.2,
        lw=1.8,
        capsize=3,
        zorder=3,
    )
    mean_difference = float(comparison["mean_difference_0p11_minus_0p16"])
    paired_p = float(comparison["paired_p_two_sided"])
    ax.text(
        0.135,
        0.576,
        rf"$\Delta Q^{{\rm conn}}={mean_difference:.4f}$"
        + "\n"
        + rf"paired $p={paired_p:.4f}$",
        ha="center",
        va="top",
        fontsize=7.6,
        color=INK,
    )
    ax.set(
        xlabel=r"positional disorder $\sigma/a$",
        title=r"matched graph seeds ($N=1024$)",
        xlim=(0.095, 0.175),
    )
    ax.set_xticks(x)
    ax.text(0.02, 0.97, "b", transform=ax.transAxes, va="top", fontweight="bold", fontsize=11)

    for axis in axes:
        axis.grid(True, color="#D9DDE2", lw=0.55, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", length=3)

    fig.tight_layout(w_pad=1.6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "n576_points": len(n576), "n1024_graphs": len(values[0])}))


if __name__ == "__main__":
    main()
