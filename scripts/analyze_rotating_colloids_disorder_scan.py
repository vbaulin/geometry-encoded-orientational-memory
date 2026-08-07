#!/usr/bin/env python3
"""Summarize the positional-disorder scan at fixed (J, g).

The Letter argues that memory is encoded by the geometry of the quenched
neighbour graph. The publication runs hold the disorder amplitude fixed at
0.16a, with 0 only as the regular-lattice control, so the dependence on the
geometric disorder itself is unmeasured. This script reduces a scan over the
amplitude to graph-averaged observables and their graph-level spread.

Usage on the GPU host, after

    for D in 0.02 0.05 0.08 0.11 0.16 0.22 0.28; do ... --output-dir colloid/disorder_$D; done

    python -B scripts/analyze_rotating_colloids_disorder_scan.py \
      --input-dir colloid --output-dir colloid/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OBSERVABLES = (
    ("S_mean", r"global order $S$", "#b2182b"),
    ("C2_mean", r"pair order $C_2$", "#2166ac"),
    ("G2_mean", r"bond-frame order $G_2$", "#1b9e77"),
    ("q_EA_mean", r"persistence $q_{\rm EA}$", "#7b3294"),
)


def load_rows(input_dir: Path) -> list[dict]:
    paths = sorted(input_dir.glob("**/capillary_pair_scan.jsonl"))
    if not paths:
        raise SystemExit(
            f"no capillary_pair_scan.jsonl under {input_dir}\n"
            "Point --input-dir at the parent of the disorder_* run directories."
        )
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--j-align", type=float, default=4.0)
    parser.add_argument("--g-capillary", type=float, default=5.0)
    parser.add_argument(
        "--node-count",
        type=int,
        help="Restrict to one rotor count. Required when the tree mixes sizes.",
    )
    args = parser.parse_args()

    rows = [
        row
        for row in load_rows(args.input_dir)
        if row.get("control", "physical") == "physical"
        and math.isclose(float(row["j_align"]), args.j_align)
        and math.isclose(float(row["g_capillary"]), args.g_capillary)
    ]
    if not rows:
        raise SystemExit(f"no physical rows at J={args.j_align}, g={args.g_capillary}")

    if args.node_count is not None:
        rows = [row for row in rows if int(row["graph"]["node_count"]) == args.node_count]
        if not rows:
            raise SystemExit(f"no rows with node_count={args.node_count}")
    sizes = sorted({int(row["graph"]["node_count"]) for row in rows})
    if len(sizes) > 1:
        # S and q_EA are size dependent, so pooling sizes into one disorder
        # point would average incomparable quantities.
        raise SystemExit(
            f"input mixes rotor counts {sizes}; the scan must vary disorder at fixed size.\n"
            f"Re-run with --node-count {sizes[0]} to select one."
        )

    groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        groups[float(row["graph"]["disorder"])].append(row)
    amplitudes = sorted(groups)

    table = []
    for amplitude in amplitudes:
        block = groups[amplitude]
        entry: dict[str, object] = {
            "disorder": amplitude,
            "graphs": len(block),
            # Scan rows carry `n` at top level; the rotor count lives in the
            # graph metadata block.
            "node_count": int(block[0]["graph"]["node_count"]),
        }
        for name, _, _ in OBSERVABLES:
            mean, sd = mean_sd(row[name] for row in block)
            entry[name] = {"mean": mean, "graph_sd": sd}
        for name in ("bond_frame_fourth_harmonic", "capillary_weighted_degree", "alignment_weighted_degree"):
            mean, sd = mean_sd(row["graph"][name] for row in block)
            entry[name] = {"mean": mean, "graph_sd": sd}
        table.append(entry)

    peak = max(table, key=lambda item: item["q_EA_mean"]["mean"])
    minimum_order = min(table, key=lambda item: item["S_mean"]["mean"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.2, "axes.labelsize": 8.5, "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.35), constrained_layout=True)

    ax = axes[0]
    for name, label, color in OBSERVABLES:
        means = [item[name]["mean"] for item in table]
        errors = [item[name]["graph_sd"] for item in table]
        ax.errorbar(amplitudes, means, yerr=errors, color=color, marker="o", ms=3.2, lw=1.2, capsize=1.5, label=label)
    ax.set(
        xlabel=r"positional disorder $\sigma/a$",
        ylabel="order parameter",
        title=rf"$J={args.j_align:g}$, $g={args.g_capillary:g}\,k_{{\rm B}}T$",
        ylim=(-0.05, 1.02),
    )
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.15, lw=0.5)

    ax = axes[1]
    harmonic = [item["bond_frame_fourth_harmonic"]["mean"] for item in table]
    ax.plot(amplitudes, harmonic, color="#444444", marker="s", ms=3.2, lw=1.2)
    ax.set(
        xlabel=r"positional disorder $\sigma/a$",
        ylabel=r"$|\langle e^{4i\phi_{ij}}\rangle_E|$",
        title="residual bond-angle anisotropy",
        yscale="log",
    )
    ax.grid(alpha=0.15, lw=0.5)

    for label, ax in zip("ab", axes):
        ax.text(0.02, 0.97, label, transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=9.5)

    stem = args.output_dir / "disorder_scan"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", facecolor="white", dpi=400)
    plt.close(fig)

    report = {
        "j_align": args.j_align,
        "g_capillary": args.g_capillary,
        "rows": len(rows),
        "disorder_amplitudes": amplitudes,
        "graphs_per_amplitude": {f"{value:g}": len(groups[value]) for value in amplitudes},
        "table": table,
        "peak_persistence": {"disorder": peak["disorder"], "q_EA_mean": peak["q_EA_mean"]["mean"]},
        "minimum_global_order": {"disorder": minimum_order["disorder"], "S_mean": minimum_order["S_mean"]["mean"]},
        "interior_maximum": bool(
            peak["disorder"] not in (amplitudes[0], amplitudes[-1])
        ),
    }
    (args.output_dir / "disorder_scan_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "amplitudes": amplitudes,
        "peak_persistence_at": report["peak_persistence"],
        "interior_maximum": report["interior_maximum"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
