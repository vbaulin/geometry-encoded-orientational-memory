#!/usr/bin/env python3
"""Plot the two-operation write-release-read discriminant for the Supplement."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"contested": "#c43c39", "partitioned": "#356fa8"}
LABELS = {"contested": "overlapping, opposed", "partitioned": "separated supports"}


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def graph_summary(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    sem = (
        float(values.std(ddof=1) / math.sqrt(values.size))
        if values.size > 1 else 0.0
    )
    return mean, sem


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--noise-mode",
        default="common",
        choices=("common", "independent"),
    )
    args = parser.parse_args()

    source = Path(args.input)
    rows = load_rows(source)
    noise_rows = [
        row for row in rows if str(row.get("noise_mode", "common")) == args.noise_mode
    ]
    primary_rows = [
        row
        for row in noise_rows
        if row["mode"] == "partitioned"
        or math.isclose(float(row.get("contest_fraction_requested", 0.25)), 0.25)
    ]
    if not primary_rows:
        raise RuntimeError("input has no partitioned or quarter-support contested rows")
    fields = sorted({float(row["field"]) for row in primary_rows})
    max_field = max(fields)
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25))

    report: Dict[str, object] = {
        "source": str(source.resolve()),
        "node_count": int(rows[0]["n"]) ** 2,
        "release_time": float(rows[0]["release_steps"]) * float(rows[0]["dt"]),
        "noise_mode": args.noise_mode,
        "fields": {},
    }
    grouped = defaultdict(list)
    for row in primary_rows:
        grouped[(float(row["field"]), str(row["mode"]))].append(row)

    for mode in ("partitioned", "contested"):
        means = []
        sems = []
        for field in fields:
            selected = grouped[(field, mode)]
            values = np.asarray(
                [float(row["terminal_order_readout"]) for row in selected], dtype=float
            )
            mean, sem = graph_summary(values)
            means.append(mean)
            sems.append(sem)
            report["fields"][f"{field:g}:{mode}"] = {
                "graph_count": int(values.size),
                "mean": mean,
                "sem": sem,
                "values": values.tolist(),
            }
            decode = [
                float(row["terminal_decode_accuracy_zero_threshold"])
                for row in selected
                if "terminal_decode_accuracy_zero_threshold" in row
            ]
            if decode:
                report["fields"][f"{field:g}:{mode}"]["decode_accuracy"] = graph_summary(
                    np.asarray(decode, dtype=float)
                )[0]
        axes[0].errorbar(
            fields,
            means,
            yerr=sems,
            marker="o" if mode == "contested" else "s",
            color=COLORS[mode],
            lw=1.8,
            ms=5.0,
            capsize=2.5,
            label=LABELS[mode],
        )

    axes[0].axhline(0.0, color="0.4", lw=0.8)
    axes[0].set(
        xlabel=r"write-field amplitude $h/k_{\rm B}T$",
        ylabel=r"retained order readout $M_{\rm ord}$",
        title="a  Contest creates a readable order bit",
    )
    axes[0].legend(frameon=False, loc="upper left", fontsize=8.5)

    for mode in ("partitioned", "contested"):
        selected = grouped[(max_field, mode)]
        curves = np.asarray(
            [np.asarray(row["order_readout_mean"], dtype=float) for row in selected]
        )
        time = np.asarray(selected[0]["time"], dtype=float)
        mean = curves.mean(axis=0)
        sem = (
            curves.std(axis=0, ddof=1) / math.sqrt(curves.shape[0])
            if curves.shape[0] > 1 else np.zeros(curves.shape[1])
        )
        axes[1].plot(time, mean, color=COLORS[mode], lw=2.0, label=LABELS[mode])
        axes[1].fill_between(time, mean - sem, mean + sem, color=COLORS[mode], alpha=0.18)
        report[f"release_curve_h_{max_field:g}:{mode}"] = {
            "time": time.tolist(),
            "mean": mean.tolist(),
            "sem": sem.tolist(),
        }
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].set(
        xlabel=r"field-free time $D_rt$",
        ylabel=r"order readout $M_{\rm ord}(t)$",
        title=rf"b  Order survives removal of $h={max_field:g}k_{{\rm B}}T$ fields",
    )
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.5)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
    figure.tight_layout(w_pad=2.4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    report_path = output.with_name(output.name + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "node_count": report["node_count"],
        "release_time": report["release_time"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
