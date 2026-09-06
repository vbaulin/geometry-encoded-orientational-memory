#!/usr/bin/env python3
"""Verify and plot the publication-size, shared-target release comparison."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    "matched_physical", "matched_no_capillary", "matched_equal_weight_relative",
    "own_write_no_capillary", "own_write_equal_weight_relative",
)
METRICS = ("Q", "Q_connected", "Q_rotation_aligned", "S", "Q_unrelated")
REQUIRED = ("manifest.json", "target_relaxation.npz", "write.npz",
            "preparation.npz", "release.npz", "observables.npz", "spatial_final.npz")
COLORS = ("#2C6DA4", "#C44337", "#7B3294")
LABELS = ("Capillary pair interaction", r"No capillary term ($g=0$)",
          "Equal-weight relative alignment")


def graph_statistics(values):
    """Replicas are averaged within graphs before estimating uncertainty."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite graph means are required")
    return values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))


def endpoint_overlaps(theta, target, unrelated):
    z = np.exp(2j * np.asarray(theta, dtype=float))
    target_z = np.exp(2j * np.asarray(target, dtype=float))
    q = (z * target_z.conj()).mean(axis=-1)
    m = z.mean(axis=-1)
    return {"Q": q.real, "Q_connected": (q - m * target_z.mean().conjugate()).real,
            "Q_rotation_aligned": abs(q), "S": abs(m),
            "Q_unrelated": (z * np.exp(-2j * unrelated)).mean(axis=-1).real}


def load_cases(directory):
    cases, seen = [], set()
    expected = set(product((17, 29, 43, 71, 97), (0.11, 0.16), ("relaxed", "quarter_turn")))
    for path in sorted(directory.glob("graphs_*/*/summary.json")):
        row = json.loads(path.read_text())
        config = row["config"]
        key = (config["graph_seed"], config["disorder"], config["target"])
        if key in seen:
            raise ValueError(f"Duplicate graph/disorder/target case: {key}")
        seen.add(key)
        required_config = {"n": 32, "replicas": 48, "coupling_scale": 1.0,
                           "dt": 0.0025, "release_steps": 250000,
                           "write_steps": 50000, "equilibration_steps": 50000,
                           "write_field": 1.5}
        if any(config[k] != v for k, v in required_config.items()):
            raise ValueError(f"Unexpected publication configuration: {path}")
        for name in REQUIRED:
            source = path.parent / name
            if not source.is_file() or not source.stat().st_size:
                raise ValueError(f"Missing or empty source: {source}")
        manifest = json.loads((path.parent / "manifest.json").read_text())
        if manifest["config"] != config or tuple(manifest["arms"]) != ARMS:
            raise ValueError(f"Manifest disagrees with summary: {path}")
        with np.load(path.parent / "preparation.npz") as prep:
            initial, target = prep["release_initial"], prep["target"]
            unrelated = prep["unrelated_target"].astype(float)
        np.testing.assert_array_equal(initial[:3], np.repeat(initial[:1], 3, axis=0))
        if initial.shape != (5, 48, 1024):
            raise ValueError(f"Wrong release shape: {path}")
        digest = hashlib.sha256(np.ascontiguousarray(target).tobytes()).hexdigest()
        if digest != row["target_sha256"]:
            raise ValueError(f"Target hash mismatch: {path}")
        with np.load(path.parent / "observables.npz") as saved:
            time = saved["time"]
            obs = {name: saved[name] for name in METRICS}
        with np.load(path.parent / "release.npz") as release:
            np.testing.assert_array_equal(time, release["time"])
            terminal = release["terminal_theta"]
        if time[0] != 0 or time[-1] != 625 or not np.all(np.diff(time) > 0):
            raise ValueError(f"Incomplete or unsorted time axis: {path}")
        for index, angles in ((0, initial), (-1, terminal)):
            actual = endpoint_overlaps(angles, target, unrelated)
            for name in METRICS:
                np.testing.assert_allclose(obs[name][index], actual[name], rtol=1e-7, atol=1e-7)
        for name in METRICS:
            if not np.isfinite(obs[name]).all() or obs[name].shape != (len(time), 5, 48):
                raise ValueError(f"Invalid observable {name}: {path}")
            for arm_index, arm in enumerate(ARMS):
                np.testing.assert_allclose(obs[name][-1, arm_index],
                                           row["arms"][arm][name]["per_replica"], atol=1e-12)
        cases.append({"config": config, "time": time, "observables": obs, "target_angles": target,
                      "manifest": manifest, "source": str(path)})
    if seen != expected:
        raise ValueError(f"Incomplete case grid: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    for name in ("code_sha256", "core_sha256"):
        if len({case["manifest"][name] for case in cases}) != 1:
            raise ValueError(f"Mixed simulation implementations: {name}")
    return cases


def summarize(cases):
    groups = defaultdict(list)
    for case in cases:
        c = case["config"]
        groups[(c["disorder"], c["target"])].append(case)
    rows, curves, contrasts = [], {}, []
    for (sigma, target), members in sorted(groups.items()):
        time = members[0]["time"]
        for case in members:
            np.testing.assert_array_equal(case["time"], time)
        for arm_index, arm in enumerate(ARMS):
            row = {"disorder": sigma, "target": target, "arm": arm, "graph_count": len(members)}
            curves[(sigma, target, arm)] = {"time": time}
            for name in METRICS:
                graph_means = np.asarray([case["observables"][name][:, arm_index].mean(axis=-1)
                                          for case in members])
                mean, sem = graph_statistics(graph_means)
                curves[(sigma, target, arm)][name] = (mean, sem)
                row[name] = {"mean": float(mean[-1]), "graph_sem": float(sem[-1])}
            rows.append(row)
        for arm_index in (1, 2, 3, 4):
            delta = [float(case["observables"]["Q_connected"][-1, 0].mean()
                           - case["observables"]["Q_connected"][-1, arm_index].mean()) for case in members]
            mean, sem = graph_statistics(delta)
            contrasts.append({"disorder": sigma, "target": target, "control": ARMS[arm_index],
                              "mean_difference": float(mean), "graph_sem": float(sem)})
    return rows, curves, contrasts


def plot(curves, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 10,
                         "legend.fontsize": 8, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), layout="constrained")
    for ax, sigma, metric, letter in zip(axes, (0.11, 0.16, 0.16),
                                         ("Q_connected", "Q_connected", "S"), "abc"):
        for arm, color, label in zip(ARMS[:3], COLORS, LABELS):
            curve = curves[(sigma, "relaxed", arm)]
            mean, sem = curve[metric]
            ax.plot(curve["time"], mean, color=color, lw=1.8, label=label)
            ax.fill_between(curve["time"], mean-sem, mean+sem, color=color, alpha=0.16, lw=0)
        ax.axhline(0, color="#989DA3", lw=0.6, zorder=0)
        ax.set(xlim=(0, 625), ylim=(-0.04, 1.0), xlabel=r"release time $D_rt$",
               title=rf"$\sigma/a={sigma:g}$",
               ylabel=r"connected target overlap $Q_{\mathrm{target}}^{\mathrm{conn}}$" if metric == "Q_connected"
               else r"global nematic order $S$")
        ax.set_xticks((0, 200, 400, 625))
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
        ax.set_box_aspect(1)
        ax.text(0.04, 0.83 if letter == "c" else 0.95, letter,
                transform=ax.transAxes, va="top", fontweight="bold", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="center right", bbox_to_anchor=(0.99, 0.28), frameon=False,
                   handlelength=1.6, borderaxespad=0.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.input_dir)
    rows, curves, contrasts = summarize(cases)
    original = json.loads((args.input_dir / "matched_preparation_report.json").read_text())
    if len(original["groups"]) != len(rows):
        raise ValueError("Merged report has missing or extra groups")
    reference = {(r["disorder"], r["target"], r["arm"]): r for r in original["groups"]}
    for row in rows:
        old = reference[(row["disorder"], row["target"], row["arm"])]
        for name in METRICS:
            for stat in ("mean", "graph_sem"):
                np.testing.assert_allclose(row[name][stat], old[name][stat], rtol=1e-10, atol=1e-12)
    plot(curves, args.output)
    targets = {(c["config"]["graph_seed"], c["config"]["disorder"], c["config"]["target"]):
               c["target_angles"].astype(float) for c in cases}
    rotation_error = max(float(np.max(abs(np.angle(np.exp(2j * (
        targets[(seed, sigma, "quarter_turn")] - targets[(seed, sigma, "relaxed")] - np.pi/2)))) / 2))
        for seed, sigma in product((17, 29, 43, 71, 97), (0.11, 0.16)))
    report = {"complete": True, "cases": len(cases), "node_count": 1024,
              "replicas_per_case_per_arm": 48, "release_time": 625,
              "matched_starts_identical": True, "endpoints_recomputed_from_angles": True,
              "merged_report_reproduced": True, "groups": rows,
              "paired_graph_contrasts": contrasts,
              "target_relation": "quarter_turn is the pi/2 symmetry partner, not an independent target family",
              "max_quarter_turn_target_error_radians": rotation_error,
              "uncertainty": "SEM of five graph means after averaging 48 thermal replicas per graph",
              "code_sha256": cases[0]["manifest"]["code_sha256"],
              "core_sha256": cases[0]["manifest"]["core_sha256"]}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("complete", "cases", "node_count",
                                           "matched_starts_identical", "merged_report_reproduced")}))


if __name__ == "__main__":
    main()
