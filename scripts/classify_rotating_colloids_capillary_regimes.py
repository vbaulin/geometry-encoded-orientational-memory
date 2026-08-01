#!/usr/bin/env python3
"""Resolve finite-size capillary-rotor regimes from a dense parameter scan.

The classification is performed in observable space, not imposed in the
coupling plane.  Parameter cells are represented by

    (S, C2, G2, q_EA, finite-window autocorrelation).

The script selects the number of clusters by silhouette score unless an
explicit value is supplied, reports random-initialization stability, and
plots both the coupling-plane regime diagram and the corresponding
observable-space clouds.  The output is a finite-size regime taxonomy, not
a claim of thermodynamic phase coexistence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation")
FEATURE_LABELS = (r"$S$", r"$C_2$", r"$G_2$", r"$q_{\rm EA}$", r"$C_{\rm window}$")

REGIME_COLORS = {
    "Brownian crossover": "#8da0cb",
    "relative-aligned state": "#e78ac3",
    "capillary-frame memory": "#fc8d62",
    "hidden mixed memory": "#1b9e77",
    "mixed persistent state": "#7570b3",
}


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def aggregate_cells(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[float, float], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(float(row["j_align"]), float(row["g_capillary"]))].append(row)
    cells: List[Dict[str, object]] = []
    for (j_value, g_value), group in sorted(groups.items()):
        cell: Dict[str, object] = {
            "j_align": j_value,
            "g_capillary": g_value,
            "graph_count": len({int(row["graph_seed"]) for row in group}),
            "row_count": len(group),
        }
        for feature in FEATURES:
            values = np.asarray([float(row[feature]) for row in group], dtype=float)
            cell[feature] = float(values.mean())
            cell[feature + "_std"] = float(values.std())
        cells.append(cell)
    return cells


def choose_clusters(x_scaled: np.ndarray, requested: int) -> Tuple[int, Dict[int, float]]:
    if requested > 0:
        candidates = [requested]
    else:
        candidates = list(range(2, min(7, x_scaled.shape[0])))
    scores: Dict[int, float] = {}
    for count in candidates:
        labels = KMeans(n_clusters=count, random_state=11, n_init=100).fit_predict(x_scaled)
        scores[count] = float(silhouette_score(x_scaled, labels))
    return max(scores, key=scores.get), scores


def regime_name(centroid: np.ndarray) -> str:
    s, c2, g2, qea, window = centroid
    if qea >= 0.52 and window >= 0.40 and c2 >= 0.32 and g2 >= 0.30 and s <= 0.25:
        return "hidden mixed memory"
    if qea >= 0.45 and window >= 0.30 and g2 >= c2 + 0.10:
        return "capillary-frame memory"
    if c2 >= g2 + 0.25:
        return "relative-aligned state"
    if qea < 0.40 or window < 0.20:
        return "Brownian crossover"
    return "mixed persistent state"


def unique_regime_names(centroids: np.ndarray) -> Dict[int, str]:
    assigned: Dict[int, str] = {}
    used: Dict[str, int] = defaultdict(int)
    for index, centroid in enumerate(centroids):
        base = regime_name(centroid)
        used[base] += 1
        assigned[index] = base if used[base] == 1 else f"{base} {used[base]}"
    return assigned


def cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]))


def grid(cells: Sequence[Dict[str, object]], values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    js = np.asarray(sorted({float(cell["j_align"]) for cell in cells}))
    gs = np.asarray(sorted({float(cell["g_capillary"]) for cell in cells}))
    output = np.full((gs.size, js.size), np.nan)
    lookup = {(float(cell["j_align"]), float(cell["g_capillary"])): value for cell, value in zip(cells, values)}
    for gi, g_value in enumerate(gs):
        for ji, j_value in enumerate(js):
            if (j_value, g_value) in lookup:
                output[gi, ji] = float(lookup[(j_value, g_value)])
    return js, gs, output


def label_order(names: Sequence[str]) -> List[str]:
    preferred = [
        "Brownian crossover",
        "relative-aligned state",
        "capillary-frame memory",
        "hidden mixed memory",
        "mixed persistent state",
    ]
    return [name for name in preferred if name in names] + sorted(name for name in names if name not in preferred)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_regime_diagram(
    cells: Sequence[Dict[str, object]],
    regime_names: Sequence[str],
    confidence: np.ndarray,
    output_dir: Path,
) -> None:
    order = label_order(regime_names)
    index = {name: value for value, name in enumerate(order)}
    numeric = [index[name] for name in regime_names]
    colors = [REGIME_COLORS.get(name.split(" 2")[0], "#66c2a5") for name in order]
    js, gs, regime_grid = grid(cells, numeric)
    _, _, confidence_grid = grid(cells, confidence)

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.25), constrained_layout=True)
    ax = axes[0]
    cmap = ListedColormap(colors)
    ax.pcolormesh(cell_edges(js), cell_edges(gs), regime_grid, cmap=cmap, vmin=-0.5, vmax=len(order) - 0.5, shading="flat")
    if len(order) > 1:
        ax.contour(js, gs, regime_grid, levels=np.arange(0.5, len(order) - 0.5, 1.0), colors="white", linewidths=0.9)
    ax.plot(4.0, 5.0, marker="*", ms=9, color="white", mec="black", mew=0.7)
    ax.annotate(
        "dynamical test\n$(J,g)=(4,5)$",
        xy=(4.0, 5.0),
        xytext=(3.05, 6.15),
        fontsize=6.6,
        ha="center",
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": "0.25"},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
    )
    ax.set(xlabel=r"alignment $J/k_{\rm B}T$", ylabel=r"capillary coupling $g/k_{\rm B}T$", title="finite-size dynamical regime map")
    direct_labels = {
        "Brownian crossover": (0.78, 2.20, "Brownian\ncrossover"),
        "relative-aligned state": (3.75, 0.56, "relative-aligned\nstate"),
        "capillary-frame memory": (0.82, 6.35, "capillary-frame\nmemory"),
        "hidden mixed memory": (4.10, 3.15, "hidden mixed\nmemory"),
    }
    for name in order:
        if name not in direct_labels:
            continue
        x_label, y_label, text_label = direct_labels[name]
        ax.text(
            x_label,
            y_label,
            text_label,
            ha="center",
            va="center",
            fontsize=6.4,
            fontweight="semibold",
            linespacing=0.95,
            color="0.12",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
                "pad": 1.0,
            },
        )
    ax.text(0.02, 0.03, "a", transform=ax.transAxes, fontweight="bold", fontsize=9.5)

    ax = axes[1]
    im = ax.pcolormesh(
        cell_edges(js),
        cell_edges(gs),
        confidence_grid,
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
        shading="flat",
    )
    ax.contour(js, gs, regime_grid, levels=np.arange(0.5, len(order) - 0.5, 1.0), colors="white", linewidths=0.75)
    ax.set(xlabel=r"alignment $J/k_{\rm B}T$", title="classification confidence")
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.055, label="centroid margin")
    ax.text(0.02, 0.03, "b", transform=ax.transAxes, fontweight="bold", fontsize=9.5, color="white")

    ax = axes[2]
    cut_g = float(gs[np.argmin(np.abs(gs - 5.0))])
    cut_cells = sorted([cell for cell in cells if math.isclose(float(cell["g_capillary"]), cut_g)], key=lambda cell: float(cell["j_align"]))
    cut_names = [regime_names[cells.index(cell)] for cell in cut_cells]
    x = np.asarray([float(cell["j_align"]) for cell in cut_cells])
    for left, right, name in zip(cell_edges(x)[:-1], cell_edges(x)[1:], cut_names):
        ax.axvspan(left, right, color=REGIME_COLORS.get(name.split(" 2")[0], "#66c2a5"), alpha=0.20, lw=0)
    for feature, label, color in zip(FEATURES[:4], FEATURE_LABELS[:4], ("#4d4d4d", "#277da1", "#43aa8b", "#8e44ad")):
        ax.plot(x, [float(cell[feature]) for cell in cut_cells], marker="o", ms=2.5, lw=1.2, color=color, label=label)
    ax.axvline(4.0, color="0.25", lw=0.75, ls=":")
    ax.text(4.0, 0.96, "dynamical test", rotation=90, va="top", ha="right", fontsize=6.5, color="0.25")
    ax.set(xlabel=r"alignment $J/k_{\rm B}T$", ylabel="observable value", ylim=(-0.42, 1.02), title=rf"cut at $g={cut_g:g}k_{{\rm B}}T$")
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.4, loc="lower right")
    ax.text(0.02, 0.03, "c", transform=ax.transAxes, fontweight="bold", fontsize=9.5)
    save(fig, output_dir, "capillary_regime_diagram")


def hull(ax: plt.Axes, xy: np.ndarray, color: str) -> None:
    if xy.shape[0] < 3:
        return
    try:
        boundary = ConvexHull(xy)
    except Exception:
        return
    polygon = xy[boundary.vertices]
    ax.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.10, lw=0)
    ax.plot(np.r_[polygon[:, 0], polygon[0, 0]], np.r_[polygon[:, 1], polygon[0, 1]], color=color, lw=0.8, alpha=0.8)


def plot_state_space(cells: Sequence[Dict[str, object]], regime_names: Sequence[str], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.22), constrained_layout=True)
    specifications = (
        ("S_mean", "G2_mean", r"global order $S$", r"bond-frame order $G_2$"),
        ("C2_mean", "G2_mean", r"relative order $C_2$", r"bond-frame order $G_2$"),
        ("S_mean", "q_EA_mean", r"global order $S$", r"finite-window memory $q_{\rm EA}$"),
    )
    order = label_order(regime_names)
    for panel, (ax, (x_key, y_key, x_label, y_label)) in enumerate(zip(axes, specifications)):
        for name in order:
            selected = [cell for cell, regime in zip(cells, regime_names) if regime == name]
            xy = np.asarray([[float(cell[x_key]), float(cell[y_key])] for cell in selected])
            color = REGIME_COLORS.get(name.split(" 2")[0], "#66c2a5")
            hull(ax, xy, color)
            sizes = 12.0 + 28.0 * np.asarray([float(cell["q_EA_mean"]) for cell in selected])
            ax.scatter(xy[:, 0], xy[:, 1], s=sizes, color=color, edgecolor="white", linewidth=0.3, alpha=0.86, label=name)
        ax.set(xlabel=x_label, ylabel=y_label)
        ax.grid(alpha=0.15)
        ax.text(0.02, 0.97, "def"[panel], transform=ax.transAxes, va="top", fontweight="bold", fontsize=9.5)
    axes[0].set_title("laboratory vs bond-frame order")
    axes[1].set_title("competing pair harmonics")
    axes[2].set_title("hidden-memory population")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        framealpha=0.88,
        edgecolor="none",
        loc="best",
        handletextpad=0.35,
        labelspacing=0.25,
    )
    save(fig, output_dir, "capillary_regime_state_space")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clusters", type=int, default=0, help="0 selects k by silhouette score")
    args = parser.parse_args()

    configure()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(input_path)
    cells = aggregate_cells(rows)
    x = np.asarray([[float(cell[key]) for key in FEATURES] for cell in cells], dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    cluster_count, silhouettes = choose_clusters(x_scaled, args.clusters)
    model = KMeans(n_clusters=cluster_count, random_state=11, n_init=200).fit(x_scaled)
    labels = model.labels_
    unscaled_centroids = scaler.inverse_transform(model.cluster_centers_)
    names_by_cluster = unique_regime_names(unscaled_centroids)
    regime_names = [names_by_cluster[int(label)] for label in labels]

    distance = np.linalg.norm(x_scaled[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    ordered_distance = np.sort(distance, axis=1)
    confidence = (ordered_distance[:, 1] - ordered_distance[:, 0]) / np.maximum(ordered_distance[:, 1], 1e-12)

    reference = labels
    ari = []
    for seed in range(20):
        comparison = KMeans(n_clusters=cluster_count, random_state=seed, n_init=30).fit_predict(x_scaled)
        ari.append(float(adjusted_rand_score(reference, comparison)))

    plot_regime_diagram(cells, regime_names, confidence, output_dir)
    plot_state_space(cells, regime_names, output_dir)

    report = {
        "input": str(input_path),
        "raw_rows": len(rows),
        "parameter_cells": len(cells),
        "graph_counts": sorted({int(cell["graph_count"]) for cell in cells}),
        "features": list(FEATURES),
        "cluster_count": int(cluster_count),
        "silhouette_scores": {str(key): value for key, value in silhouettes.items()},
        "initialization_ARI_mean": float(np.mean(ari)),
        "initialization_ARI_min": float(np.min(ari)),
        "classification_confidence_mean": float(confidence.mean()),
        "classification_confidence_min": float(confidence.min()),
        "regimes": [],
        "disclaimer": "Finite-size data-resolved regimes; no thermodynamic phase boundary is inferred.",
    }
    for cluster_index in range(cluster_count):
        mask = labels == cluster_index
        report["regimes"].append(
            {
                "name": names_by_cluster[cluster_index],
                "cell_count": int(mask.sum()),
                "centroid": {label: float(value) for label, value in zip(FEATURES, unscaled_centroids[cluster_index])},
                "J_range": [float(min(cells[i]["j_align"] for i in np.flatnonzero(mask))), float(max(cells[i]["j_align"] for i in np.flatnonzero(mask)))],
                "g_range": [float(min(cells[i]["g_capillary"] for i in np.flatnonzero(mask))), float(max(cells[i]["g_capillary"] for i in np.flatnonzero(mask)))],
            }
        )
    (output_dir / "capillary_regime_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "cluster_count": cluster_count, "silhouette": silhouettes[cluster_count]}, sort_keys=True))


if __name__ == "__main__":
    main()
