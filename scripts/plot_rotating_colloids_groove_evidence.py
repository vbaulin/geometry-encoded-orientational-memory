#!/usr/bin/env python3
"""Build evidence-centered figures for the grooved colloid manuscript."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_cache = Path(tempfile.gettempdir()) / "hyperion_matplotlib_cache"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")
OUT = Path("tex/rotating_colloids/grooved_prl_figures")
PROTOCOL_CANDIDATES = (
    ROOT / "rotating_colloids_grooved_protocols_n16_validation/groove_protocols.json",
    ROOT / "rotating_colloids_grooved_protocols_quick_mosaic/groove_protocols.json",
)

RUNS = (
    ("uniform", "rotating_colloids_grooved_uniform_scan_n16", "#3B6FB6"),
    ("long-range", "rotating_colloids_grooved_longrange_disorder", "#D9852B"),
    ("triangular", "rotating_colloids_grooved_triangular_frustrated_n16", "#C95656"),
    ("mosaic", "rotating_colloids_grooved_mosaic_hidden_search_n32", "#7C2D12"),
)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.6,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.25,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def fnum(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def load_jsonl(folder: Path) -> list[dict[str, Any]]:
    paths = sorted(folder.glob("*_points.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one points JSONL in {folder}, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    with paths[0].open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def load_scans() -> dict[str, dict[str, Any]]:
    scans: dict[str, dict[str, Any]] = {}
    for label, folder, color in RUNS:
        rows = load_jsonl(ROOT / folder)
        scans[label] = {"rows": rows, "color": color}
    return scans


def load_protocol() -> tuple[Path, dict[str, Any]]:
    for path in PROTOCOL_CANDIDATES:
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("No independent-field protocol result found")


def panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.035, label, transform=ax.transAxes, fontweight="bold", fontsize=8.2)


def draw_segment(ax, center: tuple[float, float], angle: float, length: float, **kwargs):
    direction = 0.5 * length * np.asarray([math.cos(angle), math.sin(angle)])
    point = np.asarray(center)
    return ax.plot([point[0] - direction[0], point[0] + direction[0]],
                   [point[1] - direction[1], point[1] + direction[1]], **kwargs)


def draw_groove_tile(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    angle: float,
    *,
    line_color: str = "#D97706",
    face_color: str = "#F8FAFC",
    edge_color: str = "#CBD5E1",
    line_count: int = 7,
    line_alpha: float = 0.55,
    line_style: str = "-",
) -> Rectangle:
    """Draw a clipped patch of parallel substrate grooves."""
    tile = Rectangle((x, y), width, height, facecolor=face_color, edgecolor=edge_color, lw=0.75)
    ax.add_patch(tile)
    center = np.asarray([x + 0.5 * width, y + 0.5 * height])
    normal = np.asarray([-math.sin(angle), math.cos(angle)])
    span = 0.78 * (abs(width * math.sin(angle)) + abs(height * math.cos(angle)) + 1e-6)
    length = 2.2 * math.hypot(width, height)
    for offset in np.linspace(-span, span, line_count):
        groove_center = center + offset * normal
        lines = draw_segment(
            ax,
            (float(groove_center[0]), float(groove_center[1])),
            angle,
            length,
            color=line_color,
            lw=0.65,
            alpha=line_alpha,
            linestyle=line_style,
            solid_capstyle="round",
            zorder=1,
        )
        for line in lines:
            line.set_clip_path(tile.get_path(), tile.get_transform())
    return tile


def plot_model_application() -> str:
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.35, 2.35),
        gridspec_kw={"width_ratios": [1.10, 1.08, 1.42, 1.12]},
        constrained_layout=True,
    )

    ax = axes[0]
    draw_groove_tile(ax, 0.05, 0.18, 0.78, 1.32, 0.0, line_count=8)
    draw_groove_tile(ax, 0.83, 0.18, 0.78, 1.32, math.pi / 4.0, line_count=8)
    centers = [(0.33, 0.52), (0.58, 1.17), (1.05, 0.52), (1.34, 1.16)]
    rod_angles = [0.05, -0.08, math.pi / 4.0 - 0.08, math.pi / 4.0 + 0.06]
    for center, theta in zip(centers, rod_angles):
        draw_segment(ax, center, theta, 0.38, color="#174A7E", lw=4.0, solid_capstyle="round", zorder=4)
        ax.add_patch(Circle(center, 0.035, facecolor="white", edgecolor="#174A7E", lw=0.65, zorder=5))
    for left, right in ((0, 1), (0, 2), (1, 3), (2, 3)):
        ax.plot(
            [centers[left][0], centers[right][0]],
            [centers[left][1], centers[right][1]],
            color="#64748B",
            lw=0.65,
            alpha=0.75,
            zorder=2,
        )
    ax.text(0.82, 0.08, r"groove torque $h$; coupling $J$", ha="center", fontsize=6.3)
    ax.set_title("geometry-written grooves")
    ax.set_xlim(-0.02, 1.68)
    ax.set_ylim(0.02, 1.60)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "a")

    ax = axes[1]
    patch_size = 0.5
    angular_offsets = (0.19, -0.13, 0.08, -0.17)
    for bx in range(4):
        for by in range(4):
            alpha = ((bx + 2 * by) % 4) * math.pi / 4.0
            draw_groove_tile(
                ax,
                bx * patch_size,
                by * patch_size,
                patch_size,
                patch_size,
                alpha,
                line_count=3,
                line_alpha=0.27,
                line_style="--",
                edge_color="#E2E8F0",
            )
            jitter_x = 0.035 * math.sin(2.3 * bx + 1.7 * by)
            jitter_y = 0.035 * math.cos(1.9 * bx - 1.3 * by)
            center = (
                (bx + 0.5) * patch_size + jitter_x,
                (by + 0.5) * patch_size + jitter_y,
            )
            angular_offset = angular_offsets[(bx + by) % len(angular_offsets)]
            draw_segment(
                ax,
                center,
                alpha + angular_offset,
                0.22,
                color="#174A7E",
                lw=2.5,
                solid_capstyle="round",
                zorder=4,
            )
    ax.text(1.0, -0.12, r"hidden state: $S\simeq0$", ha="center", fontsize=6.5)
    ax.text(1.0, -0.27, r"$C_2,G_2,q_{\rm EA}$ high", ha="center", fontsize=6.5, color="#7C2D12")
    ax.set_title("prescribed groove mosaic")
    ax.set_xlim(-0.04, 2.04)
    ax.set_ylim(-0.34, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "b")

    ax = axes[2]
    row_y = (0.80, 0.50, 0.20)
    labels = ("steady", "release", "rewrite")
    for y, label in zip(row_y, labels):
        ax.text(0.02, y, label, ha="left", va="center", fontsize=6.7, fontweight="bold")
    draw_segment(ax, (0.47, row_y[0]), 0.0, 0.28, color="#D97706", lw=1.8, alpha=0.65)
    draw_segment(ax, (0.47, row_y[0]), 0.05, 0.20, color="#174A7E", lw=3.0, solid_capstyle="round")
    t = np.linspace(0.0, 1.0, 80)
    ax.plot(0.77 + 0.90 * t, row_y[0] - 0.02 * (1.0 - np.exp(-5.0 * t)), color="#174A7E", lw=1.2)
    ax.text(1.22, row_y[0] + 0.07, r"$C_{\rm self}\to q_{\rm EA}$", ha="center", fontsize=6.1)

    draw_segment(ax, (0.47, row_y[1]), 0.0, 0.28, color="#D97706", lw=1.5, alpha=0.30)
    ax.plot([0.35, 0.59], [row_y[1] - 0.12, row_y[1] + 0.12], color="#9A3412", lw=1.0)
    draw_segment(ax, (0.47, row_y[1]), 0.14, 0.20, color="#64748B", lw=3.0, solid_capstyle="round")
    ax.plot(0.77 + 0.90 * t, row_y[1] + 0.10 * (2.0 * np.exp(-4.2 * t) - 1.0), color="#2A9D8F", lw=1.2)
    ax.text(1.22, row_y[1] + 0.12, r"$h\to0$: $Q_A(t)$", ha="center", fontsize=6.1)

    draw_segment(ax, (0.38, row_y[2]), math.pi / 2.0, 0.18, color="#174A7E", lw=3.0, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((0.50, row_y[2]), (0.64, row_y[2]), arrowstyle="-|>", mutation_scale=8,
                                 color="#334155", lw=0.8))
    draw_segment(ax, (0.72, row_y[2]), 0.0, 0.18, color="#D95F02", lw=3.0, solid_capstyle="round")
    ax.plot(0.90 + 0.77 * t, row_y[2] + 0.09 * np.tanh(7.0 * (t - 0.50)), color="#D95F02", lw=1.2)
    ax.text(1.28, row_y[2] + 0.13, r"$A\to B$: $G_2^B(t)$", ha="center", fontsize=6.1)
    ax.set_title("three distinct operations")
    ax.set_xlim(0.0, 1.72)
    ax.set_ylim(0.02, 1.02)
    ax.axis("off")
    panel_label(ax, "c")

    ax = axes[3]
    pair_centers = ((0.28, 0.38), (1.18, 0.52), (0.72, 1.25))
    for first, second in ((0, 1), (0, 2), (1, 2)):
        color = "#D97706" if (first, second) == (0, 1) else "#94A3B8"
        width = 1.3 if (first, second) == (0, 1) else 0.75
        ax.plot(
            [pair_centers[first][0], pair_centers[second][0]],
            [pair_centers[first][1], pair_centers[second][1]],
            color=color,
            lw=width,
            zorder=1,
        )
    for center, theta in zip(pair_centers, (0.30, -0.18, math.pi / 2.0)):
        draw_segment(ax, center, theta, 0.32, color="#174A7E", lw=3.7, solid_capstyle="round", zorder=3)
        ax.add_patch(Circle(center, 0.035, facecolor="white", edgecolor="#174A7E", lw=0.65, zorder=4))
    ax.plot([pair_centers[0][0], pair_centers[0][0] + 0.34],
            [pair_centers[0][1], pair_centers[0][1]], color="#64748B", lw=0.6, linestyle=":")
    ax.add_patch(Arc(pair_centers[0], 0.38, 0.28, theta1=0, theta2=9, color="#9A3412", lw=0.8))
    ax.text(0.52, 0.43, r"$\phi_{ij}$", fontsize=6.5, color="#9A3412")
    ax.text(0.72, 0.13, r"$\cos2(\theta_i+\theta_j-2\phi_{ij})$", ha="center", fontsize=5.8)
    ax.text(0.72, 0.00, "interaction graph stores the code", ha="center", fontsize=5.9)
    ax.set_title("geometry-generated bonds")
    ax.set_xlim(0.02, 1.45)
    ax.set_ylim(-0.08, 1.48)
    ax.set_aspect("equal")
    ax.axis("off")
    panel_label(ax, "d")

    output = OUT / "fig1_model_schematic.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return str(output)


def physical_controls(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    noise = np.asarray([fnum(row, "noise") for row in rows])
    j = np.asarray([fnum(row, "j_align") for row in rows])
    eps = np.asarray([fnum(row, "eps_geom") for row in rows])
    return j / noise, eps * j / noise


def scan_rows(scans: dict[str, dict[str, Any]], label: str) -> list[dict[str, Any]]:
    rows = scans[label]["rows"]
    if label == "long-range":
        rows = [row for row in rows if int(fnum(row, "n")) == 48]
    return rows


def hidden_score(row: dict[str, Any]) -> float:
    return (1.0 - fnum(row, "nematic_order_mean")) * min(
        fnum(row, "orientational_corr_nn_mean"),
        fnum(row, "geometry_lock_mean"),
        fnum(row, "q_EA_mean"),
    )


def plot_regime_densities(scans: dict[str, dict[str, Any]]) -> str:
    """Compare geometry-specific state populations without drawing scan paths."""
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.15), sharex=True, sharey=True, constrained_layout=True)
    labels = ("uniform", "long-range", "triangular", "mosaic")
    titles = (
        "uniform grooves",
        "long-range control",
        "triangular frustration",
        "mosaic grooves",
    )
    cmaps = {
        label: LinearSegmentedColormap.from_list(
            f"{label}-density", ["#FFFFFF", scans[label]["color"]]
        )
        for label in labels
    }
    x_edges = np.linspace(0.0, 1.0, 151)
    y_edges = np.linspace(0.0, 1.0, 151)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    levels = (0.025, 0.08, 0.18, 0.35, 0.58, 0.82, 1.001)

    for ax, label, title in zip(axes.ravel(), labels, titles):
        rows = scan_rows(scans, label)
        s = np.asarray([fnum(row, "nematic_order_mean") for row in rows])
        g = np.asarray([fnum(row, "geometry_lock_mean") for row in rows])
        density, _, _ = np.histogram2d(s, g, bins=(x_edges, y_edges))
        density = gaussian_filter(density, sigma=3.2, mode="constant")
        density /= max(float(np.nanmax(density)), 1e-12)
        ax.contourf(
            x_centers,
            y_centers,
            density.T,
            levels=levels,
            cmap=cmaps[label],
            antialiased=True,
        )
        ax.contour(
            x_centers,
            y_centers,
            density.T,
            levels=(0.18, 0.58),
            colors=scans[label]["color"],
            linewidths=(0.55, 0.85),
            alpha=0.90,
        )
        centroid = (float(np.mean(s)), float(np.mean(g)))
        ax.scatter(
            [centroid[0]],
            [centroid[1]],
            s=34,
            facecolor="white",
            edgecolor=scans[label]["color"],
            linewidth=1.0,
            zorder=5,
        )
        q_mean = float(np.mean([fnum(row, "q_EA_mean") for row in rows]))
        c_mean = float(np.mean([fnum(row, "orientational_corr_nn_mean") for row in rows]))
        ax.text(
            0.035,
            0.965,
            rf"$N_{{\rm cell}}={len(rows)}$" + "\n" + rf"$\langle C_2\rangle={c_mean:.2f}$, $\langle q_{{\rm EA}}\rangle={q_mean:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color="#1F2937",
        )
        ax.set_title(title)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.10, zorder=0)

    for label, ax in zip(("a", "b", "c", "d"), axes.ravel()):
        panel_label(ax, label)
    fig.supxlabel(r"global nematic order $S$", fontsize=8.2)
    fig.supylabel(r"registration to the local groove pattern $G_2$", fontsize=8.2)
    output = OUT / "fig2_regime_densities.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return str(output)


def plot_static_state_space(scans: dict[str, dict[str, Any]]) -> str:
    """Project all four static observables without showing parameter-grid paths."""
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.72), constrained_layout=True)
    labels = ("uniform", "long-range", "triangular", "mosaic")
    display_labels = {
        "uniform": "uniform grooves",
        "long-range": "long-range",
        "triangular": "triangular",
        "mosaic": "mosaic",
    }
    panels = (
        ("nematic_order_mean", "geometry_lock_mean", r"global nematic order $S$",
         r"groove registration $G_2$", "global order versus registration"),
        ("orientational_corr_nn_mean", "q_EA_mean", r"neighbour coherence $C_2$",
         r"field-on persistence $q_{\rm EA}$", "coherence versus persistence"),
    )
    edges = np.linspace(0.0, 1.0, 181)
    centers = 0.5 * (edges[:-1] + edges[1:])
    label_offsets = {
        0: {
            "uniform": (-0.22, 0.07),
            "long-range": (-0.31, 0.05),
            "triangular": (-0.31, 0.06),
            "mosaic": (0.035, 0.035),
        },
        1: {
            "uniform": (-0.31, -0.12),
            "long-range": (-0.31, 0.035),
            "triangular": (-0.31, -0.14),
            "mosaic": (-0.30, 0.045),
        },
    }

    for panel_index, (ax, panel) in enumerate(zip(axes[:2], panels)):
        x_key, y_key, x_label, y_label, title = panel
        for label in labels:
            rows = scan_rows(scans, label)
            x = np.asarray([fnum(row, x_key) for row in rows])
            y = np.asarray([fnum(row, y_key) for row in rows])
            density, _, _ = np.histogram2d(x, y, bins=(edges, edges))
            density = gaussian_filter(density, sigma=3.2, mode="constant")
            density /= max(float(np.nanmax(density)), 1e-12)
            color = scans[label]["color"]
            cmap = LinearSegmentedColormap.from_list(
                f"{label}-{panel_index}-overlay", ["#FFFFFF", color]
            )
            ax.contourf(
                centers,
                centers,
                density.T,
                levels=(0.12, 0.32, 0.58, 0.82, 1.001),
                cmap=cmap,
                alpha=0.46,
                antialiased=True,
            )
            ax.contour(
                centers,
                centers,
                density.T,
                levels=(0.12, 0.58),
                colors=color,
                linewidths=(0.65, 1.0),
            )
            centroid = (float(np.mean(x)), float(np.mean(y)))
            ax.scatter(
                [centroid[0]],
                [centroid[1]],
                s=34,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=6,
            )
            dx, dy = label_offsets[panel_index][label]
            ax.annotate(
                display_labels[label],
                xy=centroid,
                xytext=(centroid[0] + dx, centroid[1] + dy),
                textcoords="data",
                color=color,
                fontsize=6.4,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
                ha="left",
                va="center",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(alpha=0.10, zorder=0)
        panel_label(ax, "a" if panel_index == 0 else "b")

    mosaic = scan_rows(scans, "mosaic")
    hidden = [
        row
        for row in mosaic
        if fnum(row, "nematic_order_mean") <= 0.35
        and fnum(row, "orientational_corr_nn_mean") >= 0.70
        and fnum(row, "geometry_lock_mean") >= 0.70
        and fnum(row, "q_EA_mean") >= 0.50
    ]
    metric_keys = (
        "nematic_order_mean",
        "orientational_corr_nn_mean",
        "geometry_lock_mean",
        "q_EA_mean",
    )
    metric_matrix = np.asarray([[fnum(row, key) for key in metric_keys] for row in hidden])
    median = np.nanmedian(metric_matrix, axis=0)
    scale = np.maximum(np.nanstd(metric_matrix, axis=0), 1e-9)
    representative = hidden[int(np.argmin(np.sum(((metric_matrix - median) / scale) ** 2, axis=1)))]
    r = np.asarray(representative.get("radial_r", []), dtype=float)
    c = np.asarray(representative.get("radial_C2_mean", []), dtype=float)
    c_std = np.asarray(representative.get("radial_C2_std", []), dtype=float)
    valid = np.isfinite(r) & np.isfinite(c) & np.isfinite(c_std)
    ax = axes[2]
    ax.fill_between(r[valid], c[valid] - c_std[valid], c[valid] + c_std[valid], color="#7C2D12", alpha=0.14)
    ax.plot(r[valid], c[valid], color="#7C2D12", marker="o", markersize=2.4)
    domain_width = float(representative.get("cluster_size", 8))
    ax.axhline(0.0, color="#64748B", linewidth=0.7)
    ax.axvline(domain_width, color="#D97706", linestyle="--", linewidth=0.9)
    ax.text(domain_width + 0.35, 0.76, "domain\nwidth", color="#9A3412", fontsize=6.1)
    ax.text(1.0, 0.58, "same local\nreference frame", color="#7C2D12", fontsize=6.0)
    ax.text(9.1, -0.24, "incompatible\ndomains", color="#475569", fontsize=6.0)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(-0.55, 1.0)
    ax.set_xlabel(r"separation $r/a$")
    ax.set_ylabel(r"orientational correlation $C_2(r)$")
    ax.set_title("domain-scale sign reversal")
    ax.grid(alpha=0.10)
    panel_label(ax, "c")
    output = OUT / "fig2_static_state_space.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return str(output)


def plot_registered_state(scans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mosaic = scans["mosaic"]["rows"]
    j, h = physical_controls(mosaic)
    g = np.asarray([fnum(row, "geometry_lock_mean") for row in mosaic])
    triangulation = mtri.Triangulation(j, h)

    score = np.asarray(
        [fnum(row, "geometry_lock_mean") * (1.0 - fnum(row, "nematic_order_mean")) for row in mosaic]
    )
    best = mosaic[int(np.nanargmax(score))]

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True)

    ax = axes[0]
    levels = np.linspace(float(np.nanmin(g)), float(np.nanmax(g)), 13)
    contour = ax.tricontourf(triangulation, g, levels=levels, cmap="viridis")
    ax.tricontour(triangulation, g, levels=[0.50, 0.70, 0.85], colors="#1F2937", linewidths=0.55)
    ax.set_xlabel(r"interparticle coupling $J/D_r$")
    ax.set_ylabel(r"substrate field $h/D_r$")
    ax.set_title(r"mosaic registration $G_2$")
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025, label=r"$G_2$", format="%.2f")
    ax.text(0.04, 0.91, "not sampled", transform=ax.transAxes, color="#64748B", fontsize=6.2)
    ax.text(0.69, 0.05, "not sampled", transform=ax.transAxes, color="#64748B", fontsize=6.2)
    panel_label(ax, "a")

    ax = axes[1]
    for label in ("uniform", "long-range", "triangular", "mosaic"):
        rows = scans[label]["rows"]
        if label == "long-range":
            rows = [row for row in rows if int(fnum(row, "n")) == 48]
        x = np.asarray([fnum(row, "geometry_lock_mean") ** 2 for row in rows])
        y = np.asarray([fnum(row, "q_EA_mean") for row in rows])
        ax.scatter(
            x,
            y,
            s=7 if label != "mosaic" else 9,
            color=scans[label]["color"],
            alpha=0.26 if label != "mosaic" else 0.42,
            edgecolors="none",
            rasterized=True,
            label=label,
        )
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=0.9, label=r"$q_{\rm EA}=G_2^2$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"static registration $G_2^2$")
    ax.set_ylabel(r"field-on $q_{\rm EA}$")
    ax.set_title("pinning identity")
    ax.legend(frameon=False, loc="lower right", fontsize=5.9)
    ax.grid(alpha=0.13)
    panel_label(ax, "b")

    ax = axes[2]
    r = np.asarray(best.get("radial_r", []), dtype=float)
    c = np.asarray([np.nan if value is None else float(value) for value in best.get("radial_C2_mean", [])])
    valid = np.isfinite(r) & np.isfinite(c)
    ax.plot(r[valid], c[valid], marker="o", markersize=3.0, color="#7C2D12")
    ax.axhline(0.0, color="#6B7280", linewidth=0.75)
    domain_width = float(best.get("cluster_size", 8))
    ax.axvline(domain_width, color="#6B7280", linestyle=":", linewidth=0.8)
    ax.text(domain_width + 0.18, 0.04, "domain width", rotation=90, color="#475569", fontsize=6.2,
            ha="left", va="bottom")
    ax.set_xlabel(r"separation $r/a$")
    ax.set_ylabel(r"orientational correlation $C_2(r)$")
    ax.set_title("finite-domain correlation")
    ax.grid(alpha=0.13)
    panel_label(ax, "c")

    output = OUT / "fig2_registered_state.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return {
        "figure": str(output),
        "best_existing_cell": {
            "J_over_Dr": fnum(best, "j_align") / fnum(best, "noise"),
            "h_over_Dr": fnum(best, "eps_geom") * fnum(best, "j_align") / fnum(best, "noise"),
            "S": fnum(best, "nematic_order_mean"),
            "C2": fnum(best, "orientational_corr_nn_mean"),
            "G2": fnum(best, "geometry_lock_mean"),
            "qEA_field_on": fnum(best, "q_EA_mean"),
        },
    }


def protocol_groups(rows: list[dict[str, Any]], key: str) -> list[float]:
    return sorted({fnum(row, key) for row in rows})


def crossing_time(rows: list[dict[str, Any]], target: float = 0.5) -> float | None:
    """Estimate the first downward crossing using log-time interpolation."""
    ordered = sorted(rows, key=lambda row: fnum(row, "release_time_Dr"))
    for left, right in zip(ordered, ordered[1:]):
        y0 = fnum(left, "Q_rem_mean")
        y1 = fnum(right, "Q_rem_mean")
        if y0 >= target > y1:
            x0 = fnum(left, "release_time_Dr")
            x1 = fnum(right, "release_time_Dr")
            fraction = (target - y0) / (y1 - y0)
            if x0 > 0.0 and x1 > 0.0:
                return float(math.exp(math.log(x0) + fraction * (math.log(x1) - math.log(x0))))
            return float(x0 + fraction * (x1 - x0))
    return None


def plot_amplifier(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = protocol.get("amplifier", [])
    if not rows:
        raise RuntimeError("Protocol result has no amplifier rows")
    n_value = max(int(fnum(row, "n")) for row in rows)
    rows = [row for row in rows if int(fnum(row, "n")) == n_value]
    h_values = protocol_groups(rows, "h_over_Dr")
    colors = plt.cm.cividis(np.linspace(0.15, 0.88, len(h_values)))

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45), constrained_layout=True)
    for h, color in zip(h_values, colors):
        subset = sorted([row for row in rows if math.isclose(fnum(row, "h_over_Dr"), h)], key=lambda row: fnum(row, "J_over_Dr"))
        x = np.asarray([fnum(row, "J_over_Dr") for row in subset])
        for ax, key, std_key in (
            (axes[0], "G2_mean", "G2_std"),
            (axes[1], "S_mean", "S_std"),
            (axes[2], "cooperative_gain", None),
        ):
            y = np.asarray([fnum(row, key) for row in subset])
            ax.plot(x, y, marker="o", markersize=3.0, color=color, label=rf"$h/D_r={h:g}$")
            if std_key is not None:
                err = np.asarray([fnum(row, std_key) for row in subset])
                ax.fill_between(x, y - err, y + err, color=color, alpha=0.13, linewidth=0)

    axes[0].set_ylabel(r"registration $G_2$")
    axes[0].set_title("collective registration")
    axes[1].set_ylabel(r"global nematic order $S$")
    axes[1].set_title("global cancellation")
    axes[2].set_ylabel(r"gain $\Delta G_2=G_2(J)-G_2(0)$")
    axes[2].set_title("substrate-signal gain")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.set_xlabel(r"interparticle coupling $J/D_r$")
        ax.set_ylim(-0.04, 1.02 if ax is not axes[2] else 0.82)
        ax.grid(alpha=0.14)
        panel_label(ax, label)
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.1)

    output = OUT / "fig3_collective_amplifier.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    best = max(rows, key=lambda row: fnum(row, "cooperative_gain"))
    return {
        "figure": str(output),
        "n": n_value,
        "maximum_gain": {
            key: best.get(key)
            for key in ("J_over_Dr", "h_over_Dr", "S_mean", "C2_mean", "G2_mean", "G2_noninteracting", "cooperative_gain")
        },
    }


def plot_release(protocol: dict[str, Any]) -> dict[str, Any]:
    rows = protocol.get("release", [])
    if not rows:
        raise RuntimeError("Protocol result has no release rows")
    n_value = max(int(fnum(row, "n")) for row in rows)
    rows = [row for row in rows if int(fnum(row, "n")) == n_value]
    h_value = min(protocol_groups(rows, "h_write_over_Dr"), key=lambda value: abs(value - 0.6))
    rows = [row for row in rows if math.isclose(fnum(row, "h_write_over_Dr"), h_value)]
    j_values = protocol_groups(rows, "J_over_Dr")
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(j_values)))

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.45), constrained_layout=True)
    end_rows: list[dict[str, Any]] = []
    half_lives: dict[str, float | None] = {}
    for fraction, linestyle, fraction_label in ((0.0, "-", "field off"), (0.15, "--", "15% hold")):
        for j, color in zip(j_values, colors):
            subset = sorted(
                [
                    row
                    for row in rows
                    if math.isclose(fnum(row, "J_over_Dr"), j)
                    and math.isclose(fnum(row, "h_release_fraction"), fraction)
                ],
                key=lambda row: fnum(row, "release_time_Dr"),
            )
            if not subset:
                continue
            x = np.asarray([fnum(row, "release_time_Dr") for row in subset])
            q = np.asarray([fnum(row, "Q_rem_mean") for row in subset])
            q_std = np.asarray([fnum(row, "Q_rem_std") for row in subset])
            g = np.asarray([fnum(row, "G2_mean") for row in subset])
            g_std = np.asarray([fnum(row, "G2_std") for row in subset])
            if fraction == 0.0:
                axes[0].plot(x, q, linestyle=linestyle, color=color, label=rf"$J/D_r={j:g}$")
                axes[0].fill_between(x, q - q_std, q + q_std, color=color, alpha=0.09, linewidth=0)
                half_lives[f"J_over_Dr_{j:g}"] = crossing_time(subset)
            else:
                axes[1].plot(x, q, linestyle=linestyle, color=color, label=rf"$J/D_r={j:g}$")
                axes[2].plot(x, g, linestyle=linestyle, color=color, label=rf"$J/D_r={j:g}$")
                axes[1].fill_between(x, q - q_std, q + q_std, color=color, alpha=0.09, linewidth=0)
                axes[2].fill_between(x, g - g_std, g + g_std, color=color, alpha=0.09, linewidth=0)
            end_rows.append(subset[-1])

    t_line = np.geomspace(1e-3, max(fnum(row, "release_time_Dr") for row in rows), 200)
    axes[0].plot(t_line, np.exp(-4.0 * t_line), color="#111827", linestyle=":", linewidth=1.0,
                 label=r"free rotor $e^{-4D_rt}$")
    axes[0].set_title("writing field removed")
    axes[1].set_title("weak holding field")
    axes[2].set_title("registered response under hold")
    axes[0].set_ylabel(r"written-state overlap $Q_{\rm rem}$")
    axes[1].set_ylabel(r"written-state overlap $Q_{\rm rem}$")
    axes[2].set_ylabel(r"groove registration $G_2$")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.set_xscale("symlog", linthresh=0.03)
        ax.set_xticks([0.0, 0.1, 1.0, 10.0, 100.0])
        ax.set_xticklabels(["0", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
        ax.set_xlabel(r"release time $D_rt$")
        ax.set_ylim(-0.12, 1.04)
        ax.grid(alpha=0.14)
        panel_label(ax, label)
    axes[0].legend(frameon=False, fontsize=5.9, loc="upper right")
    axes[1].legend(frameon=False, fontsize=5.9, loc="upper right")

    output = OUT / "fig4_write_release_read.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    free_rotor_half_life = math.log(2.0) / 4.0
    return {
        "figure": str(output),
        "n": n_value,
        "h_write_over_Dr": h_value,
        "free_rotor_half_life_Drt": free_rotor_half_life,
        "field_off_half_life_Drt": half_lives,
        "half_life_gain_over_free_rotor": {
            key: (value / free_rotor_half_life if value is not None else None)
            for key, value in half_lives.items()
        },
        "end_rows": end_rows,
    }


def plot_hidden_memory_dynamics(protocol: dict[str, Any]) -> dict[str, Any]:
    """Combine field-on persistence, field-off release, and incompatible rewriting."""
    amplifier = protocol.get("amplifier", [])
    release = protocol.get("release", [])
    switch = protocol.get("switch", [])
    if not amplifier or not release or not switch:
        raise RuntimeError("Hidden-memory dynamics requires amplifier, release, and switch results")

    n_value = max(int(fnum(row, "n")) for row in amplifier)
    amplifier = [row for row in amplifier if int(fnum(row, "n")) == n_value]
    release = [row for row in release if int(fnum(row, "n")) == n_value]
    switch = [row for row in switch if int(fnum(row, "n")) == n_value]
    j_values = [value for value in (0.0, 2.0, 3.0) if any(math.isclose(fnum(r, "J_over_Dr"), value) for r in amplifier)]
    colors = dict(zip(j_values, plt.cm.viridis(np.linspace(0.12, 0.88, len(j_values)))))

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.55), constrained_layout=True)

    ax = axes[0]
    lag_keys = ((1, "temporal_C2_lag1"), (5, "temporal_C2_lag5"),
                (20, "temporal_C2_lag20"), (50, "temporal_C2_lag50"))
    sample_dt = (
        float(protocol.get("config", {}).get("sample_stride", 40))
        * float(protocol.get("config", {}).get("dt", 0.015))
        * float(protocol.get("config", {}).get("Dr", 0.45))
    )
    for j in j_values:
        candidates = [
            row
            for row in amplifier
            if math.isclose(fnum(row, "J_over_Dr"), j)
            and math.isclose(fnum(row, "h_over_Dr"), 0.2)
        ]
        if not candidates:
            continue
        row = candidates[0]
        x: list[float] = []
        y: list[float] = []
        for lag, key in lag_keys:
            value = row.get(key)
            if value is not None and math.isfinite(float(value)):
                x.append(lag * sample_dt)
                y.append(float(value))
        ax.plot(x, y, marker="o", markersize=2.8, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.axhline(fnum(row, "qEA_field_on_mean"), color=colors[j], linestyle=":", linewidth=0.75, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlim(0.22, 17.0)
    ax.set_xticks([0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.3", "1", "3", "10"])
    ax.set_ylim(-0.05, 1.04)
    ax.set_xlabel(r"lag $D_r\Delta t$")
    ax.set_ylabel(r"self-correlation $C_{\rm self}$")
    ax.set_title("field-supported memory")
    ax.legend(frameon=False, fontsize=5.8, loc="lower left")
    ax.grid(alpha=0.14)
    panel_label(ax, "a")

    ax = axes[1]
    half_lives: dict[str, float | None] = {}
    for j in j_values:
        subset = sorted(
            [
                row
                for row in release
                if math.isclose(fnum(row, "J_over_Dr"), j)
                and math.isclose(fnum(row, "h_write_over_Dr"), 0.6)
                and math.isclose(fnum(row, "h_release_fraction"), 0.0)
            ],
            key=lambda row: fnum(row, "release_time_Dr"),
        )
        x = np.asarray([fnum(row, "release_time_Dr") for row in subset])
        y = np.asarray([fnum(row, "Q_rem_mean") for row in subset])
        err = np.asarray([fnum(row, "Q_rem_std") for row in subset])
        ax.plot(x, y, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.fill_between(x, y - err, y + err, color=colors[j], alpha=0.09, linewidth=0)
        half_lives[f"J_over_Dr_{j:g}"] = crossing_time(subset)
    t_line = np.geomspace(1e-3, max(fnum(row, "release_time_Dr") for row in release), 200)
    ax.plot(t_line, np.exp(-4.0 * t_line), color="#111827", linestyle=":", linewidth=0.9,
            label="free rotor")
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.65)
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlim(0.0, max(fnum(row, "release_time_Dr") for row in release))
    ax.set_xticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
    ax.set_ylim(-0.12, 1.04)
    ax.set_xlabel(r"time after field removal $D_rt$")
    ax.set_ylabel(r"written-state overlap $Q_A$")
    ax.set_title("memory after release")
    ax.legend(frameon=False, fontsize=5.7, loc="upper right")
    ax.grid(alpha=0.14)
    panel_label(ax, "b")

    ax = axes[2]
    rewrite_half: dict[str, float | None] = {}
    for j in j_values:
        subset = sorted(
            [
                row
                for row in switch
                if math.isclose(fnum(row, "J_over_Dr"), j)
                and math.isclose(fnum(row, "h_over_Dr"), 0.6)
            ],
            key=lambda row: fnum(row, "switch_time_Dr"),
        )
        x = np.asarray([fnum(row, "switch_time_Dr") for row in subset])
        y = np.asarray([fnum(row, "Q_written_mean") for row in subset])
        err = np.asarray([fnum(row, "Q_written_std") for row in subset])
        ax.plot(x, y, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.fill_between(x, y - err, y + err, color=colors[j], alpha=0.09, linewidth=0)
        rewrite_half[f"J_over_Dr_{j:g}"] = crossing_time(
            [
                {
                    "release_time_Dr": fnum(row, "switch_time_Dr"),
                    "Q_rem_mean": fnum(row, "Q_written_mean"),
                }
                for row in subset
            ]
        )
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.65)
    ax.axhline(0.0, color="#64748B", linewidth=0.65)
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlim(0.0, max(fnum(row, "switch_time_Dr") for row in switch))
    ax.set_xticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
    ax.set_ylim(-0.9, 1.04)
    ax.set_xlabel(r"time after pattern switch $D_rt$")
    ax.set_ylabel(r"overlap with written pattern $Q_A$")
    ax.set_title(r"rewriting $A\!\rightarrow\!B$")
    ax.legend(frameon=False, fontsize=5.7, loc="upper right")
    ax.grid(alpha=0.14)
    panel_label(ax, "c")

    output = OUT / "fig4_hidden_memory_dynamics.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return {
        "figure": str(output),
        "n": n_value,
        "field_off_half_life_Drt": half_lives,
        "rewrite_half_time_Drt": rewrite_half,
    }


def plot_hidden_memory_evidence(
    scans: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> dict[str, Any]:
    """Put the static population and three independent memory tests in one figure."""
    amplifier = protocol.get("amplifier", [])
    release = protocol.get("release", [])
    switch = protocol.get("switch", [])
    if not amplifier or not release or not switch:
        raise RuntimeError("Hidden-memory evidence requires amplifier, release, and switch results")

    n_value = max(int(fnum(row, "n")) for row in amplifier)
    amplifier = [row for row in amplifier if int(fnum(row, "n")) == n_value]
    release = [row for row in release if int(fnum(row, "n")) == n_value]
    switch = [row for row in switch if int(fnum(row, "n")) == n_value]
    j_values = [
        value
        for value in (0.0, 2.0, 3.0)
        if any(math.isclose(fnum(row, "J_over_Dr"), value) for row in amplifier)
    ]
    colors = dict(zip(j_values, plt.cm.viridis(np.linspace(0.12, 0.88, len(j_values)))))
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.05), constrained_layout=True)

    ax = axes[0, 0]
    for label in ("uniform", "long-range", "triangular", "mosaic"):
        values = np.sort(np.asarray([hidden_score(row) for row in scan_rows(scans, label)]))
        survival = (len(values) - np.arange(len(values))) / len(values)
        ax.step(
            values,
            survival,
            where="post",
            color=scans[label]["color"],
            label=label,
        )
    ax.axvline(0.4, color="#64748B", linestyle=":", linewidth=0.8)
    ax.text(0.41, 0.08, "all mosaic cells\nlie above 0.458", fontsize=6.2, color="#7C2D12")
    ax.set_xlim(-0.03, 0.92)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"continuous hidden-memory score $H$")
    ax.set_ylabel(r"fraction of cells with score $geq H$")
    ax.set_title("extended static population")
    ax.legend(frameon=False, fontsize=5.8, loc="upper right")
    ax.grid(alpha=0.13)
    panel_label(ax, "a")

    ax = axes[0, 1]
    lag_keys = (
        (1, "temporal_C2_lag1"),
        (5, "temporal_C2_lag5"),
        (20, "temporal_C2_lag20"),
        (50, "temporal_C2_lag50"),
    )
    sample_dt = (
        float(protocol.get("config", {}).get("sample_stride", 40))
        * float(protocol.get("config", {}).get("dt", 0.015))
        * float(protocol.get("config", {}).get("Dr", 0.45))
    )
    steady_rows: dict[str, dict[str, Any]] = {}
    for j in j_values:
        candidates = [
            row
            for row in amplifier
            if math.isclose(fnum(row, "J_over_Dr"), j)
            and math.isclose(fnum(row, "h_over_Dr"), 0.2)
        ]
        if not candidates:
            continue
        row = candidates[0]
        steady_rows[f"J_over_Dr_{j:g}"] = row
        x = np.asarray([lag * sample_dt for lag, key in lag_keys if row.get(key) is not None])
        y = np.asarray([float(row[key]) for _, key in lag_keys if row.get(key) is not None])
        ax.plot(x, y, marker="o", markersize=3.0, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.axhline(
            fnum(row, "qEA_field_on_mean"),
            color=colors[j],
            linestyle=":",
            linewidth=0.8,
            alpha=0.85,
        )
    row_j3 = steady_rows.get("J_over_Dr_3")
    if row_j3 is not None:
        ax.annotate(
            rf"$C_{{\rm self}}=0.689$" + "\n" + rf"$q_{{\rm EA}}=0.695$",
            xy=(13.5, fnum(row_j3, "temporal_C2_lag50")),
            xytext=(4.2, 0.87),
            arrowprops={"arrowstyle": "->", "lw": 0.7, "color": colors[3.0]},
            fontsize=6.2,
            color="#1F2937",
        )
        ax.text(
            0.04,
            0.07,
            rf"early--late $q_{{\rm EA}}$ drift $={fnum(row_j3, 'qEA_half_drift'):.3f}$",
            transform=ax.transAxes,
            fontsize=6.1,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.22, 17.0)
    ax.set_xticks([0.3, 1.0, 3.0, 10.0])
    ax.set_xticklabels(["0.3", "1", "3", "10"])
    ax.set_ylim(-0.05, 1.04)
    ax.set_xlabel(r"lag $D_r\Delta t$")
    ax.set_ylabel(r"self-correlation $C_{\rm self}$")
    ax.set_title("steady pattern memory")
    ax.legend(frameon=False, fontsize=5.7, loc="center left")
    ax.grid(alpha=0.13)
    panel_label(ax, "b")

    ax = axes[1, 0]
    half_lives: dict[str, float | None] = {}
    for j in j_values:
        subset = sorted(
            [
                row
                for row in release
                if math.isclose(fnum(row, "J_over_Dr"), j)
                and math.isclose(fnum(row, "h_write_over_Dr"), 0.6)
                and math.isclose(fnum(row, "h_release_fraction"), 0.0)
            ],
            key=lambda row: fnum(row, "release_time_Dr"),
        )
        x = np.asarray([fnum(row, "release_time_Dr") for row in subset])
        y = np.asarray([fnum(row, "Q_rem_mean") for row in subset])
        err = np.asarray([fnum(row, "Q_rem_std") for row in subset])
        ax.plot(x, y, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.fill_between(x, y - err, y + err, color=colors[j], alpha=0.09, linewidth=0)
        half_life = crossing_time(subset)
        half_lives[f"J_over_Dr_{j:g}"] = half_life
        if half_life is not None:
            ax.scatter([half_life], [0.5], s=18, facecolor="white", edgecolor=colors[j], zorder=5)
    free_half_life = math.log(2.0) / 4.0
    t_line = np.geomspace(1e-3, max(fnum(row, "release_time_Dr") for row in release), 200)
    ax.plot(t_line, np.exp(-4.0 * t_line), color="#111827", linestyle=":", linewidth=0.9, label="free rotor")
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.65)
    ax.text(0.46, 0.10, r"$22\times$", transform=ax.transAxes, color=colors[2.0], fontsize=6.3)
    ax.text(0.59, 0.10, r"$30\times$", transform=ax.transAxes, color=colors[3.0], fontsize=6.3)
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlim(0.0, max(fnum(row, "release_time_Dr") for row in release))
    ax.set_xticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
    ax.set_ylim(-0.12, 1.04)
    ax.set_xlabel(r"time after groove removal $D_rt$")
    ax.set_ylabel(r"written-state overlap $Q_A$")
    ax.set_title("memory after release")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax.grid(alpha=0.13)
    panel_label(ax, "c")

    ax = axes[1, 1]
    rewrite_half: dict[str, float | None] = {}
    for j in j_values:
        subset = sorted(
            [
                row
                for row in switch
                if math.isclose(fnum(row, "J_over_Dr"), j)
                and math.isclose(fnum(row, "h_over_Dr"), 0.6)
            ],
            key=lambda row: fnum(row, "switch_time_Dr"),
        )
        x = np.asarray([fnum(row, "switch_time_Dr") for row in subset])
        y = np.asarray([fnum(row, "Q_written_mean") for row in subset])
        err = np.asarray([fnum(row, "Q_written_std") for row in subset])
        ax.plot(x, y, color=colors[j], label=rf"$J/D_r={j:g}$")
        ax.fill_between(x, y - err, y + err, color=colors[j], alpha=0.09, linewidth=0)
        half_time = crossing_time(
            [
                {"release_time_Dr": fnum(row, "switch_time_Dr"), "Q_rem_mean": fnum(row, "Q_written_mean")}
                for row in subset
            ]
        )
        rewrite_half[f"J_over_Dr_{j:g}"] = half_time
        if half_time is not None:
            ax.scatter([half_time], [0.5], s=18, facecolor="white", edgecolor=colors[j], zorder=5)
    ax.axhline(0.5, color="#94A3B8", linestyle="--", linewidth=0.65)
    ax.axhline(0.0, color="#64748B", linewidth=0.65)
    ax.text(0.05, 0.08, r"$\tau_{1/2}=0.156,\ 0.590,\ 0.730$", transform=ax.transAxes, fontsize=6.2)
    ax.set_xscale("symlog", linthresh=0.03)
    ax.set_xlim(0.0, max(fnum(row, "switch_time_Dr") for row in switch))
    ax.set_xticks([0.0, 0.1, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$"])
    ax.set_ylim(-0.9, 1.04)
    ax.set_xlabel(r"time after pattern switch $D_rt$")
    ax.set_ylabel(r"overlap with written pattern $Q_A$")
    ax.set_title(r"history-dependent rewriting $A\!\rightarrow\!B$")
    ax.legend(frameon=False, fontsize=5.6, loc="upper right")
    ax.grid(alpha=0.13)
    panel_label(ax, "d")

    output = OUT / "fig3_hidden_memory_evidence.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return {
        "figure": str(output),
        "n": n_value,
        "steady_rows": steady_rows,
        "field_off_half_life_Drt": half_lives,
        "half_life_gain_over_free_rotor": {
            key: (value / free_half_life if value is not None else None)
            for key, value in half_lives.items()
        },
        "rewrite_half_time_Drt": rewrite_half,
    }


def plot_control_provenance(scans: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.25), constrained_layout=True)
    for ax, label in zip(axes.ravel(), ("uniform", "long-range", "triangular", "mosaic")):
        rows = scans[label]["rows"]
        if label == "long-range":
            rows = [row for row in rows if int(fnum(row, "n")) == 48]
        j, h = physical_controls(rows)
        s = np.asarray([fnum(row, "nematic_order_mean") for row in rows])
        g = np.asarray([fnum(row, "geometry_lock_mean") for row in rows])
        metric = g * (1.0 - s)
        tri = mtri.Triangulation(j, h)
        levels = np.linspace(0.0, 0.95, 14)
        im = ax.tricontourf(tri, metric, levels=levels, cmap="magma", extend="max")
        ax.set_title(label)
        ax.set_xlabel(r"$J/D_r$")
        ax.set_ylabel(r"$h/D_r$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label=r"$G_2(1-S)$")
    output = OUT / "supp_control_response.png"
    fig.savefig(output, dpi=360)
    plt.close(fig)


def plot_threshold_sensitivity(scans: dict[str, dict[str, Any]]) -> str:
    """Show that the hidden mosaic population is not a single-threshold artifact."""
    mosaic = scans["mosaic"]["rows"]

    def selected(row: dict[str, Any], s_max: float, c_min: float, g_min: float, q_min: float) -> bool:
        return (
            fnum(row, "nematic_order_mean") <= s_max
            and fnum(row, "orientational_corr_nn_mean") >= c_min
            and fnum(row, "geometry_lock_mean") >= g_min
            and fnum(row, "q_EA_mean") >= q_min
        )

    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.50), constrained_layout=True)

    ax = axes[0]
    q_thresholds = np.linspace(0.45, 0.85, 17)
    colors = plt.cm.cividis(np.linspace(0.18, 0.88, 5))
    for local_min, color in zip((0.65, 0.70, 0.75, 0.80, 0.85), colors):
        counts = [
            sum(selected(row, 0.05, local_min, local_min, float(q_min)) for row in mosaic)
            for q_min in q_thresholds
        ]
        ax.plot(q_thresholds, counts, color=color, label=rf"$C_2,G_2\geq{local_min:.2f}$")
    ax.set_xlabel(r"persistence threshold $q_{\rm EA}^{\min}$")
    ax.set_ylabel("retained mosaic cells")
    ax.set_title(r"threshold survival at $S\leq0.05$")
    ax.set_ylim(0, 870)
    ax.grid(alpha=0.14)
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    panel_label(ax, "a")

    ax = axes[1]
    c_thresholds = np.linspace(0.65, 0.85, 9)
    g_thresholds = np.linspace(0.65, 0.85, 9)
    count_grid = np.zeros((len(g_thresholds), len(c_thresholds)))
    for gi, g_min in enumerate(g_thresholds):
        for ci, c_min in enumerate(c_thresholds):
            count_grid[gi, ci] = sum(selected(row, 0.05, float(c_min), float(g_min), 0.70) for row in mosaic)
    im = ax.imshow(
        count_grid,
        origin="lower",
        aspect="auto",
        extent=[c_thresholds.min(), c_thresholds.max(), g_thresholds.min(), g_thresholds.max()],
        cmap="magma",
        vmin=0,
        vmax=len(mosaic),
        interpolation="nearest",
    )
    ax.scatter([0.80], [0.80], marker="o", s=24, facecolor="none", edgecolor="white", linewidth=0.8)
    ax.set_xlabel(r"local-coherence threshold $C_2^{\min}$")
    ax.set_ylabel(r"registration threshold $G_2^{\min}$")
    ax.set_title(r"cells retained at $q_{\rm EA}\geq0.70$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025, label="mosaic cells")
    panel_label(ax, "b")

    ax = axes[2]
    standard = [row for row in mosaic if selected(row, 0.35, 0.70, 0.70, 0.50)]
    strict = [row for row in mosaic if selected(row, 0.05, 0.80, 0.80, 0.70)]
    j_standard, h_standard = physical_controls(standard)
    j_strict, h_strict = physical_controls(strict)
    ax.scatter(j_standard, h_standard, s=8, color="#BFD3E6", alpha=0.65, edgecolors="none",
               label=f"standard: {len(standard)}")
    ax.scatter(j_strict, h_strict, s=8, color="#7C2D12", alpha=0.75, edgecolors="none",
               label=f"strict core: {len(strict)}")
    ax.set_xlabel(r"interparticle coupling $J/D_r$")
    ax.set_ylabel(r"substrate field $h/D_r$")
    ax.set_title("connected strict core")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    ax.grid(alpha=0.14)
    panel_label(ax, "c")

    output = OUT / "supp_threshold_sensitivity.png"
    fig.savefig(output, dpi=420)
    plt.close(fig)
    return str(output)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    scans = load_scans()
    protocol_path, protocol = load_protocol()
    summary = {
        "protocol_source": str(protocol_path),
        "model_figure": plot_model_application(),
        "regime_densities": plot_regime_densities(scans),
        "static_state_space": plot_static_state_space(scans),
        "registered_state": plot_registered_state(scans),
        "amplifier": plot_amplifier(protocol),
        "release": plot_release(protocol),
        "hidden_memory_dynamics": plot_hidden_memory_dynamics(protocol),
        "hidden_memory_evidence": plot_hidden_memory_evidence(scans, protocol),
        "threshold_sensitivity": plot_threshold_sensitivity(scans),
    }
    plot_control_provenance(scans)
    output = OUT / "groove_evidence_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(output)}, indent=2))


if __name__ == "__main__":
    main()
