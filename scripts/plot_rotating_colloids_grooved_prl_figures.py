#!/usr/bin/env python3
"""Generate PRL-scale figures for grooved rotating-colloid simulations.

The output figures avoid workflow-specific terminology and expose only the
simulation observables used in the manuscript: S, C2, G2, and qEA.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "savefig.bbox": "tight",
    }
)


ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")
OUT = Path("tex/rotating_colloids/grooved_prl_figures")

RUNS = [
    ("uniform grooves", "rotating_colloids_grooved_uniform_scan_n16", "#4c78a8"),
    ("uniform grooves memory", "rotating_colloids_grooved_uniform_memory_zoom_n16", "#72b7b2"),
    ("uniform grooves finite size", "rotating_colloids_grooved_uniform_finite_size", "#54a24b"),
    ("long-range random grooves", "rotating_colloids_grooved_longrange_disorder", "#f58518"),
    ("triangular grooves", "rotating_colloids_grooved_triangular_frustrated_n16", "#e45756"),
    ("mosaic grooves", "rotating_colloids_grooved_mosaic_hidden_search_n32", "#7c2d12"),
]

LABEL_SHORT = {
    "uniform grooves": "uniform",
    "uniform grooves memory": "uniform\nmemory",
    "uniform grooves finite size": "uniform\nfinite size",
    "long-range random grooves": "random\nlong range",
    "triangular grooves": "triangular",
    "mosaic grooves": "mosaic\nhidden",
}


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, folder, color in RUNS:
        path = ROOT / folder / "rotating_colloids_maximal.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("rows", []):
            item = dict(row)
            item["_label"] = label
            item["_color"] = color
            rows.append(item)
    return rows


def fnum(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def j_over_dr(row: dict[str, Any]) -> float:
    diffusion = fnum(row, "noise")
    coupling = fnum(row, "j_align")
    if not math.isfinite(diffusion) or diffusion <= 0.0:
        return math.nan
    return coupling / diffusion


def finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [fnum(row, key) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return np.asarray(vals, dtype=float)


def is_hidden(row: dict[str, Any]) -> bool:
    return (
        fnum(row, "nematic_order_mean") <= 0.35
        and fnum(row, "orientational_corr_nn_mean") >= 0.70
        and fnum(row, "geometry_lock_mean") >= 0.70
        and fnum(row, "q_EA_mean", 0.0) >= 0.50
    )


def hidden_strength(row: dict[str, Any]) -> float:
    vals = [
        fnum(row, "nematic_order_mean"),
        fnum(row, "orientational_corr_nn_mean"),
        fnum(row, "geometry_lock_mean"),
        fnum(row, "q_EA_mean"),
    ]
    if not all(math.isfinite(v) for v in vals):
        return math.nan
    s, c2, g2, qea = vals
    return max(0.0, 1.0 - s) * min(c2, g2, qea)


def hidden_count(rows: list[dict[str, Any]]) -> int:
    return sum(is_hidden(row) for row in rows)


def hidden_score(row: dict[str, Any]) -> float:
    vals = [
        fnum(row, "orientational_corr_nn_mean"),
        fnum(row, "geometry_lock_mean"),
        fnum(row, "q_EA_mean", 0.0),
    ]
    if not all(math.isfinite(v) for v in vals):
        return -math.inf
    return max(0.0, 1.0 - fnum(row, "nematic_order_mean")) * min(vals)


def limiting_gate_category(row: dict[str, Any]) -> int:
    """Return the failed gate that best explains the cell classification.

    Categories:
    0 hidden memory; 1 global director too high; 2 weak local coherence;
    3 weak groove registration; 4 weak memory.
    """
    s_ok = fnum(row, "nematic_order_mean") <= 0.35
    c_ok = fnum(row, "orientational_corr_nn_mean") >= 0.70
    g_ok = fnum(row, "geometry_lock_mean") >= 0.70
    q_ok = fnum(row, "q_EA_mean", -math.inf) >= 0.50
    if s_ok and c_ok and g_ok and q_ok:
        return 0
    if not g_ok:
        return 3
    if not c_ok:
        return 2
    if not q_ok:
        return 4
    return 1


def grid(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = sorted({round(fnum(row, "eps_geom"), 10) for row in rows if math.isfinite(fnum(row, "eps_geom"))})
    js = sorted({round(j_over_dr(row), 10) for row in rows if math.isfinite(j_over_dr(row))})
    z = np.full((len(eps), len(js)), np.nan, dtype=float)
    eps_i = {value: idx for idx, value in enumerate(eps)}
    j_i = {value: idx for idx, value in enumerate(js)}
    for row in rows:
        e = round(fnum(row, "eps_geom"), 10)
        j = round(j_over_dr(row), 10)
        if e in eps_i and j in j_i:
            z[eps_i[e], j_i[j]] = fnum(row, key)
    return np.asarray(eps, dtype=float), np.asarray(js, dtype=float), z


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the monotone-chain convex hull of 2D points."""
    if points.shape[0] <= 2:
        return points
    pts = sorted({(float(x), float(y)) for x, y in points if math.isfinite(x) and math.isfinite(y)})
    if len(pts) <= 2:
        return np.asarray(pts, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def central_envelope_points(x: np.ndarray, y: np.ndarray, keep_fraction: float = 0.90) -> np.ndarray:
    """Select a robust central subset for the state-space envelope."""
    points = np.column_stack([x, y])
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if points.shape[0] <= 12:
        return points
    center = np.nanmedian(points, axis=0)
    scale = np.nanpercentile(np.abs(points - center), 75, axis=0)
    scale = np.where(scale > 1e-6, scale, np.nanstd(points, axis=0) + 1e-6)
    dist = np.sqrt(np.sum(((points - center) / scale) ** 2, axis=1))
    cutoff = float(np.nanquantile(dist, keep_fraction))
    selected = points[dist <= cutoff]
    return selected if selected.shape[0] >= 3 else points


def plot_model_schematic(rows: list[dict[str, Any]]) -> None:
    """Draw a compact physical schematic for the PRL introduction."""
    rng = np.random.default_rng(20260618)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    ri = np.array([0.0, 0.0])
    rj = np.array([1.85, 0.80])
    alpha_i = math.pi / 9.0
    alpha_j = 3.0 * math.pi / 8.0
    theta_i = alpha_i + 0.08
    theta_j = alpha_j - 0.10
    ax.plot([ri[0], rj[0]], [ri[1], rj[1]], color="#94a3b8", linewidth=1.4, zorder=1)
    for center, alpha in ((ri, alpha_i), (rj, alpha_j)):
        normal = np.array([-math.sin(alpha), math.cos(alpha)])
        groove = 0.52 * np.array([math.cos(alpha), math.sin(alpha)])
        for offset in (-0.11, 0.0, 0.11):
            p = center + offset * normal
            ax.plot(
                [p[0] - groove[0], p[0] + groove[0]],
                [p[1] - groove[1], p[1] + groove[1]],
                color="#f97316",
                linewidth=1.05,
                alpha=0.48,
                zorder=1,
            )
    for center, theta, alpha, color, label in (
        (ri, theta_i, alpha_i, "#2563eb", r"$\theta_i$"),
        (rj, theta_j, alpha_j, "#c2410c", r"$\theta_j$"),
    ):
        vec = 0.42 * np.array([math.cos(theta), math.sin(theta)])
        easy = 0.34 * np.array([math.cos(alpha), math.sin(alpha)])
        ax.plot([center[0] - easy[0], center[0] + easy[0]], [center[1] - easy[1], center[1] + easy[1]],
                color="#c2410c", linewidth=1.7, solid_capstyle="round", alpha=0.72, zorder=2)
        ax.plot([center[0] - vec[0], center[0] + vec[0]], [center[1] - vec[1], center[1] + vec[1]],
                color=color, linewidth=5.0, solid_capstyle="round", zorder=3)
        ax.scatter([center[0]], [center[1]], s=28, color="#111827", zorder=4)
        ax.text(center[0] + 0.08, center[1] + 0.26, label, fontsize=9, color=color)
    ax.text(-0.08, 0.46, r"$\alpha_i$", fontsize=9, color="#9a3412")
    ax.text(1.82, 1.20, r"$\alpha_j$", fontsize=9, color="#9a3412")
    ax.text(
        -0.18,
        -0.55,
        r"$-J\cos 2(\theta_i-\theta_j)$" "\n"
        r"$-\epsilon J\cos 2(\theta_i-\alpha_i)$",
        fontsize=8.2,
        ha="left",
        va="top",
    )
    ax.set_xlim(-0.45, 2.35)
    ax.set_ylim(-0.78, 1.40)
    ax.set_title("two route correlators: $C_2$ and $G_2$")
    ax.text(0.015, 0.985, "a", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.set_aspect("equal")
    ax.axis("off")
    blocks = [(0, 0, 0.0), (1, 0, math.pi / 4), (0, 1, math.pi / 2), (1, 1, 3 * math.pi / 4)]
    centers: list[np.ndarray] = []
    block_id: list[int] = []
    for bid, (bx, by, alpha) in enumerate(blocks):
        base = np.array([bx * 1.45, by * 1.25])
        for u in range(3):
            for v in range(3):
                local = np.array([u, v], dtype=float) * 0.34
                c = base + local + rng.normal(scale=0.012, size=2)
                centers.append(c)
                block_id.append(bid)
    centers_a = np.asarray(centers)
    for bid, (bx, by, alpha) in enumerate(blocks):
        idx = [i for i, b in enumerate(block_id) if b == bid]
        pts = centers_a[idx]
        for i in idx:
            for j in idx:
                if j <= i:
                    continue
                d = np.linalg.norm(centers_a[j] - centers_a[i])
                if d < 0.39:
                    ax.plot([centers_a[i, 0], centers_a[j, 0]], [centers_a[i, 1], centers_a[j, 1]],
                            color="#94a3b8", linewidth=0.6, alpha=0.65, zorder=1)
        for c in pts:
            theta = alpha + rng.normal(scale=0.09)
            normal = np.array([-math.sin(alpha), math.cos(alpha)])
            groove = 0.13 * np.array([math.cos(alpha), math.sin(alpha)])
            for offset in (-0.035, 0.035):
                p = c + offset * normal
                ax.plot([p[0] - groove[0], p[0] + groove[0]], [p[1] - groove[1], p[1] + groove[1]],
                        color="#f97316", linewidth=0.55, alpha=0.42, zorder=2)
            vec = 0.13 * np.array([math.cos(theta), math.sin(theta)])
            ax.plot([c[0] - vec[0], c[0] + vec[0]], [c[1] - vec[1], c[1] + vec[1]],
                    color="#7c2d12", linewidth=2.2, solid_capstyle="round", zorder=3)
            ax.scatter([c[0]], [c[1]], s=8, color="#111827", zorder=4)
    cross_edges = [(8, 9), (7, 10), (17, 18), (15, 20)]
    for a, b in cross_edges:
        ax.plot([centers_a[a, 0], centers_a[b, 0]], [centers_a[a, 1], centers_a[b, 1]],
                color="#ea580c", linewidth=0.9, alpha=0.85, zorder=2)
    ax.text(0.02, -0.20, "locally aligned groove domains\nincompatible domains suppress a global director",
            fontsize=7.4, ha="left", va="top")
    ax.set_xlim(-0.18, 2.45)
    ax.set_ylim(-0.35, 2.18)
    ax.set_title("mosaic grooves principle")
    ax.text(0.015, 0.985, "b", transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")

    fig.savefig(OUT / "fig1_model_schematic.png", dpi=360)
    plt.close(fig)


def plot_state_space(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(5.35, 4.15), constrained_layout=True)

    scan_payload: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for label, _folder, color in RUNS:
        sub = [row for row in rows if row["_label"] == label]
        if not sub:
            continue
        points = np.asarray(
            [
                [fnum(row, "nematic_order_mean"), fnum(row, "geometry_lock_mean"), fnum(row, "q_EA_mean")]
                for row in sub
            ],
            dtype=float,
        )
        finite = np.isfinite(points[:, :2]).all(axis=1)
        points = points[finite]
        if points.shape[0] == 0:
            continue
        x = points[:, 0]
        y = points[:, 1]
        q = points[:, 2]
        scan_payload.append((label, color, x, y, q))

    for label, color, x, y, q in scan_payload:
        q_finite = np.isfinite(q)
        alpha = 0.16 if x.size > 200 else 0.42
        ax.scatter(x, y, s=10, color=color, alpha=alpha, edgecolors="none", rasterized=True, zorder=2.1)
        cx = float(np.mean(x))
        cy = float(np.mean(y))
        q_text = ""
        if np.any(q_finite):
            q_text = rf", $q_{{\rm EA}}={float(np.nanmean(q)):.2f}$"
        ax.scatter([cx], [cy], s=70, color=color, edgecolor="#111827", linewidth=0.55, zorder=4)
        if label in {"mosaic grooves", "long-range random grooves", "triangular grooves", "uniform grooves finite size"}:
            label_text = {
                "mosaic grooves": "mosaic",
                "long-range random grooves": "long-range",
                "triangular grooves": "triangular",
                "uniform grooves finite size": "square finite",
            }[label]
            dx, dy, ha = {
                "mosaic grooves": (0.018, -0.030, "left"),
                "long-range random grooves": (-0.17, 0.020, "left"),
                "triangular grooves": (-0.10, 0.030, "left"),
                "uniform grooves finite size": (0.018, 0.016, "left"),
            }[label]
            ax.text(
                cx + dx,
                cy + dy,
                f"{label_text}{q_text}",
                fontsize=6.3,
                ha=ha,
                va="top" if label == "mosaic grooves" else "bottom",
                color="#111827",
            )

    ax.text(
        0.030,
        0.900,
        "mosaic hidden-memory sector",
        ha="left",
        va="center",
        fontsize=7.0,
        color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        zorder=6,
    )
    ax.text(
        0.575,
        0.395,
        "long-range control:\npersistent but weakly registered",
        fontsize=6.4,
        ha="left",
        va="center",
        color="#7c2d12",
        bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
        zorder=6,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 0.98)
    ax.set_xlabel(r"global nematic order $S$")
    ax.set_ylabel(r"groove alignment $G_2$")
    ax.set_title("Simulation state space")
    ax.grid(alpha=0.16)
    fig.savefig(OUT / "fig2_state_space.png", dpi=360)
    plt.close(fig)


def plot_mosaic_heatmaps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mosaic = [row for row in rows if row["_label"] == "mosaic grooves"]
    if not mosaic:
        raise RuntimeError("Mosaic data not found")
    hidden = [row for row in mosaic if is_hidden(row)]
    best = max(hidden, key=hidden_strength) if hidden else max(mosaic, key=hidden_score)

    eps, js, S = grid(mosaic, "nematic_order_mean")
    _eps, _js, C2 = grid(mosaic, "orientational_corr_nn_mean")
    _eps, _js, G2 = grid(mosaic, "geometry_lock_mean")
    _eps, _js, Q = grid(mosaic, "q_EA_mean")
    H = np.maximum(0.0, 1.0 - S) * np.minimum.reduce([C2, G2, Q])
    hidden_mask = ((S <= 0.35) & (C2 >= 0.70) & (G2 >= 0.70) & (Q >= 0.50)).astype(float)
    dG_de, dG_dj = np.gradient(np.nan_to_num(G2, nan=np.nanmean(G2)), eps, js)
    dQ_de, dQ_dj = np.gradient(np.nan_to_num(Q, nan=np.nanmean(Q)), eps, js)
    response = np.hypot(dG_de, dG_dj) + np.hypot(dQ_de, dQ_dj)

    fig, axes = plt.subplots(1, 3, figsize=(7.45, 2.75), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        extent=[float(js.min()), float(js.max()), float(eps.min()), float(eps.max())],
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=0.90,
    )
    ax.contour(js, eps, hidden_mask, levels=[0.5], colors="white", linewidths=1.0)
    ax.scatter([j_over_dr(best)], [fnum(best, "eps_geom")], marker="*", s=70, color="#fde68a",
               edgecolor="#111827", linewidth=0.45, zorder=5)
    ax.set_title(r"hidden score $H$")
    ax.set_xlabel(r"$J/D_r$")
    ax.set_ylabel(r"groove ratio $\epsilon=h/J$")
    fig.colorbar(im, ax=ax, shrink=0.80, pad=0.014, label=r"$H=(1-S)\min(C_2,G_2,q_{\rm EA})$")

    ax = axes[1]
    im = ax.imshow(
        response,
        origin="lower",
        aspect="auto",
        extent=[float(js.min()), float(js.max()), float(eps.min()), float(eps.max())],
        interpolation="nearest",
        cmap="viridis",
    )
    ax.contour(js, eps, hidden_mask, levels=[0.5], colors="white", linewidths=1.0)
    ax.scatter([j_over_dr(best)], [fnum(best, "eps_geom")], marker="*", s=70, color="#fde68a",
               edgecolor="#111827", linewidth=0.45, zorder=5)
    ax.set_title(r"entry response")
    ax.set_xlabel(r"$J/D_r$")
    ax.set_ylabel(r"groove ratio $\epsilon=h/J$")
    fig.colorbar(im, ax=ax, shrink=0.80, pad=0.014, label=r"$|\nabla G_2|+|\nabla q_{\rm EA}|$")

    ax = axes[2]
    eps_cut = float(eps.min())
    cut = [row for row in mosaic if abs(fnum(row, "eps_geom") - eps_cut) < 1e-8]
    cut.sort(key=j_over_dr)
    x = [j_over_dr(row) for row in cut]
    for key, label, color in [
        ("nematic_order_mean", r"$S$", "#4c78a8"),
        ("orientational_corr_nn_mean", r"$C_2$", "#2ca02c"),
        ("geometry_lock_mean", r"$G_2$", "#d62728"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#9467bd"),
    ]:
        ax.plot(x, [fnum(row, key) for row in cut], marker="o", markersize=2.5, linewidth=1.15, color=color, label=label)
    ax.axhline(0.70, color="#64748b", linestyle=":", linewidth=0.8)
    ax.axhline(0.50, color="#64748b", linestyle="--", linewidth=0.8)
    ax.axhline(0.35, color="#64748b", linestyle="-.", linewidth=0.8)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(r"low-$\epsilon$ entry cut")
    ax.set_xlabel(r"$J/D_r$ at $\epsilon=h/J=0.75$")
    ax.set_ylabel("observable value")
    ax.grid(alpha=0.14)
    ax.legend(frameon=False, ncols=2, loc="lower right", fontsize=6.5)

    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.savefig(OUT / "fig3_mosaic_memory_response.png", dpi=360)
    plt.close(fig)

    return {
        "mosaic_rows": len(mosaic),
        "mosaic_hidden_rows": len(hidden),
        "best": {
            "eps_geom": fnum(best, "eps_geom"),
            "j_align": fnum(best, "j_align"),
            "J_over_Dr": j_over_dr(best),
            "S": fnum(best, "nematic_order_mean"),
            "C2": fnum(best, "orientational_corr_nn_mean"),
            "G2": fnum(best, "geometry_lock_mean"),
            "qEA": fnum(best, "q_EA_mean"),
            "H": hidden_strength(best),
        },
        "hidden_mean": {
            "S": float(np.mean(finite_values(hidden, "nematic_order_mean"))),
            "C2": float(np.mean(finite_values(hidden, "orientational_corr_nn_mean"))),
            "G2": float(np.mean(finite_values(hidden, "geometry_lock_mean"))),
            "qEA": float(np.mean(finite_values(hidden, "q_EA_mean"))),
        },
        "hidden_std": {
            "S": float(np.std(finite_values(hidden, "nematic_order_mean"))),
            "C2": float(np.std(finite_values(hidden, "orientational_corr_nn_mean"))),
            "G2": float(np.std(finite_values(hidden, "geometry_lock_mean"))),
            "qEA": float(np.std(finite_values(hidden, "q_EA_mean"))),
        },
    }


def run_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, _folder, _color in RUNS:
        sub = [row for row in rows if row["_label"] == label]
        if not sub:
            continue
        q = finite_values(sub, "q_EA_mean")
        out.append(
            {
                "label": label,
                "rows": len(sub),
                "hidden_rows": hidden_count(sub),
                "mean_S": float(np.mean(finite_values(sub, "nematic_order_mean"))),
                "min_S": float(np.min(finite_values(sub, "nematic_order_mean"))),
                "mean_C2": float(np.mean(finite_values(sub, "orientational_corr_nn_mean"))),
                "mean_G2": float(np.mean(finite_values(sub, "geometry_lock_mean"))),
                "max_G2": float(np.max(finite_values(sub, "geometry_lock_mean"))),
                "mean_qEA": float(np.mean(q)) if q.size else None,
            }
        )
    return out


def control_map_rows(rows: list[dict[str, Any]], label: str, n_value: int | None = None) -> list[dict[str, Any]]:
    sub = [row for row in rows if row["_label"] == label]
    if n_value is not None:
        sub = [row for row in sub if int(fnum(row, "n", -1)) == n_value]
    return sub


def plot_control_maps(rows: list[dict[str, Any]]) -> None:
    panels = [
        ("uniform grooves memory", 16, "uniform grooves"),
        ("long-range random grooves", 48, "random grooves"),
        ("triangular grooves", 16, "triangular grooves"),
        ("mosaic grooves", 32, "mosaic grooves"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.25), constrained_layout=True)
    for ax, (label, n_value, title) in zip(axes.ravel(), panels):
        sub = control_map_rows(rows, label, n_value)
        if not sub:
            ax.axis("off")
            continue
        eps = sorted({round(fnum(row, "eps_geom"), 10) for row in sub if math.isfinite(fnum(row, "eps_geom"))})
        js = sorted({round(j_over_dr(row), 10) for row in sub if math.isfinite(j_over_dr(row))})
        z = np.full((len(eps), len(js)), np.nan)
        hidden = np.zeros((len(eps), len(js)))
        eps_i = {v: i for i, v in enumerate(eps)}
        j_i = {v: i for i, v in enumerate(js)}
        for row in sub:
            e = round(fnum(row, "eps_geom"), 10)
            j = round(j_over_dr(row), 10)
            if e in eps_i and j in j_i:
                z[eps_i[e], j_i[j]] = hidden_strength(row)
                hidden[eps_i[e], j_i[j]] = 1.0 if is_hidden(row) else 0.0
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            extent=[float(min(js)), float(max(js)), float(min(eps)), float(max(eps))],
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=0.78,
        )
        if np.nanmax(hidden) > 0.5:
            ax.contour(np.asarray(js), np.asarray(eps), hidden, levels=[0.5], colors="white", linewidths=1.0)
        count = sum(is_hidden(row) for row in sub)
        ax.text(
            0.03,
            0.95,
            f"{count}/{len(sub)} hidden cells",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="white",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": "#111827", "alpha": 0.65, "edgecolor": "none"},
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$J/D_r$", fontsize=8)
        ax.set_ylabel(r"$\epsilon=h/J$", fontsize=8)
        ax.tick_params(labelsize=6.5)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.012, label=r"$H=(1-S)\min(C_2,G_2,q_{\rm EA})$")
    fig.suptitle("2D control maps: the hidden-glass score requires the mosaic grooves", fontsize=10)
    fig.savefig(OUT / "fig4_control_maps.png", dpi=360)
    plt.close(fig)


def plot_regime_comparison(rows: list[dict[str, Any]]) -> None:
    """Compare the four groove geometries by the gate that limits each cell."""
    panels = [
        ("uniform grooves", 16, "uniform grooves"),
        ("long-range random grooves", 48, "random / long-range grooves"),
        ("triangular grooves", 16, "triangular / frustrated grooves"),
        ("mosaic grooves", 32, "mosaic grooves"),
    ]
    category_names = {
        0: "hidden memory",
        1: "global director",
        2: "weak local order",
        3: "weak groove registration",
        4: "weak memory",
    }
    category_colors = {
        0: "#CC6F47",
        1: "#5477C4",
        2: "#C5CAD3",
        3: "#71B436",
        4: "#BD569B",
    }
    cmap = ListedColormap([category_colors[i] for i in range(5)])
    cmap.set_bad("#FCFCFD")
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(7.35, 5.95), constrained_layout=True)
    for ax, (label, n_value, title) in zip(axes.ravel(), panels):
        sub = control_map_rows(rows, label, n_value)
        if not sub:
            ax.axis("off")
            continue

        eps = sorted({round(fnum(row, "eps_geom"), 10) for row in sub if math.isfinite(fnum(row, "eps_geom"))})
        js = sorted({round(j_over_dr(row), 10) for row in sub if math.isfinite(j_over_dr(row))})
        z = np.full((len(eps), len(js)), np.nan)
        eps_i = {value: idx for idx, value in enumerate(eps)}
        j_i = {value: idx for idx, value in enumerate(js)}
        for row in sub:
            e = round(fnum(row, "eps_geom"), 10)
            j = round(j_over_dr(row), 10)
            if e in eps_i and j in j_i:
                z[eps_i[e], j_i[j]] = limiting_gate_category(row)

        ax.imshow(
            np.ma.masked_invalid(z),
            origin="lower",
            aspect="auto",
            extent=[float(min(js)), float(max(js)), float(min(eps)), float(max(eps))],
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        counts = {idx: sum(limiting_gate_category(row) == idx for row in sub) for idx in range(5)}
        hidden_count_local = counts[0]
        fail_counts = {idx: count for idx, count in counts.items() if idx != 0}
        dominant_fail = max(fail_counts, key=fail_counts.get)
        mean_s = float(np.mean(finite_values(sub, "nematic_order_mean")))
        mean_g = float(np.mean(finite_values(sub, "geometry_lock_mean")))
        mean_q = float(np.mean(finite_values(sub, "q_EA_mean")))
        ax.text(
            0.035,
            0.955,
            f"{hidden_count_local}/{len(sub)} hidden\n"
            f"main limit: {category_names[dominant_fail]}\n"
            rf"$\bar S={mean_s:.2f}$, $\bar G_2={mean_g:.2f}$, $\bar q_{{EA}}={mean_q:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            color="white",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#111827", "alpha": 0.70, "edgecolor": "none"},
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(r"$J/D_r$", fontsize=8)
        ax.set_ylabel(r"groove ratio $\epsilon=h/J$", fontsize=8)
        ax.tick_params(labelsize=6.6)
        ax.grid(False)
        ax.set_box_aspect(0.78)

    legend_handles = [
        Patch(facecolor=category_colors[idx], edgecolor="#111827", linewidth=0.45, label=category_names[idx])
        for idx in range(5)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.50, 1.035),
        fontsize=7.0,
        handlelength=1.2,
        columnspacing=1.0,
    )
    fig.savefig(OUT / "fig5_regime_comparison.png", dpi=360)
    plt.close(fig)


def plot_hidden_phase_properties(rows: list[dict[str, Any]]) -> None:
    """Summarize the internal properties of the mosaic hidden phase."""
    mosaic = [row for row in rows if row["_label"] == "mosaic grooves"]
    if not mosaic:
        raise RuntimeError("Mosaic data not found")
    best = max(mosaic, key=hidden_score)
    eps_star = fnum(best, "eps_geom")
    j_star_stored = fnum(best, "j_align")
    j_star = j_over_dr(best)

    metrics = [
        ("nematic_order_mean", r"$S$", "#1f77b4"),
        ("orientational_corr_nn_mean", r"$C_2$", "#2ca02c"),
        ("geometry_lock_mean", r"$G_2$", "#d62728"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#9467bd"),
    ]

    fig, (ax_e, ax_rad, ax_cloud) = plt.subplots(
        1,
        3,
        figsize=(7.35, 3.15),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )

    eps_cut = [row for row in mosaic if abs(fnum(row, "j_align") - j_star_stored) < 1e-6]
    eps_cut.sort(key=lambda row: fnum(row, "eps_geom"))
    eps_x = [fnum(row, "eps_geom") for row in eps_cut]

    # Single-panel broken y-axis: keep panel size fixed while separating the
    # low-S band from the high C2/G2/qEA band.
    low_data = (0.00, 0.08)
    high_data = (0.80, 0.96)
    low_plot = (0.00, 0.42)
    high_plot = (0.58, 1.00)

    def map_broken_y(values: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        out = np.full(arr.shape, np.nan, dtype=float)
        low = arr <= low_data[1]
        high = arr >= high_data[0]
        out[low] = low_plot[0] + (arr[low] - low_data[0]) / (low_data[1] - low_data[0]) * (low_plot[1] - low_plot[0])
        out[high] = high_plot[0] + (arr[high] - high_data[0]) / (high_data[1] - high_data[0]) * (high_plot[1] - high_plot[0])
        return out

    for key, label, color in metrics:
        values = [fnum(row, key) for row in eps_cut]
        ax_e.plot(
            eps_x,
            map_broken_y(values),
            marker="o",
            markersize=2.3,
            linewidth=1.15,
            color=color,
            label=label,
        )
    ax_e.axhline(float(map_broken_y([0.35])[0]), color="#64748b", linestyle="--", linewidth=0.8)
    ax_e.axhspan(low_plot[1], high_plot[0], color="white", zorder=2)
    ax_e.plot([-0.018, 0.018], [0.47, 0.53], transform=ax_e.transAxes,
              color="#334155", lw=0.9, clip_on=False)
    ax_e.plot([0.982, 1.018], [0.47, 0.53], transform=ax_e.transAxes,
              color="#334155", lw=0.9, clip_on=False)
    tick_values = [0.00, 0.02, 0.04, 0.06, 0.08, 0.80, 0.88, 0.96]
    ax_e.set_yticks(map_broken_y(tick_values))
    ax_e.set_yticklabels([f"{value:.2f}" for value in tick_values])
    ax_e.set_ylim(-0.02, 1.03)
    ax_e.set_xlabel(r"groove ratio $\epsilon=h/J$")
    ax_e.set_ylabel("observable value")
    ax_e.grid(alpha=0.15)
    ax_e.set_title(rf"robustness at $J/D_r={j_star:.2f}$")
    ax_e.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        ncols=2,
        loc="center right",
        fontsize=6.3,
    )

    r = np.asarray(best.get("radial_r", []), dtype=float)
    c2 = np.asarray([
        math.nan if value is None else float(value)
        for value in best.get("radial_C2_mean", [])
    ], dtype=float)
    valid = np.isfinite(r) & np.isfinite(c2)
    ax_rad.plot(r[valid], c2[valid], marker="o", markersize=3.0, linewidth=1.25,
                color="#7c2d12", label=r"mosaic $C_2(r)$")
    fit_mask = valid & (c2 > 0.0)
    if np.count_nonzero(fit_mask) >= 5:
        r_fit_data = r[fit_mask]
        y_fit_data = c2[fit_mask]
        best_fit: tuple[float, float, float, float] | None = None
        for beta in np.linspace(0.7, 2.8, 106):
            for xi in np.linspace(2.0, 8.5, 261):
                basis = np.exp(-((r_fit_data / xi) ** beta))
                denom = float(np.dot(basis, basis))
                if denom <= 0.0:
                    continue
                amp = float(np.dot(y_fit_data, basis) / denom)
                pred = amp * basis
                sse = float(np.sum((y_fit_data - pred) ** 2))
                if best_fit is None or sse < best_fit[0]:
                    best_fit = (sse, amp, xi, beta)
        if best_fit is not None:
            _sse, amp, xi, beta = best_fit
            r_line = np.linspace(float(np.min(r_fit_data)), float(np.max(r_fit_data)), 160)
            y_line = amp * np.exp(-((r_line / xi) ** beta))
            ax_rad.plot(
                r_line,
                y_line,
                color="#111827",
                linestyle="--",
                linewidth=1.0,
                label=rf"stretched decay, $\beta\simeq{beta:.1f}$",
            )
    ax_rad.axhline(0.0, color="#94a3b8", linewidth=0.75)
    ax_rad.set_xlabel(r"separation $r/a$")
    ax_rad.set_ylabel(r"all-pair $C_2(r)$")
    ax_rad.set_ylim(-0.04, 0.9)
    ax_rad.grid(alpha=0.15)
    ax_rad.set_title("finite spatial range")
    ax_rad.legend(frameon=False, fontsize=6.6)

    s_vals = np.asarray([fnum(row, "nematic_order_mean") for row in mosaic])
    q_vals = np.asarray([fnum(row, "q_EA_mean") for row in mosaic])
    g_vals = np.asarray([fnum(row, "geometry_lock_mean") for row in mosaic])
    sc = ax_cloud.scatter(s_vals, q_vals, c=g_vals, cmap="viridis", s=15, alpha=0.88,
                          edgecolors="none", rasterized=True, vmin=0.68, vmax=0.96)
    ax_cloud.scatter([fnum(best, "nematic_order_mean")], [fnum(best, "q_EA_mean")],
                     marker="*", s=80, color="#fde68a", edgecolor="#111827", linewidth=0.45, zorder=5)
    ax_cloud.axvline(0.35, color="#9a3412", linestyle="--", linewidth=0.85)
    ax_cloud.axhline(0.50, color="#9a3412", linestyle="--", linewidth=0.85)
    ax_cloud.set_xlabel(r"global nematic order $S$")
    ax_cloud.set_ylabel(r"memory $q_{\rm EA}$")
    ax_cloud.set_xlim(0.00, 0.05)
    ax_cloud.set_ylim(0.45, 0.92)
    ax_cloud.grid(alpha=0.15)
    ax_cloud.set_title(r"low $S$, high memory")
    fig.colorbar(sc, ax=ax_cloud, shrink=0.64, fraction=0.040, pad=0.012, label=r"$G_2$")

    for label, ax in zip(("a", "b", "c"), (ax_e, ax_rad, ax_cloud)):
        ax.set_box_aspect(1)
        ax.text(
            -0.10,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="bottom",
        )

    fig.savefig(OUT / "fig4_hidden_phase_properties.png", dpi=360)
    plt.close(fig)

    j_cut = [row for row in mosaic if abs(fnum(row, "eps_geom") - eps_star) < 1e-6]
    j_cut.sort(key=j_over_dr)
    j_x = [j_over_dr(row) for row in j_cut]
    fig_s, ax_s = plt.subplots(figsize=(3.6, 2.55), constrained_layout=True)
    for key, label, color in metrics:
        ax_s.plot(
            j_x,
            [fnum(row, key) for row in j_cut],
            marker="o",
            markersize=2.4,
            linewidth=1.15,
            color=color,
            label=label,
        )
    ax_s.axhline(0.70, color="#64748b", linestyle=":", linewidth=0.8)
    ax_s.axhline(0.35, color="#64748b", linestyle="--", linewidth=0.8)
    ax_s.set_ylim(0.05, 0.98)
    ax_s.set_xlabel(r"alignment strength $J/D_r$")
    ax_s.set_ylabel("observable value")
    ax_s.set_title(rf"fixed-$\epsilon$ cut at $\epsilon=h/J={eps_star:.2f}$")
    ax_s.grid(alpha=0.15)
    ax_s.legend(frameon=False, ncols=2, loc="lower right", fontsize=6.6)
    fig_s.savefig(OUT / "rotating_colloids_coupling_cut_supp.png", dpi=360)
    plt.close(fig_s)


def plot_supplement_controls(rows: list[dict[str, Any]]) -> None:
    panels = [
        ("uniform grooves memory", "rotating_colloids_square_memory_supp.png", (1.05, 1.25), (0.75, 1.05)),
        ("long-range random grooves", "rotating_colloids_longrange_supp.png", (0.8, 1.6), (0.45, 1.15)),
        ("triangular grooves", "rotating_colloids_triangular_supp.png", (0.5, 1.8), (0.3, 1.5)),
    ]
    keys = [
        ("nematic_order_mean", r"$S$", 0.0, 1.0, "viridis"),
        ("geometry_lock_mean", r"$G_2$", 0.0, 1.0, "viridis"),
        ("q_EA_mean", r"$q_{\rm EA}$", 0.0, 1.0, "magma"),
    ]
    for label, filename, _eps_range, _j_range in panels:
        sub = [row for row in rows if row["_label"] == label]
        if not sub:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
        for ax, (key, title, vmin, vmax, cmap_name) in zip(axes, keys):
            eps, js, z = grid(sub, key)
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("#f8fafc")
            im = ax.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[float(js.min()), float(js.max()), float(eps.min()), float(eps.max())],
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
            )
            ax.set_title(title, fontsize=8.5)
            ax.set_xlabel(r"$J/D_r$", fontsize=7.5)
            ax.set_ylabel(r"$\epsilon=h/J$", fontsize=7.5)
            ax.tick_params(labelsize=6.5)
            fig.colorbar(im, ax=ax, shrink=0.78, pad=0.015)
        fig.suptitle(label, fontsize=10)
        fig.savefig(OUT / filename, dpi=340)
        plt.close(fig)


def plot_finite_size(rows: list[dict[str, Any]]) -> None:
    sub = [row for row in rows if row["_label"] == "uniform grooves finite size"]
    if not sub:
        return
    ns = sorted({int(fnum(row, "n")) for row in sub})
    metrics = [
        ("nematic_order_mean", r"$S$", "#f58518"),
        ("orientational_corr_nn_mean", r"$C_2$", "#4c78a8"),
        ("geometry_lock_mean", r"$G_2$", "#7c2d12"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#54a24b"),
    ]
    fig, ax = plt.subplots(figsize=(4.3, 2.8), constrained_layout=True)
    for key, label, color in metrics:
        means = []
        stds = []
        for n in ns:
            vals = finite_values([row for row in sub if int(fnum(row, "n")) == n], key)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        ax.errorbar(ns, means, yerr=stds, marker="o", linewidth=1.25, color=color, label=label)
    ax.set_xlabel("linear system size n")
    ax.set_ylabel("observable")
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.16)
    ax.legend(fontsize=7, ncols=2, frameon=False)
    ax.set_title("finite-size uniform-groove memory scan", fontsize=9)
    fig.savefig(OUT / "rotating_colloids_finite_size_supp.png", dpi=340)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    if not rows:
        raise SystemExit("No rotating-colloid simulation rows found")
    plot_model_schematic(rows)
    plot_state_space(rows)
    mosaic_summary = plot_mosaic_heatmaps(rows)
    plot_control_maps(rows)
    plot_regime_comparison(rows)
    plot_hidden_phase_properties(rows)
    plot_supplement_controls(rows)
    plot_finite_size(rows)
    summary = {
        "total_rows": len(rows),
        "runs": run_summary(rows),
        "mosaic": mosaic_summary,
        "figures": [
            "fig1_model_schematic.png",
            "fig2_state_space.png",
            "fig3_mosaic_memory_response.png",
            "fig4_hidden_phase_properties.png",
            "fig4_control_maps.png",
            "fig5_regime_comparison.png",
            "rotating_colloids_square_memory_supp.png",
            "rotating_colloids_longrange_supp.png",
            "rotating_colloids_triangular_supp.png",
            "rotating_colloids_finite_size_supp.png",
        ],
    }
    (OUT / "rotating_colloids_prl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
