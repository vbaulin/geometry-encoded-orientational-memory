#!/usr/bin/env python3
"""Build PRL figures for the caged capillary-pair colloid model."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Ellipse

from rotating_colloids_capillary_pair import make_caged_graph, resolve_device, simulate_ensemble


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "discoveries/theory_experiment_interface/rotating_colloids_hyperion"
TEX = ROOT / "tex/rotating_colloids"
OUT = TEX / "capillary_prl_figures"

PILOT = DATA / "rotating_colloids_capillary_pair_prl_pilot/capillary_pair_scan.jsonl"
DENSE = DATA / "rotating_colloids_capillary_pair_prl_dense_scan/capillary_pair_scan.jsonl"
CONTROLS = DATA / "rotating_colloids_capillary_pair_prl_controls/capillary_pair_scan.jsonl"
NO_CAPILLARY = DATA / "rotating_colloids_capillary_pair_prl_no_capillary/capillary_pair_scan.jsonl"
DYNAMICS = DATA / "rotating_colloids_capillary_pair_prl_dynamics/capillary_pair_protocols.json"
GPU = DATA / "rotating_colloids_capillary_pair_prl_gpu"
INTERNAL = (
    DATA
    / "rotating_colloids_capillary_pair_prl_internal/capillary_internal_correlations.json"
)
SIZE_PATHS = {
    n: GPU / f"finite_size_n{n}/capillary_pair_scan.jsonl"
    for n in (12, 16, 24, 32, 48)
}
CONTROLS = GPU / "matched_controls_n32/capillary_pair_scan.jsonl"
DENSE = GPU / "dense_map_n20/capillary_pair_scan.jsonl"
DYNAMICS_PATHS = [GPU / f"dynamics_seed_{seed}/capillary_pair_protocols.json" for seed in (17, 29, 43)]


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
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = 0.02, y: float = 0.98) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9.5,
        va="top",
        ha="left",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )


def aggregate(rows: Sequence[Dict[str, object]], keys: Sequence[str]) -> Dict[Tuple[float, ...], Dict[str, float]]:
    groups: Dict[Tuple[float, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(float(row[key]) for key in keys)].append(row)
    result: Dict[Tuple[float, ...], Dict[str, float]] = {}
    for key, group in groups.items():
        summary: Dict[str, float] = {}
        for metric in ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation"):
            values = np.asarray([float(row[metric]) for row in group])
            summary[metric] = float(values.mean())
            summary[metric + "_std"] = float(values.std())
        values = np.asarray([float(row["replica_overlap"]["magnitude_mean"]) for row in group])
        summary["replica_overlap_magnitude"] = float(values.mean())
        summary["replica_overlap_magnitude_std"] = float(values.std())
        result[key] = summary
    return result


def draw_capillary_schematic(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    centers = np.asarray([[-1.25, 0.0], [1.25, 0.0]])
    theta = (0.60, 2.15)
    colors = ("#1769aa", "#ca4f3d")
    for (x, y), angle, color in zip(centers, theta, colors):
        for radius, alpha in ((0.78, 0.12), (1.04, 0.07)):
            for sign, lobe_color in ((1.0, "#e7735b"), (-1.0, "#4c8cca")):
                phase = angle + (0.0 if sign > 0 else math.pi / 2.0)
                dx = radius * math.cos(phase)
                dy = radius * math.sin(phase)
                ax.add_patch(Ellipse((x + dx, y + dy), 0.70, 0.30, angle=math.degrees(phase), color=lobe_color, alpha=alpha, lw=0))
                ax.add_patch(Ellipse((x - dx, y - dy), 0.70, 0.30, angle=math.degrees(phase), color=lobe_color, alpha=alpha, lw=0))
        ax.add_patch(Ellipse((x, y), 1.15, 0.36, angle=math.degrees(angle), facecolor=color, edgecolor="black", lw=0.8, zorder=4))
        ax.plot([x, x + 0.72 * math.cos(angle)], [y, y + 0.72 * math.sin(angle)], color="black", lw=0.8, zorder=5)
    ax.plot(centers[:, 0], centers[:, 1], color="0.25", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.0, -0.18, r"$\phi_{ij}$", ha="center", va="top")
    ax.annotate("", xy=(0.42, -0.48), xytext=(-0.42, -0.48), arrowprops={"arrowstyle": "<->", "color": "#5b6f7a", "lw": 1.0})
    ax.text(0.0, -0.61, "repulsion maintains separation", ha="center", va="top", fontsize=7.1, color="#40545e")
    ax.add_patch(Arc(tuple(centers[0]), 0.85, 0.85, theta1=0, theta2=math.degrees(theta[0]), lw=0.8))
    ax.text(-0.82, 0.23, r"$\theta_i$", ha="center")
    ax.text(0.0, 1.20, r"residual capillary torque: $-g(R/r_{ij})^4\cos2(\theta_i+\theta_j-2\phi_{ij})$", ha="center", fontsize=7.4)
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-1.05, 1.55)
    ax.axis("off")


def representative_state() -> Tuple[object, np.ndarray]:
    graph = make_caged_graph(
        12,
        disorder=0.16,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    run = simulate_ensemble(
        graph,
        j_align=4.0,
        g_capillary=5.0,
        replicas=1,
        burn_in_steps=5000,
        sample_steps=100,
        sample_stride=100,
        dt=0.0025,
        seed=20260712,
        device=resolve_device("cpu"),
    )
    return graph, np.asarray(run["final_theta"])[0]


def draw_state(ax: plt.Axes, graph, theta: np.ndarray) -> None:
    pos = graph.positions
    box = graph.box
    delta = pos[graph.tgt] - pos[graph.src]
    delta = delta - box * np.round(delta / box)
    length = 0.62
    for (x, y), angle in zip(pos, theta):
        ax.add_patch(plt.Circle((x, y), 0.42, facecolor="#8ecae6", edgecolor="none", alpha=0.08, zorder=1))
        dx = 0.5 * length * math.cos(float(angle))
        dy = 0.5 * length * math.sin(float(angle))
        color = mpl.colormaps["twilight"]((float(angle) % math.pi) / math.pi)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.8, solid_capstyle="round", zorder=3)
    ax.set(xlim=(0, box[0]), ylim=(0, box[1]), xlabel=r"$x/a$", ylabel=r"$y/a$")
    ax.set_aspect("equal")
    ax.set_title(r"repulsively caged interfacial monolayer", pad=3)


def figure1() -> None:
    graph, theta = representative_state()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.30),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )
    draw_capillary_schematic(axes[0])
    panel_label(axes[0], "a", x=0.01, y=0.12)
    draw_state(axes[1], graph, theta)
    axes[1].set_box_aspect(1.0)
    panel_label(axes[1], "b")

    ax = axes[2]
    rr = np.linspace(2.2, 6.0, 250)
    gamma = 0.01
    kbt = 4.11e-21
    for du_nm, color in zip((2.0, 4.0, 6.0), ("#277da1", "#f8961e", "#c0392b")):
        energy = 3.0 * math.pi * gamma * (du_nm * 1e-9) ** 2 * rr ** -4 / kbt
        ax.plot(rr, energy, lw=1.6, color=color, label=rf"$\Delta u={du_nm:.0f}$ nm")
    ax.axhspan(1.0, 5.0, color="#6a994e", alpha=0.12, lw=0)
    ax.axhline(1.0, color="0.45", lw=0.7, ls="--")
    ax.set(xlabel=r"separation $r/R$", ylabel=r"$A_4(r)/k_{\rm B}T$", yscale="log", ylim=(0.025, 40))
    ax.set_box_aspect(1.0)
    ax.legend(
        frameon=False,
        loc="lower left",
        fontsize=6.4,
        handlelength=1.45,
        handletextpad=0.55,
        labelspacing=0.25,
        borderaxespad=0.45,
    )
    ax.set_title("Brownian coupling window")
    ax.text(5.88, 8.0, "strong aggregation", ha="right", va="bottom", fontsize=6.8, color="#9f2f24")
    ax.text(5.88, 3.4, "target coupling", ha="right", va="center", fontsize=6.8, color="#496b34")
    ax.text(5.88, 0.72, "thermal rotation", ha="right", va="top", fontsize=6.8, color="0.35")
    panel_label(ax, "c")
    save(fig, "fig1_capillary_realization")


def _grid(rows: Sequence[Dict[str, object]], metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    js = np.asarray(sorted({float(row["j_align"]) for row in rows}))
    gs = np.asarray(sorted({float(row["g_capillary"]) for row in rows}))
    grid = np.full((gs.size, js.size), np.nan)
    for row in rows:
        i = int(np.where(gs == float(row["g_capillary"]))[0][0])
        j = int(np.where(js == float(row["j_align"]))[0][0])
        grid[i, j] = float(row[metric])
    return js, gs, grid


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    midpoint = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(
        ([centers[0] - (midpoint[0] - centers[0])], midpoint, [centers[-1] + (centers[-1] - midpoint[-1])])
    )


def hidden_fields(rows: Sequence[Dict[str, object]]):
    js, gs, s = _grid(rows, "S_mean")
    _, _, c2 = _grid(rows, "C2_mean")
    _, _, g2 = _grid(rows, "G2_mean")
    _, _, qea = _grid(rows, "q_EA_mean")
    score = (1.0 - s) * np.minimum.reduce(
        [np.maximum(c2, 0.0), np.maximum(g2, 0.0), np.maximum(qea, 0.0)]
    )
    response_sq = np.zeros_like(score)
    for field in (s, c2, g2, qea):
        derivative_g, derivative_j = np.gradient(field, gs, js, edge_order=1)
        response_sq += derivative_g**2 + derivative_j**2
    response = np.sqrt(response_sq)
    return js, gs, s, c2, g2, qea, score, response


def figure2() -> None:
    rows = read_jsonl(DENSE)
    js, gs, s, c2, g2, qea, score, response = hidden_fields(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.22), constrained_layout=True)

    ax = axes[0]
    levels = np.linspace(0.0, max(0.45, float(np.nanmax(score))), 19)
    im = ax.contourf(js, gs, score, levels=levels, cmap="viridis", extend="max")
    boundary = ax.contour(js, gs, score, levels=[0.30], colors="white", linewidths=1.1)
    ax.clabel(boundary, fmt={0.30: r"$\mathcal{H}=0.30$"}, fontsize=6.8, inline=True)
    ax.plot(4.0, 5.0, marker="*", ms=8.5, mec="black", mew=0.65, color="white")
    ax.set(
        xlabel=r"alignment $J/k_{\rm B}T$",
        ylabel=r"capillary coupling $g/k_{\rm B}T$",
        title="hidden-memory state map",
    )
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.055, label=r"score $\mathcal{H}$")
    panel_label(ax, "a")

    ax = axes[1]
    response_levels = np.linspace(0.0, float(np.nanmax(response)), 19)
    im = ax.contourf(js, gs, response, levels=response_levels, cmap="magma", extend="max")
    ax.contour(js, gs, score, levels=[0.30], colors="#4cc9f0", linewidths=1.1)
    ax.plot(4.0, 5.0, marker="*", ms=8.5, mec="black", mew=0.65, color="white")
    ax.set(xlabel=r"alignment $J/k_{\rm B}T$", title="entry-response field")
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.055, label=r"response $\mathcal{R}$")
    panel_label(ax, "c")

    ax = axes[2]
    cut_index = int(np.argmin(np.abs(gs - 5.0)))
    cut_response = response[cut_index]
    normalized_response = cut_response / max(float(cut_response.max()), 1e-12)
    cut_score = score[cut_index]
    accepted = cut_score >= 0.30
    if np.any(accepted):
        ax.axvspan(float(js[accepted].min()), float(js[accepted].max()), color="#b7e4c7", alpha=0.28, lw=0)
    for values, label, color in (
        (s[cut_index], r"$S$", "#4d4d4d"),
        (c2[cut_index], r"$C_2$", "#277da1"),
        (g2[cut_index], r"$G_2$", "#43aa8b"),
        (qea[cut_index], r"$q_{\rm EA}$", "#8e44ad"),
    ):
        ax.plot(js, values, marker="o", ms=2.6, lw=1.25, color=color, label=label)
    ax.plot(js, normalized_response, color="#f8961e", lw=1.15, ls="--", label=r"$\mathcal{R}/\mathcal{R}_{\max}$")
    ax.axvline(4.0, color="0.25", lw=0.75, ls=":")
    ax.set(
        xlabel=r"alignment $J/k_{\rm B}T$",
        ylabel="observable value",
        ylim=(-0.42, 1.02),
        title=r"cut at $g=5k_{\rm B}T$",
    )
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.5, loc="lower right")
    panel_label(ax, "b")
    save(fig, "fig2_capillary_parameter_map")


def figure3() -> None:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in DYNAMICS_PATHS]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.15), constrained_layout=True)

    def ensemble_curve(section: str, xkey: str, ykey: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.asarray(runs[0][section][xkey], dtype=float)
        values = np.asarray([run[section][ykey] for run in runs], dtype=float)
        return x, values.mean(axis=0), values.std(axis=0, ddof=1)

    ax = axes[0]
    for key, label, color in (
        ("split_replica", r"capillary $g=5$", "#2166ac"),
        ("no_capillary_split_replica", r"control $g=0$", "#b2182b"),
    ):
        t, y, e = ensemble_curve(key, "time", "overlap_mean")
        ax.plot(t, y, color=color, lw=1.5, label=label)
        ax.fill_between(t, y - e, y + e, color=color, alpha=0.14, lw=0)
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.axhline(0.5, color="0.55", lw=0.65, ls=":")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set(xlabel=r"time $D_rt$", ylabel=r"split-replica overlap $Q_{\rm split}$", xlim=(0, 650), ylim=(-0.12, 1.03), title="independent-noise retention")
    ax.text(0.97, 0.70, r"$>600\times$ slower", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#2166ac")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.58, 0.35))
    panel_label(ax, "a")

    ax = axes[2]
    common_lag_max = min(
        float(run["aging"]["curves"][index]["lag_time"][-1])
        for run in runs
        for index in range(3)
    )
    for index, color in enumerate(("#90be6d", "#f8961e", "#9b5de5")):
        curves = [run["aging"]["curves"][index] for run in runs]
        t = np.asarray(curves[0]["lag_time"], dtype=float)
        values = np.asarray([curve["correlation"] for curve in curves], dtype=float)
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        keep = t <= common_lag_max + 1e-9
        ax.plot(t[keep], mean[keep], lw=1.5, color=color, label=rf"$D_rt_w={curves[0]['waiting_time']:.0f}$")
        ax.fill_between(t[keep], (mean - std)[keep], (mean + std)[keep], color=color, alpha=0.12, lw=0)
    ax.set(xlabel=r"lag $D_r\Delta t$", ylabel=r"$C_2(t_w+\Delta t,t_w)$", xlim=(0, common_lag_max), ylim=(0.42, 1.02), title="waiting-time aging")
    ax.axvline(30, color="0.55", lw=0.65, ls=":")
    ax.annotate("older states retain more", xy=(30, 0.70), xytext=(70, 0.75), arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "0.35"}, fontsize=6.8)
    ax.legend(frameon=False)
    panel_label(ax, "c")

    ax = axes[1]
    rt, ry, re = ensemble_curve("write_release", "release_time", "release_overlap")
    elapsed = rt - rt[0]
    ax.plot(elapsed, ry, color="#1b9e77", lw=1.5, label=r"capillary $g=5$")
    ax.fill_between(elapsed, ry - re, ry + re, color="#1b9e77", alpha=0.12, lw=0)
    ct, cy, ce = ensemble_curve("no_capillary_write_release", "release_time", "release_overlap")
    control_elapsed = ct - ct[0]
    ax.plot(control_elapsed, cy, color="#b2182b", lw=1.2, label=r"control $g=0$")
    ax.fill_between(control_elapsed, cy - ce, cy + ce, color="#b2182b", alpha=0.10, lw=0)
    ax.axvline(0.0, color="0.35", lw=0.85, ls=":", zorder=5)
    ax.axhline(0.5, color="0.55", lw=0.65, ls=":")
    ax.set_xscale("symlog", linthresh=1.0, linscale=0.8)
    ax.set(xlabel=r"time after field removal $D_r(t-t_{\rm off})$", ylabel="written-state overlap", xlim=(0, 520), ylim=(-0.12, 1.03), title="field-free written memory")
    ax.text(0.97, 0.70, r"$>300\times$ slower", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#1b9e77")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.72, 0.35))
    panel_label(ax, "b")
    save(fig, "fig3_capillary_memory_dynamics")


def _mean_std(rows: Sequence[Dict[str, object]], metric: str) -> Tuple[float, float]:
    values = np.asarray([float(row[metric]) for row in rows])
    return float(values.mean()), float(values.std())


def row_hidden_score(row: Dict[str, object]) -> float:
    local_floor = max(
        0.0,
        min(float(row["C2_mean"]), float(row["G2_mean"]), float(row["q_EA_mean"])),
    )
    return (1.0 - float(row["S_mean"])) * local_floor


def figure4() -> None:
    rows = read_jsonl(DENSE)
    internal = json.loads(INTERNAL.read_text(encoding="utf-8"))["aggregate"]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.18), constrained_layout=True)

    ax = axes[0]
    cut = sorted(
        [row for row in rows if math.isclose(float(row["j_align"]), 4.0)],
        key=lambda row: float(row["g_capillary"]),
    )
    x = np.asarray([float(row["g_capillary"]) for row in cut])
    accepted = np.asarray([row_hidden_score(row) >= 0.30 for row in cut])
    if np.any(accepted):
        ax.axvspan(float(x[accepted].min()), float(x[accepted].max()), color="#b7e4c7", alpha=0.28, lw=0)
    for metric, label, color, marker in (
        ("S_mean", r"$S$", "#4d4d4d", "o"),
        ("C2_mean", r"$C_2$", "#277da1", "s"),
        ("G2_mean", r"$G_2$", "#43aa8b", "D"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#8e44ad", "^"),
    ):
        ax.plot(
            x,
            [float(row[metric]) for row in cut],
            color=color,
            marker=marker,
            ms=2.7,
            lw=1.25,
            label=label,
        )
    ax.axvline(5.0, color="0.25", lw=0.75, ls=":")
    ax.set(
        xlabel=r"capillary coupling $g/k_{\rm B}T$",
        ylabel="observable value",
        ylim=(-0.05, 0.88),
        title=r"robustness at $J=4k_{\rm B}T$",
    )
    ax.legend(frameon=False, ncol=2, columnspacing=0.8, handlelength=1.4)
    panel_label(ax, "a")

    ax = axes[1]
    r = np.asarray(internal["bin_centers"], dtype=float)
    c_mean = np.asarray(internal["relative_connected_mean"], dtype=float)
    c_std = np.asarray(internal["relative_connected_std"], dtype=float)
    g_mean = np.asarray(internal["bond_frame_correlation_mean"], dtype=float)
    g_std = np.asarray(internal["bond_frame_correlation_std"], dtype=float)
    keep = r >= 0.5
    ax.plot(r[keep], c_mean[keep], color="#277da1", marker="o", ms=2.6, lw=1.25, label=r"connected $C_2(r)$")
    ax.fill_between(r[keep], c_mean[keep] - c_std[keep], c_mean[keep] + c_std[keep], color="#277da1", alpha=0.13, lw=0)
    ax.plot(r[keep], g_mean[keep], color="#43aa8b", marker="s", ms=2.4, lw=1.25, label=r"bond-frame $G_2(r)$")
    ax.fill_between(r[keep], g_mean[keep] - g_std[keep], g_mean[keep] + g_std[keep], color="#43aa8b", alpha=0.13, lw=0)
    ax.axhline(0.0, color="0.4", lw=0.65, ls="--")
    ax.set(xlabel=r"separation $r/a$", ylabel="pair correlation", title="finite spatial range", ylim=(-0.10, 0.64))
    ax.legend(frameon=False)
    panel_label(ax, "b")

    ax = axes[2]
    score = np.asarray([row_hidden_score(row) for row in rows])
    s_values = np.asarray([float(row["S_mean"]) for row in rows])
    q_values = np.asarray([float(row["q_EA_mean"]) for row in rows])
    g_values = np.asarray([float(row["G2_mean"]) for row in rows])
    hidden = score >= 0.30
    ax.scatter(s_values[~hidden], q_values[~hidden], s=12, color="0.78", alpha=0.62, edgecolor="none")
    points = ax.scatter(
        s_values[hidden],
        q_values[hidden],
        c=g_values[hidden],
        cmap="viridis",
        vmin=0.35,
        vmax=0.55,
        s=24,
        edgecolor="white",
        linewidth=0.25,
    )
    selected = min(
        rows,
        key=lambda row: abs(float(row["j_align"]) - 4.0) + abs(float(row["g_capillary"]) - 5.0),
    )
    ax.plot(float(selected["S_mean"]), float(selected["q_EA_mean"]), marker="*", ms=9, color="white", mec="black", mew=0.7)
    ax.axvline(0.20, color="0.35", lw=0.65, ls="--")
    ax.axhline(0.60, color="0.35", lw=0.65, ls="--")
    ax.text(0.97, 0.05, f"{int(hidden.sum())}/{len(rows)} cells", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0)
    ax.set(
        xlabel=r"global nematic order $S$",
        ylabel=r"finite-window memory $q_{\rm EA}$",
        xlim=(0.02, 0.32),
        ylim=(0.02, 0.84),
        title=r"low $S$, persistent population",
    )
    fig.colorbar(points, ax=ax, pad=0.02, fraction=0.055, label=r"bond-frame order $G_2$")
    panel_label(ax, "c")
    save(fig, "fig4_capillary_internal_structure")


def figure4_scaling_mechanism() -> None:
    """Synthesize size scaling, matched controls, and independent memory tests."""
    size_rows: Dict[int, List[Dict[str, object]]] = {}
    for n, path in sorted(SIZE_PATHS.items()):
        size_rows[n * n] = [
            row for row in read_jsonl(path) if math.isclose(float(row["g_capillary"]), 5.0)
        ]
    sizes = np.asarray(sorted(size_rows), dtype=float)

    controls = read_jsonl(CONTROLS)
    by_control: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in controls:
        by_control[str(row["control"])].append(row)
    g0_n32 = [
        row
        for row in read_jsonl(SIZE_PATHS[32])
        if math.isclose(float(row["g_capillary"]), 0.0)
    ]
    by_control["no_capillary"] = g0_n32

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in DYNAMICS_PATHS]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.28), constrained_layout=True)

    # (a) The defining finite-size signature: global order vanishes while
    # local-frame order and two independent persistence measures remain finite.
    ax = axes[0]
    metric_style = (
        ("S_mean", r"$S$", "#b2182b", "o"),
        ("G2_mean", r"$G_2$", "#1b9e77", "D"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#6a3d9a", "^"),
        ("window_autocorrelation", r"$C_{\rm window}$", "#2166ac", "s"),
    )
    for metric, label, color, marker in metric_style:
        means = np.asarray([_mean_std(size_rows[int(n)], metric)[0] for n in sizes])
        errors = np.asarray([_mean_std(size_rows[int(n)], metric)[1] for n in sizes])
        ax.errorbar(
            sizes,
            means,
            yerr=errors,
            color=color,
            marker=marker,
            ms=4.2,
            lw=1.35,
            capsize=2,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(int(n)) for n in sizes])
    ax.set(xlabel="number of rotors $N$", ylabel="observable", ylim=(-0.02, 0.75), title="order vanishes; memory persists")
    ax.legend(frameon=False, ncol=2, loc="lower left", columnspacing=0.8, handlelength=1.4)
    panel_label(ax, "a")

    # (b) The physical size trajectory approaches the low-S/high-memory
    # quadrant; controls identify global ordering and stronger random trapping.
    ax = axes[1]
    trajectory_s = np.asarray([_mean_std(size_rows[int(n)], "S_mean")[0] for n in sizes])
    trajectory_q = np.asarray([_mean_std(size_rows[int(n)], "q_EA_mean")[0] for n in sizes])
    ax.plot(trajectory_s, trajectory_q, color="#1b9e77", lw=1.3, zorder=2)
    ax.scatter(
        trajectory_s,
        trajectory_q,
        color="#1b9e77",
        s=42,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.annotate(r"$N=144$", (trajectory_s[0], trajectory_q[0]), xytext=(5, 6), textcoords="offset points", fontsize=7.0)
    ax.annotate(r"$N=2304$", (trajectory_s[-1], trajectory_q[-1]), xytext=(5, -12), textcoords="offset points", fontsize=7.0)
    ax.annotate("", xy=(trajectory_s[-1], trajectory_q[-1]), xytext=(trajectory_s[0], trajectory_q[0]), arrowprops={"arrowstyle": "->", "color": "#1b9e77", "lw": 1.0})
    control_style = {
        "no_capillary": ("same graph, $g=0$", "#7570b3", "X"),
        "regular": ("regular", "#d95f02", "s"),
        "shuffled_frames": ("shuffled frames", "#e7298a", "P"),
    }
    for key, (label, color, marker) in control_style.items():
        x, xe = _mean_std(by_control[key], "S_mean")
        y, ye = _mean_std(by_control[key], "q_EA_mean")
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=marker, ms=6.0, color=color, mec="white", mew=0.5, capsize=2, label=label, zorder=4)
    ax.axvspan(0.0, 0.10, color="#1b9e77", alpha=0.07, lw=0)
    ax.axhspan(0.55, 0.86, color="#1b9e77", alpha=0.07, lw=0)
    ax.set(xlabel=r"global order $S$", ylabel=r"finite-window memory $q_{\rm EA}$", xlim=(0, 1.0), ylim=(0, 0.86), title="size trajectory and controls")
    ax.legend(frameon=False, loc="center right", handletextpad=0.35)
    panel_label(ax, "b")

    # (c) Four observables that test different notions of memory. Dot intervals
    # show graph-to-graph variation rather than conflating them into one score.
    ax = axes[2]
    physical = by_control["physical"]
    static_metrics = {
        r"$q_{\rm EA}$": (
            [float(row["q_EA_mean"]) for row in physical],
            [float(row["q_EA_mean"]) for row in g0_n32],
        ),
        r"$C_{\rm window}$": (
            [float(row["window_autocorrelation"]) for row in physical],
            [float(row["window_autocorrelation"]) for row in g0_n32],
        ),
        r"$Q_{\rm split}(625)$": (
            [float(run["split_replica"]["overlap_mean"][-1]) for run in runs],
            [float(run["no_capillary_split_replica"]["overlap_mean"][-1]) for run in runs],
        ),
        r"$Q_{\rm write}(625)$": (
            [float(run["write_release"]["release_overlap"][-1]) for run in runs],
            [float(run["no_capillary_write_release"]["release_overlap"][-1]) for run in runs],
        ),
    }
    labels = list(static_metrics)
    y = np.arange(len(labels), dtype=float)
    for offset, index, color, marker, label in (
        (-0.11, 0, "#2166ac", "o", r"capillary $g=5$"),
        (0.11, 1, "#b2182b", "s", r"control $g=0$"),
    ):
        means = []
        errors = []
        for values in static_metrics.values():
            array = np.asarray(values[index], dtype=float)
            means.append(float(array.mean()))
            errors.append(float(array.std(ddof=1)))
        ax.errorbar(means, y + offset, xerr=errors, fmt=marker, color=color, ms=4.5, capsize=2, lw=1.1, label=label)
    ax.axvline(0.0, color="0.45", lw=0.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="memory observable", xlim=(-0.05, 0.75), title="four independent memory tests")
    ax.text(0.97, 0.97, r"$g=5$", color="#2166ac", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    ax.text(0.97, 0.88, r"$g=0$", color="#b2182b", transform=ax.transAxes, ha="right", va="top", fontweight="bold")
    panel_label(ax, "c")
    save(fig, "fig4_capillary_scaling_mechanism")


def figure_supplement_controls() -> Dict[str, object]:
    controls = read_jsonl(CONTROLS)
    by_control: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in controls:
        by_control[str(row["control"])].append(row)
    no_cap = []
    for path in SIZE_PATHS.values():
        no_cap.extend(row for row in read_jsonl(path) if math.isclose(float(row["g_capillary"]), 0.0))
    by_control["no_capillary"] = no_cap

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.15), constrained_layout=True)
    colors = {
        "physical": "#1b9e77",
        "no_capillary": "#7570b3",
        "regular": "#d95f02",
        "shuffled_frames": "#e7298a",
    }
    labels = {
        "physical": "disordered capillary",
        "no_capillary": r"same graph, $g=0$",
        "regular": "regular lattice",
        "shuffled_frames": "shuffled bond frames",
    }
    ax = axes[0]
    for key in ("physical", "no_capillary", "regular", "shuffled_frames"):
        rows = by_control[key]
        ax.scatter(
            [float(row["S_mean"]) for row in rows],
            [float(row["q_EA_mean"]) for row in rows],
            s=27,
            color=colors[key],
            edgecolor="white",
            linewidth=0.4,
            label=labels[key],
        )
    ax.set(xlabel=r"global order $S$", ylabel=r"finite-window memory $q_{\rm EA}$", xlim=(0, 1.02), ylim=(0, 1.02))
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3)
    panel_label(ax, "a")

    sizes = []
    size_metrics: Dict[str, List[float]] = defaultdict(list)
    size_errors: Dict[str, List[float]] = defaultdict(list)
    size_rows: Dict[int, List[Dict[str, object]]] = {}
    for n, path in sorted(SIZE_PATHS.items()):
        rows = [row for row in read_jsonl(path) if math.isclose(float(row["g_capillary"]), 5.0)]
        size_rows[n] = rows
        sizes.append(n * n)
        for metric in ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation"):
            mean, std = _mean_std(rows, metric)
            size_metrics[metric].append(mean)
            size_errors[metric].append(std)

    ax = axes[1]
    for metric, label, color, marker in (
        ("S_mean", r"$S$", "#d73027", "o"),
        ("C2_mean", r"$C_2$", "#4575b4", "s"),
        ("G2_mean", r"$G_2$", "#1a9850", "D"),
        ("q_EA_mean", r"$q_{\rm EA}$", "#7b3294", "^"),
    ):
        ax.errorbar(sizes, size_metrics[metric], yerr=size_errors[metric], color=color, marker=marker, ms=4, lw=1.2, capsize=2, label=label)
    ax.set(xlabel="number of rotors $N$", ylabel="observable", ylim=(0, 0.85))
    ax.set_xscale("log")
    ax.set_xlim(min(sizes) * 0.88, max(sizes) * 1.12)
    ax.set_xticks(sizes, [str(x) for x in sizes])
    ax.legend(frameon=False, ncol=2)
    panel_label(ax, "b")

    ax = axes[2]
    qab = []
    qab_err = []
    for n in sorted(size_rows):
        vals = np.asarray([float(row["replica_overlap"]["magnitude_mean"]) for row in size_rows[n]])
        qab.append(float(vals.mean()))
        qab_err.append(float(vals.std()))
    floor = [math.sqrt(math.pi) / (2.0 * math.sqrt(N)) for N in sizes]
    ax.errorbar(sizes, qab, yerr=qab_err, color="#2166ac", marker="o", ms=4, lw=1.3, capsize=2, label=r"independent replicas $|Q_{ab}|$")
    ax.plot(sizes, floor, color="0.35", lw=1.0, ls="--", label=r"random $N^{-1/2}$ floor")
    ax.set(xlabel="number of rotors $N$", ylabel="overlap magnitude", ylim=(0, max(qab) * 1.28))
    ax.set_xscale("log")
    ax.set_xlim(min(sizes) * 0.88, max(sizes) * 1.12)
    ax.set_xticks(sizes, [str(x) for x in sizes])
    ax.legend(frameon=False)
    panel_label(ax, "c")
    save(fig, "figS1_capillary_controls_scaling")

    summary: Dict[str, object] = {"controls": {}, "finite_size": {}}
    for key, rows in by_control.items():
        summary["controls"][key] = {
            metric: {"mean": _mean_std(rows, metric)[0], "std": _mean_std(rows, metric)[1]}
            for metric in ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation")
        }
    for n, rows in size_rows.items():
        summary["finite_size"][str(n * n)] = {
            metric: {"mean": _mean_std(rows, metric)[0], "std": _mean_std(rows, metric)[1]}
            for metric in ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation")
        }
    return summary


def main() -> None:
    configure()
    # Figure 1 is a deterministic schematic plus a representative simulation
    # snapshot. Preserve an existing publication rendering when PyTorch is not
    # installed in the lightweight plotting environment.
    if os.environ.get("FORCE_FIG1") == "1" or not (OUT / "fig1_capillary_realization.pdf").exists():
        figure1()
    if os.environ.get("FIGURE1_ONLY") == "1":
        return
    figure2()
    figure3()
    figure4()
    figure4_scaling_mechanism()
    summary = figure_supplement_controls()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "capillary_prl_figure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(OUT), "main_figures": 4, "supplement_figures": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
