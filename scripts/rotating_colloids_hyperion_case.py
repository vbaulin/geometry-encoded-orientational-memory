#!/usr/bin/env python3
"""Hyperion-style theory construction case: fixed-center rotating colloids.

This script is intentionally small and reproducible.  It takes the sparse-
attention lesson from the discoveries folder seriously: a candidate theory is
not only an interaction law, but a coupled operator + flow + closure + boundary
object.  For fixed-center elongated colloids, the minimal construction is an
orientation-only Langevin/rotational-Smoluchowski kernel on a fixed spatial
graph.

The code compares a geometry-free nematic rotor kernel against a geometry-
embodied closure term that couples rod orientation to the bond directions of
the fixed lattice.  The output is a compact JSON/Markdown/PNG case study.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import matplotlib.pyplot as plt
import numpy as np


def make_lattice(n: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    pos = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    pos -= pos.mean(axis=0, keepdims=True)
    return pos


def make_triangular_lattice(n: int) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    x = xs + 0.5 * ys
    y = (math.sqrt(3.0) / 2.0) * ys
    pos = np.column_stack([x.ravel(), y.ravel()]).astype(float)
    pos -= pos.mean(axis=0, keepdims=True)
    return pos


def make_edges(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src: List[int] = []
    tgt: List[int] = []
    phi: List[float] = []
    for i in range(n):
        for j in range(n):
            a = i * n + j
            for di, dj in ((1, 0), (0, 1)):
                ii = (i + di) % n
                jj = (j + dj) % n
                b = ii * n + jj
                src.append(a)
                tgt.append(b)
                phi.append(math.atan2(dj, di))
    return np.asarray(src), np.asarray(tgt), np.asarray(phi)


def triangular_displacement(di: int, dj: int) -> Tuple[float, float]:
    return float(di) + 0.5 * float(dj), (math.sqrt(3.0) / 2.0) * float(dj)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-6, None)
    mean = float(np.mean(weights)) if weights.size else 1.0
    if mean <= 0.0 or not math.isfinite(mean):
        return np.ones_like(weights)
    return weights / mean


@lru_cache(maxsize=128)
def normalize_graph_mode(graph_mode: str) -> str:
    key = graph_mode.replace("_", "-").lower()
    aliases = {
        "patchy": "mosaic",
        "patchwork": "mosaic",
        "patchwork-square": "mosaic",
        "mosaic-square": "mosaic",
        "clustered-patches": "mosaic",
    }
    return aliases.get(key, key)


@lru_cache(maxsize=128)
def normalize_constraint_mode(constraint_mode: str) -> str:
    key = constraint_mode.replace("_", "-").lower()
    aliases = {
        "bond": "bond",
        "pair": "bond",
        "pair-bond": "bond",
        "bond-frame": "bond",
        "substrate": "grooved",
        "groove": "grooved",
        "grooves": "grooved",
        "grooved": "grooved",
        "easy-axis": "grooved",
    }
    if key not in aliases:
        raise ValueError(f"Unknown constraint mode: {constraint_mode}")
    return aliases[key]


def make_easy_axes(
    n: int,
    *,
    graph_mode: str,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    patch_angle_step: float = math.pi / 4.0,
    easy_axis_disorder: float = 0.0,
) -> np.ndarray:
    """Return local apolar groove axes alpha_i for grooved-surface runs.

    Square, long-range, and random controls use a uniform groove axis unless
    easy_axis_disorder is supplied.  Triangular controls use a deterministic
    three-sublattice groove-axis pattern.  Mosaic runs use the same domain
    angle pattern as the final pair-model mosaic graph, making the groove
    realization a local-surface version of the same final run family.
    """
    mode = normalize_graph_mode(graph_mode)
    axes = np.zeros(n * n, dtype=float)
    if mode == "triangular":
        for i in range(n):
            for j in range(n):
                axes[i * n + j] = ((i + 2 * j) % 3) * (math.pi / 3.0)
    elif mode == "mosaic":
        cluster = max(2, int(cluster_size))
        blocks = max(1, int(math.ceil(n / cluster)))
        for i in range(n):
            for j in range(n):
                bx = min(blocks - 1, i // cluster)
                by = min(blocks - 1, j // cluster)
                axes[i * n + j] = ((bx + 2 * by) % 4) * float(patch_angle_step)
    if easy_axis_disorder > 0.0:
        rng = np.random.default_rng(graph_seed + 424243 * n + 19)
        axes += rng.normal(scale=float(easy_axis_disorder), size=axes.size)
    return (axes + math.pi) % math.pi


@lru_cache(maxsize=128)
def make_graph(
    n: int,
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Build a pinned-center graph with bond angles and normalized couplings.

    graph_mode:
      square      nearest-neighbor square graph, the historical default.
      triangular  non-bipartite triangular-neighbor graph on a triangular lattice.
      long-range  square positions with all periodic pairs within graph_radius.
      random      random pinned centers with k-nearest or radius graph.
      mosaic      rotated locally compatible square domains with weaker frustrated cross-links.
    """
    mode = normalize_graph_mode(graph_mode)
    src: List[int] = []
    tgt: List[int] = []
    phi: List[float] = []
    dist: List[float] = []

    if mode == "square":
        pos = make_lattice(n)
        for i in range(n):
            for j in range(n):
                a = i * n + j
                for di, dj in ((1, 0), (0, 1)):
                    b = ((i + di) % n) * n + ((j + dj) % n)
                    src.append(a)
                    tgt.append(b)
                    phi.append(math.atan2(dj, di))
                    dist.append(1.0)
    elif mode == "triangular":
        pos = make_triangular_lattice(n)
        for i in range(n):
            for j in range(n):
                a = i * n + j
                for di, dj in ((1, 0), (0, 1), (1, -1)):
                    b = ((i + di) % n) * n + ((j + dj) % n)
                    dx, dy = triangular_displacement(di, dj)
                    src.append(a)
                    tgt.append(b)
                    phi.append(math.atan2(dy, dx))
                    dist.append(math.hypot(dx, dy))
    elif mode == "long-range":
        pos = make_lattice(n)
        radius = graph_radius if graph_radius > 0.0 else 2.25
        for a in range(n * n):
            ia, ja = divmod(a, n)
            for b in range(a + 1, n * n):
                ib, jb = divmod(b, n)
                dx = ((ib - ia + n / 2.0) % n) - n / 2.0
                dy = ((jb - ja + n / 2.0) % n) - n / 2.0
                r = math.hypot(dx, dy)
                if 1e-12 < r <= radius:
                    src.append(a)
                    tgt.append(b)
                    phi.append(math.atan2(dy, dx))
                    dist.append(r)
    elif mode == "random":
        rng = np.random.default_rng(graph_seed + 104729 * n)
        pos = rng.uniform(0.0, float(n), size=(n * n, 2))
        pos -= pos.mean(axis=0, keepdims=True)
        pairs: set[Tuple[int, int]] = set()
        radius = graph_radius
        k = max(1, int(graph_k))
        for a in range(n * n):
            delta = pos - pos[a]
            r = np.linalg.norm(delta, axis=1)
            order = np.argsort(r)
            if radius > 0.0:
                neighbors = [int(b) for b in order if 1e-12 < r[b] <= radius]
            else:
                neighbors = [int(b) for b in order[1 : k + 1]]
            for b in neighbors:
                i, j = sorted((a, b))
                if i != j:
                    pairs.add((i, j))
        for a, b in sorted(pairs):
            dx, dy = pos[b] - pos[a]
            r = math.hypot(float(dx), float(dy))
            if r > 1e-12:
                src.append(a)
                tgt.append(b)
                phi.append(math.atan2(float(dy), float(dx)))
                dist.append(r)
    elif mode == "mosaic":
        cluster = max(2, int(cluster_size))
        blocks = max(1, int(math.ceil(n / cluster)))
        rng = np.random.default_rng(graph_seed + 314159 * n + 23)
        pos = np.zeros((n * n, 2), dtype=float)
        block_nodes: Dict[Tuple[int, int], List[int]] = {}
        spacing = float(cluster) * 1.8
        local_center = 0.5 * float(cluster - 1)
        for i in range(n):
            for j in range(n):
                bx = min(blocks - 1, i // cluster)
                by = min(blocks - 1, j // cluster)
                u = i - bx * cluster
                v = j - by * cluster
                a = i * n + j
                alpha = ((bx + 2 * by) % 4) * float(patch_angle_step)
                ca = math.cos(alpha)
                sa = math.sin(alpha)
                x = float(u) - local_center
                y = float(v) - local_center
                jitter = rng.normal(scale=0.035, size=2)
                pos[a, 0] = bx * spacing + ca * x - sa * y + jitter[0]
                pos[a, 1] = by * spacing + sa * x + ca * y + jitter[1]
                block_nodes.setdefault((bx, by), []).append(a)

        # Strong, locally compatible square edges inside each rotated domain.
        for i in range(n):
            for j in range(n):
                bx = min(blocks - 1, i // cluster)
                by = min(blocks - 1, j // cluster)
                a = i * n + j
                for di, dj in ((1, 0), (0, 1)):
                    ii = i + di
                    jj = j + dj
                    if ii >= n or jj >= n:
                        continue
                    if min(blocks - 1, ii // cluster) != bx or min(blocks - 1, jj // cluster) != by:
                        continue
                    b = ii * n + jj
                    dx, dy = pos[b] - pos[a]
                    src.append(a)
                    tgt.append(b)
                    phi.append(math.atan2(float(dy), float(dx)))
                    dist.append(math.hypot(float(dx), float(dy)))

        # Weak cross-links between neighboring domains. These obstruct global
        # director choice without overwhelming local graph registration.
        cross_pairs: set[Tuple[int, int]] = set()
        k_cross = max(0, int(crosslink_k))
        for bx in range(blocks):
            for by in range(blocks):
                nodes_a = block_nodes.get((bx, by), [])
                for nb in ((bx + 1, by), (bx, by + 1)):
                    nodes_b = block_nodes.get(nb, [])
                    if not nodes_a or not nodes_b:
                        continue
                    candidates: List[Tuple[float, int, int]] = []
                    for a in nodes_a:
                        delta = pos[nodes_b] - pos[a]
                        rr = np.linalg.norm(delta, axis=1)
                        for idx_b in np.argsort(rr)[: max(1, k_cross)]:
                            b = int(nodes_b[int(idx_b)])
                            candidates.append((float(rr[int(idx_b)]), a, b))
                    for _, a, b in sorted(candidates)[:k_cross]:
                        i0, j0 = sorted((a, b))
                        if i0 != j0:
                            cross_pairs.add((i0, j0))
        for a, b in sorted(cross_pairs):
            dx, dy = pos[b] - pos[a]
            r = math.hypot(float(dx), float(dy))
            if r > 1e-12:
                src.append(a)
                tgt.append(b)
                phi.append(math.atan2(float(dy), float(dx)))
                dist.append(r)
    else:
        raise ValueError(f"Unknown graph mode: {graph_mode}")

    src_a = np.asarray(src, dtype=np.int64)
    tgt_a = np.asarray(tgt, dtype=np.int64)
    phi_a = np.asarray(phi, dtype=float)
    dist_a = np.asarray(dist, dtype=float)
    if dist_a.size == 0:
        raise ValueError(f"Graph mode {graph_mode!r} produced no edges for n={n}")

    if coupling_decay > 0.0:
        r0 = float(np.min(dist_a))
        weights = np.exp(-(dist_a - r0) / float(coupling_decay))
    else:
        weights = np.ones_like(dist_a)
    if bond_disorder > 0.0:
        rng = np.random.default_rng(graph_seed + 271828 * n + 17)
        sigma = float(bond_disorder)
        weights *= np.exp(rng.normal(loc=-0.5 * sigma * sigma, scale=sigma, size=weights.size))
    if mode == "mosaic" and crosslink_k > 0:
        local_cut = np.percentile(dist_a, 70)
        weights = np.where(dist_a > local_cut, weights * max(1e-4, float(crosslink_weight)), weights)
    weights = normalize_weights(weights)

    meta = {
        "graph_mode": mode,
        "graph_radius": float(graph_radius),
        "graph_k": int(graph_k),
        "coupling_decay": float(coupling_decay),
        "bond_disorder": float(bond_disorder),
        "graph_seed": int(graph_seed),
        "cluster_size": int(cluster_size),
        "crosslink_k": int(crosslink_k),
        "crosslink_weight": float(crosslink_weight),
        "patch_angle_step": float(patch_angle_step),
        "edge_count": int(src_a.size),
        "mean_degree": float(2.0 * src_a.size / (n * n)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
    }
    return pos.astype(float), src_a, tgt_a, phi_a, weights.astype(float), meta


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if math.isfinite(out):
        return out
    return default


def json_ready(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return finite_or_none(float(obj))
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def nematic_order(theta: np.ndarray) -> float:
    return float(abs(np.mean(np.exp(2j * theta))))


def weighted_mean(vals: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    if weights is None:
        return float(np.mean(vals))
    denom = float(np.sum(weights))
    if denom <= 0.0 or not math.isfinite(denom):
        return float(np.mean(vals))
    return float(np.sum(weights * vals) / denom)


def geometry_lock(
    theta: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    # Bond-locking order: high when each pair aligns with the bond frame.
    vals = np.cos(2.0 * (theta[src] + theta[tgt] - 2.0 * phi))
    return weighted_mean(vals, weights)


def groove_lock(theta: np.ndarray, easy_axes: np.ndarray) -> float:
    # Local groove-locking order: high when rods align with their surface axes.
    vals = np.cos(2.0 * (theta - easy_axes))
    return float(np.mean(vals))


def orientational_correlator(
    theta: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    # Nearest-neighbor nematic correlator C_2(a) = <cos 2(theta_i - theta_j)>.
    vals = np.cos(2.0 * (theta[src] - theta[tgt]))
    return weighted_mean(vals, weights)


def temporal_correlator(snaps: List[np.ndarray], lag: int = 1) -> float:
    # Late-time angular autocorrelation C_2(tau) = <cos 2(theta_i(t+tau)-theta_i(t))>.
    if len(snaps) <= lag:
        return float("nan")
    vals = []
    for a, b in zip(snaps[:-lag], snaps[lag:]):
        vals.append(np.mean(np.cos(2.0 * (b - a))))
    return float(np.mean(vals))


def edwards_anderson_parameter(snaps: List[np.ndarray]) -> float:
    """Nematic Edwards-Anderson memory q_EA = N^-1 sum_i |<exp(2i theta_i)>_t|^2."""
    if not snaps:
        return float("nan")
    z = np.exp(2j * np.stack(snaps, axis=0))
    local_time_mean = np.mean(z, axis=0)
    return float(np.mean(np.abs(local_time_mean) ** 2))


def edge_frustration(
    theta: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    eps: float,
    weights: Optional[np.ndarray] = None,
) -> float:
    align = np.cos(2.0 * (theta[src] - theta[tgt]))
    geom = np.cos(2.0 * (theta[src] + theta[tgt] - 2.0 * phi))
    return weighted_mean(1.0 - align + eps * (1.0 - geom), weights)


def groove_frustration(
    theta: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    eps: float,
    easy_axes: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    align = np.cos(2.0 * (theta[src] - theta[tgt]))
    local = np.cos(2.0 * (theta - easy_axes))
    return weighted_mean(1.0 - align, weights) + float(eps) * (1.0 - float(np.mean(local)))


def defect_count(theta: np.ndarray, n: int) -> int:
    """Count coarse nematic half-defects on plaquettes.

    This is a diagnostic only.  It unwraps doubled angles around each plaquette
    and counts nonzero winding of the director field.
    """
    a = (2.0 * theta.reshape(n, n) + math.pi) % (2.0 * math.pi) - math.pi
    count = 0
    for i in range(n):
        for j in range(n):
            loop = [
                a[i, j],
                a[(i + 1) % n, j],
                a[(i + 1) % n, (j + 1) % n],
                a[i, (j + 1) % n],
                a[i, j],
            ]
            winding = 0.0
            for u, v in zip(loop[:-1], loop[1:]):
                d = (v - u + math.pi) % (2.0 * math.pi) - math.pi
                winding += d
            if abs(round(winding / (2.0 * math.pi))) >= 1:
                count += 1
    return count


def binder_cumulant(samples: Iterable[float]) -> float:
    vals = np.asarray(list(samples), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    second = float(np.mean(vals**2))
    fourth = float(np.mean(vals**4))
    if second <= 1e-12:
        return float("nan")
    return float(1.0 - fourth / (3.0 * second * second))


def susceptibility(samples: Iterable[float], node_count: int) -> float:
    vals = np.asarray(list(samples), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(node_count * np.var(vals))


def make_radial_pair_cache(
    n: int,
    *,
    n_bins: int,
    max_pairs: int,
    seed: int,
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
) -> Dict[str, np.ndarray]:
    """Build sampled all-pair bins for C2(r) and G2(r).

    The cache uses periodic minimum-image distances on the square pinned array.
    Sampling keeps maximal scans tractable for large lattices.
    """
    pos, _, _, _, _, _ = make_graph(
        n,
        graph_mode=graph_mode,
        graph_radius=graph_radius,
        graph_k=graph_k,
        coupling_decay=coupling_decay,
        bond_disorder=bond_disorder,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        crosslink_k=crosslink_k,
        crosslink_weight=crosslink_weight,
        patch_angle_step=patch_angle_step,
    )
    src: List[int] = []
    tgt: List[int] = []
    dist: List[float] = []
    phi: List[float] = []
    for i in range(pos.shape[0]):
        xi, yi = pos[i]
        for j in range(i + 1, pos.shape[0]):
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            if normalize_graph_mode(graph_mode) in {"square", "long-range"}:
                dx = ((dx + n / 2.0) % n) - n / 2.0
                dy = ((dy + n / 2.0) % n) - n / 2.0
            r = math.hypot(dx, dy)
            if r <= 1e-12:
                continue
            src.append(i)
            tgt.append(j)
            dist.append(r)
            phi.append(math.atan2(dy, dx))

    src_a = np.asarray(src, dtype=np.int32)
    tgt_a = np.asarray(tgt, dtype=np.int32)
    dist_a = np.asarray(dist, dtype=float)
    phi_a = np.asarray(phi, dtype=float)
    if src_a.size > max_pairs:
        rng = np.random.default_rng(seed)
        keep = rng.choice(src_a.size, size=max_pairs, replace=False)
        src_a = src_a[keep]
        tgt_a = tgt_a[keep]
        dist_a = dist_a[keep]
        phi_a = phi_a[keep]

    # Avoid the singular zero-distance bin; use half-integer lattice bins.
    r_max = min(float(n) / 2.0, float(np.max(dist_a)))
    edges = np.linspace(0.5, r_max, n_bins + 1)
    bin_idx = np.digitize(dist_a, edges) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "src": src_a[valid],
        "tgt": tgt_a[valid],
        "phi": phi_a[valid],
        "bin_idx": bin_idx[valid].astype(np.int16),
        "r_centers": centers,
    }


def radial_correlators(
    theta: np.ndarray,
    cache: Optional[Dict[str, np.ndarray]],
    *,
    constraint_mode: str = "bond",
    easy_axes: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float], List[float]]:
    if cache is None:
        return [], [], []
    src = cache["src"]
    tgt = cache["tgt"]
    phi = cache["phi"]
    bin_idx = cache["bin_idx"]
    centers = cache["r_centers"]
    c_vals = np.cos(2.0 * (theta[src] - theta[tgt]))
    if normalize_constraint_mode(constraint_mode) == "grooved" and easy_axes is not None:
        g_vals = 0.5 * (
            np.cos(2.0 * (theta[src] - easy_axes[src]))
            + np.cos(2.0 * (theta[tgt] - easy_axes[tgt]))
        )
    else:
        g_vals = np.cos(2.0 * (theta[src] + theta[tgt] - 2.0 * phi))
    c_out: List[float] = []
    g_out: List[float] = []
    for b in range(len(centers)):
        mask = bin_idx == b
        if np.any(mask):
            c_out.append(float(np.mean(c_vals[mask])))
            g_out.append(float(np.mean(g_vals[mask])))
        else:
            c_out.append(float("nan"))
            g_out.append(float("nan"))
    return centers.astype(float).tolist(), c_out, g_out


SCALAR_REP_METRICS = (
    "nematic_order",
    "nematic_order_time_std",
    "geometry_lock",
    "geometry_lock_time_std",
    "orientational_corr_nn",
    "orientational_corr_nn_time_std",
    "temporal_corr_lag1",
    "temporal_corr_lag5",
    "temporal_corr_lag20",
    "q_EA",
    "edge_frustration",
    "edge_frustration_time_std",
    "defect_count",
    "defect_count_time_std",
    "defect_density",
    "defect_density_time_std",
    "susceptibility_S",
    "susceptibility_G",
    "binder_S",
    "binder_G",
)


def add_hidden_scores(row: Dict[str, object]) -> None:
    s = safe_float(row.get("nematic_order_mean", 0.0))
    g = safe_float(row.get("geometry_lock_mean", 0.0))
    c = safe_float(row.get("orientational_corr_nn_mean", 0.0))
    q = safe_float(row.get("q_EA_mean", 0.0))
    defects = safe_float(row.get("defect_count_mean", 0.0))
    row["hidden_geometry_score"] = max(0.0, g) * max(0.0, 1.0 - s) * (1.0 + 0.04 * defects)
    row["geometry_without_global_order"] = max(0.0, g - s) * max(0.0, c)
    row["hidden_pocket_flag"] = float(g >= 0.70 and c >= 0.70 and q >= 0.50 and s <= 0.35)


def summarize_replicates(
    *,
    n: int,
    eps: float,
    j: float,
    reps: List[Dict[str, object]],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "n": n,
        "eps_geom": float(eps),
        "j_align": float(j),
        "replicates": len(reps),
    }
    for key in SCALAR_REP_METRICS:
        vals = np.asarray([r.get(key, float("nan")) for r in reps], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            row[key + "_mean"] = float(vals.mean())
            row[key + "_std"] = float(vals.std())
        else:
            row[key + "_mean"] = float("nan")
            row[key + "_std"] = float("nan")

    # Backward-compatible aliases used by plotting and older reports.
    row["nematic_order_mean"] = row["nematic_order_mean"]
    row["geometry_lock_mean"] = row["geometry_lock_mean"]
    row["orientational_corr_nn_mean"] = row["orientational_corr_nn_mean"]
    if reps:
        for key in (
            "graph_mode",
            "constraint_mode",
            "geometry_observable",
            "easy_axis_disorder",
            "graph_edge_count",
            "graph_mean_degree",
            "j_over_Dr",
            "groove_strength",
            "groove_over_Dr",
        ):
            if key in reps[0]:
                row[key] = reps[0][key]

    radial_r = reps[0].get("radial_r", []) if reps else []
    if radial_r:
        row["radial_r"] = radial_r
        for src_key, dst_key in (("radial_C2", "radial_C2"), ("radial_G2", "radial_G2")):
            arr = np.asarray([r.get(src_key, []) for r in reps], dtype=float)
            if arr.ndim == 2 and arr.shape[1] > 0:
                means: List[float] = []
                stds: List[float] = []
                for col in range(arr.shape[1]):
                    vals = arr[:, col]
                    vals = vals[np.isfinite(vals)]
                    if vals.size:
                        means.append(float(vals.mean()))
                        stds.append(float(vals.std()))
                    else:
                        means.append(float("nan"))
                        stds.append(float("nan"))
                row[dst_key + "_mean"] = means
                row[dst_key + "_std"] = stds
    add_hidden_scores(row)
    return row


def point_key(n: int, eps: float, j: float) -> str:
    return f"n={n}|eps={eps:.8g}|j={j:.8g}"


def graph_id(
    graph_mode: str,
    graph_radius: float,
    graph_k: int,
    coupling_decay: float,
    bond_disorder: float,
    graph_seed: int,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
    constraint_mode: str = "bond",
    easy_axis_disorder: float = 0.0,
) -> str:
    mode = normalize_graph_mode(graph_mode)
    constraint = normalize_constraint_mode(constraint_mode)
    if (
        mode == "square"
        and graph_radius == 0.0
        and coupling_decay == 0.0
        and bond_disorder == 0.0
        and constraint == "bond"
    ):
        return ""
    return (
        f"|graph={mode}|radius={graph_radius:.4g}|k={graph_k}"
        f"|decay={coupling_decay:.4g}|disorder={bond_disorder:.4g}|seed={graph_seed}"
        f"|cluster={cluster_size}|cross={crosslink_k}|cw={crosslink_weight:.4g}|astep={patch_angle_step:.4g}"
        f"|constraint={constraint}|axisdis={easy_axis_disorder:.4g}"
    )


def read_completed_points(path: Path) -> Dict[str, Dict[str, object]]:
    completed: Dict[str, Dict[str, object]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("point_key", ""))
            if key:
                completed[key] = row
    return completed


def resume_config_mismatches(row: Dict[str, object], expected: Dict[str, object]) -> List[str]:
    """Return stored scan settings that disagree with the requested run.

    Historical point keys omit the integration length, noise, time step, and
    replica count.  Without this check, resume can silently mix simulations
    generated under different dynamics.
    """
    mismatches: List[str] = []
    for key, wanted in expected.items():
        stored = row.get(key)
        if isinstance(wanted, float):
            try:
                if not math.isclose(float(stored), wanted, rel_tol=1e-10, abs_tol=1e-12):
                    mismatches.append(f"{key}: stored={stored!r}, requested={wanted!r}")
            except (TypeError, ValueError):
                mismatches.append(f"{key}: stored={stored!r}, requested={wanted!r}")
        elif stored != wanted:
            mismatches.append(f"{key}: stored={stored!r}, requested={wanted!r}")
    return mismatches


def simulate(
    *,
    n: int,
    j_align: float,
    eps_geom: float,
    noise: float,
    drive: float,
    dt: float,
    steps: int,
    seed: int,
    burn_in_steps: int = 0,
    initial_theta: Optional[np.ndarray] = None,
    radial_cache: Optional[Dict[str, np.ndarray]] = None,
    sample_stride: int = 20,
    return_final_theta: bool = False,
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
    constraint_mode: str = "bond",
    easy_axis_disorder: float = 0.0,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    constraint = normalize_constraint_mode(constraint_mode)
    _, src, tgt, phi, weights, graph_meta = make_graph(
        n,
        graph_mode=graph_mode,
        graph_radius=graph_radius,
        graph_k=graph_k,
        coupling_decay=coupling_decay,
        bond_disorder=bond_disorder,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        crosslink_k=crosslink_k,
        crosslink_weight=crosslink_weight,
        patch_angle_step=patch_angle_step,
    )
    easy_axes = make_easy_axes(
        n,
        graph_mode=graph_mode,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        patch_angle_step=patch_angle_step,
        easy_axis_disorder=easy_axis_disorder,
    )
    if initial_theta is None:
        theta = rng.uniform(-math.pi, math.pi, size=n * n)
    else:
        theta = np.asarray(initial_theta, dtype=float).copy()
    defect_supported = graph_meta["graph_mode"] not in {"random", "mosaic"}

    sqrt_noise = math.sqrt(2.0 * noise * dt)
    snapshots: List[np.ndarray] = []
    s_samples: List[float] = []
    g_samples: List[float] = []
    c_samples: List[float] = []
    frustration_samples: List[float] = []
    defect_samples: List[float] = []

    total_steps = burn_in_steps + steps
    for step in range(total_steps):
        torque = np.zeros_like(theta)
        d = theta[src] - theta[tgt]
        g = theta[src] + theta[tgt] - 2.0 * phi

        # Bond mode:
        #   U_ij = -J cos(2(theta_i - theta_j))
        #          -eps J cos(2(theta_i + theta_j - 2 phi_ij)).
        # Grooved mode:
        #   U = -J sum_ij cos(2(theta_i - theta_j))
        #       -eps J sum_i cos(2(theta_i - alpha_i)),
        # where alpha_i is the local surface-groove easy axis.
        # dtheta/dt = -dU/dtheta + drive + noise.
        if constraint == "grooved":
            t_src = weights * (-2.0 * j_align * np.sin(2.0 * d))
            t_tgt = weights * (+2.0 * j_align * np.sin(2.0 * d))
        else:
            t_src = weights * (
                -2.0 * j_align * np.sin(2.0 * d) - 2.0 * eps_geom * j_align * np.sin(2.0 * g)
            )
            t_tgt = weights * (
                +2.0 * j_align * np.sin(2.0 * d) - 2.0 * eps_geom * j_align * np.sin(2.0 * g)
            )
        np.add.at(torque, src, t_src)
        np.add.at(torque, tgt, t_tgt)
        if constraint == "grooved":
            torque += -2.0 * eps_geom * j_align * np.sin(2.0 * (theta - easy_axes))

        theta += dt * (torque + drive) + sqrt_noise * rng.normal(size=theta.size)
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
        if step >= burn_in_steps and (step - burn_in_steps) % sample_stride == 0:
            snapshots.append(theta.copy())
            s_samples.append(nematic_order(theta))
            if constraint == "grooved":
                g_samples.append(groove_lock(theta, easy_axes))
            else:
                g_samples.append(geometry_lock(theta, src, tgt, phi, weights))
            c_samples.append(orientational_correlator(theta, src, tgt, weights))
            if constraint == "grooved":
                frustration_samples.append(groove_frustration(theta, src, tgt, eps_geom, easy_axes, weights))
            else:
                frustration_samples.append(edge_frustration(theta, src, tgt, phi, eps_geom, weights))
            defect_samples.append(float(defect_count(theta, n)) if defect_supported else float("nan"))

    if not s_samples:
        snapshots.append(theta.copy())
        s_samples.append(nematic_order(theta))
        if constraint == "grooved":
            g_samples.append(groove_lock(theta, easy_axes))
        else:
            g_samples.append(geometry_lock(theta, src, tgt, phi, weights))
        c_samples.append(orientational_correlator(theta, src, tgt, weights))
        if constraint == "grooved":
            frustration_samples.append(groove_frustration(theta, src, tgt, eps_geom, easy_axes, weights))
        else:
            frustration_samples.append(edge_frustration(theta, src, tgt, phi, eps_geom, weights))
        defect_samples.append(float(defect_count(theta, n)) if defect_supported else float("nan"))

    r_centers, c2_r, g2_r = radial_correlators(
        theta,
        radial_cache,
        constraint_mode=constraint,
        easy_axes=easy_axes,
    )
    node_count = n * n
    defect_arr = np.asarray(defect_samples, dtype=float)

    result: Dict[str, object] = {
        "n": n,
        "j_align": j_align,
        "eps_geom": eps_geom,
        "noise": noise,
        "j_over_Dr": float(j_align / noise) if noise > 0.0 else float("nan"),
        "groove_strength": float(eps_geom * j_align) if constraint == "grooved" else float("nan"),
        "groove_over_Dr": float(eps_geom * j_align / noise)
        if constraint == "grooved" and noise > 0.0
        else float("nan"),
        "drive": drive,
        "burn_in_steps": burn_in_steps,
        "sample_steps": steps,
        "sample_stride": sample_stride,
        "sample_count": len(s_samples),
        "graph_mode": graph_meta["graph_mode"],
        "constraint_mode": constraint,
        "geometry_observable": "groove_lock" if constraint == "grooved" else "bond_frame_lock",
        "easy_axis_disorder": float(easy_axis_disorder),
        "graph_edge_count": graph_meta["edge_count"],
        "graph_mean_degree": graph_meta["mean_degree"],
        "nematic_order": float(np.mean(s_samples)),
        "nematic_order_time_std": float(np.std(s_samples)),
        "geometry_lock": float(np.mean(g_samples)),
        "geometry_lock_time_std": float(np.std(g_samples)),
        "orientational_corr_nn": float(np.mean(c_samples)),
        "orientational_corr_nn_time_std": float(np.std(c_samples)),
        "temporal_corr_lag1": temporal_correlator(snapshots, lag=1),
        "temporal_corr_lag5": temporal_correlator(snapshots, lag=5),
        "temporal_corr_lag20": temporal_correlator(snapshots, lag=20),
        "q_EA": edwards_anderson_parameter(snapshots),
        "edge_frustration": float(np.mean(frustration_samples)),
        "edge_frustration_time_std": float(np.std(frustration_samples)),
        "defect_count": float(np.nanmean(defect_samples)) if np.any(np.isfinite(defect_arr)) else float("nan"),
        "defect_count_time_std": float(np.nanstd(defect_samples)) if np.any(np.isfinite(defect_arr)) else float("nan"),
        "defect_density": float(np.nanmean(defect_arr) / node_count) if np.any(np.isfinite(defect_arr)) else float("nan"),
        "defect_density_time_std": float(np.nanstd(defect_arr / node_count)) if np.any(np.isfinite(defect_arr)) else float("nan"),
        "susceptibility_S": susceptibility(s_samples, node_count),
        "susceptibility_G": susceptibility(g_samples, node_count),
        "binder_S": binder_cumulant(s_samples),
        "binder_G": binder_cumulant(g_samples),
        "radial_r": r_centers,
        "radial_C2": c2_r,
        "radial_G2": g2_r,
    }
    if return_final_theta:
        result["final_theta"] = theta
    return result


def simulate_torch_replicates(
    *,
    n: int,
    j_align: float,
    eps_geom: float,
    noise: float,
    drive: float,
    dt: float,
    steps: int,
    burn_in_steps: int,
    sample_stride: int,
    reps_per_point: int,
    seed: int,
    radial_cache: Optional[Dict[str, np.ndarray]],
    device_name: str,
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
    constraint_mode: str = "bond",
    easy_axis_disorder: float = 0.0,
) -> List[Dict[str, object]]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --device cuda") from exc

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    constraint = normalize_constraint_mode(constraint_mode)

    _, src_np, tgt_np, phi_np, weight_np, graph_meta = make_graph(
        n,
        graph_mode=graph_mode,
        graph_radius=graph_radius,
        graph_k=graph_k,
        coupling_decay=coupling_decay,
        bond_disorder=bond_disorder,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        crosslink_k=crosslink_k,
        crosslink_weight=crosslink_weight,
        patch_angle_step=patch_angle_step,
    )
    easy_axes_np = make_easy_axes(
        n,
        graph_mode=graph_mode,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        patch_angle_step=patch_angle_step,
        easy_axis_disorder=easy_axis_disorder,
    )
    src = torch.as_tensor(src_np, dtype=torch.long, device=device)
    tgt = torch.as_tensor(tgt_np, dtype=torch.long, device=device)
    phi = torch.as_tensor(phi_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weight_np, dtype=torch.float32, device=device)
    easy_axes = torch.as_tensor(easy_axes_np, dtype=torch.float32, device=device)
    weight_denom = torch.clamp(weights.sum(), min=1e-6)
    node_count = n * n
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    theta = (2.0 * math.pi) * torch.rand((reps_per_point, node_count), device=device, generator=gen) - math.pi
    sqrt_noise = math.sqrt(2.0 * noise * dt)
    two_pi = 2.0 * math.pi
    defect_supported = graph_meta["graph_mode"] not in {"random", "mosaic"}

    def wrap_angle(x):
        return torch.remainder(x + math.pi, two_pi) - math.pi

    def sample_metrics(theta_cur):
        cos2 = torch.cos(2.0 * theta_cur).mean(dim=1)
        sin2 = torch.sin(2.0 * theta_cur).mean(dim=1)
        s = torch.sqrt(cos2 * cos2 + sin2 * sin2)
        d = theta_cur[:, src] - theta_cur[:, tgt]
        g = theta_cur[:, src] + theta_cur[:, tgt] - 2.0 * phi
        align = torch.cos(2.0 * d)
        if constraint == "grooved":
            local = torch.cos(2.0 * (theta_cur - easy_axes))
            geom = local
        else:
            geom = torch.cos(2.0 * g)
        c2 = (align * weights).sum(dim=1) / weight_denom
        if constraint == "grooved":
            g2 = geom.mean(dim=1)
            fr = ((1.0 - align) * weights).sum(dim=1) / weight_denom + eps_geom * (1.0 - g2)
        else:
            g2 = (geom * weights).sum(dim=1) / weight_denom
            fr = ((1.0 - align + eps_geom * (1.0 - geom)) * weights).sum(dim=1) / weight_denom

        if defect_supported:
            a = wrap_angle(2.0 * theta_cur.reshape(reps_per_point, n, n))
            a00 = a
            a10 = torch.roll(a, shifts=-1, dims=1)
            a11 = torch.roll(a10, shifts=-1, dims=2)
            a01 = torch.roll(a, shifts=-1, dims=2)
            winding = (
                wrap_angle(a10 - a00)
                + wrap_angle(a11 - a10)
                + wrap_angle(a01 - a11)
                + wrap_angle(a00 - a01)
            )
            defects = (torch.abs(torch.round(winding / two_pi)) >= 1).sum(dim=(1, 2)).float()
        else:
            defects = torch.full((reps_per_point,), float("nan"), dtype=torch.float32, device=device)
        return s, g2, c2, fr, defects

    s_samples: List[object] = []
    g_samples: List[object] = []
    c_samples: List[object] = []
    fr_samples: List[object] = []
    defect_samples: List[object] = []
    lag_buffers: Dict[int, List[object]] = {1: [], 5: [], 20: []}
    lag_sums = {lag: torch.zeros(reps_per_point, dtype=torch.float32, device=device) for lag in lag_buffers}
    lag_counts = {lag: 0 for lag in lag_buffers}
    qea_cos_sum = torch.zeros((reps_per_point, node_count), dtype=torch.float32, device=device)
    qea_sin_sum = torch.zeros((reps_per_point, node_count), dtype=torch.float32, device=device)
    qea_count = 0

    total_steps = burn_in_steps + steps
    for step in range(total_steps):
        d = theta[:, src] - theta[:, tgt]
        g = theta[:, src] + theta[:, tgt] - 2.0 * phi
        if constraint == "grooved":
            t_src = weights * (-2.0 * j_align * torch.sin(2.0 * d))
            t_tgt = weights * (+2.0 * j_align * torch.sin(2.0 * d))
        else:
            t_src = weights * (
                -2.0 * j_align * torch.sin(2.0 * d) - 2.0 * eps_geom * j_align * torch.sin(2.0 * g)
            )
            t_tgt = weights * (
                +2.0 * j_align * torch.sin(2.0 * d) - 2.0 * eps_geom * j_align * torch.sin(2.0 * g)
            )
        torque = torch.zeros_like(theta)
        torque.index_add_(1, src, t_src)
        torque.index_add_(1, tgt, t_tgt)
        if constraint == "grooved":
            torque = torque - 2.0 * eps_geom * j_align * torch.sin(2.0 * (theta - easy_axes))
        theta = theta + dt * (torque + drive) + sqrt_noise * torch.randn(theta.shape, device=device, generator=gen)
        theta = wrap_angle(theta)

        if step >= burn_in_steps and (step - burn_in_steps) % sample_stride == 0:
            s, g2, c2, fr, defects = sample_metrics(theta)
            s_samples.append(s.detach().cpu())
            g_samples.append(g2.detach().cpu())
            c_samples.append(c2.detach().cpu())
            fr_samples.append(fr.detach().cpu())
            defect_samples.append(defects.detach().cpu())
            qea_cos_sum += torch.cos(2.0 * theta)
            qea_sin_sum += torch.sin(2.0 * theta)
            qea_count += 1
            for lag, buf in lag_buffers.items():
                if len(buf) >= lag:
                    prev = buf[-lag]
                    lag_sums[lag] += torch.cos(2.0 * (theta - prev)).mean(dim=1)
                    lag_counts[lag] += 1
                buf.append(theta.detach().clone())
                if len(buf) > max(lag_buffers):
                    del buf[0]

    if not s_samples:
        s, g2, c2, fr, defects = sample_metrics(theta)
        s_samples.append(s.detach().cpu())
        g_samples.append(g2.detach().cpu())
        c_samples.append(c2.detach().cpu())
        fr_samples.append(fr.detach().cpu())
        defect_samples.append(defects.detach().cpu())
        qea_cos_sum += torch.cos(2.0 * theta)
        qea_sin_sum += torch.sin(2.0 * theta)
        qea_count += 1

    s_arr = np.stack([x.numpy() for x in s_samples], axis=0)
    g_arr = np.stack([x.numpy() for x in g_samples], axis=0)
    c_arr = np.stack([x.numpy() for x in c_samples], axis=0)
    fr_arr = np.stack([x.numpy() for x in fr_samples], axis=0)
    defect_arr = np.stack([x.numpy() for x in defect_samples], axis=0)

    radial_r: List[float] = []
    radial_c: List[List[float]] = [[] for _ in range(reps_per_point)]
    radial_g: List[List[float]] = [[] for _ in range(reps_per_point)]
    if radial_cache is not None:
        r_centers = radial_cache["r_centers"]
        radial_r = r_centers.astype(float).tolist()
        r_src = torch.as_tensor(radial_cache["src"], dtype=torch.long, device=device)
        r_tgt = torch.as_tensor(radial_cache["tgt"], dtype=torch.long, device=device)
        r_phi = torch.as_tensor(radial_cache["phi"], dtype=torch.float32, device=device)
        r_bin = np.asarray(radial_cache["bin_idx"], dtype=np.int16)
        c_vals = torch.cos(2.0 * (theta[:, r_src] - theta[:, r_tgt])).detach().cpu().numpy()
        if constraint == "grooved":
            r_easy = easy_axes
            g_vals = 0.5 * (
                torch.cos(2.0 * (theta[:, r_src] - r_easy[r_src]))
                + torch.cos(2.0 * (theta[:, r_tgt] - r_easy[r_tgt]))
            ).detach().cpu().numpy()
        else:
            g_vals = torch.cos(2.0 * (theta[:, r_src] + theta[:, r_tgt] - 2.0 * r_phi)).detach().cpu().numpy()
        for b in range(len(radial_r)):
            mask = r_bin == b
            for rep_i in range(reps_per_point):
                if np.any(mask):
                    radial_c[rep_i].append(float(np.mean(c_vals[rep_i, mask])))
                    radial_g[rep_i].append(float(np.mean(g_vals[rep_i, mask])))
                else:
                    radial_c[rep_i].append(float("nan"))
                    radial_g[rep_i].append(float("nan"))

    results: List[Dict[str, object]] = []
    lag_values = {}
    for lag, acc in lag_sums.items():
        if lag_counts[lag] > 0:
            lag_values[lag] = (acc / lag_counts[lag]).detach().cpu().numpy()
        else:
            lag_values[lag] = np.full(reps_per_point, np.nan, dtype=float)
    if qea_count > 0:
        qea_values = (
            (qea_cos_sum / qea_count) ** 2 + (qea_sin_sum / qea_count) ** 2
        ).mean(dim=1).detach().cpu().numpy()
    else:
        qea_values = np.full(reps_per_point, np.nan, dtype=float)

    for rep_i in range(reps_per_point):
        s_series = s_arr[:, rep_i]
        g_series = g_arr[:, rep_i]
        c_series = c_arr[:, rep_i]
        fr_series = fr_arr[:, rep_i]
        defects = defect_arr[:, rep_i]
        results.append(
            {
                "n": n,
                "j_align": j_align,
                "eps_geom": eps_geom,
                "noise": noise,
                "j_over_Dr": float(j_align / noise) if noise > 0.0 else float("nan"),
                "groove_strength": float(eps_geom * j_align) if constraint == "grooved" else float("nan"),
                "groove_over_Dr": float(eps_geom * j_align / noise)
                if constraint == "grooved" and noise > 0.0
                else float("nan"),
                "drive": drive,
                "burn_in_steps": burn_in_steps,
                "sample_steps": steps,
                "sample_stride": sample_stride,
                "sample_count": int(s_series.size),
                "graph_mode": graph_meta["graph_mode"],
                "constraint_mode": constraint,
                "geometry_observable": "groove_lock" if constraint == "grooved" else "bond_frame_lock",
                "easy_axis_disorder": float(easy_axis_disorder),
                "graph_edge_count": graph_meta["edge_count"],
                "graph_mean_degree": graph_meta["mean_degree"],
                "nematic_order": float(np.mean(s_series)),
                "nematic_order_time_std": float(np.std(s_series)),
                "geometry_lock": float(np.mean(g_series)),
                "geometry_lock_time_std": float(np.std(g_series)),
                "orientational_corr_nn": float(np.mean(c_series)),
                "orientational_corr_nn_time_std": float(np.std(c_series)),
                "temporal_corr_lag1": float(lag_values[1][rep_i]),
                "temporal_corr_lag5": float(lag_values[5][rep_i]),
                "temporal_corr_lag20": float(lag_values[20][rep_i]),
                "q_EA": float(qea_values[rep_i]),
                "edge_frustration": float(np.mean(fr_series)),
                "edge_frustration_time_std": float(np.std(fr_series)),
                "defect_count": float(np.nanmean(defects)) if np.any(np.isfinite(defects)) else float("nan"),
                "defect_count_time_std": float(np.nanstd(defects)) if np.any(np.isfinite(defects)) else float("nan"),
                "defect_density": float(np.nanmean(defects) / node_count) if np.any(np.isfinite(defects)) else float("nan"),
                "defect_density_time_std": float(np.nanstd(defects / node_count)) if np.any(np.isfinite(defects)) else float("nan"),
                "susceptibility_S": susceptibility(s_series, node_count),
                "susceptibility_G": susceptibility(g_series, node_count),
                "binder_S": binder_cumulant(s_series),
                "binder_G": binder_cumulant(g_series),
                "radial_r": radial_r,
                "radial_C2": radial_c[rep_i] if radial_r else [],
                "radial_G2": radial_g[rep_i] if radial_r else [],
            }
        )
    return results


def run_sweep(
    out_dir: Path,
    *,
    n: int = 24,
    sizes: Optional[List[int]] = None,
    noise: float = 0.45,
    dt: float = 0.015,
    steps: int = 2600,
    burn_in_steps: int = 0,
    reps_per_point: int = 4,
    refined: bool = False,
    maximal: bool = False,
    eps_values_override: Optional[List[float]] = None,
    j_values_override: Optional[List[float]] = None,
    radial_bins: int = 0,
    radial_max_pairs: int = 250_000,
    sample_stride: int = 20,
    resume: bool = True,
    progress_every: int = 10,
    device_name: str = "cpu",
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
    constraint_mode: str = "bond",
    easy_axis_disorder: float = 0.0,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    drive = 0.0
    if eps_values_override is not None:
        eps_values = eps_values_override
    elif maximal:
        eps_values = np.linspace(0.0, 1.0, 41).round(6).tolist()
    elif refined:
        eps_values = np.linspace(0.45, 1.00, 12).round(4).tolist()
    else:
        eps_values = [0.0, 0.15, 0.35, 0.60, 0.90]

    if j_values_override is not None:
        j_values = np.asarray(j_values_override, dtype=float)
    elif maximal:
        j_values = np.linspace(0.05, 0.85, 61)
    elif refined:
        j_values = np.linspace(0.32, 0.78, 19)
    else:
        j_values = np.linspace(0.05, 0.75, 11)

    scan_sizes = sizes if sizes else [n]
    if maximal and not sizes:
        scan_sizes = [16, 24, 32, 48]
    if maximal and radial_bins == 0:
        radial_bins = 12

    if maximal:
        stem = "rotating_colloids_maximal"
    elif refined:
        stem = "rotating_colloids_refined"
    else:
        stem = "rotating_colloids_hyperion_case"

    points_path = out_dir / f"{stem}_points.jsonl"
    completed = read_completed_points(points_path) if resume else {}
    rows: List[Dict[str, object]] = list(completed.values())

    total_points = len(scan_sizes) * len(eps_values) * len(j_values)
    start_time = time.time()
    done_now = 0
    mode = "maximal" if maximal else "refined" if refined else "coarse"
    graph_tag = graph_id(
        graph_mode,
        graph_radius,
        graph_k,
        coupling_decay,
        bond_disorder,
        graph_seed,
        cluster_size,
        crosslink_k,
        crosslink_weight,
        patch_angle_step,
        constraint_mode,
        easy_axis_disorder,
    )
    constraint = normalize_constraint_mode(constraint_mode)
    print(
        json.dumps(
            {
                "event": "scan_start",
                "mode": mode,
                "sizes": scan_sizes,
                "points": total_points,
                "completed_resume": len(completed),
                "reps_per_point": reps_per_point,
                "burn_in_steps": burn_in_steps,
                "sample_steps": steps,
                "radial_bins": radial_bins,
                "graph_mode": graph_mode,
                "graph_radius": graph_radius,
                "graph_k": graph_k,
                "coupling_decay": coupling_decay,
                "bond_disorder": bond_disorder,
                "cluster_size": cluster_size,
                "crosslink_k": crosslink_k,
                "crosslink_weight": crosslink_weight,
                "patch_angle_step": patch_angle_step,
                "constraint_mode": constraint,
                "easy_axis_disorder": easy_axis_disorder,
            }
        ),
        flush=True,
    )

    point_mode = "a" if resume else "w"
    with points_path.open(point_mode, encoding="utf-8") as point_file:
        for n_cur in scan_sizes:
            radial_cache = (
                make_radial_pair_cache(
                    n_cur,
                    n_bins=radial_bins,
                    max_pairs=radial_max_pairs,
                    seed=777 + n_cur,
                    graph_mode=graph_mode,
                    graph_radius=graph_radius,
                    graph_k=graph_k,
                    coupling_decay=coupling_decay,
                    bond_disorder=bond_disorder,
                    graph_seed=graph_seed,
                    cluster_size=cluster_size,
                    crosslink_k=crosslink_k,
                    crosslink_weight=crosslink_weight,
                    patch_angle_step=patch_angle_step,
                )
                if radial_bins > 0
                else None
            )
            for eps in eps_values:
                for j in j_values:
                    eps_f = float(eps)
                    j_f = float(j)
                    key = point_key(n_cur, eps_f, j_f) + graph_tag
                    if key in completed:
                        expected_config = {
                            "noise": float(noise),
                            "dt": float(dt),
                            "burn_in_steps": int(burn_in_steps),
                            "sample_steps": int(steps),
                            "sample_stride": int(sample_stride),
                            "replicates": int(reps_per_point),
                            "constraint_mode": constraint,
                        }
                        mismatches = resume_config_mismatches(completed[key], expected_config)
                        if mismatches:
                            raise RuntimeError(
                                f"Refusing to resume incompatible point {key}: "
                                + "; ".join(mismatches)
                                + ". Use a new output directory or --no-resume to replace the checkpoint."
                            )
                        continue
                    reps = []
                    base_seed = 1_000_003 + 100_003 * n_cur + int(10_000 * eps_f) + int(100_000 * j_f)
                    if device_name != "cpu":
                        reps = simulate_torch_replicates(
                            n=n_cur,
                            j_align=j_f,
                            eps_geom=eps_f,
                            noise=noise,
                            drive=drive,
                            dt=dt,
                            steps=steps,
                            burn_in_steps=burn_in_steps,
                            sample_stride=sample_stride,
                            reps_per_point=reps_per_point,
                            seed=base_seed,
                            radial_cache=radial_cache,
                            device_name=device_name,
                            graph_mode=graph_mode,
                            graph_radius=graph_radius,
                            graph_k=graph_k,
                            coupling_decay=coupling_decay,
                            bond_disorder=bond_disorder,
                            graph_seed=graph_seed,
                            cluster_size=cluster_size,
                            crosslink_k=crosslink_k,
                            crosslink_weight=crosslink_weight,
                            patch_angle_step=patch_angle_step,
                            constraint_mode=constraint,
                            easy_axis_disorder=easy_axis_disorder,
                        )
                    else:
                        for k in range(reps_per_point):
                            seed = base_seed + 10_007 * k
                            reps.append(
                                simulate(
                                    n=n_cur,
                                    j_align=j_f,
                                    eps_geom=eps_f,
                                    noise=noise,
                                    drive=drive,
                                    dt=dt,
                                    steps=steps,
                                    burn_in_steps=burn_in_steps,
                                    sample_stride=sample_stride,
                                    seed=seed,
                                    radial_cache=radial_cache,
                                    graph_mode=graph_mode,
                                    graph_radius=graph_radius,
                                    graph_k=graph_k,
                                    coupling_decay=coupling_decay,
                                    bond_disorder=bond_disorder,
                                    graph_seed=graph_seed,
                                    cluster_size=cluster_size,
                                    crosslink_k=crosslink_k,
                                    crosslink_weight=crosslink_weight,
                                    patch_angle_step=patch_angle_step,
                                    constraint_mode=constraint,
                                    easy_axis_disorder=easy_axis_disorder,
                                )
                            )
                    row = summarize_replicates(n=n_cur, eps=eps_f, j=j_f, reps=reps)
                    row["point_key"] = key
                    row["scan_mode"] = mode
                    row["noise"] = noise
                    row["dt"] = dt
                    row["burn_in_steps"] = burn_in_steps
                    row["sample_steps"] = steps
                    row["sample_stride"] = sample_stride
                    row["radial_bins"] = radial_bins
                    row["graph_mode"] = normalize_graph_mode(graph_mode)
                    row["graph_radius"] = graph_radius
                    row["graph_k"] = graph_k
                    row["coupling_decay"] = coupling_decay
                    row["bond_disorder"] = bond_disorder
                    row["graph_seed"] = graph_seed
                    row["cluster_size"] = cluster_size
                    row["crosslink_k"] = crosslink_k
                    row["crosslink_weight"] = crosslink_weight
                    row["patch_angle_step"] = patch_angle_step
                    row["constraint_mode"] = constraint
                    row["geometry_observable"] = "groove_lock" if constraint == "grooved" else "bond_frame_lock"
                    row["easy_axis_disorder"] = float(easy_axis_disorder)
                    rows.append(row)
                    point_file.write(json.dumps(json_ready(row), allow_nan=False) + "\n")
                    point_file.flush()
                    done_now += 1
                    if progress_every > 0 and (done_now % progress_every == 0 or done_now == total_points):
                        elapsed = time.time() - start_time
                        print(
                            json.dumps(
                                {
                                    "event": "scan_progress",
                                    "new_points": done_now,
                                    "total_points": total_points,
                                    "elapsed_sec": round(elapsed, 2),
                                    "last_point": key,
                                }
                            ),
                            flush=True,
                        )

    rows = sorted(rows, key=lambda r: (int(r.get("n", n)), float(r["eps_geom"]), float(r["j_align"])))
    best = max(rows, key=lambda r: r["hidden_geometry_score"])

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            json_ready({
                "scan": {
                    "sizes": scan_sizes,
                    "noise": noise,
                    "dt": dt,
                    "burn_in_steps": burn_in_steps,
                    "sample_steps": steps,
                    "sample_stride": sample_stride,
                    "reps_per_point": reps_per_point,
                    "refined": refined,
                    "maximal": maximal,
                    "radial_bins": radial_bins,
                    "device": device_name,
                    "radial_max_pairs": radial_max_pairs,
                    "point_jsonl": str(points_path),
                    "graph_mode": graph_mode,
                    "graph_radius": graph_radius,
                    "graph_k": graph_k,
                    "coupling_decay": coupling_decay,
                    "bond_disorder": bond_disorder,
                    "graph_seed": graph_seed,
                    "cluster_size": cluster_size,
                    "crosslink_k": crosslink_k,
                    "crosslink_weight": crosslink_weight,
                    "patch_angle_step": patch_angle_step,
                    "constraint_mode": constraint,
                    "geometry_observable": "groove_lock" if constraint == "grooved" else "bond_frame_lock",
                    "easy_axis_disorder": float(easy_axis_disorder),
                },
                "rows": rows,
                "best_hidden_geometry_regime": best,
            }),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    plot_paths: List[str] = []
    for n_cur in scan_sizes:
        rows_n = [r for r in rows if int(r.get("n", n_cur)) == n_cur]
        if not rows_n:
            continue
        eps_n = sorted({float(r["eps_geom"]) for r in rows_n})
        j_n = np.asarray(sorted({float(r["j_align"]) for r in rows_n}), dtype=float)
        suffix = f"_n{n_cur}" if len(scan_sizes) > 1 else ""
        plot_path = out_dir / f"{stem}{suffix}_phase_scan.png"
        plot_phase_scan(rows_n, eps_n, j_n, plot_path, refined=(refined or maximal))
        plot_paths.append(str(plot_path))

    md_path = out_dir / f"{stem}.md"
    md_plot = Path(plot_paths[-1]).name if plot_paths else ""
    md_path.write_text(build_markdown(best, json_path.name, md_plot), encoding="utf-8")

    return {"json": str(json_path), "jsonl": str(points_path), "markdown": str(md_path), "plots": plot_paths, "best": best}


def run_hysteresis(
    out_dir: Path,
    *,
    n: int,
    sizes: Optional[List[int]],
    eps_values: Optional[List[float]],
    j_values: Optional[List[float]],
    noise: float,
    dt: float,
    steps: int,
    burn_in_steps: int,
    reps_per_path: int,
    sample_stride: int,
    radial_bins: int,
    radial_max_pairs: int,
    graph_mode: str = "square",
    graph_radius: float = 0.0,
    graph_k: int = 6,
    coupling_decay: float = 0.0,
    bond_disorder: float = 0.0,
    graph_seed: int = 12345,
    cluster_size: int = 6,
    crosslink_k: int = 2,
    crosslink_weight: float = 0.25,
    patch_angle_step: float = math.pi / 4.0,
    constraint_mode: str = "bond",
    easy_axis_disorder: float = 0.0,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_sizes = sizes if sizes else [n]
    eps_scan = eps_values if eps_values else [0.45, 0.60, 0.75, 0.90, 1.00]
    j_scan = np.asarray(j_values if j_values else np.linspace(0.25, 0.85, 25), dtype=float)
    stem = "rotating_colloids_hysteresis"
    constraint = normalize_constraint_mode(constraint_mode)
    jsonl_path = out_dir / f"{stem}_paths.jsonl"
    rows: List[Dict[str, object]] = []
    start_time = time.time()

    with jsonl_path.open("w", encoding="utf-8") as f:
        for n_cur in scan_sizes:
            radial_cache = (
                make_radial_pair_cache(
                    n_cur,
                    n_bins=radial_bins,
                    max_pairs=radial_max_pairs,
                    seed=991 + n_cur,
                    graph_mode=graph_mode,
                    graph_radius=graph_radius,
                    graph_k=graph_k,
                    coupling_decay=coupling_decay,
                    bond_disorder=bond_disorder,
                    graph_seed=graph_seed,
                    cluster_size=cluster_size,
                    crosslink_k=crosslink_k,
                    crosslink_weight=crosslink_weight,
                    patch_angle_step=patch_angle_step,
                )
                if radial_bins > 0
                else None
            )
            for eps in eps_scan:
                up_by_j: Dict[float, List[Dict[str, object]]] = {float(j): [] for j in j_scan}
                down_by_j: Dict[float, List[Dict[str, object]]] = {float(j): [] for j in j_scan}
                for rep in range(reps_per_path):
                    theta: Optional[np.ndarray] = None
                    for j in j_scan:
                        sim = simulate(
                            n=n_cur,
                            j_align=float(j),
                            eps_geom=float(eps),
                            noise=noise,
                            drive=0.0,
                            dt=dt,
                            steps=steps,
                            burn_in_steps=burn_in_steps,
                            sample_stride=sample_stride,
                            seed=3_000_001 + 101 * rep + int(1000 * n_cur) + int(10_000 * eps) + int(100_000 * float(j)),
                            radial_cache=radial_cache,
                            initial_theta=theta,
                            return_final_theta=True,
                            graph_mode=graph_mode,
                            graph_radius=graph_radius,
                            graph_k=graph_k,
                            coupling_decay=coupling_decay,
                            bond_disorder=bond_disorder,
                            graph_seed=graph_seed,
                            cluster_size=cluster_size,
                            crosslink_k=crosslink_k,
                            crosslink_weight=crosslink_weight,
                            patch_angle_step=patch_angle_step,
                            constraint_mode=constraint,
                            easy_axis_disorder=easy_axis_disorder,
                        )
                        theta = np.asarray(sim.pop("final_theta"))
                        up_by_j[float(j)].append(sim)
                    for j in reversed(j_scan):
                        sim = simulate(
                            n=n_cur,
                            j_align=float(j),
                            eps_geom=float(eps),
                            noise=noise,
                            drive=0.0,
                            dt=dt,
                            steps=steps,
                            burn_in_steps=0,
                            sample_stride=sample_stride,
                            seed=4_000_001 + 101 * rep + int(1000 * n_cur) + int(10_000 * eps) + int(100_000 * float(j)),
                            radial_cache=radial_cache,
                            initial_theta=theta,
                            return_final_theta=True,
                            graph_mode=graph_mode,
                            graph_radius=graph_radius,
                            graph_k=graph_k,
                            coupling_decay=coupling_decay,
                            bond_disorder=bond_disorder,
                            graph_seed=graph_seed,
                            cluster_size=cluster_size,
                            crosslink_k=crosslink_k,
                            crosslink_weight=crosslink_weight,
                            patch_angle_step=patch_angle_step,
                            constraint_mode=constraint,
                            easy_axis_disorder=easy_axis_disorder,
                        )
                        theta = np.asarray(sim.pop("final_theta"))
                        down_by_j[float(j)].append(sim)

                for direction, bucket in (("up", up_by_j), ("down", down_by_j)):
                    for j, reps in bucket.items():
                        row = summarize_replicates(n=n_cur, eps=float(eps), j=float(j), reps=reps)
                        row["direction"] = direction
                        row["scan_mode"] = "hysteresis"
                        row["graph_mode"] = normalize_graph_mode(graph_mode)
                        row["graph_radius"] = graph_radius
                        row["graph_k"] = graph_k
                        row["coupling_decay"] = coupling_decay
                        row["bond_disorder"] = bond_disorder
                        row["graph_seed"] = graph_seed
                        row["cluster_size"] = cluster_size
                        row["crosslink_k"] = crosslink_k
                        row["crosslink_weight"] = crosslink_weight
                        row["patch_angle_step"] = patch_angle_step
                        row["constraint_mode"] = constraint
                        row["geometry_observable"] = "groove_lock" if constraint == "grooved" else "bond_frame_lock"
                        row["easy_axis_disorder"] = float(easy_axis_disorder)
                        rows.append(row)
                        f.write(json.dumps(json_ready(row), allow_nan=False) + "\n")
                print(
                    json.dumps(
                        {
                            "event": "hysteresis_progress",
                            "n": n_cur,
                            "eps": eps,
                            "elapsed_sec": round(time.time() - start_time, 2),
                        }
                    ),
                    flush=True,
                )

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            json_ready({
                "scan": {
                    "sizes": scan_sizes,
                    "eps_values": eps_scan,
                    "j_values": j_scan.astype(float).tolist(),
                    "noise": noise,
                    "dt": dt,
                    "burn_in_steps": burn_in_steps,
                    "sample_steps": steps,
                    "reps_per_path": reps_per_path,
                    "radial_bins": radial_bins,
                    "graph_mode": graph_mode,
                    "graph_radius": graph_radius,
                    "graph_k": graph_k,
                    "coupling_decay": coupling_decay,
                    "bond_disorder": bond_disorder,
                    "graph_seed": graph_seed,
                    "cluster_size": cluster_size,
                    "crosslink_k": crosslink_k,
                    "crosslink_weight": crosslink_weight,
                    "patch_angle_step": patch_angle_step,
                    "constraint_mode": constraint,
                    "geometry_observable": "groove_lock" if constraint == "grooved" else "bond_frame_lock",
                    "easy_axis_disorder": float(easy_axis_disorder),
                },
                "rows": rows,
            }),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    plot_path = out_dir / f"{stem}_loops.png"
    plot_hysteresis(rows, plot_path)
    return {"json": str(json_path), "jsonl": str(jsonl_path), "plot": str(plot_path)}


def grid(rows: List[dict], eps_values: List[float], j_values: np.ndarray, key: str) -> np.ndarray:
    out = np.zeros((len(eps_values), len(j_values)))
    lookup = {(r["eps_geom"], round(r["j_align"], 10)): r for r in rows}
    for i, eps in enumerate(eps_values):
        for j, val in enumerate(j_values):
            out[i, j] = lookup[(eps, round(float(val), 10))][key]
    return out


def plot_phase_scan(rows: List[dict], eps_values: List[float], j_values: np.ndarray, path: Path, *, refined: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    if refined:
        panels = [
            ("nematic_order_mean", "global nematic order S"),
            ("orientational_corr_nn_mean", "nearest-neighbor C2(a)"),
            ("geometry_lock_mean", "bond-geometry G2(a)"),
            ("geometry_without_global_order", "(G2 - S)+ * C2"),
        ]
    else:
        panels = [
            ("nematic_order_mean", "global nematic order"),
            ("orientational_corr_nn_mean", "nearest-neighbor C2"),
            ("geometry_lock_mean", "bond-geometry locking"),
            ("hidden_geometry_score", "hidden-geometry score"),
        ]
    for ax, (key, title) in zip(axes.ravel(), panels):
        im = ax.imshow(
            grid(rows, eps_values, j_values, key),
            origin="lower",
            aspect="auto",
            extent=[j_values.min(), j_values.max(), min(eps_values), max(eps_values)],
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("alignment strength J / noise")
        ax.set_ylabel("geometry embodiment epsilon")
        pocket = grid(rows, eps_values, j_values, "hidden_pocket_flag")
        if np.any(pocket > 0):
            ax.contour(
                j_values,
                eps_values,
                pocket,
                levels=[0.5],
                colors="white",
                linewidths=1.4,
            )
        fig.colorbar(im, ax=ax, shrink=0.85)
    if refined:
        title = "Refined hidden-geometry scan: white contour = G2,C2 >= 0.70 and S <= 0.35"
    else:
        title = "Fixed-center rotating elongated colloids: orientation-only kernel vs geometry closure"
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_hysteresis(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    sizes = sorted({int(r["n"]) for r in rows})
    eps_values = sorted({float(r["eps_geom"]) for r in rows})
    fig, axes = plt.subplots(len(sizes), 2, figsize=(11, 4 * len(sizes)), squeeze=False, constrained_layout=True)
    for row_i, n in enumerate(sizes):
        for eps in eps_values:
            for direction, style in (("up", "-"), ("down", "--")):
                sub = [
                    r
                    for r in rows
                    if int(r["n"]) == n and float(r["eps_geom"]) == eps and r.get("direction") == direction
                ]
                if not sub:
                    continue
                sub = sorted(sub, key=lambda r: float(r["j_align"]))
                js = [float(r["j_align"]) for r in sub]
                axes[row_i, 0].plot(js, [float(r["geometry_lock_mean"]) for r in sub], style, label=f"eps={eps:g} {direction}")
                axes[row_i, 1].plot(js, [float(r["nematic_order_mean"]) for r in sub], style, label=f"eps={eps:g} {direction}")
        axes[row_i, 0].set_title(f"n={n}: bond-geometry G2(a)")
        axes[row_i, 1].set_title(f"n={n}: global nematic S")
        for ax in axes[row_i]:
            ax.set_xlabel("alignment strength J / noise")
            ax.set_ylabel("order/correlation")
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    fig.suptitle("Rotating-colloid hysteresis: solid=up sweep, dashed=down sweep", fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_markdown(best: dict, json_name: str, plot_name: str) -> str:
    best_json = json.dumps(json_ready(best), indent=2, allow_nan=False)
    template = r"""# Rotating Fixed-Center Colloids: Hyperion Theory-Construction Case

## Sparse-Attention Reading

The relevant Hyperion rule is:

```text
valid transferable theory = operator role + flow role + closure role + boundary/geometry role
```

For elongated colloids whose centers are fixed, the translational geometry is frozen.  The theory should therefore not start from particle transport.  It should start from an orientation-only transport law on a fixed graph.

## Minimal Kernel

Let each particle have a center `r_i` and angle `theta_i`.  The center is fixed, so the only state variable is orientation:

```math
\dot{\theta}_i
= -\mu_r \frac{\partial U}{\partial \theta_i}
+ \Omega_i(t)
+ \sqrt{2D_r}\,\eta_i(t).
```

The corresponding rotational Smoluchowski equation, equivalently the angular
Fokker-Planck equation on the orientation torus, is:

```math
\partial_t P
= \sum_i \partial_{\theta_i}
\left[
D_r\partial_{\theta_i}P
+ \mu_r P\,\partial_{\theta_i}U
- \Omega_i P
\right].
```

In flux form:

```math
\partial_t P = -\sum_i\partial_{\theta_i}J_i,
\qquad
J_i =
-D_r\partial_{\theta_i}P
-\mu_rP\,\partial_{\theta_i}U
+\Omega_iP.
```

## Hyperion Addition

The A09-like kernel is the geometry-free nematic operator:

```math
U_0 =
-\sum_{ij} J_{ij}\cos 2(\theta_i-\theta_j).
```

The A06/A08-style re-embodiment term is the fixed-center geometry closure:

```math
U_g =
-\epsilon\sum_{ij} J_{ij}
\cos 2(\theta_i+\theta_j-2\phi_{ij}),
```

where `phi_ij` is the bond angle between fixed particle centers.

This term is the experiment's crucial object: it lets a pure orientation kernel become a geometry-bearing theory without allowing center-of-mass motion.

## Correlators

The experimental observable is a correlator, not only an order parameter.  The
simulation therefore reports three correlation channels:

```math
C_2(a)=
\left\langle\cos 2(\theta_i-\theta_j)\right\rangle_{\langle ij\rangle},
```

```math
G_2(a)=
\left\langle
\cos 2(\theta_i+\theta_j-2\phi_{ij})
\right\rangle_{\langle ij\rangle},
```

and the late-time rotational autocorrelator

```math
C_2(\tau)=
\left\langle
\cos 2(\theta_i(t+\tau)-\theta_i(t))
\right\rangle_{i,t}.
```

The spin-glass-like diagnostic is the nematic Edwards-Anderson memory

```math
q_{\rm EA}
=
\frac{1}{N}
\sum_i
\left|
\left\langle e^{2i\theta_i(t)}\right\rangle_t
\right|^2.
```

`C_2(a)` asks whether neighboring rods align with one another.  `G_2(a)` asks
whether neighboring rods align with the bond geometry.  The surprising phase
is where `G_2(a)` is large even when global nematic order is modest.  The
spin-glass-like phase is where `S` is low, but `q_EA` stays high.

## Simulation Result

The code scanned alignment strength and geometry embodiment.  The selected
cell below maximizes the hidden-geometry score used by the scan; it must be
interpreted together with the sparse-attention gate report, because a
frustrated control can score highly by suppressing global order even when it
does not form a true geometry-locked phase.

```json
__BEST_JSON__
```

![phase scan](__PLOT_NAME__)

## Striking Experimental Idea

Build a lattice of optically, magnetically, or lithographically pinned elongated colloids.  Let every colloid rotate freely but keep every center fixed.  Then tune the interaction from mostly orientation-alignment to geometry-embodied coupling by changing one physical control:

- particle aspect ratio;
- gap between rods;
- salt concentration or electrostatic screening length;
- wall distance for hydrodynamic coupling;
- rotating magnetic/electric field amplitude;
- boundary pattern around the array.

The prediction is not just global nematic ordering.  The striking regime is:

```text
moderate global nematic order
but strong bond-geometry locking
and persistent orientational defects
```

That would mean the experiment has created a phase transition in orientation space while the particles do not translate.

## What To Measure

1. Track `theta_i(t)` for all rods.
2. Compute global nematic order:

```math
S = \left|N^{-1}\sum_i e^{2i\theta_i}\right|.
```

3. Compute bond-geometry locking:

```math
G =
\left\langle
\cos 2(\theta_i+\theta_j-2\phi_{ij})
\right\rangle_{ij}.
```

4. Count director defects from the doubled angle field `2 theta_i`.
5. Measure the time autocorrelator `C_2(tau)` from `theta_i(t)`.
6. Measure `q_EA` to distinguish a fast rotational liquid from a frozen
   nematic glass.
7. Sweep drive amplitude, rod spacing, and screening length.

## Discovery Claim

If `G` rises while `S` remains modest, the system has found a hidden phase:

```text
not ordinary flocking,
not aggregation,
not particle transport,
but geometry-locked angular matter on a fixed graph.
```

This is the clean experiment suggested by Hyperion: isolate the theory kernel by removing translation, then test whether geometry reappears only as a closure term in angular Fokker-Planck flow.
"""
    return template.replace("__BEST_JSON__", best_json).replace("__PLOT_NAME__", plot_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="discoveries/theory_experiment_interface/rotating_colloids_hyperion")
    parser.add_argument("--refined", action="store_true", help="scan the high-geometry region with a denser grid")
    parser.add_argument("--maximal", action="store_true", help="publication-scale scan with finite-size, radial correlator, checkpointed JSONL output")
    parser.add_argument("--hysteresis", action="store_true", help="run up/down sweeps carrying final angle states forward")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--sizes", default="", help="comma-separated lattice sizes for finite-size scans, e.g. 16,24,32,48")
    parser.add_argument("--noise", type=float, default=0.45)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc. CUDA batches replicas with PyTorch")
    parser.add_argument("--steps", type=int, default=None, help="sampling steps after burn-in")
    parser.add_argument("--burn-in-steps", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--sample-stride", type=int, default=20)
    parser.add_argument("--radial-bins", type=int, default=None)
    parser.add_argument("--radial-max-pairs", type=int, default=250_000)
    parser.add_argument(
        "--graph-mode",
        default="square",
        choices=["square", "triangular", "long-range", "random", "mosaic", "patchy"],
        help="pinned-center graph: square default, triangular/non-bipartite, long-range square, random geometric, or rotated mosaic domains. 'patchy' is accepted only as an old alias.",
    )
    parser.add_argument("--graph-radius", type=float, default=0.0, help="edge radius for long-range/random modes; mode defaults are used when 0")
    parser.add_argument("--graph-k", type=int, default=6, help="k-nearest neighbors for --graph-mode random when radius is 0")
    parser.add_argument("--coupling-decay", type=float, default=0.0, help="if >0, use J(r)=exp[-(r-r_min)/decay] on included edges")
    parser.add_argument("--bond-disorder", type=float, default=0.0, help="log-normal quenched disorder strength applied to J_ij")
    parser.add_argument("--graph-seed", type=int, default=12345, help="seed for random pinned centers and quenched bond disorder")
    parser.add_argument("--cluster-size", type=int, default=6, help="domain side length for --graph-mode mosaic")
    parser.add_argument("--crosslink-k", type=int, default=2, help="weak nearest cross-links per neighboring mosaic-domain pair")
    parser.add_argument("--crosslink-weight", type=float, default=0.25, help="relative weight of inter-domain cross-links in mosaic mode")
    parser.add_argument(
        "--domain-angle-step",
        "--patch-angle-step",
        dest="patch_angle_step",
        type=float,
        default=math.pi / 4.0,
        help="rotation increment between mosaic-domain orientations, in radians",
    )
    parser.add_argument(
        "--constraint-mode",
        default="bond",
        choices=["bond", "pair", "pair-bond", "bond-frame", "grooved", "groove", "grooves", "easy-axis", "substrate"],
        help="geometry mechanism: pair/bond-frame constraint from the recovered model, or grooved local surface easy axes",
    )
    parser.add_argument(
        "--easy-axis-disorder",
        type=float,
        default=0.0,
        help="quenched angular disorder of local groove axes in radians for --constraint-mode grooved",
    )
    parser.add_argument("--eps-values", default="", help="explicit comma-separated epsilon values")
    parser.add_argument("--j-values", default="", help="explicit comma-separated J/noise values")
    parser.add_argument("--eps-min", type=float, default=None)
    parser.add_argument("--eps-max", type=float, default=None)
    parser.add_argument("--eps-count", type=int, default=None)
    parser.add_argument("--j-min", type=float, default=None)
    parser.add_argument("--j-max", type=float, default=None)
    parser.add_argument("--j-count", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true", help="ignore existing point JSONL checkpoints")
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.maximal:
        steps = args.steps if args.steps is not None else 15_000
        burn_in_steps = args.burn_in_steps if args.burn_in_steps is not None else 5_000
        reps = args.reps if args.reps is not None else 16
        radial_bins = args.radial_bins if args.radial_bins is not None else 12
    else:
        steps = args.steps if args.steps is not None else 2_600
        burn_in_steps = args.burn_in_steps if args.burn_in_steps is not None else 0
        reps = args.reps if args.reps is not None else 4
        radial_bins = args.radial_bins if args.radial_bins is not None else 0

    sizes = parse_int_list(args.sizes) if args.sizes else None
    eps_values = parse_float_list(args.eps_values) if args.eps_values else None
    j_values = parse_float_list(args.j_values) if args.j_values else None
    if eps_values is None and args.eps_count:
        eps_values = np.linspace(
            0.0 if args.eps_min is None else args.eps_min,
            1.0 if args.eps_max is None else args.eps_max,
            args.eps_count,
        ).round(6).tolist()
    if j_values is None and args.j_count:
        j_values = np.linspace(
            0.05 if args.j_min is None else args.j_min,
            0.85 if args.j_max is None else args.j_max,
            args.j_count,
        ).round(6).tolist()

    if args.hysteresis:
        result = run_hysteresis(
            Path(args.output_dir),
            n=args.n,
            sizes=sizes,
            eps_values=eps_values,
            j_values=j_values,
            noise=args.noise,
            dt=args.dt,
            steps=steps,
            burn_in_steps=burn_in_steps,
            reps_per_path=reps,
            sample_stride=args.sample_stride,
            radial_bins=radial_bins,
            radial_max_pairs=args.radial_max_pairs,
            graph_mode=args.graph_mode,
            graph_radius=args.graph_radius,
            graph_k=args.graph_k,
            coupling_decay=args.coupling_decay,
            bond_disorder=args.bond_disorder,
            graph_seed=args.graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=args.patch_angle_step,
            constraint_mode=args.constraint_mode,
            easy_axis_disorder=args.easy_axis_disorder,
        )
    else:
        result = run_sweep(
            Path(args.output_dir),
            n=args.n,
            sizes=sizes,
            steps=steps,
            burn_in_steps=burn_in_steps,
            reps_per_point=reps,
            refined=args.refined,
            maximal=args.maximal,
            eps_values_override=eps_values,
            j_values_override=j_values,
            noise=args.noise,
            dt=args.dt,
            radial_bins=radial_bins,
            radial_max_pairs=args.radial_max_pairs,
            sample_stride=args.sample_stride,
            resume=not args.no_resume,
            progress_every=args.progress_every,
            device_name=args.device,
            graph_mode=args.graph_mode,
            graph_radius=args.graph_radius,
            graph_k=args.graph_k,
            coupling_decay=args.coupling_decay,
            bond_disorder=args.bond_disorder,
            graph_seed=args.graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=args.patch_angle_step,
            constraint_mode=args.constraint_mode,
            easy_axis_disorder=args.easy_axis_disorder,
        )
    print(json.dumps(json_ready(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
