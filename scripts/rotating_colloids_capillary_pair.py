#!/usr/bin/env python3
"""Capillary pair model for caged interfacial colloidal ellipsoids.

The far-field quadrupolar capillary interaction between two ellipsoids at a
fluid interface has the angular form

    -g(r) cos[2(theta_i + theta_j - 2 phi_ij)],  g(r) proportional to r^-4.

This script tests that physical realization on a quenched, disordered
monolayer without prescribed easy axes or hand-built orientational domains.
It writes parameter-scan summaries and dynamical diagnostics suitable for a
claim audit: independent-replica overlap, split-replica persistence, aging,
and write-release-read retention.

All energies are measured in k_B T and time in D_r^-1.  Consequently the
dimensionless overdamped dynamics is

    d theta_i = -partial_(theta_i) U dt + sqrt(2 dt) dW_i.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CagedGraph:
    positions: np.ndarray
    box: np.ndarray
    src: np.ndarray
    tgt: np.ndarray
    phi: np.ndarray
    distance: np.ndarray
    capillary_weight: np.ndarray
    alignment_weight: np.ndarray
    metadata: Dict[str, object]


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def make_caged_graph(
    n: int,
    *,
    disorder: float,
    cutoff: float,
    alignment_range: float,
    alignment_decay: float,
    seed: int,
    shuffle_bond_frames: bool = False,
) -> CagedGraph:
    """Construct a periodically caged, disordered triangular monolayer.

    ``n`` must be even so that the staggered triangular rows close under the
    rectangular periodic boundary.  Positional disorder is a Gaussian
    displacement in units of the undistorted nearest-neighbor spacing.
    """

    if n < 4 or n % 2:
        raise ValueError("n must be an even integer >= 4")
    if not 0.0 <= disorder <= 0.28:
        raise ValueError("disorder must lie in [0, 0.28] to avoid overlaps")
    if cutoff <= 1.0:
        raise ValueError("cutoff must exceed one nearest-neighbor spacing")

    rng = np.random.default_rng(seed)
    dy = math.sqrt(3.0) / 2.0
    box = np.asarray([float(n), float(n) * dy], dtype=float)
    positions = np.empty((n * n, 2), dtype=float)
    k = 0
    for row in range(n):
        for col in range(n):
            positions[k, 0] = float(col) + 0.5 * float(row % 2)
            positions[k, 1] = float(row) * dy
            k += 1
    if disorder:
        positions += rng.normal(scale=float(disorder), size=positions.shape)
    positions %= box

    pairs = cKDTree(positions, boxsize=box).query_pairs(float(cutoff), output_type="ndarray")
    if pairs.size == 0:
        raise RuntimeError("caged graph has no edges")
    src = pairs[:, 0].astype(np.int64)
    tgt = pairs[:, 1].astype(np.int64)
    delta = _minimum_image(positions[tgt] - positions[src], box)
    distance = np.linalg.norm(delta, axis=1)

    # Very rare near-overlaps can result from the independent displacement.
    # Reject them rather than allowing a divergent r^-4 weight.
    keep = distance >= 0.55
    src = src[keep]
    tgt = tgt[keep]
    delta = delta[keep]
    distance = distance[keep]
    phi = np.arctan2(delta[:, 1], delta[:, 0])

    nearest = np.full(n * n, np.inf, dtype=float)
    np.minimum.at(nearest, src, distance)
    np.minimum.at(nearest, tgt, distance)
    finite_nearest = nearest[np.isfinite(nearest)]
    r0 = float(np.median(finite_nearest))
    if not math.isfinite(r0) or r0 <= 0.0:
        raise RuntimeError("could not determine nearest-neighbor spacing")

    capillary_weight = np.power(r0 / distance, 4.0)
    capillary_weight = np.clip(capillary_weight, 0.0, 6.0)

    scaled_gap = np.maximum(distance / r0 - 1.0, 0.0)
    alignment_weight = np.exp(-scaled_gap / max(float(alignment_decay), 1e-6))
    alignment_weight = np.where(distance <= float(alignment_range) * r0, alignment_weight, 0.0)

    if shuffle_bond_frames:
        phi = phi[rng.permutation(phi.size)]

    cap_degree = 2.0 * float(capillary_weight.sum()) / float(n * n)
    align_degree = 2.0 * float(alignment_weight.sum()) / float(n * n)
    frame_harmonic = float(abs(np.mean(np.exp(4.0j * phi))))
    metadata: Dict[str, object] = {
        "n": int(n),
        "node_count": int(n * n),
        "edge_count": int(src.size),
        "mean_degree": float(2.0 * src.size / (n * n)),
        "disorder": float(disorder),
        "cutoff": float(cutoff),
        "alignment_range": float(alignment_range),
        "alignment_decay": float(alignment_decay),
        "seed": int(seed),
        "nearest_spacing_median": r0,
        "minimum_separation": float(distance.min()),
        "maximum_separation": float(distance.max()),
        "capillary_weighted_degree": cap_degree,
        "alignment_weighted_degree": align_degree,
        "bond_frame_fourth_harmonic": frame_harmonic,
        "shuffle_bond_frames": bool(shuffle_bond_frames),
    }
    return CagedGraph(
        positions=positions,
        box=box,
        src=src,
        tgt=tgt,
        phi=phi.astype(float),
        distance=distance.astype(float),
        capillary_weight=capillary_weight.astype(float),
        alignment_weight=alignment_weight.astype(float),
        metadata=metadata,
    )


def resolve_device(requested: str):
    import torch

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
    return device


def _graph_tensors(graph: CagedGraph, device):
    import torch

    return {
        "src": torch.as_tensor(graph.src, dtype=torch.long, device=device),
        "tgt": torch.as_tensor(graph.tgt, dtype=torch.long, device=device),
        "phi": torch.as_tensor(graph.phi, dtype=torch.float32, device=device),
        "wc": torch.as_tensor(graph.capillary_weight, dtype=torch.float32, device=device),
        "wa": torch.as_tensor(graph.alignment_weight, dtype=torch.float32, device=device),
    }


def _observables(theta, tensors) -> Dict[str, object]:
    import torch

    src = tensors["src"]
    tgt = tensors["tgt"]
    phi = tensors["phi"]
    wa = tensors["wa"]
    wc = tensors["wc"]
    z = torch.exp(2.0j * theta)
    s = torch.abs(z.mean(dim=1))
    d = theta[:, src] - theta[:, tgt]
    b = theta[:, src] + theta[:, tgt] - 2.0 * phi
    c2 = (torch.cos(2.0 * d) * wa).sum(dim=1) / torch.clamp(wa.sum(), min=1e-8)
    g2 = (torch.cos(2.0 * b) * wc).sum(dim=1) / torch.clamp(wc.sum(), min=1e-8)
    return {"S": s, "C2": c2, "G2": g2}


def _torque(
    theta,
    tensors,
    j_align: float,
    g_capillary: float,
    write_axis=None,
    write_field: float = 0.0,
    write_weight=None,
):
    import torch

    src = tensors["src"]
    tgt = tensors["tgt"]
    phi = tensors["phi"]
    wa = tensors["wa"]
    wc = tensors["wc"]
    d = theta[:, src] - theta[:, tgt]
    b = theta[:, src] + theta[:, tgt] - 2.0 * phi
    align = 2.0 * float(j_align) * wa * torch.sin(2.0 * d)
    cap = 2.0 * float(g_capillary) * wc * torch.sin(2.0 * b)
    edge_src = -align - cap
    edge_tgt = +align - cap
    out = torch.zeros_like(theta)
    out.index_add_(1, src, edge_src)
    out.index_add_(1, tgt, edge_tgt)
    if write_axis is not None and write_field:
        write_torque = -2.0 * float(write_field) * torch.sin(
            2.0 * (theta - write_axis)
        )
        if write_weight is not None:
            write_torque = write_torque * write_weight
        out = out + write_torque
    return out


def simulate_ensemble(
    graph: CagedGraph,
    *,
    j_align: float,
    g_capillary: float,
    replicas: int,
    burn_in_steps: int,
    sample_steps: int,
    sample_stride: int,
    dt: float,
    seed: int,
    device,
    initial_theta: Optional[np.ndarray] = None,
    write_axis: Optional[np.ndarray] = None,
    write_field: float = 0.0,
    write_weight: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    import torch

    tensors = _graph_tensors(graph, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    node_count = int(graph.positions.shape[0])
    if initial_theta is None:
        theta = math.pi * torch.rand((replicas, node_count), generator=generator, device=device)
    else:
        initial = np.asarray(initial_theta, dtype=np.float32)
        if initial.ndim == 1:
            initial = np.repeat(initial[None, :], replicas, axis=0)
        if initial.shape != (replicas, node_count):
            raise ValueError(f"initial_theta shape {initial.shape} != {(replicas, node_count)}")
        theta = torch.as_tensor(initial, dtype=torch.float32, device=device).clone()
    axis_t = None
    if write_axis is not None:
        axis = np.asarray(write_axis, dtype=np.float32)
        if axis.ndim == 1:
            axis = np.repeat(axis[None, :], replicas, axis=0)
        if axis.shape != (replicas, node_count):
            raise ValueError(f"write_axis shape {axis.shape} != {(replicas, node_count)}")
        axis_t = torch.as_tensor(axis, dtype=torch.float32, device=device)
    weight_t = None
    if write_weight is not None:
        weight = np.asarray(write_weight, dtype=np.float32)
        if weight.ndim == 1:
            weight = np.repeat(weight[None, :], replicas, axis=0)
        if weight.shape != (replicas, node_count):
            raise ValueError(
                f"write_weight shape {weight.shape} != {(replicas, node_count)}"
            )
        weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)

    snapshots: List[np.ndarray] = []
    metrics = {"time": [], "S": [], "C2": [], "G2": []}
    noise_scale = math.sqrt(2.0 * float(dt))
    total_steps = int(burn_in_steps) + int(sample_steps)
    for step in range(total_steps):
        drift = _torque(
            theta,
            tensors,
            j_align=float(j_align),
            g_capillary=float(g_capillary),
            write_axis=axis_t,
            write_field=float(write_field),
            write_weight=weight_t,
        )
        theta = theta + float(dt) * drift + noise_scale * torch.randn(
            theta.shape, generator=generator, device=device
        )
        theta = torch.remainder(theta, math.pi)
        if step >= burn_in_steps and (step - burn_in_steps) % sample_stride == 0:
            obs = _observables(theta, tensors)
            metrics["time"].append(float((step - burn_in_steps) * dt))
            for key in ("S", "C2", "G2"):
                metrics[key].append(obs[key].detach().cpu().numpy())
            snapshots.append(theta.detach().cpu().numpy().astype(np.float32))

    if not snapshots:
        obs = _observables(theta, tensors)
        metrics["time"].append(0.0)
        for key in ("S", "C2", "G2"):
            metrics[key].append(obs[key].detach().cpu().numpy())
        snapshots.append(theta.detach().cpu().numpy().astype(np.float32))

    snap = np.stack(snapshots, axis=0)
    metric_arrays = {key: np.asarray(value) for key, value in metrics.items()}
    z_mean = np.mean(np.exp(2.0j * snap), axis=0)
    q_ea = np.mean(np.abs(z_mean) ** 2, axis=1)
    return {
        "snapshots": snap,
        "metrics": metric_arrays,
        "q_EA": q_ea,
        "final_theta": snap[-1],
        "state_after_steps": theta.detach().cpu().numpy().astype(np.float32),
    }


def replica_overlap(theta: np.ndarray) -> Dict[str, object]:
    z = np.exp(2.0j * np.asarray(theta, dtype=float))
    signed: List[float] = []
    magnitude: List[float] = []
    for a in range(z.shape[0]):
        for b in range(a + 1, z.shape[0]):
            overlap = np.mean(z[a] * np.conjugate(z[b]))
            signed.append(float(np.real(overlap)))
            magnitude.append(float(abs(overlap)))
    return {
        "signed": signed,
        "magnitude": magnitude,
        "signed_mean": float(np.mean(signed)) if signed else float("nan"),
        "signed_std": float(np.std(signed)) if signed else float("nan"),
        "magnitude_mean": float(np.mean(magnitude)) if magnitude else float("nan"),
        "magnitude_std": float(np.std(magnitude)) if magnitude else float("nan"),
    }


def summarize_run(run: Dict[str, object]) -> Dict[str, object]:
    metrics = run["metrics"]
    assert isinstance(metrics, dict)
    summary: Dict[str, object] = {}
    for key in ("S", "C2", "G2"):
        values = np.asarray(metrics[key], dtype=float)
        per_rep = values.mean(axis=0)
        summary[key + "_mean"] = float(per_rep.mean())
        summary[key + "_std"] = float(per_rep.std())
        summary[key + "_time_std_mean"] = float(values.std(axis=0).mean())
    q_ea = np.asarray(run["q_EA"], dtype=float)
    summary["q_EA_mean"] = float(q_ea.mean())
    summary["q_EA_std"] = float(q_ea.std())
    summary["replica_overlap"] = replica_overlap(np.asarray(run["final_theta"]))
    snap = np.asarray(run["snapshots"], dtype=float)
    if snap.shape[0] > 1:
        summary["window_autocorrelation"] = float(
            np.mean(np.cos(2.0 * (snap[-1] - snap[0])))
        )
    else:
        summary["window_autocorrelation"] = float("nan")
    return summary


def split_replica_protocol(
    graph: CagedGraph,
    *,
    j_align: float,
    g_capillary: float,
    parents: int,
    equilibration_steps: int,
    observation_steps: int,
    stride: int,
    dt: float,
    seed: int,
    device,
) -> Dict[str, object]:
    parent_run = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=parents,
        burn_in_steps=equilibration_steps,
        sample_steps=max(1, stride),
        sample_stride=max(1, stride),
        dt=dt,
        seed=seed,
        device=device,
    )
    parent = np.asarray(parent_run["final_theta"], dtype=np.float32)
    initial = np.repeat(parent, 2, axis=0)
    split = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=2 * parents,
        burn_in_steps=0,
        sample_steps=observation_steps,
        sample_stride=stride,
        dt=dt,
        seed=seed + 100003,
        device=device,
        initial_theta=initial,
    )
    snap = np.asarray(split["snapshots"], dtype=float)
    overlap = []
    for sample in snap:
        first = sample[0::2]
        second = sample[1::2]
        overlap.append(np.mean(np.cos(2.0 * (first - second)), axis=1))
    overlap_a = np.asarray(overlap)
    return {
        "time": np.asarray(split["metrics"]["time"], dtype=float).tolist(),
        "overlap_mean": overlap_a.mean(axis=1).tolist(),
        "overlap_std": overlap_a.std(axis=1).tolist(),
        "parents": int(parents),
    }


def aging_protocol(
    graph: CagedGraph,
    *,
    j_align: float,
    g_capillary: float,
    replicas: int,
    steps: int,
    stride: int,
    dt: float,
    seed: int,
    device,
) -> Dict[str, object]:
    run = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=replicas,
        burn_in_steps=0,
        sample_steps=steps,
        sample_stride=stride,
        dt=dt,
        seed=seed,
        device=device,
    )
    snap = np.asarray(run["snapshots"], dtype=float)
    times = np.asarray(run["metrics"]["time"], dtype=float)
    wait_indices = sorted(set(max(0, min(snap.shape[0] - 2, int(frac * snap.shape[0]))) for frac in (0.1, 0.3, 0.6)))
    curves: List[Dict[str, object]] = []
    for wait_idx in wait_indices:
        reference = snap[wait_idx]
        corr = np.mean(np.cos(2.0 * (snap[wait_idx:] - reference[None, :, :])), axis=(1, 2))
        curves.append(
            {
                "waiting_time": float(times[wait_idx]),
                "lag_time": (times[wait_idx:] - times[wait_idx]).tolist(),
                "correlation": corr.tolist(),
            }
        )
    return {"curves": curves, "replicas": int(replicas)}


def write_release_protocol(
    graph: CagedGraph,
    *,
    j_align: float,
    g_capillary: float,
    replicas: int,
    equilibration_steps: int,
    write_steps: int,
    release_steps: int,
    stride: int,
    dt: float,
    write_field: float,
    seed: int,
    device,
) -> Dict[str, object]:
    target_run = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=1,
        burn_in_steps=equilibration_steps,
        sample_steps=max(1, stride),
        sample_stride=max(1, stride),
        dt=dt,
        seed=seed,
        device=device,
    )
    target = np.asarray(target_run["final_theta"], dtype=np.float32)[0]
    write = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=replicas,
        burn_in_steps=0,
        sample_steps=write_steps,
        sample_stride=stride,
        dt=dt,
        seed=seed + 300007,
        device=device,
        write_axis=target,
        write_field=write_field,
    )
    release = simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=replicas,
        burn_in_steps=0,
        sample_steps=release_steps,
        sample_stride=stride,
        dt=dt,
        seed=seed + 600011,
        device=device,
        initial_theta=np.asarray(write["final_theta"], dtype=np.float32),
    )
    write_snap = np.asarray(write["snapshots"], dtype=float)
    release_snap = np.asarray(release["snapshots"], dtype=float)
    write_overlap = np.mean(np.cos(2.0 * (write_snap - target[None, None, :])), axis=(1, 2))
    release_overlap = np.mean(np.cos(2.0 * (release_snap - target[None, None, :])), axis=(1, 2))
    write_time = np.asarray(write["metrics"]["time"], dtype=float)
    release_time = np.asarray(release["metrics"]["time"], dtype=float)
    offset = float(write_time[-1] + dt * stride) if write_time.size else 0.0
    return {
        "write_time": write_time.tolist(),
        "write_overlap": write_overlap.tolist(),
        "release_time": (release_time + offset).tolist(),
        "release_overlap": release_overlap.tolist(),
        "release_S": np.asarray(release["metrics"]["S"], dtype=float).mean(axis=1).tolist(),
        "release_G2": np.asarray(release["metrics"]["G2"], dtype=float).mean(axis=1).tolist(),
        "write_field": float(write_field),
        "replicas": int(replicas),
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_scan(args, device, out_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    scan_path = out_dir / "capillary_pair_scan.jsonl"
    if scan_path.exists() and args.no_resume:
        scan_path.unlink()
    completed = set()
    if scan_path.exists():
        for line in scan_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((row["n"], row["graph_seed"], row["j_align"], row["g_capillary"], row["control"]))
            rows.append(row)

    graph_seeds = parse_int_list(args.graph_seeds)
    j_values = parse_float_list(args.j_values)
    g_values = parse_float_list(args.g_values)
    controls = ("physical", "shuffled_frames", "regular") if args.include_controls else ("physical",)
    total = len(graph_seeds) * len(j_values) * len(g_values) * len(controls)
    done = 0
    started = time.time()
    for graph_seed in graph_seeds:
        for control in controls:
            graph = make_caged_graph(
                args.n,
                disorder=0.0 if control == "regular" else args.disorder,
                cutoff=args.cutoff,
                alignment_range=args.alignment_range,
                alignment_decay=args.alignment_decay,
                seed=graph_seed,
                shuffle_bond_frames=control == "shuffled_frames",
            )
            for j_align in j_values:
                for g_capillary in g_values:
                    key = (args.n, graph_seed, j_align, g_capillary, control)
                    done += 1
                    if key in completed:
                        continue
                    run = simulate_ensemble(
                        graph,
                        j_align=j_align,
                        g_capillary=g_capillary,
                        replicas=args.replicas,
                        burn_in_steps=args.burn_in_steps,
                        sample_steps=args.sample_steps,
                        sample_stride=args.sample_stride,
                        dt=args.dt,
                        seed=args.seed + 1009 * graph_seed + int(1000 * j_align) + int(10000 * g_capillary),
                        device=device,
                    )
                    summary = summarize_run(run)
                    row: Dict[str, object] = {
                        "n": int(args.n),
                        "graph_seed": int(graph_seed),
                        "j_align": float(j_align),
                        "g_capillary": float(g_capillary),
                        "control": control,
                        "replicas": int(args.replicas),
                        "burn_in_steps": int(args.burn_in_steps),
                        "sample_steps": int(args.sample_steps),
                        "sample_stride": int(args.sample_stride),
                        "dt": float(args.dt),
                        "graph": graph.metadata,
                        **summary,
                    }
                    append_jsonl(scan_path, [row])
                    rows.append(row)
                    if done % max(1, args.progress_every) == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "scan_progress",
                                    "completed": done,
                                    "total": total,
                                    "elapsed_sec": round(time.time() - started, 2),
                                    "control": control,
                                    "j": j_align,
                                    "g": g_capillary,
                                }
                            ),
                            flush=True,
                        )
    return rows


def run_protocols(args, device, out_dir: Path) -> Dict[str, object]:
    graph_seed = parse_int_list(args.graph_seeds)[0]
    graph = make_caged_graph(
        args.n,
        disorder=args.disorder,
        cutoff=args.cutoff,
        alignment_range=args.alignment_range,
        alignment_decay=args.alignment_decay,
        seed=graph_seed,
    )
    common = {
        "graph": graph,
        "j_align": args.selected_j,
        "g_capillary": args.selected_g,
        "dt": args.dt,
        "seed": args.seed + 700001,
        "device": device,
    }
    split = split_replica_protocol(
        **common,
        parents=max(2, args.replicas // 2),
        equilibration_steps=args.protocol_equilibration_steps,
        observation_steps=args.protocol_steps,
        stride=args.protocol_stride,
    )
    aging = aging_protocol(
        **common,
        replicas=args.replicas,
        steps=args.protocol_steps,
        stride=args.protocol_stride,
    )
    write_release = write_release_protocol(
        **common,
        replicas=args.replicas,
        equilibration_steps=args.protocol_equilibration_steps,
        write_steps=args.write_steps,
        release_steps=args.protocol_steps,
        stride=args.protocol_stride,
        write_field=args.write_field,
    )
    no_capillary = split_replica_protocol(
        graph,
        j_align=args.selected_j,
        g_capillary=0.0,
        parents=max(2, args.replicas // 2),
        equilibration_steps=args.protocol_equilibration_steps,
        observation_steps=args.protocol_steps,
        stride=args.protocol_stride,
        dt=args.dt,
        seed=args.seed + 900001,
        device=device,
    )
    no_capillary_write_release = write_release_protocol(
        graph,
        j_align=args.selected_j,
        g_capillary=0.0,
        replicas=args.replicas,
        equilibration_steps=args.protocol_equilibration_steps,
        write_steps=args.write_steps,
        release_steps=args.protocol_steps,
        stride=args.protocol_stride,
        dt=args.dt,
        write_field=args.write_field,
        seed=args.seed + 1_100_003,
        device=device,
    )
    result = {
        "model": {
            "j_align": float(args.selected_j),
            "g_capillary": float(args.selected_g),
            "dt": float(args.dt),
            "graph": graph.metadata,
        },
        "split_replica": split,
        "aging": aging,
        "write_release": write_release,
        "no_capillary_split_replica": no_capillary,
        "no_capillary_write_release": no_capillary_write_release,
    }
    write_json(out_dir / "capillary_pair_protocols.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--graph-seeds", default="17,29,43")
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--cutoff", type=float, default=2.6)
    parser.add_argument("--alignment-range", type=float, default=1.35)
    parser.add_argument("--alignment-decay", type=float, default=0.20)
    parser.add_argument("--j-values", default="0,0.2,0.4,0.6,0.8")
    parser.add_argument("--g-values", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--burn-in-steps", type=int, default=12000)
    parser.add_argument("--sample-steps", type=int, default=18000)
    parser.add_argument("--sample-stride", type=int, default=60)
    parser.add_argument("--selected-j", type=float, default=0.4)
    parser.add_argument("--selected-g", type=float, default=0.8)
    parser.add_argument("--protocol-equilibration-steps", type=int, default=18000)
    parser.add_argument("--protocol-steps", type=int, default=30000)
    parser.add_argument("--protocol-stride", type=int, default=60)
    parser.add_argument("--write-steps", type=int, default=12000)
    parser.add_argument("--write-field", type=float, default=1.0)
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-protocols", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.graph_seeds = parse_int_list(args.graph_seeds)[:1]
        args.graph_seeds = ",".join(str(x) for x in args.graph_seeds)
        args.j_values = "0,0.4,0.8"
        args.g_values = "0,0.4,0.8"
        args.replicas = min(args.replicas, 6)
        args.burn_in_steps = min(args.burn_in_steps, 1200)
        args.sample_steps = min(args.sample_steps, 2400)
        args.sample_stride = min(args.sample_stride, 30)
        args.protocol_equilibration_steps = min(args.protocol_equilibration_steps, 1600)
        args.protocol_steps = min(args.protocol_steps, 3000)
        args.protocol_stride = min(args.protocol_stride, 30)
        args.write_steps = min(args.write_steps, 1200)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    manifest = {
        "command": "rotating_colloids_capillary_pair.py",
        "parameters": vars(args),
        "device": str(device),
        "energy_units": "k_B T",
        "time_units": "D_r^-1",
        "capillary_law": "g_ij = g * (r0/r_ij)^4",
    }
    write_json(out_dir / "run_manifest.json", manifest)

    result: Dict[str, object] = {"manifest": manifest}
    if not args.skip_scan:
        result["scan_rows"] = len(run_scan(args, device, out_dir))
    if not args.skip_protocols:
        result["protocols"] = run_protocols(args, device, out_dir)
    write_json(out_dir / "capillary_pair_run_summary.json", result)
    print(json.dumps({"event": "complete", "output_dir": str(out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
