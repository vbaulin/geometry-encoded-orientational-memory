#!/usr/bin/env python3
"""Measure spatial structure inside the capillary hidden-memory regime.

The parameter scans establish where low global order coexists with local,
bond-frame, and temporal correlations.  This companion analysis resolves the
real-space range of those correlations at the selected state.  Results are
cached as JSON so figure generation never silently reruns the simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rotating_colloids_capillary_pair import make_caged_graph, resolve_device, simulate_ensemble


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def pair_geometry(positions: np.ndarray, box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src, tgt = np.triu_indices(positions.shape[0], k=1)
    delta = minimum_image(positions[tgt] - positions[src], box)
    distance = np.linalg.norm(delta, axis=1)
    phi = np.arctan2(delta[:, 1], delta[:, 0])
    return src.astype(np.int64), tgt.astype(np.int64), distance, phi


def binned_correlations(
    snapshots: np.ndarray,
    positions: np.ndarray,
    box: np.ndarray,
    *,
    bins: int,
) -> Dict[str, object]:
    """Average relative- and bond-frame correlations over all particle pairs."""

    src, tgt, distance, phi = pair_geometry(positions, box)
    upper = 0.5 * float(min(box))
    edges = np.linspace(0.0, upper, bins + 1)
    which = np.digitize(distance, edges) - 1
    valid = (which >= 0) & (which < bins)
    src = src[valid]
    tgt = tgt[valid]
    distance = distance[valid]
    phi = phi[valid]
    which = which[valid]

    sample = snapshots.reshape(-1, snapshots.shape[-1]).astype(float)
    z = np.exp(2.0j * sample)
    s2 = np.abs(z.mean(axis=1)) ** 2
    relative_sum = np.zeros(bins, dtype=float)
    bond_sum = np.zeros(bins, dtype=float)
    connected_sum = np.zeros(bins, dtype=float)
    pair_count = np.zeros(bins, dtype=np.int64)

    # Chunk pairs so the analysis stays bounded for publication-scale N.
    chunk = 4096
    for start in range(0, src.size, chunk):
        stop = min(src.size, start + chunk)
        sl = slice(start, stop)
        rel = np.real(z[:, src[sl]] * np.conjugate(z[:, tgt[sl]])).mean(axis=0)
        bond = np.real(z[:, src[sl]] * z[:, tgt[sl]] * np.exp(-4.0j * phi[sl])[None, :]).mean(axis=0)
        connected = rel - float(s2.mean())
        for b in range(bins):
            mask = which[sl] == b
            if not np.any(mask):
                continue
            relative_sum[b] += float(rel[mask].sum())
            bond_sum[b] += float(bond[mask].sum())
            connected_sum[b] += float(connected[mask].sum())
            pair_count[b] += int(mask.sum())

    denom = np.maximum(pair_count, 1)
    return {
        "bin_edges": edges.tolist(),
        "bin_centers": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "relative_correlation": (relative_sum / denom).tolist(),
        "relative_connected": (connected_sum / denom).tolist(),
        "bond_frame_correlation": (bond_sum / denom).tolist(),
        "pair_count": pair_count.tolist(),
        "mean_S_squared": float(s2.mean()),
        "sample_count": int(sample.shape[0]),
    }


def aggregate(curves: List[Dict[str, object]]) -> Dict[str, object]:
    output: Dict[str, object] = {
        "bin_edges": curves[0]["bin_edges"],
        "bin_centers": curves[0]["bin_centers"],
        "graph_count": len(curves),
    }
    for key in ("relative_correlation", "relative_connected", "bond_frame_correlation"):
        values = np.asarray([curve[key] for curve in curves], dtype=float)
        output[key + "_mean"] = values.mean(axis=0).tolist()
        output[key + "_std"] = values.std(axis=0).tolist()
    output["mean_S_squared"] = float(np.mean([curve["mean_S_squared"] for curve in curves]))
    output["sample_count_per_graph"] = [int(curve["sample_count"]) for curve in curves]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "rotating_colloids_capillary_pair_prl_internal/capillary_internal_correlations.json"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--graph-seeds", default="17,29,43")
    parser.add_argument("--j", type=float, default=4.0)
    parser.add_argument("--g", type=float, default=5.0)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--burn-in-steps", type=int, default=8000)
    parser.add_argument("--sample-steps", type=int, default=12000)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--bins", type=int, default=14)
    args = parser.parse_args()

    device = resolve_device(args.device)
    seeds = [int(value) for value in args.graph_seeds.split(",") if value.strip()]
    curves: List[Dict[str, object]] = []
    graph_records: List[Dict[str, object]] = []
    for seed in seeds:
        graph = make_caged_graph(
            args.n,
            disorder=0.16,
            cutoff=2.6,
            alignment_range=1.35,
            alignment_decay=0.20,
            seed=seed,
        )
        run = simulate_ensemble(
            graph,
            j_align=args.j,
            g_capillary=args.g,
            replicas=args.replicas,
            burn_in_steps=args.burn_in_steps,
            sample_steps=args.sample_steps,
            sample_stride=args.sample_stride,
            dt=args.dt,
            seed=20260712 + 1009 * seed,
            device=device,
        )
        curve = binned_correlations(
            np.asarray(run["snapshots"]),
            graph.positions,
            graph.box,
            bins=args.bins,
        )
        curves.append(curve)
        graph_records.append({"seed": seed, "graph": graph.metadata, "correlations": curve})
        print(json.dumps({"event": "graph_complete", "seed": seed, "samples": curve["sample_count"]}), flush=True)

    result = {
        "model": {
            "n": args.n,
            "node_count": args.n * args.n,
            "j_align": args.j,
            "g_capillary": args.g,
            "replicas": args.replicas,
            "burn_in_steps": args.burn_in_steps,
            "sample_steps": args.sample_steps,
            "sample_stride": args.sample_stride,
            "dt": args.dt,
            "graph_seeds": seeds,
        },
        "aggregate": aggregate(curves),
        "graphs": graph_records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
