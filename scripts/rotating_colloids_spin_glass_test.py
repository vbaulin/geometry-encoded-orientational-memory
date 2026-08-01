#!/usr/bin/env python3
"""Finite-size spin-glass diagnostics for the capillary-pair rotor model.

The scan follows a fixed coupling ray (J, g) = lambda * (J0, g0).  For each
quenched graph and lambda, independent equilibrium replicas provide the local
complex overlap q_i = exp[2 i (theta_i^a - theta_i^b)].  We report P(|Q_ab|),
the overlap susceptibility, a two-component Binder cumulant, and the standard
second-moment overlap correlation length xi_L / L.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from rotating_colloids_capillary_pair import (
    make_caged_graph,
    parse_float_list,
    parse_int_list,
    resolve_device,
    simulate_ensemble,
)


def replica_pairs(count: int, max_pairs: int, seed: int) -> np.ndarray:
    pairs = np.asarray([(a, b) for a in range(count) for b in range(a + 1, count)], dtype=np.int64)
    if pairs.shape[0] <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    return pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]


def overlap_diagnostics(theta: np.ndarray, positions: np.ndarray, box: np.ndarray, *, max_pairs: int, seed: int) -> Dict[str, object]:
    """Compute finite-size diagnostics from independent final replicas."""

    z = np.exp(2.0j * np.asarray(theta, dtype=float))
    pairs = replica_pairs(z.shape[0], max_pairs=max_pairs, seed=seed)
    local = z[pairs[:, 0]] * np.conjugate(z[pairs[:, 1]])
    q0 = local.mean(axis=1)
    q_abs = np.abs(q0)
    q2 = float(np.mean(q_abs**2))
    q4 = float(np.mean(q_abs**4))
    node_count = int(z.shape[1])

    # For a two-component (complex) overlap, Gaussian fluctuations give U4=0.
    binder = 1.0 - q4 / max(2.0 * q2 * q2, 1e-15)
    chi0 = float(node_count * q2)

    kx = 2.0 * math.pi / float(box[0])
    ky = 2.0 * math.pi / float(box[1])
    phase_x = np.exp(1.0j * kx * positions[:, 0])
    phase_y = np.exp(1.0j * ky * positions[:, 1])
    qx = np.mean(local * phase_x[None, :], axis=1)
    qy = np.mean(local * phase_y[None, :], axis=1)
    chik_x = float(node_count * np.mean(np.abs(qx) ** 2))
    chik_y = float(node_count * np.mean(np.abs(qy) ** 2))
    ratio_x = max(chi0 / max(chik_x, 1e-15) - 1.0, 0.0)
    ratio_y = max(chi0 / max(chik_y, 1e-15) - 1.0, 0.0)
    xi_x = math.sqrt(ratio_x) / max(2.0 * math.sin(0.5 * kx), 1e-12)
    xi_y = math.sqrt(ratio_y) / max(2.0 * math.sin(0.5 * ky), 1e-12)
    xi = math.sqrt(xi_x * xi_y)
    linear_size = math.sqrt(float(box[0] * box[1]))

    hist, edges = np.histogram(q_abs, bins=np.linspace(0.0, 1.0, 41), density=True)
    return {
        "pair_count": int(pairs.shape[0]),
        "q_abs_mean": float(q_abs.mean()),
        "q_abs_std": float(q_abs.std()),
        "q2": q2,
        "q4": q4,
        "binder_complex": float(binder),
        "chi_overlap": chi0,
        "chi_kmin_x": chik_x,
        "chi_kmin_y": chik_y,
        "xi_x": float(xi_x),
        "xi_y": float(xi_y),
        "xi_L": float(xi),
        "xi_over_L": float(xi / linear_size),
        "histogram_edges": edges.tolist(),
        "histogram_density": hist.tolist(),
    }


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                keys.add(str(json.loads(line)["key"]))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sizes", default="12,16,24,32,48")
    parser.add_argument("--graph-seeds", default="17,29,43,71,97")
    parser.add_argument("--lambdas", default="0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4")
    parser.add_argument("--j0", type=float, default=4.0)
    parser.add_argument("--g0", type=float, default=5.0)
    parser.add_argument("--replicas", type=int, default=48)
    parser.add_argument("--max-replica-pairs", type=int, default=512)
    parser.add_argument("--burn-in-steps", type=int, default=50000)
    parser.add_argument("--sample-steps", type=int, default=100000)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--cutoff", type=float, default=2.6)
    parser.add_argument("--alignment-range", type=float, default=1.35)
    parser.add_argument("--alignment-decay", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spin_glass_scan.jsonl"
    done = completed_keys(output_path)
    device = resolve_device(args.device)
    sizes = parse_int_list(args.sizes)
    graph_seeds = parse_int_list(args.graph_seeds)
    lambdas = parse_float_list(args.lambdas)
    total = len(sizes) * len(graph_seeds) * len(lambdas)
    started = time.time()
    new_points = 0

    for n in sizes:
        for graph_seed in graph_seeds:
            graph = make_caged_graph(
                n,
                disorder=args.disorder,
                cutoff=args.cutoff,
                alignment_range=args.alignment_range,
                alignment_decay=args.alignment_decay,
                seed=graph_seed,
            )
            for lam in lambdas:
                key = f"n={n}|graph={graph_seed}|lambda={lam:.8g}"
                if key in done:
                    continue
                run_seed = args.seed + 1000003 * n + 1009 * graph_seed + int(round(10000 * lam))
                run = simulate_ensemble(
                    graph,
                    j_align=args.j0 * lam,
                    g_capillary=args.g0 * lam,
                    replicas=args.replicas,
                    burn_in_steps=args.burn_in_steps,
                    sample_steps=args.sample_steps,
                    sample_stride=args.sample_stride,
                    dt=args.dt,
                    seed=run_seed,
                    device=device,
                )
                row: Dict[str, object] = {
                    "key": key,
                    "n": n,
                    "node_count": n * n,
                    "graph_seed": graph_seed,
                    "lambda": lam,
                    "j": args.j0 * lam,
                    "g": args.g0 * lam,
                    "S_mean": float(np.mean(run["metrics"]["S"])),
                    "C2_mean": float(np.mean(run["metrics"]["C2"])),
                    "G2_mean": float(np.mean(run["metrics"]["G2"])),
                    "q_EA_mean": float(np.mean(run["q_EA"])),
                    "overlap": overlap_diagnostics(
                        np.asarray(run["final_theta"]),
                        graph.positions,
                        graph.box,
                        max_pairs=args.max_replica_pairs,
                        seed=run_seed + 17,
                    ),
                    "graph": graph.metadata,
                }
                append_jsonl(output_path, row)
                new_points += 1
                if new_points % max(1, args.progress_every) == 0:
                    print(json.dumps({"event": "progress", "new_points": new_points, "total_points": total, "elapsed_sec": round(time.time() - started, 2), "last_key": key}), flush=True)

    manifest = vars(args).copy()
    manifest.update({"output": str(output_path), "total_points": total, "new_points": new_points})
    (output_dir / "spin_glass_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output_path), "new_points": new_points}), flush=True)


if __name__ == "__main__":
    main()
