#!/usr/bin/env python3
"""Activated-retention and observation-window tests for capillary rotors."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np


_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:  # NumPy < 2.0
    _TRAPEZOID = np.trapz

from rotating_colloids_capillary_pair import (
    make_caged_graph,
    parse_float_list,
    parse_int_list,
    resolve_device,
    simulate_ensemble,
    split_replica_protocol,
    write_release_protocol,
)


def crossing_time(times: np.ndarray, values: np.ndarray, threshold: float) -> Dict[str, object]:
    below = np.flatnonzero(values <= threshold)
    if below.size == 0:
        return {"time": float(times[-1]), "censored": True}
    idx = int(below[0])
    if idx == 0:
        return {"time": float(times[0]), "censored": False}
    t0, t1 = float(times[idx - 1]), float(times[idx])
    y0, y1 = float(values[idx - 1]), float(values[idx])
    frac = 0.0 if y1 == y0 else (threshold - y0) / (y1 - y0)
    return {"time": float(t0 + frac * (t1 - t0)), "censored": False}


def curve_summary(times: List[float], values: List[float]) -> Dict[str, object]:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size == 0:
        raise ValueError("empty retention curve")
    positive = np.maximum(y, 0.0)
    integral = float(_TRAPEZOID(positive, t) / max(abs(y[0]), 1e-12))
    tail_count = max(1, t.size // 10)
    return {
        "initial": float(y[0]),
        "final": float(y[-1]),
        "tail_mean": float(y[-tail_count:].mean()),
        "positive_integral_time": integral,
        "t_0p8": crossing_time(t, y, 0.8),
        "t_0p6": crossing_time(t, y, 0.6),
        "t_0p4": crossing_time(t, y, 0.4),
        "t_0p2": crossing_time(t, y, 0.2),
    }


def window_diagnostics(run: Dict[str, object], windows: List[float]) -> Dict[str, object]:
    snapshots = np.asarray(run["snapshots"], dtype=float)
    times = np.asarray(run["metrics"]["time"], dtype=float)
    available = []
    for window in windows:
        count = int(np.searchsorted(times, window, side="right"))
        if count < 2:
            continue
        sample = snapshots[:count]
        local_mean = np.mean(np.exp(2.0j * sample), axis=0)
        qea = np.mean(np.abs(local_mean) ** 2, axis=1)
        endpoint = np.mean(np.cos(2.0 * (sample[-1] - sample[0])), axis=1)
        available.append(
            {
                "window": float(times[count - 1]),
                "q_EA_mean": float(qea.mean()),
                "q_EA_std": float(qea.std()),
                "endpoint_correlation_mean": float(endpoint.mean()),
                "endpoint_correlation_std": float(endpoint.std()),
            }
        )
    return {"windows": available}


def append_row(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--graph-seeds", default="17,29,43,71,97")
    parser.add_argument("--lambdas", default="0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.4")
    parser.add_argument("--j0", type=float, default=4.0)
    parser.add_argument("--g0", type=float, default=5.0)
    parser.add_argument("--replicas", type=int, default=32)
    parser.add_argument("--equilibration-steps", type=int, default=50000)
    parser.add_argument("--observation-steps", type=int, default=250000)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--write-steps", type=int, default=50000)
    parser.add_argument("--write-field", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--cutoff", type=float, default=2.6)
    parser.add_argument("--alignment-range", type=float, default=1.35)
    parser.add_argument("--alignment-decay", type=float, default=0.2)
    parser.add_argument("--windows", default="5,10,25,50,100,250,500,625")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    output = out / "activated_memory_scan.jsonl"
    done = completed(output)
    device = resolve_device(args.device)
    graph_seeds = parse_int_list(args.graph_seeds)
    lambdas = parse_float_list(args.lambdas)
    windows = parse_float_list(args.windows)
    started = time.time()
    new_points = 0

    for graph_seed in graph_seeds:
        graph = make_caged_graph(
            args.n,
            disorder=args.disorder,
            cutoff=args.cutoff,
            alignment_range=args.alignment_range,
            alignment_decay=args.alignment_decay,
            seed=graph_seed,
        )
        for lam in lambdas:
            key = f"n={args.n}|graph={graph_seed}|lambda={lam:.8g}"
            if key in done:
                continue
            j = args.j0 * lam
            g = args.g0 * lam
            run_seed = args.seed + graph_seed * 1009 + int(round(lam * 10000))
            protocols = {}
            for label, capillary in (("physical", g), ("no_capillary", 0.0)):
                offset = 0 if label == "physical" else 2_000_003
                split = split_replica_protocol(
                    graph,
                    j_align=j,
                    g_capillary=capillary,
                    parents=max(2, args.replicas // 2),
                    equilibration_steps=args.equilibration_steps,
                    observation_steps=args.observation_steps,
                    stride=args.stride,
                    dt=args.dt,
                    seed=run_seed + offset,
                    device=device,
                )
                written = write_release_protocol(
                    graph,
                    j_align=j,
                    g_capillary=capillary,
                    replicas=args.replicas,
                    equilibration_steps=args.equilibration_steps,
                    write_steps=args.write_steps,
                    release_steps=args.observation_steps,
                    stride=args.stride,
                    dt=args.dt,
                    write_field=args.write_field,
                    seed=run_seed + offset + 700001,
                    device=device,
                )
                release_time = np.asarray(written["release_time"], dtype=float)
                release_time = release_time - release_time[0]
                protocols[label] = {
                    "split": split,
                    "split_summary": curve_summary(split["time"], split["overlap_mean"]),
                    "write_release": written,
                    "release_summary": curve_summary(release_time.tolist(), written["release_overlap"]),
                }

            trajectory = simulate_ensemble(
                graph,
                j_align=j,
                g_capillary=g,
                replicas=args.replicas,
                burn_in_steps=args.equilibration_steps,
                sample_steps=args.observation_steps,
                sample_stride=args.stride,
                dt=args.dt,
                seed=run_seed + 4_000_037,
                device=device,
            )
            row = {
                "key": key,
                "n": args.n,
                "node_count": args.n * args.n,
                "graph_seed": graph_seed,
                "lambda": lam,
                "j": j,
                "g": g,
                "graph": graph.metadata,
                "protocols": protocols,
                "observation_window": window_diagnostics(trajectory, windows),
            }
            append_row(output, row)
            new_points += 1
            print(json.dumps({"event": "progress", "new_points": new_points, "total_points": len(graph_seeds) * len(lambdas), "elapsed_sec": round(time.time() - started, 2), "last_key": key}), flush=True)

    manifest = vars(args).copy()
    manifest.update({"output": str(output), "new_points": new_points})
    (out / "activated_memory_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "complete", "output": str(output), "new_points": new_points}), flush=True)


if __name__ == "__main__":
    main()
