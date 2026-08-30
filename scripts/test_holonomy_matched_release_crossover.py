#!/usr/bin/env python3
"""Matched-start release test for the colloidal loop-frustration crossover.

The original and loop-flattened networks start from exactly the same angular
configuration and receive the same Brownian increments.  No writing field is
applied in either arm.  The test therefore measures post-preparation survival,
not a mixture of writeability and retention.

This is the first stage of the holonomy mechanism test.  It does not energy-
match the two Hamiltonians, so a response can still be caused by ordinary
target-energy or local-curvature differences.  Those quantities are exported
for the subsequent constrained intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:  # NumPy < 2.0
    _TRAPEZOID = np.trapz

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_rotating_colloids_pair_domain_reduction import block_labels  # noqa: E402
from rotating_colloids_hyperion_case import make_graph  # noqa: E402
from discovery.continuous_colloid_holonomy import (  # noqa: E402
    build_frame_intervention,
    pair_energy,
    simulate_write_release,
)
from test_continuous_colloid_holonomy_memory import target_catalog  # noqa: E402


def parse_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def target_family(name: str) -> str:
    if name.startswith("flip_shell_"):
        return "flip_shell"
    if name.startswith("random_domains_"):
        return "random_domains"
    return name


def normalized_survival(result: dict[str, object]) -> dict[str, object]:
    times = np.asarray(result["time"], dtype=float)
    curve = np.asarray(result["overlap_curve"], dtype=float)
    initial = float(curve[0])
    if abs(initial) < 1e-8:
        raise ValueError("matched preparation has vanishing target overlap")
    normalized = curve / initial
    duration = max(float(times[-1] - times[0]), 1e-12)
    auc = float(_TRAPEZOID(normalized, times) / duration)
    positive_auc = float(_TRAPEZOID(np.maximum(normalized, 0.0), times) / duration)
    return {
        "time": times.tolist(),
        "curve": normalized.tolist(),
        "initial_overlap": initial,
        "final": float(normalized[-1]),
        "auc": auc,
        "positive_auc": positive_auc,
    }


def target_torque_rms(
    theta: np.ndarray,
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    j_align: float,
    g_capillary: float,
) -> float:
    """RMS deterministic torque at one target configuration."""

    difference = theta[src] - theta[tgt]
    bond_frame = theta[src] + theta[tgt] - 2.0 * phi
    align = 2.0 * float(j_align) * weights * np.sin(2.0 * difference)
    capillary = 2.0 * float(g_capillary) * weights * np.sin(2.0 * bond_frame)
    torque = np.zeros(theta.size, dtype=float)
    np.add.at(torque, src, -align - capillary)
    np.add.at(torque, tgt, +align - capillary)
    return float(np.sqrt(np.mean(torque**2)))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        keys.add(str(json.loads(line)["key"]))
    return keys


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--cluster-size", type=int, default=2)
    parser.add_argument("--graph-seeds", default="12345,12346,12347,12348,12349,12350,12351,12352")
    parser.add_argument("--beta-j-values", default="0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.4")
    parser.add_argument("--epsilon", type=float, default=1.614286)
    parser.add_argument("--crosslink-k", type=int, default=2)
    parser.add_argument("--crosslink-weight", type=float, default=0.18)
    parser.add_argument("--domain-angle-step", type=float, default=math.pi / 4.0)
    parser.add_argument("--random-target-count", type=int, default=4)
    parser.add_argument("--flip-shell-sizes", default="2,4,6,8,10,12,14")
    parser.add_argument("--replicas", type=int, default=48)
    parser.add_argument("--start-sigma", type=float, default=0.05)
    parser.add_argument("--release-steps", type=int, default=40000)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "matched_release_scan.jsonl"
    done = completed_keys(output)
    graph_seeds = parse_ints(args.graph_seeds)
    beta_j_values = parse_floats(args.beta_j_values)
    flip_shell_sizes = parse_ints(args.flip_shell_sizes)
    labels, axes, blocks_per_side = block_labels(args.n, args.cluster_size)
    domain_count = blocks_per_side**2
    total = len(graph_seeds) * len(beta_j_values)
    started = time.time()
    new_rows = 0
    completed_cells = 0

    for graph_offset, graph_seed in enumerate(graph_seeds):
        _, src, tgt, original_phi, weights, _ = make_graph(
            args.n,
            graph_mode="mosaic",
            graph_seed=graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=args.domain_angle_step,
        )
        for coupling_index, beta_j in enumerate(beta_j_values):
            beta_g = float(args.epsilon) * float(beta_j)
            intervention = build_frame_intervention(
                src=src,
                tgt=tgt,
                phi=original_phi,
                weights=weights,
                labels=labels,
                axes=axes,
                j_align=beta_j,
                g_capillary=beta_g,
                domain_count=domain_count,
            )
            catalog = target_catalog(
                axes=axes,
                labels=labels,
                blocks_per_side=blocks_per_side,
                original_couplings=intervention.original_couplings,
                flat_couplings=intervention.realized_couplings,
                random_target_count=args.random_target_count,
                flip_shell_sizes=flip_shell_sizes,
                seed=args.seed + 10007 * graph_offset,
            )
            prespecified = [entry for entry in catalog if entry["selection"] == "prespecified"]
            for target_index, entry in enumerate(prespecified):
                key = (
                    f"graph={graph_seed}|beta_j={beta_j:.8g}|epsilon={args.epsilon:.8g}|"
                    f"target={entry['name']}"
                )
                if key in done:
                    continue
                target = np.asarray(entry["angles"], dtype=float)
                initial_seed = (
                    args.seed
                    + 104729 * graph_offset
                    + 1543 * coupling_index
                    + target_index
                )
                initial_rng = np.random.default_rng(initial_seed)
                initial = np.remainder(
                    target[None, :]
                    + float(args.start_sigma)
                    * initial_rng.normal(size=(args.replicas, target.size)),
                    math.pi,
                )
                noise_seed = (
                    args.seed
                    + 1000003 * graph_offset
                    + 10009 * coupling_index
                    + target_index
                )
                arm_results: dict[str, dict[str, object]] = {}
                for arm, phi in (
                    ("frustrated", original_phi),
                    ("flat", intervention.flat_phi),
                ):
                    raw = simulate_write_release(
                        src=src,
                        tgt=tgt,
                        phi=phi,
                        weights=weights,
                        target=target,
                        j_align=beta_j,
                        g_capillary=beta_g,
                        replicas=args.replicas,
                        write_steps=0,
                        release_steps=args.release_steps,
                        stride=args.stride,
                        dt=args.dt,
                        write_field=0.0,
                        initial_theta=initial,
                        noise_seed=noise_seed,
                        device=args.device,
                    )
                    survival = normalized_survival(raw)
                    arm_results[arm] = {
                        "survival": survival,
                        "q_ea": raw["q_ea"],
                        "half_life": raw["half_life"],
                        "half_life_censored": raw["half_life_censored"],
                    }

                energy_original = float(
                    pair_energy(
                        target,
                        src=src,
                        tgt=tgt,
                        phi=original_phi,
                        weights=weights,
                        j_align=beta_j,
                        g_capillary=beta_g,
                    )[0]
                )
                energy_flat = float(
                    pair_energy(
                        target,
                        src=src,
                        tgt=tgt,
                        phi=intervention.flat_phi,
                        weights=weights,
                        j_align=beta_j,
                        g_capillary=beta_g,
                    )[0]
                )
                scale = max(beta_g * float(np.sum(np.abs(weights))), 1e-12)
                frustrated = arm_results["frustrated"]["survival"]
                flat = arm_results["flat"]["survival"]
                row = {
                    "key": key,
                    "event": "holonomy_matched_release",
                    "preparation": "identical_target_plus_gaussian_perturbation",
                    "paired_noise": True,
                    "graph_seed": graph_seed,
                    "beta_j": beta_j,
                    "beta_g": beta_g,
                    "epsilon": args.epsilon,
                    "target": entry["name"],
                    "target_family": target_family(str(entry["name"])),
                    "target_selection": entry["selection"],
                    "start_sigma": args.start_sigma,
                    "initial_overlap_arm_difference": float(
                        frustrated["initial_overlap"] - flat["initial_overlap"]
                    ),
                    "energy_original_per_node": energy_original / target.size,
                    "energy_flat_per_node": energy_flat / target.size,
                    "flat_energy_advantage_per_node": (energy_original - energy_flat)
                    / target.size,
                    "normalized_compatibility": (energy_original - energy_flat) / scale,
                    "target_torque_rms_original": target_torque_rms(
                        target,
                        src=src,
                        tgt=tgt,
                        phi=original_phi,
                        weights=weights,
                        j_align=beta_j,
                        g_capillary=beta_g,
                    ),
                    "target_torque_rms_flat": target_torque_rms(
                        target,
                        src=src,
                        tgt=tgt,
                        phi=intervention.flat_phi,
                        weights=weights,
                        j_align=beta_j,
                        g_capillary=beta_g,
                    ),
                    "negative_cycle_count_original": intervention.negative_flux_original,
                    "negative_cycle_count_flat": intervention.negative_flux_realized,
                    "arms": arm_results,
                    "frustrated_minus_flat_survival_auc": float(
                        frustrated["auc"] - flat["auc"]
                    ),
                    "frustrated_minus_flat_final_survival": float(
                        frustrated["final"] - flat["final"]
                    ),
                }
                append_jsonl(output, row)
                done.add(key)
                new_rows += 1
            completed_cells += 1
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "completed_graph_coupling_cells": completed_cells,
                        "total_graph_coupling_cells": total,
                        "new_rows": new_rows,
                        "elapsed_sec": round(time.time() - started, 2),
                        "last_graph": graph_seed,
                        "last_beta_j": beta_j,
                    }
                ),
                flush=True,
            )

    manifest = {
        "complete": True,
        "output": str(output),
        "new_rows": new_rows,
        "graph_seeds": graph_seeds,
        "beta_j_values": beta_j_values,
        "epsilon": args.epsilon,
        "target_count_per_cell": len(
            [
                "uniform_director",
                "mosaic_axes",
                "checkerboard_domains",
            ]
        )
        + len(flip_shell_sizes)
        + args.random_target_count,
        "observation_time_Dr_t": args.release_steps * args.dt,
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key != "output_dir"
        },
        "scope": (
            "Matched-start release isolates post-preparation dynamics. The arms are not "
            "energy, torque, or Hessian matched, so this run does not isolate cycle topology."
        ),
    }
    (args.output_dir / "matched_release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "complete", **manifest}, default=str), flush=True)
    return manifest


def main() -> int:
    run(make_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
