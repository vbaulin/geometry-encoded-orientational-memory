#!/usr/bin/env python3
"""Test orientational retention when colloid centres undergo Brownian motion.

The positional model is deliberately minimal: capillary and alignment forces
act together with a soft excluded-volume core and an isotropic cage stiffness.
The scan measures the cage stiffness required to preserve neighbour topology
over the angular-memory observation window. Output is append-only and resumable.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rotating_colloids_capillary_pair import (
    make_caged_graph,
    parse_float_list,
    parse_int_list,
    resolve_device,
)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def minimum_image_torch(delta, box):
    import torch

    return delta - box * torch.round(delta / box)


def mobile_energy(
    theta,
    positions,
    reference,
    box,
    src,
    tgt,
    core_src,
    core_tgt,
    *,
    r0: float,
    j_align: float,
    g_capillary: float,
    cage_stiffness: float,
    alignment_range: float,
    alignment_decay: float,
    core_diameter: float,
    core_strength: float,
):
    import torch

    delta = minimum_image_torch(positions[:, tgt] - positions[:, src], box)
    radius = torch.linalg.vector_norm(delta, dim=2).clamp_min(0.35 * core_diameter)
    phi = torch.atan2(delta[..., 1], delta[..., 0])
    capillary_weight = torch.clamp((float(r0) / radius) ** 4, max=6.0)
    alignment_weight = torch.where(
        radius <= float(alignment_range) * float(r0),
        torch.exp(
            -torch.clamp(radius / float(r0) - 1.0, min=0.0)
            / float(alignment_decay)
        ),
        torch.zeros_like(radius),
    )
    relative = theta[:, src] - theta[:, tgt]
    bond_frame = theta[:, src] + theta[:, tgt] - 2.0 * phi
    energy = (
        -float(j_align) * alignment_weight * torch.cos(2.0 * relative)
        -float(g_capillary) * capillary_weight * torch.cos(2.0 * bond_frame)
    ).sum(dim=1)

    core_delta = minimum_image_torch(
        positions[:, core_tgt] - positions[:, core_src], box
    )
    core_radius = torch.linalg.vector_norm(core_delta, dim=2).clamp_min(
        0.35 * core_diameter
    )
    penetration = torch.clamp(float(core_diameter) - core_radius, min=0.0)
    soft_core = 0.5 * float(core_strength) * (
        penetration / float(core_diameter)
    ) ** 2
    energy = energy + soft_core.sum(dim=1)

    displacement = minimum_image_torch(positions - reference, box)
    energy = energy + 0.5 * float(cage_stiffness) * torch.sum(
        displacement**2, dim=(1, 2)
    )
    return energy


def integrate_mobile(
    graph,
    *,
    theta,
    positions,
    steps: int,
    stride: int,
    dt: float,
    mobility_ratio: float,
    j_align: float,
    g_capillary: float,
    cage_stiffness: float,
    core_diameter: float,
    core_strength: float,
    max_position_force: float,
    seed: int,
    device,
) -> Dict[str, object]:
    import torch

    theta = torch.as_tensor(theta, dtype=torch.float32, device=device).clone()
    positions = torch.as_tensor(positions, dtype=torch.float32, device=device).clone()
    reference = torch.as_tensor(
        graph.positions, dtype=torch.float32, device=device
    )[None, :, :]
    box = torch.as_tensor(graph.box, dtype=torch.float32, device=device)[None, None, :]
    src = torch.as_tensor(graph.src, dtype=torch.long, device=device)
    tgt = torch.as_tensor(graph.tgt, dtype=torch.long, device=device)

    # Every pair in the initial 2.6a interaction graph is retained as a core
    # candidate. The WCA contribution is exactly zero outside its short cutoff,
    # but keeping the wider candidate shell prevents an initially separated pair
    # from entering the core without acquiring a repulsive force.
    core_src = src
    core_tgt = tgt

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    theta_noise = math.sqrt(2.0 * float(dt))
    position_noise = math.sqrt(2.0 * float(mobility_ratio) * float(dt))
    overlap_rows = []
    displacement_rows = []
    clipped_force_nodes = 0
    force_nodes = 0
    for step in range(int(steps)):
        theta.requires_grad_(True)
        positions.requires_grad_(True)
        energy = mobile_energy(
            theta,
            positions,
            reference,
            box,
            src,
            tgt,
            core_src,
            core_tgt,
            r0=float(graph.metadata["nearest_spacing_median"]),
            j_align=j_align,
            g_capillary=g_capillary,
            cage_stiffness=cage_stiffness,
            alignment_range=float(graph.metadata["alignment_range"]),
            alignment_decay=float(graph.metadata["alignment_decay"]),
            core_diameter=core_diameter,
            core_strength=core_strength,
        )
        theta_grad, position_grad = torch.autograd.grad(
            energy.sum(), (theta, positions)
        )
        with torch.no_grad():
            force_norm = torch.linalg.vector_norm(position_grad, dim=2, keepdim=True)
            clipped_force_nodes += int(
                torch.count_nonzero(force_norm > float(max_position_force)).item()
            )
            force_nodes += int(force_norm.numel())
            position_grad = position_grad * torch.clamp(
                float(max_position_force) / torch.clamp(force_norm, min=1e-12),
                max=1.0,
            )
            theta = theta - float(dt) * theta_grad + theta_noise * torch.randn(
                theta.shape, generator=generator, device=device
            )
            theta = torch.remainder(theta, math.pi).detach()
            positions = positions - (
                float(mobility_ratio) * float(dt) * position_grad
            ) + position_noise * torch.randn(
                positions.shape, generator=generator, device=device
            )
            positions = torch.remainder(positions, box).detach()
            if not torch.isfinite(theta).all() or not torch.isfinite(positions).all():
                raise RuntimeError("non-finite mobile-centre trajectory")
            if step % max(1, stride) == 0 or step == int(steps) - 1:
                if theta.shape[0] % 2 == 0:
                    overlap = torch.mean(
                        torch.cos(2.0 * (theta[0::2] - theta[1::2])), dim=1
                    )
                    overlap_rows.append(overlap.detach().cpu().numpy())
                displacement = minimum_image_torch(positions - reference, box)
                displacement_rows.append(
                    torch.sqrt(torch.mean(displacement**2, dim=(1, 2)))
                    .detach().cpu().numpy()
                )

    return {
        "theta": theta.detach().cpu().numpy(),
        "positions": positions.detach().cpu().numpy(),
        "overlap": np.asarray(overlap_rows, dtype=float),
        "rms_displacement": np.asarray(displacement_rows, dtype=float),
        "position_force_clip_fraction": float(
            clipped_force_nodes / max(force_nodes, 1)
        ),
    }


def edge_set(positions: np.ndarray, box: np.ndarray, cutoff: float) -> set[Tuple[int, int]]:
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box * np.round(delta / box)
    radius = np.linalg.norm(delta, axis=2)
    i, j = np.triu_indices(positions.shape[0], k=1)
    return {
        (int(a), int(b))
        for a, b, keep in zip(i, j, radius[i, j] <= float(cutoff))
        if keep
    }


def topology_metrics(graph, positions: np.ndarray) -> Dict[str, float]:
    initial = {(int(a), int(b)) for a, b in zip(graph.src, graph.tgt)}
    jaccard = []
    retained = []
    minimum = []
    for sample in positions:
        final = edge_set(sample, graph.box, float(graph.metadata["cutoff"]))
        jaccard.append(len(initial & final) / max(len(initial | final), 1))
        retained.append(len(initial & final) / max(len(initial), 1))
        delta = sample[:, None, :] - sample[None, :, :]
        delta -= graph.box * np.round(delta / graph.box)
        radius = np.linalg.norm(delta, axis=2)
        radius += np.eye(radius.shape[0]) * 1e9
        minimum.append(float(radius.min()))
    return {
        "edge_jaccard_mean": float(np.mean(jaccard)),
        "initial_edges_retained_mean": float(np.mean(retained)),
        "minimum_separation_mean": float(np.mean(minimum)),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "rotating_colloids_mobile_cage_validation"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--graph-seeds", default="17,29,43,71,97")
    parser.add_argument("--cage-stiffness", default="100,200,500,1000,2000")
    parser.add_argument("--parents", type=int, default=6)
    parser.add_argument("--j-align", type=float, default=4.0)
    parser.add_argument("--g-capillary", type=float, default=5.0)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--equilibration-time", type=float, default=25.0)
    parser.add_argument("--observation-time", type=float, default=50.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--mobility-ratio", type=float, default=0.05)
    parser.add_argument("--core-diameter", type=float, default=0.55)
    parser.add_argument("--core-strength", type=float, default=1000.0)
    parser.add_argument("--max-position-force", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.quick:
        args.device = "cpu"
        args.n = min(args.n, 4)
        args.graph_seeds = str(parse_int_list(args.graph_seeds)[0])
        args.cage_stiffness = "5,20"
        args.parents = 2
        args.dt = 0.01
        args.equilibration_time = 0.1
        args.observation_time = 0.2
        args.sample_interval = 0.05

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "mobile_cage_validation.jsonl"
    if args.no_resume and rows_path.exists():
        rows_path.unlink()
    existing = {
        (int(row["graph_seed"]), float(row["cage_stiffness"]))
        for row in read_rows(rows_path)
    }
    device = resolve_device(args.device)
    stiffnesses = parse_float_list(args.cage_stiffness)
    graph_seeds = parse_int_list(args.graph_seeds)
    total = len(stiffnesses) * len(graph_seeds)
    completed = 0

    for graph_seed in graph_seeds:
        graph = make_caged_graph(
            args.n,
            disorder=args.disorder,
            cutoff=2.6,
            alignment_range=1.35,
            alignment_decay=0.20,
            seed=graph_seed,
        )
        reference = np.repeat(
            graph.positions[None, :, :], args.parents, axis=0
        ).astype(np.float32)
        rng = np.random.default_rng(args.seed + graph_seed * 1009)
        initial_theta = rng.uniform(
            0.0, math.pi, size=(args.parents, graph.positions.shape[0])
        ).astype(np.float32)
        for cage_stiffness in stiffnesses:
            completed += 1
            key = (graph_seed, float(cage_stiffness))
            if key in existing:
                continue
            equilibration = integrate_mobile(
                graph,
                theta=initial_theta,
                positions=reference,
                steps=max(1, int(round(args.equilibration_time / args.dt))),
                stride=max(1, int(round(args.sample_interval / args.dt))),
                dt=args.dt,
                mobility_ratio=args.mobility_ratio,
                j_align=args.j_align,
                g_capillary=args.g_capillary,
                cage_stiffness=cage_stiffness,
                core_diameter=args.core_diameter,
                core_strength=args.core_strength,
                max_position_force=args.max_position_force,
                seed=args.seed + graph_seed * 2003 + int(100 * cage_stiffness),
                device=device,
            )
            split_theta = np.repeat(equilibration["theta"], 2, axis=0)
            split_positions = np.repeat(equilibration["positions"], 2, axis=0)
            release = integrate_mobile(
                graph,
                theta=split_theta,
                positions=split_positions,
                steps=max(1, int(round(args.observation_time / args.dt))),
                stride=max(1, int(round(args.sample_interval / args.dt))),
                dt=args.dt,
                mobility_ratio=args.mobility_ratio,
                j_align=args.j_align,
                g_capillary=args.g_capillary,
                cage_stiffness=cage_stiffness,
                core_diameter=args.core_diameter,
                core_strength=args.core_strength,
                max_position_force=args.max_position_force,
                seed=args.seed + graph_seed * 3001 + int(100 * cage_stiffness),
                device=device,
            )
            topology = topology_metrics(graph, release["positions"])
            row = {
                "graph_seed": graph_seed,
                "n": args.n,
                "node_count": args.n * args.n,
                "cage_stiffness": cage_stiffness,
                "mobility_ratio": args.mobility_ratio,
                "dt": args.dt,
                "equilibration_time": args.equilibration_time,
                "observation_time": args.observation_time,
                "split_endpoint": float(np.mean(release["overlap"][-1])),
                "rms_displacement_endpoint": float(
                    np.mean(release["rms_displacement"][-1])
                ),
                "position_force_clip_fraction": float(
                    release["position_force_clip_fraction"]
                ),
                **topology,
            }
            append_jsonl(rows_path, row)
            existing.add(key)
            print(json.dumps({
                "event": "mobile_cage_progress",
                "completed": completed,
                "total": total,
                **row,
            }), flush=True)

    rows = read_rows(rows_path)
    report = {
        "parameters": vars(args),
        "row_count": len(rows),
        "criterion": (
            "A cage-realizable memory window requires high retained split overlap, "
            "initial-edge retention near unity, and RMS displacement small compared "
            "with the particle spacing over the same observation time."
        ),
        "rows": rows,
    }
    (out_dir / "mobile_cage_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", "output_dir": str(out_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
