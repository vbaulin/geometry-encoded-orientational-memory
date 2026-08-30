#!/usr/bin/env python3
"""Reduce the mosaic bond-frame rotor model to its domain-scale Ising limit.

Inside each rotated square domain, the pair Hamiltonian admits two exactly
degenerate directors separated by pi/2.  Writing that choice as s_b = +/- 1
gives an effective Ising coupling between domains.  This script derives those
couplings from the actual simulation graph and audits frustration, degeneracy,
and the number of one-domain-flip-stable states.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rotating_colloids_hyperion_case import make_graph


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def block_labels(n: int, cluster_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    blocks_per_side = int(math.ceil(n / cluster_size))
    labels = np.empty(n * n, dtype=np.int16)
    axes = np.empty(n * n, dtype=float)
    for i in range(n):
        for j in range(n):
            bx = min(blocks_per_side - 1, i // cluster_size)
            by = min(blocks_per_side - 1, j // cluster_size)
            index = i * n + j
            labels[index] = bx * blocks_per_side + by
            axes[index] = ((bx + 2 * by) % 4) * math.pi / 4.0
    return labels, axes, blocks_per_side


def induced_couplings(
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    axes: np.ndarray,
    epsilon: float,
) -> tuple[dict[tuple[int, int], float], int, float]:
    couplings: dict[tuple[int, int], float] = defaultdict(float)
    cross_edges = 0
    cross_weight = 0.0
    for node_i, node_j, bond_angle, weight in zip(src, tgt, phi, weights):
        block_i = int(labels[node_i])
        block_j = int(labels[node_j])
        if block_i == block_j:
            continue
        cross_edges += 1
        cross_weight += float(weight)
        key = tuple(sorted((block_i, block_j)))
        ordinary = math.cos(2.0 * (axes[node_i] - axes[node_j]))
        bond_frame = math.cos(2.0 * (axes[node_i] + axes[node_j] - 2.0 * bond_angle))
        couplings[key] += float(weight) * (ordinary + epsilon * bond_frame)
    return dict(couplings), cross_edges, cross_weight


def enumerate_landscape(couplings: dict[tuple[int, int], float], block_count: int) -> dict[str, Any]:
    if block_count > 20:
        return {
            "enumerated": False,
            "reason": f"2^{block_count} states exceed the exact-enumeration limit",
        }
    state_ids = np.arange(1 << block_count, dtype=np.uint32)
    spins = np.empty((state_ids.size, block_count), dtype=np.int8)
    for block in range(block_count):
        spins[:, block] = (1 - 2 * ((state_ids >> block) & 1)).astype(np.int8)

    energies = np.zeros(state_ids.size, dtype=float)
    for (block_i, block_j), coupling in couplings.items():
        energies -= coupling * spins[:, block_i] * spins[:, block_j]

    ground_energy = float(np.min(energies))
    ground = np.isclose(energies, ground_energy, rtol=0.0, atol=1e-9)
    local_minimum = np.ones(state_ids.size, dtype=bool)
    for block in range(block_count):
        local_minimum &= energies <= energies[state_ids ^ (1 << block)] + 1e-12

    levels = np.unique(np.round(energies, 10))
    sum_abs = float(sum(abs(value) for value in couplings.values()))
    unsatisfied_fraction = (
        (ground_energy + sum_abs) / (2.0 * sum_abs) if sum_abs > 0.0 else 0.0
    )
    return {
        "enumerated": True,
        "state_count": int(state_ids.size),
        "ground_energy": ground_energy,
        "ground_degeneracy": int(np.count_nonzero(ground)),
        "single_domain_flip_stable_states": int(np.count_nonzero(local_minimum)),
        "energy_level_count": int(levels.size),
        "first_excitation_gap": float(levels[1] - levels[0]) if levels.size > 1 else None,
        "weighted_unsatisfied_coupling_fraction": float(unsatisfied_fraction),
        "states_within_0.1_of_ground": int(np.count_nonzero(energies <= ground_energy + 0.1)),
        "states_within_0.5_of_ground": int(np.count_nonzero(energies <= ground_energy + 0.5)),
    }


def plaquette_frustration(
    couplings: dict[tuple[int, int], float], blocks_per_side: int
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in range(blocks_per_side - 1):
        for col in range(blocks_per_side - 1):
            a = row * blocks_per_side + col
            b = (row + 1) * blocks_per_side + col
            c = (row + 1) * blocks_per_side + col + 1
            d = row * blocks_per_side + col + 1
            edges = [tuple(sorted(pair)) for pair in ((a, b), (b, c), (c, d), (d, a))]
            values = [float(couplings.get(edge, 0.0)) for edge in edges]
            signs = [int(np.sign(value)) for value in values]
            product = int(np.prod(signs))
            records.append(
                {
                    "row": row,
                    "column": col,
                    "couplings": values,
                    "sign_product": product,
                    "frustrated": product < 0,
                }
            )
    return {
        "plaquette_count": len(records),
        "frustrated_plaquettes": sum(record["frustrated"] for record in records),
        "records": records,
    }


def markdown(report: dict[str, Any]) -> str:
    graph = report["graph"]
    lines = [
        "# Domain-scale reduction of the bond-frame rotor model",
        "",
        "Within a rotated square domain, both pair terms are minimized by either "
        "of two orthogonal directors. In the strong intra-domain-locking limit, "
        "the director choice is therefore a binary variable. Cross-domain edges "
        "induce an effective Ising model on the domain graph.",
        "",
        "## Graph audit",
        "",
        f"- Nodes: {graph['nodes']}",
        f"- Domains: {graph['block_count']}",
        f"- Total edges: {graph['total_edges']}",
        f"- Cross-domain edges: {graph['cross_edges']}",
        f"- Cross-domain coupling-weight fraction: {graph['cross_weight_fraction']:.6f}",
        f"- Doubled-angle order of prescribed domain frames: {graph['axis_order']:.3e}",
        "",
        "## Effective landscape",
        "",
        "| epsilon | frustrated plaquettes | stable states | ground degeneracy | gap | weighted frustration |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["epsilon_scan"]:
        landscape = row["landscape"]
        plaquettes = row["plaquettes"]
        lines.append(
            f"| {row['epsilon']:.6g} | {plaquettes['frustrated_plaquettes']}/{plaquettes['plaquette_count']} | "
            f"{landscape.get('single_domain_flip_stable_states', '--')} | "
            f"{landscape.get('ground_degeneracy', '--')} | "
            f"{landscape.get('first_excitation_gap', float('nan')):.5g} | "
            f"{landscape.get('weighted_unsatisfied_coupling_fraction', float('nan')):.5f} |"
        )
    seeds = report["seed_robustness"]
    lines.extend(
        [
            "",
            "## Graph-seed robustness",
            "",
            f"At epsilon={seeds['epsilon']:.6g}, {seeds['seed_count']} independently jittered "
            "mosaic graphs contain "
            f"{seeds['frustrated_plaquettes']['min']}--{seeds['frustrated_plaquettes']['max']} "
            "frustrated plaquettes and "
            f"{seeds['stable_states']['min']}--{seeds['stable_states']['max']} "
            "one-domain-flip-stable states. All have ground-state degeneracy "
            f"{seeds['ground_degeneracy_values']}.",
            "",
            "## Evidential interpretation",
            "",
            "Frustrated loops and multiple one-domain-flip-stable states establish a "
            "metastable domain-scale mechanism. They do not by themselves establish a "
            "thermodynamic glass phase. That claim additionally requires two-replica "
            "overlap statistics, waiting-time dependence, long-time mixing tests, and "
            "finite-size scaling under a graph family whose cross-domain weight does not "
            "vanish with system size.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--crosslink-k", type=int, default=2)
    parser.add_argument("--crosslink-weight", type=float, default=0.18)
    parser.add_argument("--graph-seed", type=int, default=12345)
    parser.add_argument(
        "--epsilon-values",
        type=parse_floats,
        default=parse_floats("0.75,0.95,1.05,1.15,1.35,1.55,1.614286,1.85"),
    )
    parser.add_argument("--robustness-epsilon", type=float, default=1.614286)
    parser.add_argument("--seed-count", type=int, default=20)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "pair_domain_reduction/pair_domain_reduction"
        ),
    )
    args = parser.parse_args()

    _, src, tgt, phi, weights, meta = make_graph(
        args.n,
        graph_mode="mosaic",
        graph_seed=args.graph_seed,
        cluster_size=args.cluster_size,
        crosslink_k=args.crosslink_k,
        crosslink_weight=args.crosslink_weight,
        patch_angle_step=math.pi / 4.0,
    )
    labels, axes, blocks_per_side = block_labels(args.n, args.cluster_size)
    axis_order = float(abs(np.mean(np.exp(2j * axes))))
    cross_mask = labels[src] != labels[tgt]

    epsilon_scan: list[dict[str, Any]] = []
    for epsilon in args.epsilon_values:
        couplings, cross_edges, cross_weight = induced_couplings(
            src=src,
            tgt=tgt,
            phi=phi,
            weights=weights,
            labels=labels,
            axes=axes,
            epsilon=epsilon,
        )
        epsilon_scan.append(
            {
                "epsilon": epsilon,
                "couplings": [
                    {"blocks": list(key), "value": value}
                    for key, value in sorted(couplings.items())
                ],
                "plaquettes": plaquette_frustration(couplings, blocks_per_side),
                "landscape": enumerate_landscape(couplings, blocks_per_side**2),
            }
        )

    seed_rows: list[dict[str, Any]] = []
    for offset in range(max(1, args.seed_count)):
        seed = args.graph_seed + offset
        _, seed_src, seed_tgt, seed_phi, seed_weights, _ = make_graph(
            args.n,
            graph_mode="mosaic",
            graph_seed=seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=math.pi / 4.0,
        )
        seed_couplings, _, _ = induced_couplings(
            src=seed_src,
            tgt=seed_tgt,
            phi=seed_phi,
            weights=seed_weights,
            labels=labels,
            axes=axes,
            epsilon=args.robustness_epsilon,
        )
        seed_landscape = enumerate_landscape(seed_couplings, blocks_per_side**2)
        seed_plaquettes = plaquette_frustration(seed_couplings, blocks_per_side)
        seed_rows.append(
            {
                "graph_seed": seed,
                "frustrated_plaquettes": seed_plaquettes["frustrated_plaquettes"],
                "single_domain_flip_stable_states": seed_landscape.get(
                    "single_domain_flip_stable_states"
                ),
                "ground_degeneracy": seed_landscape.get("ground_degeneracy"),
                "first_excitation_gap": seed_landscape.get("first_excitation_gap"),
                "weighted_unsatisfied_coupling_fraction": seed_landscape.get(
                    "weighted_unsatisfied_coupling_fraction"
                ),
            }
        )

    def range_summary(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in seed_rows], dtype=float)
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    report = {
        "reduction": {
            "block_variable": "exp(2 i theta_i) = s_b exp(2 i alpha_b), s_b in {-1,+1}",
            "effective_hamiltonian": "H_eff/J = -sum_(b,c) K_bc s_b s_c",
            "coupling": (
                "K_bc = sum_edges w_ij [cos 2(alpha_b-alpha_c) + "
                "epsilon cos 2(alpha_b+alpha_c-2 phi_ij)]"
            ),
        },
        "graph": {
            "n": args.n,
            "nodes": args.n**2,
            "blocks_per_side": blocks_per_side,
            "block_count": blocks_per_side**2,
            "cluster_size": args.cluster_size,
            "total_edges": int(src.size),
            "cross_edges": int(np.count_nonzero(cross_mask)),
            "cross_edge_fraction": float(np.mean(cross_mask)),
            "cross_weight_fraction": float(np.sum(weights[cross_mask]) / np.sum(weights)),
            "axis_order": axis_order,
            "graph_meta": meta,
        },
        "epsilon_scan": epsilon_scan,
        "seed_robustness": {
            "epsilon": args.robustness_epsilon,
            "seed_count": len(seed_rows),
            "frustrated_plaquettes": range_summary("frustrated_plaquettes"),
            "stable_states": range_summary("single_domain_flip_stable_states"),
            "first_excitation_gap": range_summary("first_excitation_gap"),
            "weighted_unsatisfied_coupling_fraction": range_summary(
                "weighted_unsatisfied_coupling_fraction"
            ),
            "ground_degeneracy_values": sorted(
                {int(row["ground_degeneracy"]) for row in seed_rows}
            ),
            "rows": seed_rows,
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output_prefix": str(args.output_prefix), "rows": len(epsilon_scan)}))


if __name__ == "__main__":
    main()
