#!/usr/bin/env python3
"""Intervene on loop holonomy in the colloidal domain-scale Hamiltonian.

For every quenched colloid graph, the script derives the domain couplings,
constructs the closest globally flat sign connection with identical ``|K_bc|``,
and compares exact metastable-state multiplicity and finite-temperature
write-release memory. A gauge-equivalent sign relabelling is the null control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_rotating_colloids_pair_domain_reduction import (  # noqa: E402
    block_labels,
    induced_couplings,
)
from rotating_colloids_hyperion_case import make_graph  # noqa: E402
from discovery.colloid_holonomy_memory import (  # noqa: E402
    build_intervention,
    exact_landscape,
    paired_memory_test,
    paired_metastable_retention_test,
)


def bootstrap_interval(values: Sequence[float], draws: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(max(1, draws))
    )
    return [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]


def sign_flip_p_value(values: Sequence[float]) -> float | None:
    values = tuple(float(value) for value in values)
    if not values:
        return None
    observed = sum(values) / len(values)
    if len(values) <= 20:
        null = []
        for mask in range(1 << len(values)):
            null.append(
                sum(value if (mask >> index) & 1 else -value for index, value in enumerate(values))
                / len(values)
            )
    else:
        rng = np.random.default_rng(314159)
        signs = rng.choice((-1.0, 1.0), size=(9999, len(values)))
        null = (signs @ np.asarray(values) / len(values)).tolist()
    return (1 + sum(value >= observed for value in null)) / (len(null) + 1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    labels, axes, blocks_per_side = block_labels(args.n, args.cluster_size)
    node_count = blocks_per_side**2
    for offset in range(args.seed_count):
        graph_seed = args.graph_seed + offset
        _, src, tgt, phi, weights, graph_meta = make_graph(
            args.n,
            graph_mode="mosaic",
            graph_seed=graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=args.domain_angle_step,
        )
        couplings, cross_edges, cross_weight = induced_couplings(
            src=src,
            tgt=tgt,
            phi=phi,
            weights=weights,
            labels=labels,
            axes=axes,
            epsilon=args.epsilon,
        )
        intervention = build_intervention(couplings, node_count, seed=args.seed + offset)
        original_landscape = exact_landscape(intervention.original, node_count)
        flat_landscape = exact_landscape(intervention.flat, node_count)
        gauge_landscape = exact_landscape(intervention.gauge_equivalent, node_count)
        memory = paired_memory_test(
            intervention,
            node_count,
            replicas=args.replicas,
            beta=args.beta,
            write_field=args.write_field,
            write_sweeps=args.write_sweeps,
            release_sweeps=args.release_sweeps,
            seed=args.seed + 1009 * offset,
        )
        metastable_retention = paired_metastable_retention_test(
            intervention,
            node_count,
            beta=args.beta,
            release_sweeps=args.release_sweeps,
            repeats_per_state=args.repeats_per_state,
            seed=args.seed + 2003 * offset,
        )
        magnitude_residual = max(
            abs(abs(value) - abs(intervention.flat[edge]))
            for edge, value in intervention.original.items()
        )
        row = {
            "graph_seed": graph_seed,
            "graph": graph_meta,
            "domain_count": node_count,
            "cross_edges": cross_edges,
            "cross_weight": cross_weight,
            "fundamental_cycles": len(intervention.original_fluxes),
            "negative_holonomies_original": sum(value < 0 for value in intervention.original_fluxes),
            "negative_holonomies_flat": sum(value < 0 for value in intervention.flat_fluxes),
            "coupling_magnitude_residual": magnitude_residual,
            "landscape": {
                "original": original_landscape,
                "flat": flat_landscape,
                "gauge_equivalent_control": gauge_landscape,
                "stable_state_excess": (
                    original_landscape["single_flip_stable_states"]
                    - flat_landscape["single_flip_stable_states"]
                ),
            },
            "memory": memory,
            "metastable_retention": metastable_retention,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "colloid_holonomy_intervention",
                    "seed": graph_seed,
                    "negative_cycles": row["negative_holonomies_original"],
                    "stable_original": original_landscape["single_flip_stable_states"],
                    "stable_flat": flat_landscape["single_flip_stable_states"],
                    "memory_auc_difference": memory["original_minus_flat_auc"],
                    "metastable_auc_difference": metastable_retention.get(
                        "original_minus_flat_auc"
                    ),
                }
            ),
            flush=True,
        )

    stable_excess = [float(row["landscape"]["stable_state_excess"]) for row in rows]
    memory_difference = [float(row["memory"]["original_minus_flat_auc"]) for row in rows]
    gauge_memory_residual = [float(row["memory"]["gauge_auc_residual"]) for row in rows]
    metastable_difference = [
        float(row["metastable_retention"]["original_minus_flat_auc"])
        for row in rows
        if row["metastable_retention"].get("decision") == "measured"
    ]
    metastable_gauge_residual = [
        float(row["metastable_retention"]["gauge_auc_residual"])
        for row in rows
        if row["metastable_retention"].get("decision") == "measured"
    ]
    stable_interval = bootstrap_interval(stable_excess, args.bootstrap_draws, args.seed + 1)
    memory_interval = bootstrap_interval(memory_difference, args.bootstrap_draws, args.seed + 2)
    metastable_interval = bootstrap_interval(
        metastable_difference, args.bootstrap_draws, args.seed + 3
    )
    structural_checks = {
        "original_has_nontrivial_holonomy": all(row["negative_holonomies_original"] > 0 for row in rows),
        "flat_control_has_trivial_holonomy": all(row["negative_holonomies_flat"] == 0 for row in rows),
        "coupling_magnitudes_preserved": all(row["coupling_magnitude_residual"] < 1e-12 for row in rows),
        "gauge_equivalent_landscape_preserved": all(
            row["landscape"]["original"]["single_flip_stable_states"]
            == row["landscape"]["gauge_equivalent_control"]["single_flip_stable_states"]
            and row["landscape"]["original"]["ground_degeneracy"]
            == row["landscape"]["gauge_equivalent_control"]["ground_degeneracy"]
            for row in rows
        ),
    }
    landscape_pass = bool(stable_interval[0] is not None and stable_interval[0] > 0)
    generic_memory_pass = bool(memory_interval[0] is not None and memory_interval[0] > 0)
    metastable_memory_pass = bool(
        metastable_interval[0] is not None and metastable_interval[0] > 0
    )
    gauge_pass = max(
        gauge_memory_residual + metastable_gauge_residual, default=float("inf")
    ) < 1e-12
    decision = (
        "pass_domain_reduction"
        if all(structural_checks.values())
        and landscape_pass
        and metastable_memory_pass
        and gauge_pass
        else "not_passed"
    )
    report = {
        "report_type": "colloid_holonomy_causal_intervention",
        "decision": decision,
        "claim": (
            "Nontrivial Z2 loop holonomy causes additional metastable domain states "
            "and increases history-specific write-release retention in the exact "
            "domain-scale reduction."
            if decision == "pass_domain_reduction"
            else "The paired intervention does not establish that loop holonomy causes colloidal memory."
        ),
        "scope": (
            "Causal numerical intervention on the strong-locking domain reduction; "
            "confirmation in the continuous-angle colloid dynamics remains a separate test."
        ),
        "intervention": (
            "Graph and every |K_bc| are fixed. Bond signs are projected to the closest "
            "globally flat Z2 connection. A gauge-equivalent sign relabelling controls "
            "for individual bond signs."
        ),
        "structural_checks": structural_checks,
        "landscape": {
            "decision": "pass" if landscape_pass else "not_passed",
            "mean_stable_state_excess": float(np.mean(stable_excess)) if stable_excess else 0.0,
            "graph_bootstrap_95_interval": stable_interval,
            "positive_seed_fraction": float(np.mean(np.asarray(stable_excess) > 0)) if stable_excess else 0.0,
        },
        "dynamic_memory": {
            "generic_pattern_decision": (
                "pass" if generic_memory_pass and gauge_pass else "not_passed"
            ),
            "mean_original_minus_flat_auc": float(np.mean(memory_difference)) if memory_difference else 0.0,
            "graph_bootstrap_95_interval": memory_interval,
            "paired_sign_flip_p_value": sign_flip_p_value(memory_difference),
            "positive_seed_fraction": float(np.mean(np.asarray(memory_difference) > 0)) if memory_difference else 0.0,
            "maximum_gauge_control_residual": max(gauge_memory_residual, default=None),
        },
        "holonomy_created_state_retention": {
            "decision": "pass" if metastable_memory_pass and gauge_pass else "not_passed",
            "mean_original_minus_flat_auc": (
                float(np.mean(metastable_difference)) if metastable_difference else 0.0
            ),
            "graph_bootstrap_95_interval": metastable_interval,
            "paired_sign_flip_p_value": sign_flip_p_value(metastable_difference),
            "positive_seed_fraction": (
                float(np.mean(np.asarray(metastable_difference) > 0))
                if metastable_difference
                else 0.0
            ),
            "maximum_gauge_control_residual": max(
                metastable_gauge_residual, default=None
            ),
        },
        "parameters": {
            "n": args.n,
            "cluster_size": args.cluster_size,
            "domain_count": node_count,
            "epsilon": args.epsilon,
            "seed_count": args.seed_count,
            "replicas": args.replicas,
            "beta": args.beta,
            "write_field": args.write_field,
            "write_sweeps": args.write_sweeps,
            "release_sweeps": args.release_sweeps,
            "repeats_per_state": args.repeats_per_state,
        },
        "rows": rows,
    }
    return report


def markdown(report: Mapping[str, Any]) -> str:
    landscape = report["landscape"]
    memory = report["dynamic_memory"]
    retained = report["holonomy_created_state_retention"]
    return "\n".join(
        [
            "# Causal intervention on colloidal loop holonomy",
            "",
            f"- Decision: `{report['decision']}`",
            f"- Scope: {report['scope']}",
            "",
            "The intervention preserves the domain graph and every coupling magnitude. "
            "It changes only the gauge-invariant products of coupling signs around closed paths.",
            "",
            "## Exact landscape",
            "",
            f"Mean excess of one-flip-stable states: `{landscape['mean_stable_state_excess']:.4f}`; "
            f"graph-bootstrap 95% interval: `{landscape['graph_bootstrap_95_interval']}`.",
            "",
            "## Write-release memory",
            "",
            f"Mean overlap-AUC difference: `{memory['mean_original_minus_flat_auc']:.6f}`; "
            f"graph-bootstrap 95% interval: `{memory['graph_bootstrap_95_interval']}`; "
            f"paired sign-flip p: `{memory['paired_sign_flip_p_value']}`.",
            "",
            "## Retention of holonomy-created metastable states",
            "",
            f"Mean overlap-AUC difference: `{retained['mean_original_minus_flat_auc']:.6f}`; "
            f"graph-bootstrap 95% interval: `{retained['graph_bootstrap_95_interval']}`; "
            f"paired sign-flip p: `{retained['paired_sign_flip_p_value']}`.",
            "",
            "## Interpretation",
            "",
            report["claim"],
            "",
        ]
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cluster-size", type=int, default=8)
    ap.add_argument("--crosslink-k", type=int, default=2)
    ap.add_argument("--crosslink-weight", type=float, default=0.18)
    ap.add_argument("--domain-angle-step", type=float, default=math.pi / 4.0)
    ap.add_argument("--epsilon", type=float, default=1.614286)
    ap.add_argument("--graph-seed", type=int, default=12345)
    ap.add_argument("--seed-count", type=int, default=12)
    ap.add_argument("--replicas", type=int, default=256)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--write-field", type=float, default=3.0)
    ap.add_argument("--write-sweeps", type=int, default=80)
    ap.add_argument("--release-sweeps", type=int, default=240)
    ap.add_argument("--repeats-per-state", type=int, default=8)
    ap.add_argument("--bootstrap-draws", type=int, default=4999)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument(
        "--out-json",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "holonomy_memory_intervention/holonomy_memory_intervention.json"
        ),
    )
    ap.add_argument(
        "--out-md",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "holonomy_memory_intervention/holonomy_memory_intervention.md"
        ),
    )
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = run(args)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(out_json), "markdown": str(out_md), "decision": report["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
