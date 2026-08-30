#!/usr/bin/env python3
"""Test loop holonomy in the continuous-angle rotating-colloid model.

The intervention rotates preferred capillary frames only on cross-domain
bonds.  It targets a globally flat domain-scale Z2 connection while retaining
the graph, edge weights, pair amplitudes and induced coupling magnitudes.  The
same write/release noise is used in every arm.  An exact phase-coordinate
transformation supplies the gauge-equivalent null.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from analyze_rotating_colloids_pair_domain_reduction import block_labels  # noqa: E402
from rotating_colloids_hyperion_case import make_graph  # noqa: E402
from discovery.colloid_holonomy_memory import (  # noqa: E402
    fundamental_cycle_fluxes,
    one_flip_stable_states,
)
from discovery.continuous_colloid_holonomy import (  # noqa: E402
    build_frame_intervention,
    gauge_equivalent_phases,
    induced_domain_couplings,
    simulate_write_release,
)


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def bootstrap_interval(values: Sequence[float], draws: int, seed: int) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(max(1, int(draws)))
    )
    return [
        float(samples[int(0.025 * (len(samples) - 1))]),
        float(samples[int(0.975 * (len(samples) - 1))]),
    ]


def graph_cluster_interval(
    records: Sequence[Mapping[str, Any]],
    value_key: str,
    *,
    draws: int,
    seed: int,
) -> tuple[float | None, list[float | None], dict[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for record in records:
        grouped[int(record["graph_seed"])].append(float(record[value_key]))
    graph_values = {
        graph_seed: float(np.mean(values)) for graph_seed, values in grouped.items()
    }
    values = list(graph_values.values())
    mean = float(np.mean(values)) if values else None
    return mean, bootstrap_interval(values, draws, seed), graph_values


def domain_target(axes: np.ndarray, labels: np.ndarray, spins: np.ndarray) -> np.ndarray:
    return np.remainder(axes + np.where(spins[labels] < 0, math.pi / 2.0, 0.0), math.pi)


def ground_state(couplings: Mapping[tuple[int, int], float], node_count: int) -> np.ndarray:
    state_ids = np.arange(1 << node_count, dtype=np.uint64)
    states = np.empty((state_ids.size, node_count), dtype=np.int8)
    for node in range(node_count):
        states[:, node] = (1 - 2 * ((state_ids >> node) & 1)).astype(np.int8)
    energy = np.zeros(state_ids.size, dtype=float)
    for (left, right), value in couplings.items():
        energy -= float(value) * states[:, left] * states[:, right]
    return states[int(np.argmin(energy))]


def target_catalog(
    *,
    axes: np.ndarray,
    labels: np.ndarray,
    blocks_per_side: int,
    original_couplings: Mapping[tuple[int, int], float],
    flat_couplings: Mapping[tuple[int, int], float],
    random_target_count: int,
    seed: int,
    flip_shell_sizes: Sequence[int] = (),
) -> list[dict[str, Any]]:
    domain_count = blocks_per_side**2
    targets: list[dict[str, Any]] = [
        {
            "name": "uniform_director",
            "selection": "prespecified",
            "angles": np.zeros_like(axes),
        },
        {
            "name": "mosaic_axes",
            "selection": "prespecified",
            "angles": axes.copy(),
        },
    ]
    checkerboard = np.asarray(
        [1 if ((index // blocks_per_side) + (index % blocks_per_side)) % 2 == 0 else -1
         for index in range(domain_count)],
        dtype=np.int8,
    )
    targets.append(
        {
            "name": "checkerboard_domains",
            "selection": "prespecified",
            "angles": domain_target(axes, labels, checkerboard),
        }
    )
    # Shells of k domains rotated by pi/2 from the uniform director. k = 0
    # reproduces the uniform director and k near half the domain count reaches
    # the region the random patterns occupy, so the family bridges the
    # compatibility gap that otherwise leaves the uniform target isolated.
    # Which domains are flipped is fixed by a dedicated seed and never uses a
    # measured outcome, so the family is prespecified.
    for shell in flip_shell_sizes:
        count = int(shell)
        if not 0 <= count <= domain_count:
            raise ValueError(f"flip shell {count} outside 0..{domain_count}")
        shell_rng = np.random.default_rng(90210 + 991 * count)
        flipped = shell_rng.permutation(domain_count)[:count]
        angles = np.zeros_like(axes)
        angles[np.isin(labels, flipped)] = math.pi / 2.0
        targets.append(
            {
                "name": f"flip_shell_{count:02d}",
                "selection": "prespecified",
                "angles": np.remainder(angles, math.pi),
            }
        )

    rng = np.random.default_rng(seed)
    for index in range(int(random_target_count)):
        spins = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=domain_count)
        spins[0] = 1
        targets.append(
            {
                "name": f"random_domains_{index + 1}",
                "selection": "prespecified",
                "angles": domain_target(axes, labels, spins),
            }
        )

    targets.append(
        {
            "name": "original_ground_state",
            "selection": "landscape_selected",
            "angles": domain_target(
                axes, labels, ground_state(original_couplings, domain_count)
            ),
        }
    )
    original_ids, original_states = one_flip_stable_states(original_couplings, domain_count)
    flat_ids, _ = one_flip_stable_states(flat_couplings, domain_count)
    created = np.flatnonzero(~np.isin(original_ids, flat_ids))
    if created.size:
        targets.append(
            {
                "name": "holonomy_created_stable_state",
                "selection": "landscape_selected",
                "angles": domain_target(axes, labels, original_states[int(created[0])]),
            }
        )
    return targets


def public_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "final_theta"}


def magnitude_residual(
    reference: Mapping[tuple[int, int], float],
    candidate: Mapping[tuple[int, int], float],
) -> tuple[float, float]:
    denominator = max(sum(abs(value) for value in reference.values()), 1e-12)
    errors = [
        abs(abs(candidate[edge]) - abs(reference[edge])) for edge in reference
    ]
    relative = [
        error / max(abs(reference[edge]), 1e-12)
        for edge, error in zip(reference, errors)
    ]
    return float(sum(errors) / denominator), float(max(relative, default=0.0))


def linear_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    x_a = np.asarray(x, dtype=float)
    y_a = np.asarray(y, dtype=float)
    if x_a.size < 2 or float(np.var(x_a)) < 1e-14:
        return None
    return float(np.cov(x_a, y_a, ddof=0)[0, 1] / np.var(x_a))


def run(args: argparse.Namespace) -> dict[str, Any]:
    labels, axes, blocks_per_side = block_labels(args.n, args.cluster_size)
    domain_count = blocks_per_side**2
    if domain_count > 20:
        raise ValueError("the exact target catalogue is limited to 20 domains")
    parameter_pairs = [
        (j_align, epsilon)
        for j_align in args.j_values
        for epsilon in args.epsilon_values
    ]
    records: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []

    for graph_offset in range(args.seed_count):
        graph_seed = args.graph_seed + graph_offset
        _, src, tgt, original_phi, weights, graph_meta = make_graph(
            args.n,
            graph_mode="mosaic",
            graph_seed=graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            patch_angle_step=args.domain_angle_step,
        )
        node_count = args.n * args.n
        laplacian = np.zeros((node_count, node_count), dtype=float)
        np.add.at(laplacian, (src, src), weights)
        np.add.at(laplacian, (tgt, tgt), weights)
        np.add.at(laplacian, (src, tgt), -weights)
        np.add.at(laplacian, (tgt, src), -weights)
        graph_spectrum = np.linalg.eigvalsh(laplacian)
        for parameter_index, (j_align, epsilon) in enumerate(parameter_pairs):
            g_capillary = float(j_align) * float(epsilon)
            intervention = build_frame_intervention(
                src=src,
                tgt=tgt,
                phi=original_phi,
                weights=weights,
                labels=labels,
                axes=axes,
                j_align=j_align,
                g_capillary=g_capillary,
                domain_count=domain_count,
            )
            control_rng = np.random.default_rng(args.seed + 7919 * graph_offset + parameter_index)
            control_gauge = control_rng.choice(np.asarray([-1, 1], dtype=np.int8), size=domain_count)
            control_gauge[0] = 1
            gauge_chi, gauge_phi, gauge_eta = gauge_equivalent_phases(
                src=src,
                tgt=tgt,
                phi=original_phi,
                labels=labels,
                gauge=control_gauge,
            )
            targets = target_catalog(
                axes=axes,
                labels=labels,
                blocks_per_side=blocks_per_side,
                original_couplings=intervention.original_couplings,
                flat_couplings=intervention.realized_couplings,
                random_target_count=args.random_target_count,
                flip_shell_sizes=args.flip_shell_sizes,
                seed=args.seed + 10007 * graph_offset,
            )
            dose_phases: dict[str, tuple[float, np.ndarray]] = {
                "original": (0.0, original_phi),
                "half_flattening": (0.5, original_phi + 0.5 * intervention.frame_shift),
                "flat_frame": (1.0, intervention.flat_phi),
                "inverse_frame": (-1.0, original_phi - intervention.frame_shift),
            }
            dose_structure: dict[str, dict[str, Any]] = {}
            for arm, (dose, arm_phi) in dose_phases.items():
                couplings = induced_domain_couplings(
                    src=src,
                    tgt=tgt,
                    phi=arm_phi,
                    weights=weights,
                    labels=labels,
                    axes=axes,
                    j_align=j_align,
                    g_capillary=g_capillary,
                )
                l1_residual, max_residual = magnitude_residual(
                    intervention.original_couplings, couplings
                )
                fluxes = fundamental_cycle_fluxes(couplings, domain_count)
                dose_structure[arm] = {
                    "dose": dose,
                    "negative_fluxes": int(sum(value < 0 for value in fluxes)),
                    "fundamental_cycles": len(fluxes),
                    "magnitude_relative_l1": l1_residual,
                    "magnitude_relative_max": max_residual,
                }
            structural_rows.append(
                {
                    "graph_seed": graph_seed,
                    "j_align": float(j_align),
                    "epsilon": float(epsilon),
                    "g_capillary": g_capillary,
                    "graph": graph_meta,
                    "weighted_graph_spectrum_invariants": {
                        "minimum": float(graph_spectrum[0]),
                        "maximum": float(graph_spectrum[-1]),
                        "trace": float(np.sum(graph_spectrum)),
                    },
                    "flat_frame_shift_rms": float(np.sqrt(np.mean(intervention.frame_shift**2))),
                    "flat_frame_shift_max": float(np.max(np.abs(intervention.frame_shift))),
                    "dose_structure": dose_structure,
                }
            )

            for target_index, target_record in enumerate(targets):
                target = np.asarray(target_record["angles"], dtype=float)
                initial_rng = np.random.default_rng(
                    args.seed + 104729 * graph_offset + 1543 * parameter_index + target_index
                )
                initial = initial_rng.uniform(
                    0.0, math.pi, size=(args.replicas, args.n * args.n)
                )
                noise_seed = (
                    args.seed + 1000003 * graph_offset + 10009 * parameter_index + target_index
                )
                arm_results: dict[str, dict[str, Any]] = {}
                raw_results: dict[str, Mapping[str, Any]] = {}
                for arm, (_, arm_phi) in dose_phases.items():
                    result = simulate_write_release(
                        src=src,
                        tgt=tgt,
                        phi=arm_phi,
                        weights=weights,
                        target=target,
                        j_align=j_align,
                        g_capillary=g_capillary,
                        replicas=args.replicas,
                        write_steps=args.write_steps,
                        release_steps=args.release_steps,
                        stride=args.stride,
                        dt=args.dt,
                        write_field=args.write_field,
                        initial_theta=initial,
                        noise_seed=noise_seed,
                    )
                    raw_results[arm] = result
                    arm_results[arm] = public_metrics(result)
                gauge_result = simulate_write_release(
                    src=src,
                    tgt=tgt,
                    phi=gauge_phi,
                    chi=gauge_chi,
                    weights=weights,
                    target=target + gauge_eta,
                    j_align=j_align,
                    g_capillary=g_capillary,
                    replicas=args.replicas,
                    write_steps=args.write_steps,
                    release_steps=args.release_steps,
                    stride=args.stride,
                    dt=args.dt,
                    write_field=args.write_field,
                    initial_theta=initial + gauge_eta[None, :],
                    noise_seed=noise_seed,
                )
                arm_results["gauge_equivalent"] = public_metrics(gauge_result)
                original = raw_results["original"]
                flat = raw_results["flat_frame"]
                gauge_curve_residual = float(
                    np.max(
                        np.abs(
                            np.asarray(original["overlap_curve"])
                            - np.asarray(gauge_result["overlap_curve"])
                        )
                    )
                )
                dose_arms = ("original", "half_flattening", "flat_frame")
                flux_fraction = [
                    dose_structure[arm]["negative_fluxes"]
                    / max(dose_structure[arm]["fundamental_cycles"], 1)
                    for arm in dose_arms
                ]
                dose_auc = [float(raw_results[arm]["overlap_auc"]) for arm in dose_arms]
                records.append(
                    {
                        "graph_seed": graph_seed,
                        "j_align": float(j_align),
                        "epsilon": float(epsilon),
                        "g_capillary": g_capillary,
                        "target": target_record["name"],
                        "target_selection": target_record["selection"],
                        "arms": arm_results,
                        "original_minus_flat_auc": float(
                            original["overlap_auc"] - flat["overlap_auc"]
                        ),
                        "original_minus_flat_q_ea": float(
                            original["q_ea"] - flat["q_ea"]
                        ),
                        "original_minus_flat_half_life": float(
                            original["half_life"] - flat["half_life"]
                        ),
                        "original_minus_flat_final_overlap": float(
                            original["final_overlap"] - flat["final_overlap"]
                        ),
                        "inverse_minus_original_auc": float(
                            raw_results["inverse_frame"]["overlap_auc"]
                            - original["overlap_auc"]
                        ),
                        "auc_slope_per_negative_flux_fraction": linear_slope(
                            flux_fraction, dose_auc
                        ),
                        "gauge_overlap_curve_residual": gauge_curve_residual,
                        "gauge_q_ea_residual": float(
                            abs(float(original["q_ea"]) - float(gauge_result["q_ea"]))
                        ),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "continuous_colloid_holonomy_target",
                            "graph_seed": graph_seed,
                            "j_align": j_align,
                            "epsilon": epsilon,
                            "target": target_record["name"],
                            "selection": target_record["selection"],
                            "original_minus_flat_auc": records[-1]["original_minus_flat_auc"],
                            "gauge_residual": gauge_curve_residual,
                        }
                    ),
                    flush=True,
                )

    fixed = [record for record in records if record["target_selection"] == "prespecified"]
    selected = [record for record in records if record["target_selection"] == "landscape_selected"]
    fixed_mean, fixed_interval, graph_effects = graph_cluster_interval(
        fixed,
        "original_minus_flat_auc",
        draws=args.bootstrap_draws,
        seed=args.seed + 1,
    )
    selected_mean, selected_interval, _ = graph_cluster_interval(
        selected,
        "original_minus_flat_auc",
        draws=args.bootstrap_draws,
        seed=args.seed + 2,
    )
    fixed_q_ea_mean, fixed_q_ea_interval, _ = graph_cluster_interval(
        fixed,
        "original_minus_flat_q_ea",
        draws=args.bootstrap_draws,
        seed=args.seed + 3,
    )
    fixed_half_life_mean, fixed_half_life_interval, _ = graph_cluster_interval(
        fixed,
        "original_minus_flat_half_life",
        draws=args.bootstrap_draws,
        seed=args.seed + 4,
    )
    fixed_final_mean, fixed_final_interval, _ = graph_cluster_interval(
        fixed,
        "original_minus_flat_final_overlap",
        draws=args.bootstrap_draws,
        seed=args.seed + 5,
    )
    target_summaries = {}
    for target_name in sorted({record["target"] for record in records}):
        subset = [record for record in records if record["target"] == target_name]
        mean, interval, _ = graph_cluster_interval(
            subset,
            "original_minus_flat_auc",
            draws=args.bootstrap_draws,
            seed=args.seed + 100 + len(target_summaries),
        )
        slopes = [
            float(record["auc_slope_per_negative_flux_fraction"])
            for record in subset
            if record["auc_slope_per_negative_flux_fraction"] is not None
        ]
        target_summaries[target_name] = {
            "selection": subset[0]["target_selection"],
            "mean_original_minus_flat_auc": mean,
            "graph_bootstrap_95_interval": interval,
            "mean_auc_slope_per_negative_flux_fraction": (
                float(np.mean(slopes)) if slopes else None
            ),
        }

    max_gauge_residual = max(
        max(record["gauge_overlap_curve_residual"], record["gauge_q_ea_residual"])
        for record in records
    )
    flat_rows = [row["dose_structure"]["flat_frame"] for row in structural_rows]
    structural_pass = all(
        row["negative_fluxes"] == 0
        and row["magnitude_relative_l1"] <= args.max_magnitude_residual
        for row in flat_rows
    )
    gauge_pass = max_gauge_residual <= args.max_gauge_residual
    written_fraction = float(
        np.mean(
            [
                record["arms"]["original"]["written_overlap"] >= args.min_written_overlap
                and record["arms"]["flat_frame"]["written_overlap"] >= args.min_written_overlap
                for record in fixed
            ]
        )
    )
    original_half_life_censored_fraction = float(
        np.mean(
            [record["arms"]["original"]["half_life_censored"] for record in fixed]
        )
    )
    flat_half_life_censored_fraction = float(
        np.mean(
            [record["arms"]["flat_frame"]["half_life_censored"] for record in fixed]
        )
    )
    positive_targets = sum(
        summary["selection"] == "prespecified"
        and summary["graph_bootstrap_95_interval"][0] is not None
        and summary["graph_bootstrap_95_interval"][0] > 0
        for summary in target_summaries.values()
    )
    sized = (
        args.seed_count >= args.min_causal_seed_count
        and len({record["target"] for record in fixed}) >= 4
        and len(parameter_pairs) >= 2
    )
    general_memory_pass = bool(
        sized
        and structural_pass
        and gauge_pass
        and written_fraction >= 0.9
        and fixed_interval[0] is not None
        and fixed_interval[0] > 0
        and positive_targets >= 2
    )
    if general_memory_pass:
        decision = "holonomy_increases_continuous_angle_memory"
    elif not sized:
        decision = "cpu_pilot_not_sized_for_causal_claim"
    else:
        decision = "general_memory_causation_not_established"

    return {
        "report_type": "continuous_angle_colloid_holonomy_intervention",
        "decision": decision,
        "scientific_scope": (
            "Causal intervention on cross-domain capillary frames in the continuous-angle "
            "mosaic pair Hamiltonian. The primary estimate averages only prespecified written "
            "targets; landscape-selected targets are reported separately."
        ),
        "reconciliation": (
            "The exact domain reduction measures landscape capacity. Dynamic retention measures "
            "the stability of a written basin under thermal release. Additional metastable basins "
            "do not imply that an arbitrary or selected basin has a longer retention time."
        ),
        "reconciliation_with_targeted_dose_test": (
            "The earlier dose construction chose a separate gauge representative at every dose "
            "to favor one all-positive target and then kept that target fixed. Its negative slope "
            "is therefore a target-network compatibility effect. In the gauge null used here, "
            "the Hamiltonian, state and written target transform together, and the full dynamic "
            "curve is unchanged."
        ),
        "intervention": {
            "physical_change": (
                "Preferred quadrupolar interaction phases are rotated on cross-domain bonds."
            ),
            "fixed": [
                "particle graph",
                "local degree distribution",
                "edge weights",
                "alignment and capillary amplitudes",
                "positional disorder",
                "weighted graph spectrum",
                "write and release schedule",
                "thermal-noise realization",
            ],
            "matched": (
                "The full-dose arm has a globally flat domain-scale Z2 connection and matches "
                "the original induced |K_bc| within the reported residual."
            ),
            "gauge_control": (
                "An exact phase-coordinate transformation changes domain signs without changing "
                "the Hamiltonian or any transformed-target trajectory."
            ),
            "realization_level": (
                "The intervention is exact for a bond-addressable continuous-angle Hamiltonian. "
                "A laboratory implementation requires patterned boundary anisotropy that controls "
                "the cross-domain interaction phase, or a geometric embedding that realizes the "
                "same bond-angle changes at fixed distances."
            ),
            "unpatterned_capillary_limit": (
                "For an unpatterned capillary monolayer, phi_ij is the line-of-centres angle and "
                "cannot be varied independently at fixed particle coordinates. No coordinate-level "
                "embedding is asserted by this calculation."
            ),
        },
        "structural_checks": {
            "flat_flux_and_magnitude_match": structural_pass,
            "maximum_flat_magnitude_relative_l1": max(
                (row["magnitude_relative_l1"] for row in flat_rows), default=None
            ),
            "maximum_flat_magnitude_relative_max": max(
                (row["magnitude_relative_max"] for row in flat_rows), default=None
            ),
            "maximum_gauge_dynamic_residual": max_gauge_residual,
            "gauge_control_pass": gauge_pass,
            "written_target_fraction": written_fraction,
        },
        "primary_prespecified_targets": {
            "mean_original_minus_flat_auc": fixed_mean,
            "graph_bootstrap_95_interval": fixed_interval,
            "graph_equal_effects": graph_effects,
            "positive_target_families": positive_targets,
            "mean_original_minus_flat_q_ea": fixed_q_ea_mean,
            "q_ea_graph_bootstrap_95_interval": fixed_q_ea_interval,
            "mean_original_minus_flat_half_life": fixed_half_life_mean,
            "half_life_graph_bootstrap_95_interval": fixed_half_life_interval,
            "original_half_life_censored_fraction": original_half_life_censored_fraction,
            "flat_half_life_censored_fraction": flat_half_life_censored_fraction,
            "mean_original_minus_flat_final_overlap": fixed_final_mean,
            "final_overlap_graph_bootstrap_95_interval": fixed_final_interval,
            "decision": "pass" if general_memory_pass else "not_passed",
        },
        "landscape_selected_targets": {
            "mean_original_minus_flat_auc": selected_mean,
            "graph_bootstrap_95_interval": selected_interval,
            "interpretation": (
                "Selection-conditioned basin response; it is not an estimate of general memory."
            ),
        },
        "target_summaries": target_summaries,
        "parameters": {
            "n": args.n,
            "cluster_size": args.cluster_size,
            "domain_count": domain_count,
            "graph_seed": args.graph_seed,
            "seed_count": args.seed_count,
            "j_values": args.j_values,
            "epsilon_values": args.epsilon_values,
            "replicas": args.replicas,
            "write_steps": args.write_steps,
            "release_steps": args.release_steps,
            "stride": args.stride,
            "dt": args.dt,
            "write_field": args.write_field,
            "min_causal_seed_count": args.min_causal_seed_count,
            "crosslink_k": args.crosslink_k,
            "crosslink_weight": args.crosslink_weight,
            "domain_angle_step": args.domain_angle_step,
            "random_target_count": args.random_target_count,
            "flip_shell_sizes": args.flip_shell_sizes,
            "seed": args.seed,
        },
        "structural_rows": structural_rows,
        "records": records,
    }


def markdown(report: Mapping[str, Any]) -> str:
    primary = report["primary_prespecified_targets"]
    selected = report["landscape_selected_targets"]
    checks = report["structural_checks"]
    lines = [
        "# Continuous-angle intervention on colloidal loop holonomy",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Scope: {report['scientific_scope']}",
        "",
        "## Intervention",
        "",
        report["intervention"]["physical_change"],
        "The particle graph, every edge amplitude, positional disorder and the complete "
        "write-release history are identical between arms. Only the preferred capillary "
        "frame of cross-domain bonds is rotated.",
        "",
        "## Structural match",
        "",
        f"- Flat loop flux and coupling-magnitude match: `{checks['flat_flux_and_magnitude_match']}`",
        f"- Maximum relative L1 residual in |K|: `{checks['maximum_flat_magnitude_relative_l1']}`",
        f"- Maximum gauge-control dynamic residual: `{checks['maximum_gauge_dynamic_residual']}`",
        f"- Fraction of prespecified targets written above threshold: `{checks['written_target_fraction']:.3f}`",
        "",
        "## Prespecified written states",
        "",
        f"Mean original-minus-flat retention AUC: `{primary['mean_original_minus_flat_auc']}`; "
        f"graph-bootstrap 95% interval: `{primary['graph_bootstrap_95_interval']}`.",
        f"Mean qEA difference: `{primary['mean_original_minus_flat_q_ea']}`; "
        f"95% interval: `{primary['q_ea_graph_bootstrap_95_interval']}`.",
        f"Mean half-life difference: `{primary['mean_original_minus_flat_half_life']}`; "
        f"95% interval: `{primary['half_life_graph_bootstrap_95_interval']}`; "
        f"censored fractions original/flat: `{primary['original_half_life_censored_fraction']}`/"
        f"`{primary['flat_half_life_censored_fraction']}`.",
        f"Mean final-overlap difference: `{primary['mean_original_minus_flat_final_overlap']}`; "
        f"95% interval: `{primary['final_overlap_graph_bootstrap_95_interval']}`.",
        "",
        "## Landscape-selected states",
        "",
        f"Mean original-minus-flat retention AUC: `{selected['mean_original_minus_flat_auc']}`; "
        f"graph-bootstrap 95% interval: `{selected['graph_bootstrap_95_interval']}`. "
        f"{selected['interpretation']}",
        "",
        "## Reconciliation",
        "",
        report["reconciliation"],
        report["reconciliation_with_targeted_dose_test"],
        "",
        "The calculation is exact at the bond-phase Hamiltonian level. "
        f"{report['intervention']['unpatterned_capillary_limit']}",
        "",
        "## Target-resolved effects",
        "",
        "| target | selection | original - flat AUC | graph-bootstrap 95% interval | dose slope |",
        "|---|---|---:|---:|---:|",
    ]
    for name, summary in report["target_summaries"].items():
        lines.append(
            f"| {name} | {summary['selection']} | {summary['mean_original_minus_flat_auc']} | "
            f"{summary['graph_bootstrap_95_interval']} | "
            f"{summary['mean_auc_slope_per_negative_flux_fraction']} |"
        )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--cluster-size", type=int, default=2)
    ap.add_argument("--crosslink-k", type=int, default=2)
    ap.add_argument("--crosslink-weight", type=float, default=0.18)
    ap.add_argument("--domain-angle-step", type=float, default=math.pi / 4.0)
    ap.add_argument("--graph-seed", type=int, default=12345)
    ap.add_argument("--seed-count", type=int, default=3)
    ap.add_argument("--j-values", type=parse_floats, default=parse_floats("0.8,1.6"))
    ap.add_argument("--epsilon-values", type=parse_floats, default=parse_floats("1.614286"))
    ap.add_argument("--random-target-count", type=int, default=1)
    ap.add_argument(
        "--flip-shell-sizes",
        type=lambda text: [int(part) for part in text.split(",") if part.strip()],
        default=[],
        help="Prespecified shells of k domains rotated from the uniform director, "
             "e.g. 0,2,4,6,8,10,12,14,16.",
    )
    ap.add_argument("--replicas", type=int, default=12)
    ap.add_argument("--write-steps", type=int, default=120)
    ap.add_argument("--release-steps", type=int, default=300)
    ap.add_argument("--stride", type=int, default=15)
    ap.add_argument("--dt", type=float, default=0.003)
    ap.add_argument("--write-field", type=float, default=6.0)
    ap.add_argument("--min-written-overlap", type=float, default=0.70)
    ap.add_argument("--max-magnitude-residual", type=float, default=1e-8)
    ap.add_argument("--max-gauge-residual", type=float, default=1e-8)
    ap.add_argument("--min-causal-seed-count", type=int, default=8)
    ap.add_argument("--bootstrap-draws", type=int, default=4999)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument(
        "--out-json",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "continuous_holonomy_memory/continuous_holonomy_memory.json"
        ),
    )
    ap.add_argument(
        "--out-md",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "continuous_holonomy_memory/continuous_holonomy_memory.md"
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
    print(
        json.dumps(
            {
                "json": str(out_json),
                "markdown": str(out_md),
                "decision": report["decision"],
                "primary": report["primary_prespecified_targets"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
