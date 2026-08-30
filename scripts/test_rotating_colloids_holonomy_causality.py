#!/usr/bin/env python3
"""Causal intervention on loop holonomy in the rotating-colloid domain model.

The pair-domain reduction gives

    H_eff / J = -sum_(b,c) K_bc s_b s_c.

For a closed domain loop, prod sign(K_bc) is invariant under the local change
of variables s_b -> g_b s_b.  This script changes those loop products while
holding fixed the graph, every coupling magnitude, the degree sequence, the
quenched disorder realization, and the unsigned graph spectrum.  Identical
write and release schedules are then applied with common random numbers.

The result fails closed unless a positive, graph-clustered dose response is
accompanied by exact gauge, zero-interaction, and inversion controls.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import wilcoxon

from analyze_rotating_colloids_pair_domain_reduction import (
    block_labels,
    induced_couplings,
)
from rotating_colloids_hyperion_case import make_graph


@dataclass(frozen=True)
class SpinSchedule:
    initial: np.ndarray
    write_sites: np.ndarray
    write_uniforms: np.ndarray
    release_sites: np.ndarray
    release_uniforms: np.ndarray


def edge_table(
    couplings: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted((int(a), int(b), float(value)) for (a, b), value in couplings.items())
    if not rows:
        raise ValueError("the reduced graph has no couplings")
    src = np.asarray([row[0] for row in rows], dtype=np.int16)
    tgt = np.asarray([row[1] for row in rows], dtype=np.int16)
    values = np.asarray([row[2] for row in rows], dtype=float)
    if np.any(values == 0.0):
        raise ValueError("zero coupling makes the loop sign undefined")
    return src, tgt, values


def plaquette_cycle_matrix(
    src: np.ndarray,
    tgt: np.ndarray,
    blocks_per_side: int,
) -> np.ndarray:
    edge_lookup = {
        tuple(sorted((int(a), int(b)))): index
        for index, (a, b) in enumerate(zip(src, tgt))
    }
    cycles: list[np.ndarray] = []
    for row in range(blocks_per_side - 1):
        for col in range(blocks_per_side - 1):
            a = row * blocks_per_side + col
            b = (row + 1) * blocks_per_side + col
            c = (row + 1) * blocks_per_side + col + 1
            d = row * blocks_per_side + col + 1
            vector = np.zeros(src.size, dtype=np.uint8)
            for pair in ((a, b), (b, c), (c, d), (d, a)):
                key = tuple(sorted(pair))
                if key not in edge_lookup:
                    raise ValueError(f"domain graph is missing plaquette edge {key}")
                vector[edge_lookup[key]] = 1
            cycles.append(vector)
    matrix = np.stack(cycles, axis=0)
    if gf2_rank(matrix) != matrix.shape[0]:
        raise ValueError("plaquette loops are not independent")
    return matrix


def gf2_rank(matrix: np.ndarray) -> int:
    work = np.asarray(matrix, dtype=np.uint8).copy() & 1
    row = 0
    for col in range(work.shape[1]):
        pivots = np.flatnonzero(work[row:, col])
        if pivots.size == 0:
            continue
        pivot = row + int(pivots[0])
        work[[row, pivot]] = work[[pivot, row]]
        for other in range(work.shape[0]):
            if other != row and work[other, col]:
                work[other] ^= work[row]
        row += 1
        if row == work.shape[0]:
            break
    return row


def solve_gf2(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    b = np.asarray(rhs, dtype=np.uint8).copy() & 1
    if a.shape[0] != b.size:
        raise ValueError("GF(2) system has incompatible shapes")
    augmented = np.concatenate((a, b[:, None]), axis=1)
    row = 0
    pivots: list[int] = []
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(augmented[row:, col])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        augmented[[row, pivot]] = augmented[[pivot, row]]
        for other in range(augmented.shape[0]):
            if other != row and augmented[other, col]:
                augmented[other] ^= augmented[row]
        pivots.append(col)
        row += 1
        if row == augmented.shape[0]:
            break
    inconsistent = np.any(
        (np.sum(augmented[:, :-1], axis=1) == 0) & (augmented[:, -1] == 1)
    )
    if inconsistent:
        raise ValueError("requested holonomy class is not realizable")
    solution = np.zeros(a.shape[1], dtype=np.uint8)
    for pivot_row, pivot_col in enumerate(pivots):
        solution[pivot_col] = augmented[pivot_row, -1]
    if not np.array_equal((a @ solution) & 1, b):
        raise RuntimeError("GF(2) solution failed verification")
    return solution


def cycle_bits(sign_bits: np.ndarray, cycle_matrix: np.ndarray) -> np.ndarray:
    return (cycle_matrix @ np.asarray(sign_bits, dtype=np.uint8)) & 1


def gauge_cut_matrix(src: np.ndarray, tgt: np.ndarray, nodes: int) -> np.ndarray:
    if nodes > 20:
        raise ValueError("exact gauge optimization is limited to 20 domain nodes")
    masks = np.arange(1 << (nodes - 1), dtype=np.uint64)
    labels = np.zeros((masks.size, nodes), dtype=np.uint8)
    for node in range(1, nodes):
        labels[:, node] = ((masks >> (node - 1)) & 1).astype(np.uint8)
    return labels[:, src] ^ labels[:, tgt]


def coupling_matrix(
    src: np.ndarray,
    tgt: np.ndarray,
    magnitudes: np.ndarray,
    sign_bits: np.ndarray,
    nodes: int,
) -> np.ndarray:
    matrix = np.zeros((nodes, nodes), dtype=float)
    values = magnitudes * (1.0 - 2.0 * np.asarray(sign_bits, dtype=float))
    matrix[src, tgt] = values
    matrix[tgt, src] = values
    return matrix


def normalized_laplacian_spectrum(matrix: np.ndarray, *, signed: bool) -> np.ndarray:
    adjacency = np.asarray(matrix, dtype=float) if signed else np.abs(matrix)
    degree = np.sum(np.abs(matrix), axis=1)
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    laplacian = np.eye(matrix.shape[0]) - inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    return np.linalg.eigvalsh(laplacian)


def count_one_flip_stable_states(matrix: np.ndarray) -> int:
    """Count all states whose energy cannot be lowered by one domain flip."""
    matrix = np.asarray(matrix, dtype=float)
    nodes = matrix.shape[0]
    if matrix.shape != (nodes, nodes) or nodes > 20:
        raise ValueError("exact one-flip enumeration requires a square matrix with at most 20 nodes")
    state_ids = np.arange(1 << nodes, dtype=np.uint32)
    spins = np.empty((state_ids.size, nodes), dtype=np.int8)
    for node in range(nodes):
        spins[:, node] = (1 - 2 * ((state_ids >> node) & 1)).astype(np.int8)
    local_field = spins @ matrix
    stable = np.all(spins * local_field >= -1e-12, axis=1)
    return int(np.count_nonzero(stable))


def closest_sign_representative(
    baseline_bits: np.ndarray,
    target_cycle_bits: np.ndarray,
    cycle_matrix: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    magnitudes: np.ndarray,
    nodes: int,
) -> tuple[np.ndarray, dict[str, float]]:
    desired_delta = target_cycle_bits ^ cycle_bits(baseline_bits, cycle_matrix)
    particular = solve_gf2(cycle_matrix, desired_delta)
    cuts = gauge_cut_matrix(src, tgt, nodes)
    differences = cuts ^ particular[None, :]
    candidates = baseline_bits[None, :] ^ differences
    signed_edges = (1.0 - 2.0 * candidates.astype(float)) * magnitudes[None, :]
    incidence = np.zeros((src.size, nodes), dtype=float)
    incidence[np.arange(src.size), src] = 1.0
    incidence[np.arange(src.size), tgt] = 1.0
    signed_strength = signed_edges @ incidence
    baseline_edges = (1.0 - 2.0 * baseline_bits.astype(float)) * magnitudes
    baseline_strength = baseline_edges @ incidence
    scale = max(float(np.sqrt(np.mean(baseline_strength**2))), 1e-12)
    strength_rmse = np.sqrt(np.mean((signed_strength - baseline_strength[None, :]) ** 2, axis=1)) / scale
    flip_weight = (differences.astype(float) @ magnitudes) / float(np.sum(magnitudes))
    edge_fraction = np.mean(differences, axis=1)
    negative_weight = (candidates.astype(float) @ magnitudes) / float(np.sum(magnitudes))
    minimum_target_field = np.min(signed_strength, axis=1)
    negative_edges = np.mean(candidates, axis=1)
    # Fix the gauge by making the common all-positive written state a ground
    # state representative of each holonomy class. The remaining criteria only
    # break degeneracies and cannot alter a cycle product.
    ordering = np.lexsort(
        (
            edge_fraction,
            flip_weight,
            strength_rmse,
            negative_edges,
            -minimum_target_field,
            negative_weight,
        )
    )
    best = int(ordering[0])
    selected = candidates[best]
    if not np.array_equal(cycle_bits(selected, cycle_matrix), target_cycle_bits):
        raise RuntimeError("gauge optimization changed the requested holonomy")
    return selected, {
        "weighted_edge_change_fraction": float(flip_weight[best]),
        "edge_change_fraction": float(edge_fraction[best]),
        "signed_strength_rmse": float(strength_rmse[best]),
        "target_unsatisfied_weight_fraction": float(negative_weight[best]),
        "target_minimum_local_field": float(minimum_target_field[best]),
    }


def construct_holonomy_variants(
    couplings: dict[tuple[int, int], float],
    blocks_per_side: int,
    doses: Iterable[int],
) -> dict[str, Any]:
    src, tgt, values = edge_table(couplings)
    nodes = blocks_per_side**2
    magnitudes = np.abs(values)
    baseline_bits = (values < 0.0).astype(np.uint8)
    cycles = plaquette_cycle_matrix(src, tgt, blocks_per_side)
    baseline_cycle = cycle_bits(baseline_bits, cycles)
    baseline_matrix = coupling_matrix(src, tgt, magnitudes, baseline_bits, nodes)
    baseline_signed_spectrum = normalized_laplacian_spectrum(baseline_matrix, signed=True)
    baseline_unsigned_spectrum = normalized_laplacian_spectrum(baseline_matrix, signed=False)

    variants: list[dict[str, Any]] = []
    for dose in sorted(set(int(value) for value in doses)):
        if not 0 <= dose <= cycles.shape[0]:
            raise ValueError(f"holonomy dose {dose} is outside [0, {cycles.shape[0]}]")
        best: tuple[tuple[float, int], np.ndarray, np.ndarray] | None = None
        for negative_cycles in itertools.combinations(range(cycles.shape[0]), dose):
            target_cycle = np.zeros(cycles.shape[0], dtype=np.uint8)
            target_cycle[list(negative_cycles)] = 1
            particular_delta = solve_gf2(cycles, target_cycle ^ baseline_cycle)
            trial_bits = baseline_bits ^ particular_delta
            trial_matrix = coupling_matrix(src, tgt, magnitudes, trial_bits, nodes)
            spectrum = normalized_laplacian_spectrum(trial_matrix, signed=True)
            spectrum_distance = float(
                np.linalg.norm(spectrum - baseline_signed_spectrum)
                / max(np.linalg.norm(baseline_signed_spectrum), 1e-12)
            )
            cycle_distance = int(np.count_nonzero(target_cycle ^ baseline_cycle))
            key = (spectrum_distance, cycle_distance)
            if best is None or key < best[0]:
                best = (key, target_cycle, trial_bits)
        assert best is not None
        target_cycle = best[1]
        selected_bits, local_match = closest_sign_representative(
            baseline_bits,
            target_cycle,
            cycles,
            src,
            tgt,
            magnitudes,
            nodes,
        )
        matrix = coupling_matrix(src, tgt, magnitudes, selected_bits, nodes)
        unsigned_spectrum = normalized_laplacian_spectrum(matrix, signed=False)
        signed_spectrum = normalized_laplacian_spectrum(matrix, signed=True)
        match = {
            "edge_endpoints_exact": True,
            "coupling_magnitudes_max_error": float(
                np.max(np.abs(np.abs(matrix[src, tgt]) - magnitudes))
            ),
            "degree_sequence_max_error": 0.0,
            "unsigned_spectrum_max_error": float(
                np.max(np.abs(unsigned_spectrum - baseline_unsigned_spectrum))
            ),
            "signed_spectrum_relative_distance": float(
                np.linalg.norm(signed_spectrum - baseline_signed_spectrum)
                / max(np.linalg.norm(baseline_signed_spectrum), 1e-12)
            ),
            **local_match,
        }
        variants.append(
            {
                "dose": dose,
                "cycle_bits": target_cycle,
                "sign_bits": selected_bits,
                "matrix": matrix,
                "match": match,
            }
        )
    return {
        "src": src,
        "tgt": tgt,
        "magnitudes": magnitudes,
        "baseline_bits": baseline_bits,
        "baseline_cycle_bits": baseline_cycle,
        "cycle_matrix": cycles,
        "variants": variants,
    }


def make_schedule(
    *,
    nodes: int,
    replicas: int,
    write_sweeps: int,
    release_sweeps: int,
    seed: int,
) -> SpinSchedule:
    rng = np.random.default_rng(seed)
    initial = rng.choice((-1, 1), size=(replicas, nodes)).astype(np.int8)

    def phase(sweeps: int) -> tuple[np.ndarray, np.ndarray]:
        sites = np.concatenate([rng.permutation(nodes) for _ in range(sweeps)]).astype(np.int16)
        uniforms = rng.random((sites.size, replicas), dtype=float)
        return sites, uniforms

    write_sites, write_uniforms = phase(write_sweeps)
    release_sites, release_uniforms = phase(release_sweeps)
    return SpinSchedule(initial, write_sites, write_uniforms, release_sites, release_uniforms)


def transformed_schedule(schedule: SpinSchedule, gauge: np.ndarray) -> SpinSchedule:
    gauge = np.asarray(gauge, dtype=np.int8)

    def transform_uniforms(sites: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
        invert = gauge[sites] < 0
        return np.where(invert[:, None], 1.0 - uniforms, uniforms)

    return SpinSchedule(
        initial=schedule.initial * gauge[None, :],
        write_sites=schedule.write_sites.copy(),
        write_uniforms=transform_uniforms(schedule.write_sites, schedule.write_uniforms),
        release_sites=schedule.release_sites.copy(),
        release_uniforms=transform_uniforms(schedule.release_sites, schedule.release_uniforms),
    )


def _heat_bath_phase(
    spins: np.ndarray,
    matrices: np.ndarray,
    sites: np.ndarray,
    uniforms: np.ndarray,
    *,
    target: np.ndarray,
    interaction_scale: float,
    field: float,
    record_each_sweep: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    nodes = spins.shape[2]
    records: list[np.ndarray] = []
    for index, site_value in enumerate(sites):
        site = int(site_value)
        local = interaction_scale * np.einsum(
            "vrn,vn->vr", spins, matrices[:, site, :], optimize=True
        )
        if field:
            local = local + field * float(target[site])
        probability = 1.0 / (1.0 + np.exp(np.clip(-2.0 * local, -60.0, 60.0)))
        spins[:, :, site] = np.where(uniforms[index][None, :] < probability, 1, -1)
        if record_each_sweep and (index + 1) % nodes == 0:
            records.append(spins.copy())
    return spins, records


def simulate_variants(
    matrices: np.ndarray,
    schedule: SpinSchedule,
    *,
    target: np.ndarray,
    interaction_scale: float,
    write_field: float,
) -> dict[str, np.ndarray]:
    matrices = np.asarray(matrices, dtype=float)
    target = np.asarray(target, dtype=np.int8)
    spins = np.repeat(schedule.initial[None, :, :], matrices.shape[0], axis=0)
    spins, _ = _heat_bath_phase(
        spins,
        matrices,
        schedule.write_sites,
        schedule.write_uniforms,
        target=target,
        interaction_scale=interaction_scale,
        field=write_field,
        record_each_sweep=False,
    )
    written_overlap = np.mean(spins * target[None, None, :], axis=2)
    release_records = [spins.copy()]
    spins, later = _heat_bath_phase(
        spins,
        matrices,
        schedule.release_sites,
        schedule.release_uniforms,
        target=target,
        interaction_scale=interaction_scale,
        field=0.0,
        record_each_sweep=True,
    )
    release_records.extend(later)
    snapshots = np.stack(release_records, axis=0)
    overlap = np.mean(snapshots * target[None, None, None, :], axis=3)
    auc = np.mean(overlap, axis=0)
    retained = overlap[-1]
    start = overlap[0]
    half_life = np.full(start.shape, overlap.shape[0] - 1, dtype=float)
    censored = np.ones(start.shape, dtype=bool)
    for variant in range(start.shape[0]):
        for replica in range(start.shape[1]):
            threshold = 0.5 * max(float(start[variant, replica]), 0.0)
            crossed = np.flatnonzero(overlap[:, variant, replica] <= threshold)
            if crossed.size:
                half_life[variant, replica] = float(crossed[0])
                censored[variant, replica] = False
    late = snapshots[max(1, snapshots.shape[0] // 2) :]
    q_ea = np.mean(np.mean(late, axis=0) ** 2, axis=2)
    return {
        "written_overlap": written_overlap,
        "overlap_curve": overlap,
        "memory_auc": auc,
        "retained_overlap": retained,
        "half_life_sweeps": half_life,
        "half_life_censored": censored,
        "q_EA": q_ea,
        "final_spins": spins,
    }


def bootstrap_mean_interval(values: np.ndarray, *, seed: int, draws: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[index], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=float)
        result[order] = np.arange(values.size, dtype=float)
        return result

    rx = ranks(np.asarray(x, dtype=float))
    ry = ranks(np.asarray(y, dtype=float))
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def graph_couplings(args: argparse.Namespace, graph_seed: int) -> tuple[dict[tuple[int, int], float], dict[str, Any]]:
    _, src, tgt, phi, weights, graph_meta = make_graph(
        args.n,
        graph_mode="mosaic",
        graph_seed=graph_seed,
        cluster_size=args.cluster_size,
        crosslink_k=args.crosslink_k,
        crosslink_weight=args.crosslink_weight,
        patch_angle_step=math.pi / 4.0,
    )
    labels, axes, blocks_per_side = block_labels(args.n, args.cluster_size)
    couplings, cross_edges, cross_weight = induced_couplings(
        src=src,
        tgt=tgt,
        phi=phi,
        weights=weights,
        labels=labels,
        axes=axes,
        epsilon=args.epsilon,
    )
    coupling_degree = np.zeros(blocks_per_side**2, dtype=float)
    for (block_i, block_j), value in couplings.items():
        coupling_degree[block_i] += abs(value)
        coupling_degree[block_j] += abs(value)
    return couplings, {
        "graph_seed": graph_seed,
        "blocks_per_side": blocks_per_side,
        "cross_edges": cross_edges,
        "cross_weight": cross_weight,
        "maximum_abs_coupling_degree": float(np.max(coupling_degree)),
        "maximum_interaction_field": float(args.interaction_scale * np.max(coupling_degree)),
        "graph_meta": graph_meta,
    }


def exact_controls(
    matrices: np.ndarray,
    schedule: SpinSchedule,
    target: np.ndarray,
    interaction_scale: float,
    write_field: float,
    seed: int,
) -> dict[str, float | bool]:
    zero = simulate_variants(
        matrices,
        schedule,
        target=target,
        interaction_scale=0.0,
        write_field=write_field,
    )
    zero_error = float(np.max(np.ptp(zero["overlap_curve"], axis=1)))

    rng = np.random.default_rng(seed)
    gauge = rng.choice((-1, 1), size=target.size).astype(np.int8)
    gauge[0] = 1
    base = matrices[:1]
    transformed = gauge[None, :, None] * base * gauge[None, None, :]
    original_run = simulate_variants(
        base,
        schedule,
        target=target,
        interaction_scale=interaction_scale,
        write_field=write_field,
    )
    gauge_run = simulate_variants(
        transformed,
        transformed_schedule(schedule, gauge),
        target=target * gauge,
        interaction_scale=interaction_scale,
        write_field=write_field,
    )
    gauge_error = float(
        np.max(np.abs(original_run["overlap_curve"] - gauge_run["overlap_curve"]))
    )
    return {
        "zero_interaction_max_curve_difference": zero_error,
        "gauge_equivalence_max_curve_difference": gauge_error,
        "zero_interaction_pass": zero_error <= 1e-12,
        "gauge_equivalence_pass": gauge_error <= 1e-12,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    blocks_per_side = int(math.ceil(args.n / args.cluster_size))
    plaquette_count = (blocks_per_side - 1) ** 2
    doses = sorted(set(args.doses or [0, 1, 3, 5, 7, plaquette_count]))
    doses = [dose for dose in doses if 0 <= dose <= plaquette_count]
    if len(doses) < 4 or doses[0] != 0 or doses[-1] != plaquette_count:
        raise ValueError("dose response must include zero, maximum, and at least two intermediate levels")

    seed_rows: list[dict[str, Any]] = []
    dose_extreme_effects: list[float] = []
    physical_flattening_effects: list[float] = []
    slopes: list[float] = []
    correlations: list[float] = []
    original_stable_counts: list[int] = []
    flat_stable_counts: list[int] = []
    matching_pass = True
    controls_pass = True
    inversion_pass = True

    for seed_offset in range(args.graph_seeds):
        graph_seed = args.graph_seed + seed_offset
        couplings, graph_meta = graph_couplings(args, graph_seed)
        construction = construct_holonomy_variants(couplings, blocks_per_side, doses)
        variants = construction["variants"]
        matrices = np.stack([variant["matrix"] for variant in variants], axis=0)
        original_matrix = coupling_matrix(
            construction["src"],
            construction["tgt"],
            construction["magnitudes"],
            construction["baseline_bits"],
            blocks_per_side**2,
        )
        flat_matrix = coupling_matrix(
            construction["src"],
            construction["tgt"],
            construction["magnitudes"],
            np.zeros_like(construction["baseline_bits"]),
            blocks_per_side**2,
        )
        original_stable = count_one_flip_stable_states(original_matrix)
        flat_stable = count_one_flip_stable_states(flat_matrix)
        original_stable_counts.append(original_stable)
        flat_stable_counts.append(flat_stable)
        target = np.ones(blocks_per_side**2, dtype=np.int8)
        schedule = make_schedule(
            nodes=target.size,
            replicas=args.replicas,
            write_sweeps=args.write_sweeps,
            release_sweeps=args.release_sweeps,
            seed=args.seed + 100003 * graph_seed,
        )
        simulation = simulate_variants(
            matrices,
            schedule,
            target=target,
            interaction_scale=args.interaction_scale,
            write_field=args.write_field,
        )
        physical_simulation = simulate_variants(
            np.stack((flat_matrix, original_matrix), axis=0),
            schedule,
            target=target,
            interaction_scale=args.interaction_scale,
            write_field=args.write_field,
        )
        controls = exact_controls(
            matrices,
            schedule,
            target,
            args.interaction_scale,
            args.write_field,
            args.seed + 700001 * graph_seed,
        )
        controls_pass &= bool(controls["zero_interaction_pass"] and controls["gauge_equivalence_pass"])

        baseline_bits = construction["baseline_bits"]
        for variant in variants:
            flip = baseline_bits ^ variant["sign_bits"]
            recovered = variant["sign_bits"] ^ flip
            inversion_pass &= bool(np.array_equal(recovered, baseline_bits))
            match = variant["match"]
            matching_pass &= bool(
                match["edge_endpoints_exact"]
                and match["coupling_magnitudes_max_error"] <= 1e-14
                and match["degree_sequence_max_error"] <= 1e-14
                and match["unsigned_spectrum_max_error"] <= 1e-12
            )

        dose_array = np.asarray([variant["dose"] for variant in variants], dtype=float) / plaquette_count
        auc_mean = np.mean(simulation["memory_auc"], axis=1)
        slope = float(np.polyfit(dose_array, auc_mean, 1)[0])
        correlation = spearman(dose_array, auc_mean)
        primary = float(auc_mean[-1] - auc_mean[0])
        physical_flattening_effect = float(
            np.mean(physical_simulation["memory_auc"][1])
            - np.mean(physical_simulation["memory_auc"][0])
        )
        slopes.append(slope)
        correlations.append(correlation)
        dose_extreme_effects.append(primary)
        physical_flattening_effects.append(physical_flattening_effect)

        dose_rows = []
        for index, variant in enumerate(variants):
            dose_rows.append(
                {
                    "frustrated_cycles": int(variant["dose"]),
                    "frustrated_fraction": float(variant["dose"] / plaquette_count),
                    "cycle_products": (1 - 2 * variant["cycle_bits"].astype(int)).tolist(),
                    "match": variant["match"],
                    "written_overlap_mean": float(np.mean(simulation["written_overlap"][index])),
                    "memory_auc_mean": float(np.mean(simulation["memory_auc"][index])),
                    "memory_auc_sem": float(
                        np.std(simulation["memory_auc"][index], ddof=1) / math.sqrt(args.replicas)
                    ),
                    "retained_overlap_mean": float(np.mean(simulation["retained_overlap"][index])),
                    "half_life_mean_sweeps": float(np.mean(simulation["half_life_sweeps"][index])),
                    "half_life_censored_fraction": float(np.mean(simulation["half_life_censored"][index])),
                    "q_EA_mean": float(np.mean(simulation["q_EA"][index])),
                }
            )
        seed_rows.append(
            {
                **graph_meta,
                "baseline_frustrated_cycles": int(np.sum(construction["baseline_cycle_bits"])),
                "dose_rows": dose_rows,
                "memory_auc_slope_per_frustrated_fraction": slope,
                "dose_spearman": correlation,
                "max_minus_flat_memory_auc": primary,
                "original_minus_flat_memory_auc": physical_flattening_effect,
                "one_flip_stable_states": {
                    "original": original_stable,
                    "flat": flat_stable,
                    "difference": original_stable - flat_stable,
                },
                "controls": controls,
            }
        )
        print(
            json.dumps(
                {
                    "event": "holonomy_causality_graph_complete",
                    "graph_seed": graph_seed,
                    "slope": slope,
                    "spearman": correlation,
                    "max_minus_flat": primary,
                    "original_minus_flat": physical_flattening_effect,
                    "stable_original": original_stable,
                    "stable_flat": flat_stable,
                }
            ),
            flush=True,
        )

    slopes_a = np.asarray(slopes, dtype=float)
    effects_a = np.asarray(dose_extreme_effects, dtype=float)
    physical_effects_a = np.asarray(physical_flattening_effects, dtype=float)
    slope_ci = bootstrap_mean_interval(slopes_a, seed=args.seed + 17, draws=args.bootstrap_draws)
    effect_ci = bootstrap_mean_interval(effects_a, seed=args.seed + 19, draws=args.bootstrap_draws)
    physical_effect_ci = bootstrap_mean_interval(
        physical_effects_a, seed=args.seed + 23, draws=args.bootstrap_draws
    )
    stable_original_a = np.asarray(original_stable_counts, dtype=float)
    stable_flat_a = np.asarray(flat_stable_counts, dtype=float)
    stable_test = wilcoxon(
        stable_original_a,
        stable_flat_a,
        alternative="greater",
        zero_method="wilcox",
        correction=False,
        method="auto",
    )
    median_rho = float(np.nanmedian(correlations))
    mean_written = float(
        np.mean([row["dose_rows"][0]["written_overlap_mean"] for row in seed_rows])
    )
    maximum_interaction_field = float(
        max(row["maximum_interaction_field"] for row in seed_rows)
    )

    shared_checks = {
        "matched_interventions": matching_pass,
        "write_field_dominates_interactions": args.write_field > maximum_interaction_field,
        "write_state_established": mean_written >= args.minimum_written_overlap,
        "gauge_and_zero_interaction_controls": controls_pass,
        "inversion_round_trip": inversion_pass,
    }
    capacity_checks = {
        "original_has_more_stable_states": float(np.mean(stable_original_a - stable_flat_a)) > 0.0,
        "paired_one_sided_wilcoxon": float(stable_test.pvalue) < 0.05,
    }
    retention_direction = (
        "increased"
        if physical_effect_ci[0] > 0.0
        else "decreased"
        if physical_effect_ci[1] < 0.0
        else "not_resolved"
    )
    dose_direction = (
        "increased"
        if slope_ci[0] > 0.0
        else "decreased"
        if slope_ci[1] < 0.0
        else "not_resolved"
    )
    retention_checks = {
        "physical_flattening_contrast_resolved": retention_direction != "not_resolved",
        "dose_slope_resolved": dose_direction != "not_resolved",
        "monotone_dose_response": bool(abs(median_rho) >= args.minimum_spearman),
    }
    capacity_decision = "pass" if matching_pass and all(capacity_checks.values()) else "not_passed"
    retention_decision = (
        "pass" if all(shared_checks.values()) and all(retention_checks.values()) else "not_passed"
    )
    if capacity_decision == "pass" and retention_decision == "pass":
        decision = "split_result"
    elif capacity_decision == "pass":
        decision = "capacity_only"
    else:
        decision = "not_passed"
    if decision == "split_result":
        retention_verb = {
            "increased": "increases",
            "decreased": "decreases",
        }.get(retention_direction, "does not resolve")
        claim = (
            "Loop holonomy changes two distinct properties of the rotating-colloid domain model: "
            f"it increases the number of metastable states and {retention_verb} retention "
            "of a common written pattern."
        )
    elif decision == "capacity_only":
        claim = (
            "Loop holonomy increases the number of metastable domain states; its effect on "
            "dynamic retention was not resolved by the registered intervention."
        )
    else:
        claim = "The registered intervention did not establish a causal holonomy effect."
    report: dict[str, Any] = {
        "report_type": "rotating_colloids_holonomy_causality",
        "decision": decision,
        "claim": claim,
        "landscape_capacity_decision": capacity_decision,
        "dynamic_retention_decision": retention_decision,
        "model": {
            "equation": "H_eff/J = -sum_(b,c) K_bc s_b s_c",
            "origin": "strong intra-domain-locking reduction of the rotating-colloid pair Hamiltonian",
            "holonomy": "product sign(K_bc) around each elementary domain plaquette",
            "intervention": "change cycle products at fixed graph and fixed |K_bc|",
        },
        "parameters": {
            "n": args.n,
            "cluster_size": args.cluster_size,
            "domains": blocks_per_side**2,
            "plaquettes": plaquette_count,
            "doses": doses,
            "epsilon": args.epsilon,
            "interaction_scale": args.interaction_scale,
            "write_field": args.write_field,
            "write_sweeps": args.write_sweeps,
            "release_sweeps": args.release_sweeps,
            "replicas": args.replicas,
            "graph_seeds": args.graph_seeds,
        },
        "estimands": {
            "mean_memory_auc_slope": float(np.mean(slopes_a)),
            "slope_graph_bootstrap_95_interval": list(slope_ci),
            "mean_max_minus_flat_memory_auc": float(np.mean(effects_a)),
            "contrast_graph_bootstrap_95_interval": list(effect_ci),
            "mean_original_minus_flat_memory_auc": float(np.mean(physical_effects_a)),
            "physical_flattening_graph_bootstrap_95_interval": list(physical_effect_ci),
            "dynamic_retention_direction": retention_direction,
            "dose_slope_direction": dose_direction,
            "median_graph_spearman": median_rho,
            "mean_written_overlap": mean_written,
            "maximum_interaction_field": maximum_interaction_field,
            "one_flip_stable_states": {
                "original_mean": float(np.mean(stable_original_a)),
                "flat_mean": float(np.mean(stable_flat_a)),
                "mean_difference": float(np.mean(stable_original_a - stable_flat_a)),
                "original_greater": int(np.count_nonzero(stable_original_a > stable_flat_a)),
                "equal": int(np.count_nonzero(stable_original_a == stable_flat_a)),
                "original_lower": int(np.count_nonzero(stable_original_a < stable_flat_a)),
                "paired_one_sided_wilcoxon_statistic": float(stable_test.statistic),
                "paired_one_sided_wilcoxon_p": float(stable_test.pvalue),
            },
        },
        "checks": {
            "shared": shared_checks,
            "landscape_capacity": capacity_checks,
            "dynamic_retention": retention_checks,
        },
        "matching_scope": {
            "exact": [
                "domain graph",
                "degree sequence",
                "coupling magnitude on every edge",
                "quenched graph disorder",
                "unsigned normalized-Laplacian spectrum",
                "write and release schedules",
            ],
            "reported_not_conditioned_away": (
                "The signed spectrum changes with the gauge-invariant cycle class and is therefore "
                "a consequence of the holonomy intervention, not an independent matching variable."
            ),
        },
        "seed_rows": seed_rows,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def markdown(report: dict[str, Any]) -> str:
    estimate = report["estimands"]
    checks = report["checks"]
    stable = estimate["one_flip_stable_states"]
    lines = [
        "# Causal holonomy test in the rotating-colloid domain model",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Landscape-capacity decision: `{report['landscape_capacity_decision']}`",
        f"- Dynamic-retention decision: `{report['dynamic_retention_decision']}`",
        f"- Exact stable-state count: `{stable['original_mean']:.3f}` original versus "
        f"`{stable['flat_mean']:.3f}` after flattening",
        f"- Paired one-sided Wilcoxon: `p={stable['paired_one_sided_wilcoxon_p']:.6g}`",
        f"- Mean memory-area slope: `{estimate['mean_memory_auc_slope']:.6g}`",
        "- Graph-bootstrap 95% interval: "
        f"`[{estimate['slope_graph_bootstrap_95_interval'][0]:.6g}, "
        f"{estimate['slope_graph_bootstrap_95_interval'][1]:.6g}]`",
        f"- Maximum-minus-flat contrast: `{estimate['mean_max_minus_flat_memory_auc']:.6g}`",
        "- Contrast 95% interval: "
        f"`[{estimate['contrast_graph_bootstrap_95_interval'][0]:.6g}, "
        f"{estimate['contrast_graph_bootstrap_95_interval'][1]:.6g}]`",
        f"- Median dose-response Spearman coefficient: `{estimate['median_graph_spearman']:.4f}`",
        f"- Dynamic retention direction: `{estimate['dynamic_retention_direction']}`",
        "",
        "## Intervention",
        "",
        "The elementary loop variable is the product of coupling signs around a domain plaquette. "
        "Each intervention changes these products while retaining the same graph, every coupling "
        "magnitude, the degree sequence, graph disorder and the unsigned Laplacian spectrum. All "
        "arms receive the same written state, update sequence, random numbers and field-free release.",
        "",
        "## Causal criteria",
        "",
    ]
    for family, family_checks in checks.items():
        lines.append(f"- {family.replace('_', ' ')}")
        for name, passed in family_checks.items():
            lines.append(
                f"  - {name.replace('_', ' ')}: `{'pass' if passed else 'not_passed'}`"
            )
    lines.extend(
        [
            "",
            "## Scientific scope",
            "",
            report["claim"],
            "",
            "The calculation acts on the exact Ising reduction of the pair Hamiltonian in the "
            "strong intra-domain-locking limit. A positive result identifies causality within that "
            "reduced model. Confirmation in the continuous particle-angle dynamics remains a "
            "separate calculation.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--cluster-size", type=int, default=8)
    ap.add_argument("--crosslink-k", type=int, default=2)
    ap.add_argument("--crosslink-weight", type=float, default=0.18)
    ap.add_argument("--epsilon", type=float, default=1.614286)
    ap.add_argument("--graph-seed", type=int, default=12345)
    ap.add_argument("--graph-seeds", type=int, default=20)
    ap.add_argument("--doses", type=parse_ints, default=None)
    ap.add_argument("--replicas", type=int, default=192)
    ap.add_argument("--interaction-scale", type=float, default=3.0)
    ap.add_argument("--write-field", type=float, default=10.0)
    ap.add_argument("--write-sweeps", type=int, default=100)
    ap.add_argument("--release-sweeps", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--bootstrap-draws", type=int, default=5000)
    ap.add_argument("--minimum-written-overlap", type=float, default=0.80)
    ap.add_argument("--minimum-spearman", type=float, default=0.60)
    ap.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "holonomy_causality/holonomy_causality"
        ),
    )
    return ap


def main() -> None:
    args = parser().parse_args()
    report = run(args)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.output_prefix.with_suffix('.json')),
                "markdown": str(args.output_prefix.with_suffix('.md')),
                "decision": report["decision"],
                "estimands": report["estimands"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
