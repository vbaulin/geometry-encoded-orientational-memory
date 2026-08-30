"""Causal holonomy intervention for the colloidal domain reduction.

The strong-locking reduction has the Ising form

    H = -sum_(b,c) K_bc s_b s_c.

The sign product around a closed cycle is a Z2 holonomy.  Local sign changes
``s_b -> g_b s_b`` alter individual bonds but leave every cycle product fixed.
This module compares the measured coupling network with its closest flat-sign
network while preserving the graph and every ``|K_bc|`` exactly.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


@dataclass(frozen=True)
class HolonomyIntervention:
    original: Dict[Edge, float]
    flat: Dict[Edge, float]
    gauge_equivalent: Dict[Edge, float]
    flat_gauge: np.ndarray
    control_gauge: np.ndarray
    original_fluxes: Tuple[int, ...]
    flat_fluxes: Tuple[int, ...]


def normalized_edges(couplings: Mapping[Edge, float]) -> Dict[Edge, float]:
    result = {}
    for (left, right), value in couplings.items():
        edge = tuple(sorted((int(left), int(right))))
        if edge[0] == edge[1] or not math.isfinite(float(value)) or float(value) == 0.0:
            continue
        result[edge] = float(value)
    return result


def coupling_matrix(couplings: Mapping[Edge, float], node_count: int) -> np.ndarray:
    matrix = np.zeros((node_count, node_count), dtype=float)
    for (left, right), value in normalized_edges(couplings).items():
        matrix[left, right] = matrix[right, left] = value
    return matrix


def fundamental_cycle_fluxes(
    couplings: Mapping[Edge, float], node_count: int
) -> Tuple[int, ...]:
    """Return Z2 products for a deterministic fundamental-cycle basis."""

    edges = normalized_edges(couplings)
    adjacency: Dict[int, list[tuple[int, int]]] = {node: [] for node in range(node_count)}
    for edge_index, (left, right) in enumerate(sorted(edges)):
        adjacency[left].append((right, edge_index))
        adjacency[right].append((left, edge_index))
    parent = [-1] * node_count
    parent_edge = [-1] * node_count
    depth = [0] * node_count
    tree_edges: set[int] = set()
    indexed_edges = sorted(edges)
    for root in range(node_count):
        if parent[root] != -1:
            continue
        parent[root] = root
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, edge_index in adjacency[node]:
                if parent[neighbor] != -1:
                    continue
                parent[neighbor] = node
                parent_edge[neighbor] = edge_index
                depth[neighbor] = depth[node] + 1
                tree_edges.add(edge_index)
                queue.append(neighbor)

    def path_edges(left: int, right: int) -> list[int]:
        result = []
        a, b = left, right
        while depth[a] > depth[b]:
            result.append(parent_edge[a])
            a = parent[a]
        while depth[b] > depth[a]:
            result.append(parent_edge[b])
            b = parent[b]
        while a != b:
            result.extend((parent_edge[a], parent_edge[b]))
            a, b = parent[a], parent[b]
        return result

    signs = [1 if edges[edge] > 0 else -1 for edge in indexed_edges]
    fluxes = []
    for edge_index, (left, right) in enumerate(indexed_edges):
        if edge_index in tree_edges:
            continue
        product = signs[edge_index]
        for path_index in path_edges(left, right):
            product *= signs[path_index]
        fluxes.append(int(product))
    return tuple(fluxes)


def apply_gauge(
    couplings: Mapping[Edge, float], gauge: Sequence[int]
) -> Dict[Edge, float]:
    gauge_a = np.asarray(gauge, dtype=np.int8)
    return {
        edge: float(value) * int(gauge_a[edge[0]]) * int(gauge_a[edge[1]])
        for edge, value in normalized_edges(couplings).items()
    }


def closest_flat_signs(
    couplings: Mapping[Edge, float], node_count: int
) -> Tuple[Dict[Edge, float], np.ndarray, float]:
    """Find the minimum-weight sign edit to a globally flat Z2 connection."""

    edges = normalized_edges(couplings)
    if node_count > 22:
        raise ValueError("exact flat-sign projection is limited to 22 domain variables")
    state_ids = np.arange(1 << max(node_count - 1, 0), dtype=np.uint64)
    gauge = np.ones((state_ids.size, node_count), dtype=np.int8)
    for node in range(1, node_count):
        gauge[:, node] = (1 - 2 * ((state_ids >> (node - 1)) & 1)).astype(np.int8)
    score = np.zeros(state_ids.size, dtype=float)
    for (left, right), value in edges.items():
        score += abs(value) * np.sign(value) * gauge[:, left] * gauge[:, right]
    selected = gauge[int(np.argmax(score))]
    flat = {
        edge: abs(value) * int(selected[edge[0]]) * int(selected[edge[1]])
        for edge, value in edges.items()
    }
    changed_weight = sum(
        abs(value)
        for edge, value in edges.items()
        if np.sign(value) != np.sign(flat[edge])
    )
    return flat, selected, float(changed_weight)


def build_intervention(
    couplings: Mapping[Edge, float], node_count: int, seed: int = 314159
) -> HolonomyIntervention:
    original = normalized_edges(couplings)
    flat, flat_gauge, _ = closest_flat_signs(original, node_count)
    rng = np.random.default_rng(seed)
    control_gauge = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=node_count)
    control_gauge[0] = 1
    gauge_equivalent = apply_gauge(original, control_gauge)
    return HolonomyIntervention(
        original=original,
        flat=flat,
        gauge_equivalent=gauge_equivalent,
        flat_gauge=flat_gauge,
        control_gauge=control_gauge,
        original_fluxes=fundamental_cycle_fluxes(original, node_count),
        flat_fluxes=fundamental_cycle_fluxes(flat, node_count),
    )


def exact_landscape(couplings: Mapping[Edge, float], node_count: int) -> Dict[str, Any]:
    if node_count > 22:
        raise ValueError("exact landscape enumeration is limited to 22 variables")
    matrix = coupling_matrix(couplings, node_count)
    state_ids = np.arange(1 << node_count, dtype=np.uint64)
    states = np.empty((state_ids.size, node_count), dtype=np.int8)
    for node in range(node_count):
        states[:, node] = (1 - 2 * ((state_ids >> node) & 1)).astype(np.int8)
    energies = -0.5 * np.einsum("bi,ij,bj->b", states, matrix, states, optimize=True)
    local_fields = states @ matrix
    stable = np.all(states * local_fields >= -1e-12, axis=1)
    ground = np.isclose(energies, energies.min(), rtol=0.0, atol=1e-10)
    return {
        "state_count": int(state_ids.size),
        "ground_energy": float(energies.min()),
        "ground_degeneracy": int(np.count_nonzero(ground)),
        "single_flip_stable_states": int(np.count_nonzero(stable)),
        "stable_state_fraction": float(np.mean(stable)),
        "energy_level_count": int(np.unique(np.round(energies, 10)).size),
    }


def one_flip_stable_states(
    couplings: Mapping[Edge, float], node_count: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return identifiers and spin vectors of all one-flip-stable states."""

    if node_count > 22:
        raise ValueError("exact stable-state enumeration is limited to 22 variables")
    matrix = coupling_matrix(couplings, node_count)
    state_ids = np.arange(1 << node_count, dtype=np.uint64)
    states = np.empty((state_ids.size, node_count), dtype=np.int8)
    for node in range(node_count):
        states[:, node] = (1 - 2 * ((state_ids >> node) & 1)).astype(np.int8)
    local_fields = states @ matrix
    stable = np.all(states * local_fields >= -1e-12, axis=1)
    return state_ids[stable], states[stable]


def _glauber_schedule(
    rng: np.random.Generator,
    sweeps: int,
    replicas: int,
    node_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    orders = np.stack([rng.permutation(node_count) for _ in range(sweeps)])
    uniforms = rng.random((sweeps, node_count, replicas))
    return orders, uniforms


def glauber_write_release(
    couplings: Mapping[Edge, float],
    targets: np.ndarray,
    initial: np.ndarray,
    *,
    beta: float,
    write_field: float,
    write_schedule: Tuple[np.ndarray, np.ndarray],
    release_schedule: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    matrix = coupling_matrix(couplings, targets.shape[1])
    scale = float(np.median(np.sum(np.abs(matrix), axis=1)))
    if scale <= 0:
        raise ValueError("coupling network has zero weighted degree")
    matrix = matrix / scale
    states = np.asarray(initial, dtype=np.int8).copy()
    targets = np.asarray(targets, dtype=np.int8)

    def advance(schedule: Tuple[np.ndarray, np.ndarray], field: float, collect: bool) -> list[float]:
        orders, uniforms = schedule
        overlaps = []
        for sweep, order in enumerate(orders):
            for slot, node in enumerate(order):
                local = states @ matrix[:, node] + field * targets[:, node]
                argument = np.clip(2.0 * beta * local, -60.0, 60.0)
                probability = 1.0 / (1.0 + np.exp(-argument))
                states[:, node] = np.where(uniforms[sweep, slot] < probability, 1, -1)
            if collect:
                overlaps.append(float(np.mean(np.mean(states * targets, axis=1))))
        return overlaps

    advance(write_schedule, float(write_field), False)
    initial_overlap = float(np.mean(np.mean(states * targets, axis=1)))
    release_overlap = advance(release_schedule, 0.0, True)
    positive = np.maximum(np.asarray(release_overlap, dtype=float), 0.0)
    return {
        "initial_overlap": initial_overlap,
        "release_overlap": release_overlap,
        "final_overlap": float(release_overlap[-1]) if release_overlap else initial_overlap,
        "positive_overlap_auc": float(np.mean(positive)) if positive.size else max(initial_overlap, 0.0),
    }


def paired_memory_test(
    intervention: HolonomyIntervention,
    node_count: int,
    *,
    replicas: int = 256,
    beta: float = 3.0,
    write_field: float = 3.0,
    write_sweeps: int = 80,
    release_sweeps: int = 240,
    seed: int = 314159,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    targets = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(replicas, node_count))
    initial = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(replicas, node_count))
    write_schedule = _glauber_schedule(rng, write_sweeps, replicas, node_count)
    release_schedule = _glauber_schedule(rng, release_sweeps, replicas, node_count)
    original = glauber_write_release(
        intervention.original,
        targets,
        initial,
        beta=beta,
        write_field=write_field,
        write_schedule=write_schedule,
        release_schedule=release_schedule,
    )
    flat = glauber_write_release(
        intervention.flat,
        targets,
        initial,
        beta=beta,
        write_field=write_field,
        write_schedule=write_schedule,
        release_schedule=release_schedule,
    )
    gauge = intervention.control_gauge[None, :]

    def gauge_schedule(schedule: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        orders, uniforms = schedule
        transformed = uniforms.copy()
        for sweep, order in enumerate(orders):
            for slot, node in enumerate(order):
                if intervention.control_gauge[node] < 0:
                    transformed[sweep, slot] = 1.0 - transformed[sweep, slot]
        return orders, transformed

    gauge_control = glauber_write_release(
        intervention.gauge_equivalent,
        targets * gauge,
        initial * gauge,
        beta=beta,
        write_field=write_field,
        write_schedule=gauge_schedule(write_schedule),
        release_schedule=gauge_schedule(release_schedule),
    )
    return {
        "original": original,
        "flat": flat,
        "gauge_equivalent_control": gauge_control,
        "original_minus_flat_auc": (
            original["positive_overlap_auc"] - flat["positive_overlap_auc"]
        ),
        "gauge_auc_residual": abs(
            original["positive_overlap_auc"] - gauge_control["positive_overlap_auc"]
        ),
        "parameters": {
            "replicas": replicas,
            "beta": beta,
            "write_field": write_field,
            "write_sweeps": write_sweeps,
            "release_sweeps": release_sweeps,
            "seed": seed,
        },
    }


def paired_metastable_retention_test(
    intervention: HolonomyIntervention,
    node_count: int,
    *,
    beta: float = 3.0,
    release_sweeps: int = 240,
    repeats_per_state: int = 8,
    seed: int = 314159,
) -> Dict[str, Any]:
    """Release states stabilized by holonomy but absent from the flat network."""

    original_ids, original_states = one_flip_stable_states(intervention.original, node_count)
    flat_ids, _ = one_flip_stable_states(intervention.flat, node_count)
    additional_mask = ~np.isin(original_ids, flat_ids)
    additional = original_states[additional_mask]
    if additional.size == 0:
        return {
            "decision": "not_evaluable",
            "reason": "the original network has no stable state absent from the flat control",
            "additional_stable_states": 0,
        }
    targets = np.repeat(additional, max(1, int(repeats_per_state)), axis=0)
    initial = targets.copy()
    rng = np.random.default_rng(seed)
    empty_orders = np.empty((0, node_count), dtype=np.int64)
    empty_uniforms = np.empty((0, node_count, targets.shape[0]), dtype=float)
    write_schedule = (empty_orders, empty_uniforms)
    release_schedule = _glauber_schedule(rng, release_sweeps, targets.shape[0], node_count)
    original = glauber_write_release(
        intervention.original,
        targets,
        initial,
        beta=beta,
        write_field=0.0,
        write_schedule=write_schedule,
        release_schedule=release_schedule,
    )
    flat = glauber_write_release(
        intervention.flat,
        targets,
        initial,
        beta=beta,
        write_field=0.0,
        write_schedule=write_schedule,
        release_schedule=release_schedule,
    )
    gauge = intervention.control_gauge[None, :]
    transformed_uniforms = release_schedule[1].copy()
    for sweep, order in enumerate(release_schedule[0]):
        for slot, node in enumerate(order):
            if intervention.control_gauge[node] < 0:
                transformed_uniforms[sweep, slot] = 1.0 - transformed_uniforms[sweep, slot]
    gauge_control = glauber_write_release(
        intervention.gauge_equivalent,
        targets * gauge,
        initial * gauge,
        beta=beta,
        write_field=0.0,
        write_schedule=write_schedule,
        release_schedule=(release_schedule[0], transformed_uniforms),
    )
    return {
        "decision": "measured",
        "additional_stable_states": int(additional.shape[0]),
        "replicas": int(targets.shape[0]),
        "original": original,
        "flat": flat,
        "gauge_equivalent_control": gauge_control,
        "original_minus_flat_auc": (
            original["positive_overlap_auc"] - flat["positive_overlap_auc"]
        ),
        "gauge_auc_residual": abs(
            original["positive_overlap_auc"] - gauge_control["positive_overlap_auc"]
        ),
        "parameters": {
            "beta": beta,
            "release_sweeps": release_sweeps,
            "repeats_per_state": repeats_per_state,
            "seed": seed,
        },
    }
