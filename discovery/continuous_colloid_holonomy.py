"""Continuous-angle interventions on loop holonomy in the mosaic rotor model.

The pair Hamiltonian is

    U = -J sum_e w_e cos 2(theta_i-theta_j-chi_e)
        -G sum_e w_e cos 2(theta_i+theta_j-2 phi_e).

For domain-locked directors, cross-domain bonds induce Ising couplings.  This
module changes only the preferred capillary frames ``phi_e`` on cross-domain
bonds to approach a globally flat Z2 connection while preserving the graph,
bond amplitudes and effective coupling magnitudes.  It also supplies the exact
gauge transformation of ``chi_e`` and ``phi_e`` used as a null control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:  # NumPy < 2.0
    _TRAPEZOID = np.trapz

from discovery.colloid_holonomy_memory import fundamental_cycle_fluxes


Edge = Tuple[int, int]


@dataclass(frozen=True)
class FrameIntervention:
    original_phi: np.ndarray
    flat_phi: np.ndarray
    frame_shift: np.ndarray
    original_couplings: Dict[Edge, float]
    target_couplings: Dict[Edge, float]
    realized_couplings: Dict[Edge, float]
    flat_gauge: np.ndarray
    cross_edge_mask: np.ndarray
    magnitude_relative_l1: float
    magnitude_relative_max: float
    negative_flux_original: int
    negative_flux_realized: int


def _edge_groups(
    src: np.ndarray,
    tgt: np.ndarray,
    labels: np.ndarray,
) -> Dict[Edge, np.ndarray]:
    groups: dict[Edge, list[int]] = {}
    for index, (left, right) in enumerate(zip(src, tgt)):
        a = int(labels[int(left)])
        b = int(labels[int(right)])
        if a == b:
            continue
        edge = tuple(sorted((a, b)))
        groups.setdefault(edge, []).append(index)
    return {edge: np.asarray(indices, dtype=np.int64) for edge, indices in groups.items()}


def induced_domain_couplings(
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    axes: np.ndarray,
    j_align: float,
    g_capillary: float,
) -> Dict[Edge, float]:
    """Project the continuous pair Hamiltonian onto binary domain directors."""

    result: Dict[Edge, float] = {}
    for edge, indices in _edge_groups(src, tgt, labels).items():
        left = src[indices]
        right = tgt[indices]
        ordinary = np.cos(2.0 * (axes[left] - axes[right]))
        bond = np.cos(2.0 * (axes[left] + axes[right] - 2.0 * phi[indices]))
        result[edge] = float(
            np.sum(weights[indices] * (float(j_align) * ordinary + float(g_capillary) * bond))
        )
    return result


def _minimal_shift_to_cosine(phase: float, target_cosine: float) -> float:
    target = float(np.clip(target_cosine, -1.0, 1.0))
    angle = math.acos(target)
    candidates = []
    for desired in (angle, -angle):
        delta = (float(phase) - desired) / 4.0
        delta = (delta + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
        candidates.append(delta)
    return float(min(candidates, key=lambda value: (abs(value), value)))


def _flat_target_with_minimum_physical_residual(
    *,
    original: Mapping[Edge, float],
    lower: Mapping[Edge, float],
    upper: Mapping[Edge, float],
    domain_count: int,
) -> tuple[Dict[Edge, float], np.ndarray]:
    if domain_count > 22:
        raise ValueError("exact flat target search is limited to 22 domains")
    edges = sorted(original)
    state_ids = np.arange(1 << max(domain_count - 1, 0), dtype=np.uint64)
    gauges = np.ones((state_ids.size, domain_count), dtype=np.int8)
    for node in range(1, domain_count):
        gauges[:, node] = (1 - 2 * ((state_ids >> (node - 1)) & 1)).astype(np.int8)

    residual = np.zeros(state_ids.size, dtype=float)
    edit_weight = np.zeros(state_ids.size, dtype=float)
    normalizer = max(sum(abs(float(value)) for value in original.values()), 1e-12)
    for edge in edges:
        value = float(original[edge])
        sign = gauges[:, edge[0]] * gauges[:, edge[1]]
        target = sign.astype(float) * abs(value)
        realized = np.clip(target, float(lower[edge]), float(upper[edge]))
        residual += np.abs(realized - target) / normalizer
        edit_weight += np.where(np.sign(value) == sign, 0.0, abs(value) / normalizer)
    selected_index = int(np.lexsort((edit_weight, residual))[0])
    selected = gauges[selected_index].copy()
    target = {
        edge: float(abs(original[edge]) * selected[edge[0]] * selected[edge[1]])
        for edge in edges
    }
    return target, selected


def build_frame_intervention(
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    axes: np.ndarray,
    j_align: float,
    g_capillary: float,
    domain_count: int,
) -> FrameIntervention:
    """Rotate cross-domain bond frames toward the nearest reachable flat connection."""

    if g_capillary <= 0.0:
        raise ValueError("g_capillary must be positive for a bond-frame intervention")
    src = np.asarray(src, dtype=np.int64)
    tgt = np.asarray(tgt, dtype=np.int64)
    phi = np.asarray(phi, dtype=float)
    weights = np.asarray(weights, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    axes = np.asarray(axes, dtype=float)
    groups = _edge_groups(src, tgt, labels)
    original = induced_domain_couplings(
        src=src,
        tgt=tgt,
        phi=phi,
        weights=weights,
        labels=labels,
        axes=axes,
        j_align=j_align,
        g_capillary=g_capillary,
    )
    lower: Dict[Edge, float] = {}
    upper: Dict[Edge, float] = {}
    ordinary_by_edge: Dict[Edge, float] = {}
    weight_by_edge: Dict[Edge, float] = {}
    for edge, indices in groups.items():
        left = src[indices]
        right = tgt[indices]
        ordinary = float(
            np.sum(weights[indices] * float(j_align) * np.cos(2.0 * (axes[left] - axes[right])))
        )
        radius = float(g_capillary) * float(np.sum(weights[indices]))
        ordinary_by_edge[edge] = ordinary
        weight_by_edge[edge] = float(np.sum(weights[indices]))
        lower[edge] = ordinary - radius
        upper[edge] = ordinary + radius

    target, flat_gauge = _flat_target_with_minimum_physical_residual(
        original=original,
        lower=lower,
        upper=upper,
        domain_count=domain_count,
    )
    shift = np.zeros_like(phi)
    realized_target: Dict[Edge, float] = {}
    for edge, indices in groups.items():
        desired = float(np.clip(target[edge], lower[edge], upper[edge]))
        desired_cosine = (
            (desired - ordinary_by_edge[edge])
            / (float(g_capillary) * weight_by_edge[edge])
        )
        for index in indices:
            phase = 2.0 * (
                axes[src[index]] + axes[tgt[index]] - 2.0 * phi[index]
            )
            shift[index] = _minimal_shift_to_cosine(phase, desired_cosine)
        realized_target[edge] = desired

    flat_phi = phi + shift
    realized = induced_domain_couplings(
        src=src,
        tgt=tgt,
        phi=flat_phi,
        weights=weights,
        labels=labels,
        axes=axes,
        j_align=j_align,
        g_capillary=g_capillary,
    )
    denominator = max(sum(abs(value) for value in original.values()), 1e-12)
    magnitude_errors = {
        edge: abs(abs(realized[edge]) - abs(original[edge])) for edge in original
    }
    relative_errors = [
        magnitude_errors[edge] / max(abs(original[edge]), 1e-12) for edge in original
    ]
    original_flux = fundamental_cycle_fluxes(original, domain_count)
    realized_flux = fundamental_cycle_fluxes(realized, domain_count)
    cross_mask = labels[src] != labels[tgt]
    return FrameIntervention(
        original_phi=phi.copy(),
        flat_phi=flat_phi,
        frame_shift=shift,
        original_couplings=original,
        target_couplings=target,
        realized_couplings=realized,
        flat_gauge=flat_gauge,
        cross_edge_mask=cross_mask,
        magnitude_relative_l1=float(sum(magnitude_errors.values()) / denominator),
        magnitude_relative_max=float(max(relative_errors, default=0.0)),
        negative_flux_original=int(sum(value < 0 for value in original_flux)),
        negative_flux_realized=int(sum(value < 0 for value in realized_flux)),
    )


def gauge_equivalent_phases(
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    labels: np.ndarray,
    gauge: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the exact phase-coordinate transform of the pair Hamiltonian."""

    gauge_a = np.asarray(gauge, dtype=np.int8)
    labels_a = np.asarray(labels, dtype=np.int64)
    eta = np.where(gauge_a[labels_a] < 0, math.pi / 2.0, 0.0)
    chi = eta[np.asarray(src, dtype=np.int64)] - eta[np.asarray(tgt, dtype=np.int64)]
    transformed_phi = np.asarray(phi, dtype=float) + 0.5 * (
        eta[np.asarray(src, dtype=np.int64)] + eta[np.asarray(tgt, dtype=np.int64)]
    )
    return chi, transformed_phi, eta


def pair_energy(
    theta: np.ndarray,
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    j_align: float,
    g_capillary: float,
    chi: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the pair energy for one state or a batch of states."""

    state = np.atleast_2d(np.asarray(theta, dtype=float))
    chi_a = np.zeros_like(phi, dtype=float) if chi is None else np.asarray(chi, dtype=float)
    difference = state[:, src] - state[:, tgt] - chi_a[None, :]
    bond_frame = state[:, src] + state[:, tgt] - 2.0 * np.asarray(phi)[None, :]
    value = -float(j_align) * np.sum(weights[None, :] * np.cos(2.0 * difference), axis=1)
    value -= float(g_capillary) * np.sum(weights[None, :] * np.cos(2.0 * bond_frame), axis=1)
    return value


def simulate_write_release(
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
    j_align: float,
    g_capillary: float,
    replicas: int,
    write_steps: int,
    release_steps: int,
    stride: int,
    dt: float,
    write_field: float,
    initial_theta: np.ndarray,
    noise_seed: int,
    chi: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Run overdamped continuous-angle dynamics with common-random-number support."""

    import torch

    torch_device = torch.device(device)
    src_t = torch.as_tensor(src, dtype=torch.long, device=torch_device)
    tgt_t = torch.as_tensor(tgt, dtype=torch.long, device=torch_device)
    phi_t = torch.as_tensor(phi, dtype=torch.float64, device=torch_device)
    weights_t = torch.as_tensor(weights, dtype=torch.float64, device=torch_device)
    chi_a = np.zeros_like(phi, dtype=float) if chi is None else np.asarray(chi, dtype=float)
    chi_t = torch.as_tensor(chi_a, dtype=torch.float64, device=torch_device)
    theta = torch.as_tensor(
        np.asarray(initial_theta), dtype=torch.float64, device=torch_device
    ).clone()
    target_t = torch.as_tensor(
        np.asarray(target), dtype=torch.float64, device=torch_device
    )
    if target_t.ndim == 1:
        target_t = target_t[None, :].expand(replicas, -1)
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(int(noise_seed))
    noise_scale = math.sqrt(2.0 * float(dt))

    def advance(field: float) -> None:
        difference = theta[:, src_t] - theta[:, tgt_t] - chi_t[None, :]
        bond_frame = theta[:, src_t] + theta[:, tgt_t] - 2.0 * phi_t[None, :]
        align = 2.0 * float(j_align) * weights_t[None, :] * torch.sin(2.0 * difference)
        capillary = 2.0 * float(g_capillary) * weights_t[None, :] * torch.sin(2.0 * bond_frame)
        torque = torch.zeros_like(theta)
        torque.index_add_(1, src_t, -align - capillary)
        torque.index_add_(1, tgt_t, +align - capillary)
        if field:
            torque -= 2.0 * float(field) * torch.sin(2.0 * (theta - target_t))
        noise = torch.randn(
            theta.shape,
            generator=generator,
            dtype=torch.float64,
            device=torch_device,
        )
        theta.add_(float(dt) * torque + noise_scale * noise)
        theta.remainder_(math.pi)

    for _ in range(int(write_steps)):
        advance(float(write_field))
    written = torch.mean(torch.cos(2.0 * (theta - target_t)), dim=1)
    overlap_samples = [written.detach().cpu().numpy()]
    z_samples = [torch.exp(2.0j * theta).detach().cpu().numpy()]
    times = [0.0]
    for step in range(1, int(release_steps) + 1):
        advance(0.0)
        if step % int(stride) == 0 or step == int(release_steps):
            overlap_samples.append(
                torch.mean(torch.cos(2.0 * (theta - target_t)), dim=1).detach().cpu().numpy()
            )
            z_samples.append(torch.exp(2.0j * theta).detach().cpu().numpy())
            times.append(float(step) * float(dt))

    overlap = np.asarray(overlap_samples, dtype=float)
    mean_curve = np.mean(overlap, axis=1)
    times_a = np.asarray(times, dtype=float)
    duration = max(float(times_a[-1] - times_a[0]), float(dt))
    auc = float(_TRAPEZOID(mean_curve, times_a) / duration)
    positive_auc = float(_TRAPEZOID(np.maximum(mean_curve, 0.0), times_a) / duration)
    threshold = 0.5 * max(float(mean_curve[0]), 0.0)
    crossing = np.flatnonzero(mean_curve <= threshold)
    half_life = float(times_a[crossing[0]]) if crossing.size else float(times_a[-1])
    z_mean = np.mean(np.asarray(z_samples), axis=0)
    q_ea = float(np.mean(np.abs(z_mean) ** 2))
    return {
        "time": times_a.tolist(),
        "overlap_curve": mean_curve.tolist(),
        "written_overlap": float(mean_curve[0]),
        "final_overlap": float(mean_curve[-1]),
        "overlap_auc": auc,
        "positive_overlap_auc": positive_auc,
        "half_life": half_life,
        "half_life_censored": bool(crossing.size == 0),
        "q_ea": q_ea,
        "final_theta": theta.detach().cpu().numpy(),
    }
