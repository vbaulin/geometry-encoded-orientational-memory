#!/usr/bin/env python3
"""Minimal multibasin test of order sensitivity versus retained order.

The two write vector fields can have a large Lie bracket while field-free
relaxation maps both pulse orders to the same basin.  This script reproduces
the support-fraction and coupling controls reported in the orientational-memory
Supplement without importing a substrate-specific model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


N = 8
DT = 0.025
Q_WELL = 0.65
TAU = np.geomspace(0.08, 40.0, N)
MOBILITY_DENOMINATOR = 0.25 + TAU**0.35
LOW_FREQUENCY = 0.02
HIGH_FREQUENCY = 1.5


def ring_laplacian(size: int = N) -> np.ndarray:
    laplacian = np.zeros((size, size), dtype=float)
    for index in range(size):
        for neighbor in ((index - 1) % size, (index + 1) % size):
            laplacian[index, index] += 1.0
            laplacian[index, neighbor] -= 1.0
    return laplacian


LAPLACIAN = ring_laplacian()


def common_support(k: int) -> np.ndarray:
    mask = np.zeros(N, dtype=float)
    if k > 0:
        start = (N - k) // 2
        mask[start:start + k] = 1.0
    return mask


def debye_loss(frequency: float) -> np.ndarray:
    omega_tau = 2.0 * np.pi * frequency * TAU
    return omega_tau / (1.0 + omega_tau**2)


def drive(frequency: float, amplitude: float, mask: np.ndarray, sign: float) -> np.ndarray:
    pattern = 1.0 + 0.12 * np.sin(
        1.3 * np.arange(N) + np.log10(frequency + 1e-12)
    )
    return sign * amplitude * debye_loss(frequency) * pattern * mask


def evolve(
    state: np.ndarray,
    force: np.ndarray,
    steps: int,
    *,
    coupling: float,
) -> np.ndarray:
    q = np.asarray(state, dtype=float).copy()
    for _ in range(steps):
        gradient = q * (q**2 - Q_WELL**2) + coupling * (LAPLACIAN @ q)
        q += DT * (-gradient + force) / MOBILITY_DENOMINATOR
        q = np.clip(q, -1.2, 1.2)
    return q


def forces(k: int, amplitude: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if k == 0:
        mask_a = np.zeros(N); mask_a[: N // 2] = 1.0
        mask_b = np.zeros(N); mask_b[N // 2:] = 1.0
    else:
        mask_a = common_support(k)
        mask_b = mask_a.copy()
    force_a = drive(LOW_FREQUENCY, amplitude, mask_a, +1.0)
    force_b = drive(HIGH_FREQUENCY, amplitude, mask_b, -1.0)
    return mask_a, mask_b, force_a, force_b


def run_case(
    k: int,
    coupling: float,
    *,
    amplitude: float = 12.0,
    write_steps: int = 70,
    release_steps: int = 20_000,
) -> Dict[str, object]:
    mask_a, mask_b, force_a, force_b = forces(k, amplitude)
    initial = -Q_WELL * np.ones(N)

    state_ba = evolve(
        evolve(initial, force_b, write_steps, coupling=coupling),
        force_a,
        write_steps,
        coupling=coupling,
    )
    state_ab = evolve(
        evolve(initial, force_a, write_steps, coupling=coupling),
        force_b,
        write_steps,
        coupling=coupling,
    )
    retained_ba = evolve(
        state_ba, np.zeros(N), release_steps, coupling=coupling
    )
    retained_ab = evolve(
        state_ab, np.zeros(N), release_steps, coupling=coupling
    )

    label_ba = np.sign(retained_ba).astype(int)
    label_ab = np.sign(retained_ab).astype(int)
    overlap = mask_a * mask_b
    jacobian = -np.diag(3.0 * initial**2 - Q_WELL**2) - coupling * LAPLACIAN
    bracket_norm = float(np.linalg.norm(jacobian @ (force_a - force_b)))
    different = label_ba != label_ab
    return {
        "k": int(k),
        "coupling": float(coupling),
        "support_overlap_contrast": float(overlap.var()),
        "bracket_norm": bracket_norm,
        "separation_after_write": float(np.max(np.abs(state_ba - state_ab))),
        "separation_retained": float(np.max(np.abs(retained_ba - retained_ab))),
        "retained_coordinates": int(different.sum()),
        "retained_outside_common_support": int((different & (overlap == 0)).sum()),
        "retained": bool(different.any()),
    }


def build_report() -> Dict[str, object]:
    couplings = (0.0, 0.13, 0.30)
    fractions = (0, 2, 3, 4, 5, 6, 8)
    scan = {
        f"k_{k}": {f"c_{coupling:g}": run_case(k, coupling) for coupling in couplings}
        for k in fractions
    }
    bracket_control = {
        "disjoint": run_case(0, 0.30),
        "partial_k4": run_case(4, 0.30),
        "full_k8": run_case(8, 0.30),
        "full_k8_weak_coupling": run_case(8, 0.13),
    }
    return {
        "model": "eight-coordinate coupled double well",
        "equation": "qdot_i=-q_i(q_i^2-q_w^2)-c sum_j L_ij q_j+f_i^X",
        "q_well": Q_WELL,
        "amplitude": 12.0,
        "write_steps_per_operation": 70,
        "release_steps": 20_000,
        "contested_fraction_scan": scan,
        "bracket_control": bracket_control,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "relaxed_exchange_order_minimal.json"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = build_report()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, row in report["bracket_control"].items():
        print(
            f"{name:>23}  bracket={row['bracket_norm']:.2f}  "
            f"write={row['separation_after_write']:.4f}  "
            f"retained={row['separation_retained']:.4f}"
        )
    print(json.dumps({"output": str(output), "complete": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
