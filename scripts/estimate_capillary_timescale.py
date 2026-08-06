#!/usr/bin/env python3
"""Estimate capillary meniscus relaxation time versus rotational diffusion."""

import argparse


def estimate_tau_int(eta_eff: float, gamma: float, R: float, prefactor: float = 1.0) -> float:
    """Return tau_int ≈ prefactor * eta_eff * 2R / gamma (s)."""
    return prefactor * eta_eff * (2.0 * R) / gamma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=5e-3, help="effective viscosity [Pa s]")
    parser.add_argument("--gamma", type=float, default=10e-3, help="interfacial tension [N/m]")
    parser.add_argument("--R", type=float, default=10e-6, help="particle size scale [m]")
    parser.add_argument("--Dr", type=float, default=1.0, help="rotational diffusion [s^-1]")
    parser.add_argument(
        "--prefactor",
        type=float,
        default=1.0,
        help="O(1) factor; use >1 for conservative estimate",
    )
    args = parser.parse_args()

    tau_int = estimate_tau_int(args.eta, args.gamma, args.R, args.prefactor)
    print(f"tau_int = {tau_int:.3e} s")
    print(f"tau_int * D_r = {tau_int * args.Dr:.3e}")


if __name__ == "__main__":
    main()

