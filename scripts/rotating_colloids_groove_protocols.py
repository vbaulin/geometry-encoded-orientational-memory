#!/usr/bin/env python3
"""Independent-field tests for grooved rotating colloids.

The legacy sweep uses a local groove energy h = epsilon * J.  That
parameterization cannot compare interacting and noninteracting rods at fixed
substrate strength.  This script treats J and h as independent rates and runs
two tests:

* amplifier: steady-state response at fixed h while J is varied;
* release: write with h > 0, reduce h, and measure overlap with the written
  state.  A J = 0 trace is the Brownian null control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rotating_colloids_hyperion_case import make_easy_axes, make_graph


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def summarize(values: Any) -> tuple[float, float]:
    arr = values.detach().cpu().numpy().astype(float)
    return float(np.mean(arr)), float(np.std(arr))


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the protocol simulations") from exc
    return torch


def build_system(
    *,
    n: int,
    device: str,
    graph_mode: str,
    axis_mode: str,
    graph_seed: int,
    cluster_size: int,
    crosslink_k: int,
    crosslink_weight: float,
    domain_angle_step: float,
) -> dict[str, Any]:
    torch = import_torch()
    axis_graph_mode = "square" if axis_mode == "uniform" else axis_mode
    if cluster_size >= n and (graph_mode == "mosaic" or axis_graph_mode == "mosaic"):
        raise ValueError(
            f"Mosaic protocol requires cluster_size < n; got cluster_size={cluster_size}, n={n}. "
            "Use at least four groove domains for the cancellation test."
        )
    _, src, tgt, _, weights, meta = make_graph(
        n,
        graph_mode=graph_mode,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        crosslink_k=crosslink_k,
        crosslink_weight=crosslink_weight,
        patch_angle_step=domain_angle_step,
    )
    axes = make_easy_axes(
        n,
        graph_mode=axis_graph_mode,
        graph_seed=graph_seed,
        cluster_size=cluster_size,
        patch_angle_step=domain_angle_step,
    )
    meta = dict(meta)
    meta["easy_axis_mode"] = axis_mode
    return {
        "src": torch.as_tensor(src, dtype=torch.long, device=device),
        "tgt": torch.as_tensor(tgt, dtype=torch.long, device=device),
        "weights": torch.as_tensor(weights, dtype=torch.float32, device=device),
        "axes": torch.as_tensor(axes, dtype=torch.float32, device=device),
        "meta": meta,
    }


def initialize_theta(*, replicas: int, nodes: int, device: str, seed: int):
    torch = import_torch()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    theta = 2.0 * math.pi * torch.rand((replicas, nodes), device=device, generator=generator) - math.pi
    return theta, generator


def advance(theta, *, steps: int, j: float, h: float, Dr: float, dt: float, system: dict[str, Any], generator):
    torch = import_torch()
    src = system["src"]
    tgt = system["tgt"]
    weights = system["weights"]
    axes = system["axes"]
    noise_scale = math.sqrt(2.0 * Dr * dt)
    for _ in range(int(steps)):
        delta = theta[:, src] - theta[:, tgt]
        edge_torque = -2.0 * j * weights * torch.sin(2.0 * delta)
        torque = torch.zeros_like(theta)
        torque.index_add_(1, src, edge_torque)
        torque.index_add_(1, tgt, -edge_torque)
        if h != 0.0:
            torque = torque - 2.0 * h * torch.sin(2.0 * (theta - axes))
        theta = theta + dt * torque + noise_scale * torch.randn(
            theta.shape, device=theta.device, generator=generator
        )
        theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi
    return theta


def instantaneous_metrics(theta, system: dict[str, Any]) -> dict[str, Any]:
    torch = import_torch()
    src = system["src"]
    tgt = system["tgt"]
    weights = system["weights"]
    axes = system["axes"]
    weight_sum = torch.clamp(weights.sum(), min=1e-12)
    cos_mean = torch.cos(2.0 * theta).mean(dim=1)
    sin_mean = torch.sin(2.0 * theta).mean(dim=1)
    s = torch.sqrt(cos_mean**2 + sin_mean**2)
    c2 = (torch.cos(2.0 * (theta[:, src] - theta[:, tgt])) * weights).sum(dim=1) / weight_sum
    g2 = torch.cos(2.0 * (theta - axes)).mean(dim=1)
    return {"S": s, "C2": c2, "G2": g2}


def run_stationary_point(
    *,
    n: int,
    j_over_Dr: float,
    h_over_Dr: float,
    Dr: float,
    dt: float,
    burn_steps: int,
    sample_steps: int,
    sample_stride: int,
    replicas: int,
    device: str,
    seed: int,
    system: dict[str, Any],
) -> dict[str, Any]:
    torch = import_torch()
    j = j_over_Dr * Dr
    h = h_over_Dr * Dr
    theta, generator = initialize_theta(replicas=replicas, nodes=n * n, device=device, seed=seed)
    theta = advance(theta, steps=burn_steps, j=j, h=h, Dr=Dr, dt=dt, system=system, generator=generator)

    s_sum = torch.zeros(replicas, device=device)
    c_sum = torch.zeros(replicas, device=device)
    g_sum = torch.zeros(replicas, device=device)
    s2_sum = torch.zeros(replicas, device=device)
    s4_sum = torch.zeros(replicas, device=device)
    g2_sum = torch.zeros(replicas, device=device)
    g4_sum = torch.zeros(replicas, device=device)
    q_cos = torch.zeros_like(theta)
    q_sin = torch.zeros_like(theta)
    q_cos_early = torch.zeros_like(theta)
    q_sin_early = torch.zeros_like(theta)
    q_cos_late = torch.zeros_like(theta)
    q_sin_late = torch.zeros_like(theta)
    q_early_count = 0
    q_late_count = 0
    expected_samples = max(1, int(math.ceil(sample_steps / sample_stride)))
    half_sample = max(1, expected_samples // 2)
    lag_indices = (1, 5, 20, 50)
    lag_sums = {lag: torch.zeros(replicas, device=device) for lag in lag_indices}
    lag_counts = {lag: 0 for lag in lag_indices}
    history: list[Any] = []
    count = 0
    blocks = max(1, int(math.ceil(sample_steps / sample_stride)))
    remaining = sample_steps
    for _ in range(blocks):
        step_count = min(sample_stride, remaining)
        if step_count <= 0:
            break
        theta = advance(theta, steps=step_count, j=j, h=h, Dr=Dr, dt=dt, system=system, generator=generator)
        metrics = instantaneous_metrics(theta, system)
        s_sum += metrics["S"]
        c_sum += metrics["C2"]
        g_sum += metrics["G2"]
        s2_sum += metrics["S"] ** 2
        s4_sum += metrics["S"] ** 4
        g2_sum += metrics["G2"] ** 2
        g4_sum += metrics["G2"] ** 4
        q_cos += torch.cos(2.0 * theta)
        q_sin += torch.sin(2.0 * theta)
        if count < half_sample:
            q_cos_early += torch.cos(2.0 * theta)
            q_sin_early += torch.sin(2.0 * theta)
            q_early_count += 1
        else:
            q_cos_late += torch.cos(2.0 * theta)
            q_sin_late += torch.sin(2.0 * theta)
            q_late_count += 1
        for lag in lag_indices:
            if len(history) >= lag:
                lag_sums[lag] += torch.cos(2.0 * (theta - history[-lag])).mean(dim=1)
                lag_counts[lag] += 1
        history.append(theta.detach().clone())
        if len(history) > max(lag_indices):
            history.pop(0)
        count += 1
        remaining -= step_count

    s = s_sum / count
    c2 = c_sum / count
    g2 = g_sum / count
    qea = ((q_cos / count) ** 2 + (q_sin / count) ** 2).mean(dim=1)
    if q_early_count:
        qea_early = (
            (q_cos_early / q_early_count) ** 2 + (q_sin_early / q_early_count) ** 2
        ).mean(dim=1)
    else:
        qea_early = torch.full((replicas,), float("nan"), device=device)
    if q_late_count:
        qea_late = (
            (q_cos_late / q_late_count) ** 2 + (q_sin_late / q_late_count) ** 2
        ).mean(dim=1)
    else:
        qea_late = torch.full((replicas,), float("nan"), device=device)
    s_mean, s_std = summarize(s)
    c_mean, c_std = summarize(c2)
    g_mean, g_std = summarize(g2)
    q_mean, q_std = summarize(qea)
    q_early_mean, q_early_std = summarize(qea_early)
    q_late_mean, q_late_std = summarize(qea_late)

    s_scalar_mean = float(s.mean().item())
    g_scalar_mean = float(g2.mean().item())
    s_second = float((s2_sum / count).mean().item())
    s_fourth = float((s4_sum / count).mean().item())
    g_second = float((g2_sum / count).mean().item())
    g_fourth = float((g4_sum / count).mean().item())
    node_count = n * n
    chi_s = node_count * max(0.0, s_second - s_scalar_mean**2)
    chi_g = node_count * max(0.0, g_second - g_scalar_mean**2)
    binder_s = 1.0 - s_fourth / (3.0 * s_second**2) if s_second > 1e-12 else None
    binder_g = 1.0 - g_fourth / (3.0 * g_second**2) if g_second > 1e-12 else None
    temporal = {
        f"temporal_C2_lag{lag}": (
            float((lag_sums[lag] / lag_counts[lag]).mean().item())
            if lag_counts[lag]
            else None
        )
        for lag in lag_indices
    }
    return {
        "n": n,
        "replicas": replicas,
        "J_over_Dr": j_over_Dr,
        "h_over_Dr": h_over_Dr,
        "S_mean": s_mean,
        "S_std": s_std,
        "C2_mean": c_mean,
        "C2_std": c_std,
        "G2_mean": g_mean,
        "G2_std": g_std,
        "qEA_field_on_mean": q_mean,
        "qEA_field_on_std": q_std,
        "qEA_early_mean": q_early_mean,
        "qEA_early_std": q_early_std,
        "qEA_late_mean": q_late_mean,
        "qEA_late_std": q_late_std,
        "qEA_half_drift": q_late_mean - q_early_mean,
        "qEA_minus_G2_squared": q_mean - g_mean**2,
        "susceptibility_S": chi_s,
        "susceptibility_G2": chi_g,
        "binder_S": binder_s,
        "binder_G2": binder_g,
        "chi_qEA_replicas": node_count * q_std**2,
        **temporal,
    }


def run_amplifier(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        system = build_system(
            n=n,
            device=args.device,
            graph_mode=args.graph_mode,
            axis_mode=args.axis_mode,
            graph_seed=args.graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            domain_angle_step=args.domain_angle_step,
        )
        for h_index, h_over_Dr in enumerate(args.h_values):
            block: list[dict[str, Any]] = []
            for j_index, j_over_Dr in enumerate(args.j_values):
                row = run_stationary_point(
                    n=n,
                    j_over_Dr=j_over_Dr,
                    h_over_Dr=h_over_Dr,
                    Dr=args.Dr,
                    dt=args.dt,
                    burn_steps=args.burn_steps,
                    sample_steps=args.sample_steps,
                    sample_stride=args.sample_stride,
                    replicas=args.replicas,
                    device=args.device,
                    seed=args.seed + 100003 * n + 1009 * h_index + 17 * j_index,
                    system=system,
                )
                block.append(row)
                print(
                    json.dumps(
                        {
                            "event": "stationary_point_complete",
                            "n": n,
                            "J_over_Dr": j_over_Dr,
                            "h_over_Dr": h_over_Dr,
                            "S": row["S_mean"],
                            "G2": row["G2_mean"],
                            "qEA": row["qEA_field_on_mean"],
                        }
                    ),
                    flush=True,
                )
            baseline = min(block, key=lambda row: abs(float(row["J_over_Dr"])))
            for row in block:
                row["G2_noninteracting"] = baseline["G2_mean"]
                row["cooperative_gain"] = float(row["G2_mean"]) - float(baseline["G2_mean"])
                row["contrast_gain"] = (
                    float(row["G2_mean"]) / float(baseline["G2_mean"])
                    if float(baseline["G2_mean"]) > 1e-8
                    else None
                )
                rows.append(row)
    return rows


def logarithmic_steps(max_steps: int, count: int) -> list[int]:
    if max_steps <= 0:
        return [0]
    values = np.unique(np.rint(np.geomspace(1, max_steps, count)).astype(int))
    return [0] + [int(value) for value in values]


def run_release_trace(
    *,
    n: int,
    j_over_Dr: float,
    h_write_over_Dr: float,
    h_release_fraction: float,
    args: argparse.Namespace,
    system: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    torch = import_torch()
    j = j_over_Dr * args.Dr
    h_write = h_write_over_Dr * args.Dr
    h_release = h_release_fraction * h_write
    theta, generator = initialize_theta(replicas=args.replicas, nodes=n * n, device=args.device, seed=seed)
    theta = advance(
        theta,
        steps=args.write_steps,
        j=j,
        h=h_write,
        Dr=args.Dr,
        dt=args.dt,
        system=system,
        generator=generator,
    )
    written = theta.detach().clone()
    schedule = logarithmic_steps(args.release_steps, args.release_points)
    rows: list[dict[str, Any]] = []
    previous = 0
    for step in schedule:
        if step > previous:
            theta = advance(
                theta,
                steps=step - previous,
                j=j,
                h=h_release,
                Dr=args.Dr,
                dt=args.dt,
                system=system,
                generator=generator,
            )
        metrics = instantaneous_metrics(theta, system)
        overlap = torch.cos(2.0 * (theta - written)).mean(dim=1)
        q_mean, q_std = summarize(overlap)
        s_mean, s_std = summarize(metrics["S"])
        c_mean, c_std = summarize(metrics["C2"])
        g_mean, g_std = summarize(metrics["G2"])
        rows.append(
            {
                "n": n,
                "replicas": args.replicas,
                "J_over_Dr": j_over_Dr,
                "h_write_over_Dr": h_write_over_Dr,
                "h_release_fraction": h_release_fraction,
                "release_step": step,
                "release_time_Dr": step * args.dt * args.Dr,
                "Q_rem_mean": q_mean,
                "Q_rem_std": q_std,
                "S_mean": s_mean,
                "S_std": s_std,
                "C2_mean": c_mean,
                "C2_std": c_std,
                "G2_mean": g_mean,
                "G2_std": g_std,
            }
        )
        previous = step
    return rows


def run_release(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        system = build_system(
            n=n,
            device=args.device,
            graph_mode=args.graph_mode,
            axis_mode=args.axis_mode,
            graph_seed=args.graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            domain_angle_step=args.domain_angle_step,
        )
        for h_index, h_write in enumerate(args.release_h_values):
            for j_index, j_value in enumerate(args.release_j_values):
                for fraction in args.release_fractions:
                    trace = run_release_trace(
                        n=n,
                        j_over_Dr=j_value,
                        h_write_over_Dr=h_write,
                        h_release_fraction=fraction,
                        args=args,
                        system=system,
                        seed=args.seed
                        + 700001 * n
                        + 1009 * h_index
                        + 37 * j_index,
                    )
                    rows.extend(trace)
                    print(
                        json.dumps(
                            {
                                "event": "release_trace_complete",
                                "n": n,
                                "J_over_Dr": j_value,
                                "h_write_over_Dr": h_write,
                                "release_fraction": fraction,
                                "Q_end": trace[-1]["Q_rem_mean"],
                            }
                        ),
                        flush=True,
                    )
    return rows


def make_switched_system(system: dict[str, Any], *, n: int, cluster_size: int) -> dict[str, Any]:
    """Return the same graph with the mosaic easy-axis pattern shifted by one domain."""
    torch = import_torch()
    switched = dict(system)
    axes = system["axes"].reshape(n, n)
    switched["axes"] = torch.roll(axes, shifts=cluster_size, dims=1).reshape(-1)
    switched["pattern_overlap"] = float(
        torch.cos(2.0 * (system["axes"] - switched["axes"])).mean().item()
    )
    return switched


def run_switch_trace(
    *,
    n: int,
    j_over_Dr: float,
    h_over_Dr: float,
    args: argparse.Namespace,
    system_a: dict[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Write mosaic A, switch to incompatible mosaic B, and track both memories."""
    torch = import_torch()
    j = j_over_Dr * args.Dr
    h = h_over_Dr * args.Dr
    system_b = make_switched_system(system_a, n=n, cluster_size=args.cluster_size)
    theta, generator = initialize_theta(replicas=args.replicas, nodes=n * n, device=args.device, seed=seed)
    theta = advance(
        theta,
        steps=args.write_steps,
        j=j,
        h=h,
        Dr=args.Dr,
        dt=args.dt,
        system=system_a,
        generator=generator,
    )
    written = theta.detach().clone()
    schedule = logarithmic_steps(args.switch_steps, args.switch_points)
    rows: list[dict[str, Any]] = []
    previous = 0
    for step in schedule:
        if step > previous:
            theta = advance(
                theta,
                steps=step - previous,
                j=j,
                h=h,
                Dr=args.Dr,
                dt=args.dt,
                system=system_b,
                generator=generator,
            )
        metrics_b = instantaneous_metrics(theta, system_b)
        q_write = torch.cos(2.0 * (theta - written)).mean(dim=1)
        g_a = torch.cos(2.0 * (theta - system_a["axes"])).mean(dim=1)
        q_mean, q_std = summarize(q_write)
        ga_mean, ga_std = summarize(g_a)
        s_mean, s_std = summarize(metrics_b["S"])
        c_mean, c_std = summarize(metrics_b["C2"])
        gb_mean, gb_std = summarize(metrics_b["G2"])
        rows.append(
            {
                "n": n,
                "replicas": args.replicas,
                "J_over_Dr": j_over_Dr,
                "h_over_Dr": h_over_Dr,
                "switch_step": step,
                "switch_time_Dr": step * args.dt * args.Dr,
                "pattern_overlap_A_B": system_b["pattern_overlap"],
                "Q_written_mean": q_mean,
                "Q_written_std": q_std,
                "G2_A_mean": ga_mean,
                "G2_A_std": ga_std,
                "G2_B_mean": gb_mean,
                "G2_B_std": gb_std,
                "S_mean": s_mean,
                "S_std": s_std,
                "C2_mean": c_mean,
                "C2_std": c_std,
            }
        )
        previous = step
    return rows


def run_switch(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n in args.sizes:
        system = build_system(
            n=n,
            device=args.device,
            graph_mode=args.graph_mode,
            axis_mode=args.axis_mode,
            graph_seed=args.graph_seed,
            cluster_size=args.cluster_size,
            crosslink_k=args.crosslink_k,
            crosslink_weight=args.crosslink_weight,
            domain_angle_step=args.domain_angle_step,
        )
        for h_index, h_value in enumerate(args.switch_h_values):
            for j_index, j_value in enumerate(args.switch_j_values):
                trace = run_switch_trace(
                    n=n,
                    j_over_Dr=j_value,
                    h_over_Dr=h_value,
                    args=args,
                    system_a=system,
                    seed=args.seed + 900001 * n + 1009 * h_index + 37 * j_index,
                )
                rows.extend(trace)
                print(
                    json.dumps(
                        {
                            "event": "switch_trace_complete",
                            "n": n,
                            "J_over_Dr": j_value,
                            "h_over_Dr": h_value,
                            "Q_written_end": trace[-1]["Q_written_mean"],
                            "G2_B_end": trace[-1]["G2_B_mean"],
                        }
                    ),
                    flush=True,
                )
    return rows


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(row), allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(
        "discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_grooved_protocols"
    ))
    parser.add_argument(
        "--protocol",
        choices=["amplifier", "release", "switch", "both", "all"],
        default="all",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=parse_ints, default=parse_ints("16"))
    parser.add_argument("--replicas", type=int, default=24)
    parser.add_argument("--Dr", type=float, default=0.45)
    parser.add_argument("--dt", type=float, default=0.015)
    parser.add_argument("--burn-steps", type=int, default=12000)
    parser.add_argument("--sample-steps", type=int, default=24000)
    parser.add_argument("--sample-stride", type=int, default=40)
    parser.add_argument("--j-values", type=parse_floats, default=parse_floats("0,0.5,1,1.5,2,2.5,3"))
    parser.add_argument("--h-values", type=parse_floats, default=parse_floats("0.1,0.2,0.4,0.6,0.8,1.2,1.8"))
    parser.add_argument("--write-steps", type=int, default=24000)
    parser.add_argument("--release-steps", type=int, default=60000)
    parser.add_argument("--release-points", type=int, default=30)
    parser.add_argument("--release-j-values", type=parse_floats, default=parse_floats("0,1,2,3"))
    parser.add_argument("--release-h-values", type=parse_floats, default=parse_floats("0.4,0.8,1.2"))
    parser.add_argument("--release-fractions", type=parse_floats, default=parse_floats("0,0.15"))
    parser.add_argument("--switch-steps", type=int, default=60000)
    parser.add_argument("--switch-points", type=int, default=30)
    parser.add_argument("--switch-j-values", type=parse_floats, default=parse_floats("0,1,2,3"))
    parser.add_argument("--switch-h-values", type=parse_floats, default=parse_floats("0.2,0.6"))
    parser.add_argument(
        "--graph-mode",
        choices=["square", "triangular", "long-range", "random", "mosaic"],
        default="mosaic",
        help="interaction graph; choose square with --axis-mode mosaic for the matched-boundary control",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["uniform", "square", "triangular", "mosaic"],
        default="mosaic",
        help="prescribed easy-axis pattern, selected independently of the interaction graph",
    )
    parser.add_argument("--graph-seed", type=int, default=12345)
    parser.add_argument("--seed", type=int, default=88117)
    parser.add_argument("--cluster-size", type=int, default=8)
    parser.add_argument("--crosslink-k", type=int, default=2)
    parser.add_argument("--crosslink-weight", type=float, default=0.18)
    parser.add_argument("--domain-angle-step", type=float, default=math.pi / 4.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {
            key: json_ready(value)
            for key, value in vars(args).items()
            if key != "output_dir"
        },
        "release_design": {
            "paired_written_state_across_release_fractions": True,
            "common_noise_across_release_fractions": True,
        },
    }
    if args.protocol in {"amplifier", "both", "all"}:
        amplifier = run_amplifier(args)
        payload["amplifier"] = amplifier
        write_jsonl(args.output_dir / "amplifier_points.jsonl", amplifier)
    if args.protocol in {"release", "both", "all"}:
        release = run_release(args)
        payload["release"] = release
        write_jsonl(args.output_dir / "release_traces.jsonl", release)
    if args.protocol in {"switch", "all"}:
        switch = run_switch(args)
        payload["switch"] = switch
        write_jsonl(args.output_dir / "switch_traces.jsonl", switch)

    output = args.output_dir / "groove_protocols.json"
    output.write_text(json.dumps(json_ready(payload), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
