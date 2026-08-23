#!/usr/bin/env python3
"""Test whether rotating colloids retain the order of two write operations.

Two equal-duration orienting fields, A and B, are applied as AB and BA. Both
histories then evolve for the same time with the fields removed. The contested
protocol applies opposing fields to the same rotors; the partitioned control
applies them to spatially separated rotors. Common-noise runs isolate the
deterministic order effect, whereas independent-noise runs test whether pulse
order remains decodable against ordinary replica decorrelation.
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
    simulate_ensemble,
)


def append_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def periodic_x_mask(positions: np.ndarray, box_x: float, lo: float, hi: float) -> np.ndarray:
    fraction = np.mod(positions[:, 0], box_x) / box_x
    return ((fraction >= lo) & (fraction < hi)).astype(np.float32)


def periodic_centered_x_mask(
    positions: np.ndarray, box_x: float, center: float, width: float
) -> np.ndarray:
    """Periodic stripe of fractional width around ``center`` in the x direction."""
    width = float(np.clip(width, 0.0, 1.0))
    if width == 0.0:
        return np.zeros(positions.shape[0], dtype=np.float32)
    if width == 1.0:
        return np.ones(positions.shape[0], dtype=np.float32)
    fraction = np.mod(positions[:, 0], box_x) / box_x
    distance = np.abs(np.mod(fraction - center + 0.5, 1.0) - 0.5)
    return (distance < width / 2.0).astype(np.float32)


def masks(
    positions: np.ndarray,
    box_x: float,
    mode: str,
    *,
    contest_fraction: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    if mode == "contested":
        shared = periodic_centered_x_mask(
            positions, box_x, center=0.375, width=contest_fraction
        )
        return shared, shared.copy()
    if mode == "partitioned":
        return (
            periodic_x_mask(positions, box_x, 0.00, 0.25),
            periodic_x_mask(positions, box_x, 0.50, 0.75),
        )
    raise ValueError(mode)


def run_segment(
    graph,
    state: np.ndarray,
    *,
    axis: np.ndarray | None,
    weight: np.ndarray | None,
    field: float,
    steps: int,
    stride: int,
    j_align: float,
    g_capillary: float,
    dt: float,
    seed: int,
    device,
) -> Dict[str, object]:
    return simulate_ensemble(
        graph,
        j_align=j_align,
        g_capillary=g_capillary,
        replicas=int(state.shape[0]),
        burn_in_steps=0,
        sample_steps=steps,
        sample_stride=max(1, stride),
        dt=dt,
        seed=seed,
        device=device,
        initial_theta=state,
        write_axis=axis,
        write_field=field,
        write_weight=weight,
    )


def apply_order(
    graph,
    initial: np.ndarray,
    *,
    order: str,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    field: float,
    pulse_steps: int,
    release_steps: int,
    stride: int,
    j_align: float,
    g_capillary: float,
    dt: float,
    seed: int,
    device,
) -> Dict[str, object]:
    operations = {
        "A": (axis_a, mask_a),
        "B": (axis_b, mask_b),
    }
    state = initial.copy()
    for index, label in enumerate(order):
        axis, weight = operations[label]
        pulse = run_segment(
            graph,
            state,
            axis=axis,
            weight=weight,
            field=field,
            steps=pulse_steps,
            stride=stride,
            j_align=j_align,
            g_capillary=g_capillary,
            dt=dt,
            seed=seed + 10_007 * (index + 1),
            device=device,
        )
        state = np.asarray(pulse["state_after_steps"], dtype=np.float32)
    return run_segment(
        graph,
        state,
        axis=None,
        weight=None,
        field=0.0,
        steps=release_steps,
        stride=stride,
        j_align=j_align,
        g_capillary=g_capillary,
        dt=dt,
        seed=seed + 30_011,
        device=device,
    )


def masked_overlap(theta: np.ndarray, axis: np.ndarray, mask: np.ndarray) -> np.ndarray:
    denominator = max(float(mask.sum()), 1.0)
    return (
        np.cos(2.0 * (theta - axis[None, None, :])) * mask[None, None, :]
    ).sum(axis=2) / denominator


def compare_orders(
    graph,
    initial: np.ndarray,
    axis_a: np.ndarray,
    *,
    mode: str,
    contest_fraction: float = 0.25,
    field: float,
    pulse_steps: int,
    release_steps: int,
    stride: int,
    j_align: float,
    g_capillary: float,
    dt: float,
    seed: int,
    noise_mode: str,
    device,
) -> Dict[str, object]:
    mask_a, mask_b = masks(
        graph.positions,
        float(graph.box[0]),
        mode,
        contest_fraction=contest_fraction,
    )
    direct_support_edges = (
        ((mask_a[graph.src] > 0) & (mask_b[graph.tgt] > 0))
        | ((mask_b[graph.src] > 0) & (mask_a[graph.tgt] > 0))
    )
    axis_b = np.mod(axis_a + math.pi / 2.0, math.pi).astype(np.float32)
    common = dict(
        graph=graph,
        initial=initial,
        axis_a=axis_a,
        axis_b=axis_b,
        mask_a=mask_a,
        mask_b=mask_b,
        field=field,
        pulse_steps=pulse_steps,
        release_steps=release_steps,
        stride=stride,
        j_align=j_align,
        g_capillary=g_capillary,
        dt=dt,
        seed=seed,
        device=device,
    )
    if noise_mode not in {"common", "independent"}:
        raise ValueError(f"unknown noise mode: {noise_mode}")
    ab = apply_order(order="AB", **common)
    ba_common = dict(common)
    if noise_mode == "independent":
        ba_common["seed"] = int(seed) + 70_000_019
    ba = apply_order(order="BA", **ba_common)
    theta_ab = np.asarray(ab["snapshots"], dtype=float)
    theta_ba = np.asarray(ba["snapshots"], dtype=float)
    read_mask = np.maximum(mask_a, mask_b)
    support_overlap = mask_a * mask_b
    read_count = float(read_mask.sum())
    denominator = max(read_count, 1.0)
    if read_count == 0.0:
        history_overlap = np.ones(theta_ab.shape[:2], dtype=float)
    else:
        history_overlap = (
            np.cos(2.0 * (theta_ab - theta_ba)) * read_mask[None, None, :]
        ).sum(axis=2) / denominator
    qa_ab = masked_overlap(theta_ab, axis_a, read_mask)
    qa_ba = masked_overlap(theta_ba, axis_a, read_mask)
    order_readout = 0.5 * (qa_ba - qa_ab)
    terminal_qa_ab = np.asarray(qa_ab[-1], dtype=float)
    terminal_qa_ba = np.asarray(qa_ba[-1], dtype=float)
    decode_accuracy = 0.5 * (
        float(np.mean(terminal_qa_ab < 0.0))
        + float(np.mean(terminal_qa_ba > 0.0))
    )
    pooled_variance = 0.5 * (
        float(np.var(terminal_qa_ab, ddof=1))
        + float(np.var(terminal_qa_ba, ddof=1))
    ) if terminal_qa_ab.size > 1 else 0.0
    decode_d_prime = (
        float(terminal_qa_ba.mean() - terminal_qa_ab.mean())
        / math.sqrt(max(pooled_variance, 1e-12))
    )
    time = np.asarray(ab["metrics"]["time"], dtype=float)
    return {
        "mode": mode,
        "noise_mode": noise_mode,
        "field": float(field),
        "active_a": int(mask_a.sum()),
        "active_b": int(mask_b.sum()),
        "active_overlap": int(np.minimum(mask_a, mask_b).sum()),
        "contest_fraction_requested": (
            float(contest_fraction) if mode == "contested" else 0.0
        ),
        "contest_fraction_realized": float(support_overlap.mean()),
        "support_overlap_contrast": float(support_overlap.var()),
        "direct_support_edges": int(direct_support_edges.sum()),
        "time": time.tolist(),
        "history_overlap_mean": history_overlap.mean(axis=1).tolist(),
        "history_overlap_sem": (
            history_overlap.std(axis=1, ddof=1) / math.sqrt(history_overlap.shape[1])
            if history_overlap.shape[1] > 1 else np.zeros(history_overlap.shape[0])
        ).tolist(),
        "order_readout_mean": order_readout.mean(axis=1).tolist(),
        "order_readout_sem": (
            order_readout.std(axis=1, ddof=1) / math.sqrt(order_readout.shape[1])
            if order_readout.shape[1] > 1 else np.zeros(order_readout.shape[0])
        ).tolist(),
        "terminal_history_overlap": float(history_overlap[-1].mean()),
        "terminal_order_separation": float(1.0 - history_overlap[-1].mean()),
        "terminal_order_readout": float(order_readout[-1].mean()),
        "terminal_QA_AB": terminal_qa_ab.tolist(),
        "terminal_QA_BA": terminal_qa_ba.tolist(),
        "terminal_decode_accuracy_zero_threshold": float(decode_accuracy),
        "terminal_decode_d_prime": float(decode_d_prime),
    }


def existing_keys(path: Path) -> set[Tuple[int, float, str, float, str]]:
    keys: set[Tuple[int, float, str, float, str]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mode = str(row["mode"])
        keys.add((
            int(row["graph_seed"]),
            float(row["field"]),
            mode,
            float(row.get(
                "contest_fraction_requested",
                0.25 if mode == "contested" else 0.0,
            )),
            str(row.get("noise_mode", "common")),
        ))
    return keys


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "discoveries/theory_experiment_interface/rotating_colloids_hyperion/"
            "rotating_colloids_operation_order_memory"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--graph-seeds", default="17,29,43")
    parser.add_argument("--fields", default="0.5,1,2,3,5,8")
    parser.add_argument(
        "--contest-fractions",
        default="",
        help=(
            "optional jointly driven fractions; when supplied, add a fixed-field "
            "contested-support scan"
        ),
    )
    parser.add_argument(
        "--fraction-scan-only",
        action="store_true",
        help="run only --contest-fractions at the largest --fields value",
    )
    parser.add_argument("--replicas", type=int, default=12)
    parser.add_argument("--disorder", type=float, default=0.16)
    parser.add_argument("--j-align", type=float, default=4.0)
    parser.add_argument("--g-capillary", type=float, default=5.0)
    parser.add_argument("--dt", type=float, default=0.0025)
    parser.add_argument("--equilibration-steps", type=int, default=3000)
    parser.add_argument("--pulse-steps", type=int, default=2000)
    parser.add_argument("--release-steps", type=int, default=8000)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--noise-modes",
        default="common",
        help="comma-separated: common, independent, or both values",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.quick:
        args.graph_seeds = str(parse_int_list(args.graph_seeds)[0])
        args.fields = "1,3,6"
        args.replicas = min(args.replicas, 6)
        args.equilibration_steps = min(args.equilibration_steps, 1200)
        args.pulse_steps = min(args.pulse_steps, 800)
        args.release_steps = min(args.release_steps, 2400)
        args.stride = min(args.stride, 30)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "operation_order_memory.jsonl"
    if args.no_resume and rows_path.exists():
        rows_path.unlink()
    completed = existing_keys(rows_path)
    device = resolve_device(args.device)
    fields = parse_float_list(args.fields)
    contest_fractions = (
        parse_float_list(args.contest_fractions) if args.contest_fractions else []
    )
    if any(value < 0.0 or value > 1.0 for value in contest_fractions):
        raise ValueError("--contest-fractions values must lie in [0, 1]")
    graph_seeds = parse_int_list(args.graph_seeds)
    noise_modes = [item.strip() for item in args.noise_modes.split(",") if item.strip()]
    invalid_noise_modes = sorted(set(noise_modes) - {"common", "independent"})
    if invalid_noise_modes:
        raise ValueError(f"invalid --noise-modes values: {invalid_noise_modes}")
    standard_specs = [] if args.fraction_scan_only else [
        (field, mode, 0.25)
        for field in fields
        for mode in ("partitioned", "contested")
    ]
    fraction_specs = [
        (max(fields), "contested", fraction) for fraction in contest_fractions
    ]
    specs = list(dict.fromkeys(standard_specs + fraction_specs))
    total = len(specs) * len(graph_seeds) * len(noise_modes)
    done = 0

    for graph_seed in graph_seeds:
        graph = make_caged_graph(
            args.n,
            disorder=args.disorder,
            cutoff=2.6,
            alignment_range=1.35,
            alignment_decay=0.20,
            seed=graph_seed,
        )
        target_run = simulate_ensemble(
            graph,
            j_align=args.j_align,
            g_capillary=args.g_capillary,
            replicas=1,
            burn_in_steps=args.equilibration_steps,
            sample_steps=1,
            sample_stride=1,
            dt=args.dt,
            seed=args.seed + graph_seed * 1009,
            device=device,
        )
        axis_a = np.asarray(target_run["state_after_steps"], dtype=np.float32)[0]
        initial_run = simulate_ensemble(
            graph,
            j_align=args.j_align,
            g_capillary=args.g_capillary,
            replicas=args.replicas,
            burn_in_steps=args.equilibration_steps,
            sample_steps=1,
            sample_stride=1,
            dt=args.dt,
            seed=args.seed + graph_seed * 2003,
            device=device,
        )
        initial = np.asarray(initial_run["state_after_steps"], dtype=np.float32)
        for field, mode, contest_fraction in specs:
            for noise_mode in noise_modes:
                done += 1
                key_fraction = float(contest_fraction) if mode == "contested" else 0.0
                key = (graph_seed, float(field), mode, key_fraction, noise_mode)
                if key in completed:
                    continue
                result = compare_orders(
                    graph,
                    initial,
                    axis_a,
                    mode=mode,
                    contest_fraction=contest_fraction,
                    field=field,
                    pulse_steps=args.pulse_steps,
                    release_steps=args.release_steps,
                    stride=args.stride,
                    j_align=args.j_align,
                    g_capillary=args.g_capillary,
                    dt=args.dt,
                    seed=args.seed + graph_seed * 3001 + int(field * 100),
                    noise_mode=noise_mode,
                    device=device,
                )
                row = {
                    "graph_seed": graph_seed,
                    "n": args.n,
                    "replicas": args.replicas,
                    "j_align": args.j_align,
                    "g_capillary": args.g_capillary,
                    "disorder": args.disorder,
                    "dt": args.dt,
                    "pulse_steps": args.pulse_steps,
                    "release_steps": args.release_steps,
                    **result,
                }
                append_jsonl(rows_path, row)
                completed.add(key)
                print(json.dumps({
                    "event": "order_memory_progress",
                    "completed": done,
                    "total": total,
                    "graph_seed": graph_seed,
                    "field": field,
                    "mode": mode,
                    "noise_mode": noise_mode,
                    "contest_fraction": contest_fraction,
                    "terminal_order_separation": result["terminal_order_separation"],
                    "terminal_order_readout": result["terminal_order_readout"],
                    "terminal_decode_accuracy": result[
                        "terminal_decode_accuracy_zero_threshold"
                    ],
                }), flush=True)

    manifest = {
        "output": str(rows_path),
        "parameters": vars(args),
        "operation_A": "temporary field toward an equilibrated angular pattern",
        "operation_B": "temporary field rotated by pi/2",
        "readout": "AB versus BA after both fields are removed",
        "noise_modes": noise_modes,
        "common_random_numbers": "used only for rows with noise_mode=common",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "complete", **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
