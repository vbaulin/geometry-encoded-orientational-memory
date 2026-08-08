#!/usr/bin/env python3
"""Compare retention protocols across positional-disorder amplitudes.

The static disorder scan measures the finite-window persistence q_EA, not
retention. This script reduces the split-replica and write-release-read
protocols run at two or more disorder amplitudes to the endpoint overlaps and
the information they imply, so that the collapse of the global director can be
compared against what is actually retained.

Note that `rotating_colloids_capillary_pair.py --skip-scan` builds its graph
from the FIRST entry of --graph-seeds only. One output directory therefore
holds one quenched graph; run it once per seed into separate directories to
obtain graph-to-graph error bars.

    python -B scripts/analyze_rotating_colloids_disorder_protocols.py \
      --input-dir colloid --output-dir colloid/analysis
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


def retained_bits(resultant: float) -> float:
    """Lower bound on information retained per rotor, in bits.

    Maximum-entropy (von Mises) density on the circle of the doubled angle at
    fixed mean resultant. Maximizing entropy minimizes information, so this is
    a bound rather than an estimate.
    """

    from scipy.optimize import brentq
    from scipy.special import i0, i1

    if resultant <= 1e-12:
        return 0.0
    kappa = brentq(lambda k: i1(k) / i0(k) - resultant, 1e-12, 700.0)
    return float((kappa * (i1(kappa) / i0(kappa)) - math.log(i0(kappa))) / math.log(2.0))


def tail_mean(values, fraction: float) -> float:
    """Average the last `fraction` of a slowly varying trajectory.

    A single final sample of S or of the overlap is one configuration at one
    time. Near the order-disorder crossover its scatter is dominated by
    thermal fluctuation rather than by the quenched graph, which inflates the
    apparent graph-to-graph spread.
    """

    array = np.asarray(values, dtype=float)
    count = max(1, int(round(array.size * fraction)))
    return float(array[-count:].mean())


def summarize(run: dict[str, Any], fraction: float) -> dict[str, float]:
    release_time = np.asarray(run["write_release"]["release_time"], dtype=float)
    # Two replicas drawn from the same one-body angular distribution already
    # overlap by S^2 through the common director. The connected part is what
    # the director does not explain, and is the quantity the hidden-memory
    # claim rests on. release_S is recorded on the same release trajectory as
    # release_overlap, so the subtraction is matched for the written state.
    final_S = tail_mean(run["write_release"]["release_S"], fraction)
    write_end = tail_mean(run["write_release"]["release_overlap"], fraction)
    split_end = tail_mean(run["split_replica"]["overlap_mean"], fraction)
    return {
        "connected_write_end": write_end - final_S**2,
        # The split protocol does not record S; final_S is used as a proxy.
        "connected_split_end": split_end - final_S**2,
        "split_end": split_end,
        "split_time": float(run["split_replica"]["time"][-1]),
        "write_on": float(run["write_release"]["write_overlap"][-1]),
        "write_end": write_end,
        "release_span": float(release_time[-1] - release_time[0]),
        "S_end": final_S,
        "G2_end": tail_mean(run["write_release"]["release_G2"], fraction),
        "no_capillary_split_end": tail_mean(run["no_capillary_split_replica"]["overlap_mean"], fraction),
        "no_capillary_write_end": tail_mean(run["no_capillary_write_release"]["release_overlap"], fraction),
        "final_sample_S": float(run["write_release"]["release_S"][-1]),
        "final_sample_write": float(run["write_release"]["release_overlap"][-1]),
        "final_sample_split": float(run["split_replica"]["overlap_mean"][-1]),
    }


def mean_sd(values) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return float(array.mean()), float("nan")
    return float(array.mean()), float(array.std(ddof=1))


def standard_error(block: dict[str, Any], count: int) -> float:
    """Uncertainty in a mean. The graph SD describes the spread of graphs, and
    comparing two means against it rejects real differences."""

    if count < 2 or math.isnan(block["graph_sd"]):
        return float("nan")
    return block["graph_sd"] / math.sqrt(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.1,
        help="Fraction of the trajectory tail to average. Use 0 for the final sample only.",
    )
    parser.add_argument(
        "--matched-s",
        type=float,
        default=0.10,
        help="Director threshold for the matched-S control.",
    )
    parser.add_argument(
        "--node-count",
        type=int,
        help="Restrict to one rotor count. Required when the tree mixes sizes.",
    )
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("**/capillary_pair_protocols.json"))
    if not paths:
        raise SystemExit(
            f"no capillary_pair_protocols.json under {args.input_dir}\n"
            "Point --input-dir at the parent of the protocol run directories."
        )

    groups: dict[float, list[dict[str, Any]]] = {}
    seeds: dict[float, list[int]] = {}
    node_counts: dict[float, list[int]] = {}
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        graph = run["model"]["graph"]
        size = int(graph["node_count"])
        if args.node_count is not None and size != args.node_count:
            continue
        amplitude = float(graph["disorder"])
        groups.setdefault(amplitude, []).append(summarize(run, args.tail_fraction))
        seeds.setdefault(amplitude, []).append(int(graph["seed"]))
        node_counts.setdefault(amplitude, []).append(size)

    if not groups:
        raise SystemExit(f"no protocol runs with node_count={args.node_count}")
    # S is size dependent, and so therefore is the S^2 subtracted from the
    # overlap. Pooling sizes into one amplitude would mix incomparable states.
    present = sorted({size for values in node_counts.values() for size in values})
    if len(present) > 1:
        raise SystemExit(
            f"input mixes rotor counts {present}; the connected overlap subtracts a "
            "size-dependent S^2, so the runs are not comparable.\n"
            f"Re-run with --node-count {present[0]} to select one."
        )

    amplitudes = sorted(groups)
    table = []
    for amplitude in amplitudes:
        block = groups[amplitude]
        entry: dict[str, Any] = {
            "disorder": amplitude,
            "graphs": len(block),
            "graph_seeds": sorted(seeds[amplitude]),
            "node_count": node_counts[amplitude][0],
        }
        for field in (
            "split_end", "write_on", "write_end", "S_end", "G2_end",
            "final_sample_S", "final_sample_write", "final_sample_split",
            "connected_split_end", "connected_write_end",
            "no_capillary_split_end", "no_capillary_write_end",
        ):
            mean, sd = mean_sd(item[field] for item in block)
            entry[field] = {"mean": mean, "graph_sd": sd, "sem": sd / math.sqrt(len(block)) if len(block) > 1 else float("nan")}
        for label, field in (
            ("retained_bits_per_rotor", "write_end"),
            ("connected_bits_per_rotor", "connected_write_end"),
        ):
            bits = [retained_bits(max(item[field], 0.0)) for item in block]
            mean, sd = mean_sd(bits)
            entry[label] = {"mean": mean, "graph_sd": sd, "sem": sd / math.sqrt(len(block)) if len(block) > 1 else float("nan")}
        entry["observation_time"] = block[0]["split_time"]
        # Keep the individual graphs. A wide spread can mean a broad unimodal
        # distribution or a split between graphs that order and graphs that do
        # not; only the individual values distinguish them.
        order = np.argsort(seeds[amplitude])
        entry["per_graph"] = {
            "graph_seed": [seeds[amplitude][index] for index in order],
            "S_end": [block[index]["S_end"] for index in order],
            "write_end": [block[index]["write_end"] for index in order],
            "connected_write_end": [block[index]["connected_write_end"] for index in order],
        }
        table.append(entry)

    single_graph = [entry["disorder"] for entry in table if entry["graphs"] < 2]

    # An amplitude whose graphs land in qualitatively different released states
    # has no meaningful mean. Flag it rather than averaging across the split.
    for entry in table:
        values = sorted(entry["per_graph"]["S_end"])
        span = values[-1] - values[0]
        gaps = [values[index + 1] - values[index] for index in range(len(values) - 1)]
        largest_gap = max(gaps) if gaps else 0.0
        entry["released_state_split"] = bool(span > 0.25 and largest_gap > 0.5 * span)
    split_amplitudes = [entry["disorder"] for entry in table if entry["released_state_split"]]
    comparable = [entry for entry in table if not entry["released_state_split"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "tail_fraction": args.tail_fraction,
        "amplitudes": amplitudes,
        "table": table,
        "single_graph_amplitudes": single_graph,
        "released_state_split_amplitudes": split_amplitudes,
        "graph_error_bars_available": not single_graph,
    }
    if len(comparable) >= 2:
        low, high = comparable[0], comparable[-1]
        report["endpoints"] = {
            "low_disorder": low["disorder"],
            "high_disorder": high["disorder"],
            "S_ratio": low["S_end"]["mean"] / max(high["S_end"]["mean"], 1e-12),
            "split_ratio": low["split_end"]["mean"] / max(high["split_end"]["mean"], 1e-12),
            "write_ratio": low["write_end"]["mean"] / max(high["write_end"]["mean"], 1e-12),
            # The claim is about the director-free component, so this ratio is
            # reported high-over-low: it should exceed one if disorder builds
            # memory rather than merely destroying order.
            "connected_write_gain": (
                high["connected_write_end"]["mean"] / max(low["connected_write_end"]["mean"], 1e-12)
            ),
            "connected_bits_gain": (
                high["connected_bits_per_rotor"]["mean"] / max(low["connected_bits_per_rotor"]["mean"], 1e-12)
            ),
        }
    (args.output_dir / "disorder_protocol_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    def cell(entry: dict[str, Any], field: str) -> str:
        block = entry[field]
        if math.isnan(block["graph_sd"]):
            return f"{block['mean']:.4f}"
        return f"{block['mean']:.4f}+-{block['graph_sd']:.4f}"

    header = (
        f"{'sigma/a':>8} {'n':>2} {'S_end':>16} {'Q_write':>16} {'Q-S^2':>16} "
        f"{'Qs-S^2':>16} {'bits conn':>16}"
    )
    print(header)
    print("-" * len(header))
    for entry in table:
        print(
            f"{entry['disorder']:>8g} {entry['graphs']:>2} "
            f"{cell(entry, 'S_end'):>16} {cell(entry, 'write_end'):>16} "
            f"{cell(entry, 'connected_write_end'):>16} "
            f"{cell(entry, 'connected_split_end'):>16} "
            f"{cell(entry, 'connected_bits_per_rotor'):>16}"
        )
    print()
    # Differences are only worth reading against the graph-to-graph spread,
    # which is largest near the order-disorder crossover.
    if split_amplitudes:
        print(
            "  sigma/a = "
            + ", ".join(f"{value:g}" for value in split_amplitudes)
            + ": the graphs land in qualitatively different released states, so the"
        )
        print("  mean is not a description of any of them; excluded from the comparison below.")
    if len(comparable) >= 2:
        field = "connected_write_end"
        peak = max(comparable, key=lambda item: item[field]["mean"])
        print(
            f"  connected written overlap peaks at sigma/a = {peak['disorder']:g}, "
            f"{peak[field]['mean']:.4f} +- {standard_error(peak[field], peak['graphs']):.4f} (SEM)"
        )
        # Compare every pair, not just the endpoints: a series with an interior
        # maximum has no informative endpoint difference.
        for low, high in combinations(comparable, 2):
            difference = high[field]["mean"] - low[field]["mean"]
            error = math.hypot(
                standard_error(low[field], low["graphs"]),
                standard_error(high[field], high["graphs"]),
            )
            if math.isnan(error) or error == 0.0:
                continue
            ratio = difference / error
            mark = "   SIGNIFICANT" if abs(ratio) > 2.5 else ""
            print(
                f"    {low['disorder']:g} vs {high['disorder']:g}: {difference:+.4f} "
                f"+- {error:.4f}  t={ratio:+.2f}{mark}"
            )
    print()
    # A trend in sigma could simply track the residual director. Compare only
    # graphs whose director is already suppressed.
    matched = [
        (entry["disorder"], [
            value
            for value, order in zip(entry["per_graph"]["connected_write_end"], entry["per_graph"]["S_end"])
            if order < args.matched_s
        ])
        for entry in table
    ]
    matched = [(amplitude, values) for amplitude, values in matched if values]
    if len(matched) >= 2:
        print(f"  Restricted to graphs with S < {args.matched_s:g} (rules out a trend that merely tracks S):")
        for amplitude, values in matched:
            print(
                f"    sigma/a = {amplitude:<5g} n={len(values)}  "
                f"connected written overlap {np.mean(values):.4f}"
            )
    print()
    print(
        "Q-S^2 subtracts the overlap two replicas share through a common director; "
        "it is the hidden component."
    )
    print(
        f"Values average the last {args.tail_fraction:.0%} of each trajectory; "
        "a single final sample is one configuration at one time."
    )
    print()
    print("Per graph (a wide spread may be one broad distribution or a split between states):")
    print(f"  {'sigma/a':>8} {'seed':>6} {'S_end':>9} {'Q_write':>9} {'Q-S^2':>9}")
    for entry in table:
        block = entry["per_graph"]
        for index, seed in enumerate(block["graph_seed"]):
            print(
                f"  {entry['disorder']:>8g} {seed:>6} {block['S_end'][index]:>9.4f} "
                f"{block['write_end'][index]:>9.4f} {block['connected_write_end'][index]:>9.4f}"
            )
    sizes = sorted({entry["node_count"] for entry in table})
    if len(sizes) > 1:
        print(f"WARNING: amplitudes span rotor counts {sizes}; S is size dependent, so the")
        print("         amplitudes are not directly comparable. Rerun at one size.")
    print()
    if single_graph:
        print(
            "WARNING: one quenched graph only at sigma/a = "
            + ", ".join(f"{value:g}" for value in single_graph)
            + "; no graph-to-graph error bars. Rerun once per seed into separate "
            "output directories."
        )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "observation_time_D_r_t": table[0]["observation_time"],
        "endpoints": report.get("endpoints"),
        "graph_error_bars_available": report["graph_error_bars_available"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
