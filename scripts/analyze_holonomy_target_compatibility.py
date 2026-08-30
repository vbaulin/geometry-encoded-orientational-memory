#!/usr/bin/env python3
"""Test whether loop holonomy redistributes retention by target compatibility.

The holonomy intervention gives a split result: flattening the loop structure
reliably removes metastable states, but its effect on retention of a written
pattern changes sign between targets. The target-resolved table shows the
pattern - the uniform director loses when holonomy is present, incompatible
random domain patterns gain - which suggests the consequence is not "holonomy
helps memory" but "holonomy moves stability from targets compatible with the
flattened structure to targets incompatible with it".

This script tests that as a quantitative law. For every archived record it
reconstructs the same graph, intervention and target catalogue the driver
used, evaluates how much better each written target is satisfied by the flat
frame than by the original one, and regresses the measured retention
difference on that predictor.

WHAT THIS CAN AND CANNOT SHOW. The predictor is the target's energy advantage,
E_original(theta*) - E_flat(theta*), normalized by the bond-frame weight. A
negative slope therefore says that a pattern is retained better by the
Hamiltonian in which that same pattern has lower energy. That is ordinary
energetic target selection. It is consistent with loop holonomy controlling
memory, but it does not isolate it, and no amount of resampling changes that.

Deciding the mechanism needs energy-matched controls: prepare the identical
angular configuration in both networks, remove the writing field, use paired
Brownian histories, and compare normalized survival while holding the target
energy and local torque fixed and changing only cycle frustration. If the
networks still differ, holonomy controls memory; if the difference vanishes,
this analysis was measuring energetic selection.

Results are reported per coupling. Pooling couplings can create a threshold
that exists in neither, so a pooled sign claim is withheld when the strata
disagree.

    python -B scripts/analyze_holonomy_target_compatibility.py \
      --archive discoveries/theory_experiment_interface/rotating_colloids_hyperion/continuous_holonomy_memory/continuous_holonomy_memory.json \
      --output-dir discoveries/theory_experiment_interface/rotating_colloids_hyperion/holonomy_target_compatibility
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_rotating_colloids_pair_domain_reduction import block_labels  # noqa: E402
from rotating_colloids_hyperion_case import make_graph  # noqa: E402
from discovery.continuous_colloid_holonomy import (  # noqa: E402
    build_frame_intervention,
    pair_energy,
)
from test_continuous_colloid_holonomy_memory import target_catalog  # noqa: E402


def compatibility(
    angles: np.ndarray,
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    original_phi: np.ndarray,
    flat_phi: np.ndarray,
    weights: np.ndarray,
    j_align: float,
    g_capillary: float,
) -> float:
    """How much better the flat frame satisfies this target than the original.

    Positive means the target is more compatible with the flattened loop
    structure. Energies are differences of the same functional on the same
    state, so the alignment term cancels and only the bond-frame phases differ.
    The scale is the total bond-frame weight, making the value dimensionless
    and comparable across graphs and couplings.
    """

    common = dict(src=src, tgt=tgt, weights=weights, j_align=j_align, g_capillary=g_capillary)
    energy_original = float(pair_energy(angles, phi=original_phi, **common)[0])
    energy_flat = float(pair_energy(angles, phi=flat_phi, **common)[0])
    scale = float(g_capillary) * float(np.sum(np.abs(weights)))
    if scale <= 0.0:
        return float("nan")
    return (energy_original - energy_flat) / scale


def ordinary_least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def graph_bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Resample whole graphs, since records from one graph are not independent."""

    seeds = sorted({row["graph_seed"] for row in rows})
    by_seed = {value: [row for row in rows if row["graph_seed"] == value] for value in seeds}
    rng = np.random.default_rng(seed)
    slopes, correlations = [], []
    for _ in range(int(draws)):
        picked = rng.choice(seeds, size=len(seeds), replace=True)
        sample = [row for value in picked for row in by_seed[value]]
        x = np.asarray([row["compatibility"] for row in sample], dtype=float)
        y = np.asarray([row["original_minus_flat_auc"] for row in sample], dtype=float)
        if x.size < 3 or np.allclose(x, x[0]):
            continue
        slopes.append(ordinary_least_squares(x, y)[0])
        correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if not slopes:
        return {"slope_95_interval": None, "correlation_95_interval": None, "draws": 0}
    return {
        "slope_95_interval": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "correlation_95_interval": [
            float(np.percentile(correlations, 2.5)),
            float(np.percentile(correlations, 97.5)),
        ],
        "draws": len(slopes),
    }


def within_target_fit(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Regress after removing each target's mean from both variables.

    The pooled fit can be carried entirely by the contrast between a few
    targets. Centring within target removes that contrast and leaves only
    graph-to-graph covariation, so a pooled slope that survives here is a law
    in the predictor rather than an ordering of a handful of targets.
    """

    names = sorted({row["target"] for row in rows})

    def centred(sample: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for name in names:
            subset = [row for row in sample if row["target"] == name]
            if len(subset) < 3:
                continue
            x = np.asarray([row["compatibility"] for row in subset], dtype=float)
            y = np.asarray([row["original_minus_flat_auc"] for row in subset], dtype=float)
            xs.append(x - x.mean())
            ys.append(y - y.mean())
        if not xs:
            return np.asarray([]), np.asarray([])
        return np.concatenate(xs), np.concatenate(ys)

    x, y = centred(rows)
    if x.size < 3 or np.allclose(x, x[0]):
        return {"resolved": False, "reason": "insufficient within-target variation"}
    slope, _ = ordinary_least_squares(x, y)
    pearson = float(np.corrcoef(x, y)[0, 1])

    seeds = sorted({row["graph_seed"] for row in rows})
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(int(draws)):
        picked = rng.choice(seeds, size=len(seeds), replace=True)
        sample = [row for value in picked for row in rows if row["graph_seed"] == value]
        bx, by = centred(sample)
        if bx.size < 3 or np.allclose(bx, bx[0]):
            continue
        slopes.append(ordinary_least_squares(bx, by)[0])
    if not slopes:
        return {"slope": slope, "pearson": pearson, "resolved": False}
    interval = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    return {
        "slope": slope,
        "pearson": pearson,
        "slope_95_interval": interval,
        "draws": len(slopes),
        "resolved": bool(interval[0] < 0.0 and interval[1] < 0.0),
    }


def aggregate_couplings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average the duplicated J conditions within each graph and target.

    The J values are distinct couplings, not replicates, so treating them as
    independent rows doubles the apparent sample without adding graphs or
    targets.
    """

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["graph_seed"], row["target"]), []).append(row)
    out = []
    for (graph_seed, target), block in sorted(grouped.items()):
        out.append({
            "graph_seed": graph_seed,
            "target": target,
            "selection": block[0]["selection"],
            "couplings_averaged": len(block),
            "compatibility": float(np.mean([r["compatibility"] for r in block])),
            "original_minus_flat_auc": float(np.mean([r["original_minus_flat_auc"] for r in block])),
        })
    return out


def hierarchical_bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Resample graphs and target patterns, not graphs alone.

    Resampling graphs only gives an interval conditional on this fixed target
    catalogue. A claim about a compatibility law over possible targets requires
    the targets to be resampled too.
    """

    seeds = sorted({row["graph_seed"] for row in rows})
    names = sorted({row["target"] for row in rows})
    index: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["graph_seed"], row["target"]), []).append(row)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(int(draws)):
        picked_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        picked_names = rng.choice(names, size=len(names), replace=True)
        sample = [row
                  for s in picked_seeds
                  for n in picked_names
                  for row in index.get((s, n), [])]
        if len(sample) < 4:
            continue
        x = np.asarray([r["compatibility"] for r in sample], dtype=float)
        y = np.asarray([r["original_minus_flat_auc"] for r in sample], dtype=float)
        if np.allclose(x, x[0]):
            continue
        slopes.append(ordinary_least_squares(x, y)[0])
    if not slopes:
        return {"draws": 0, "resolved": False}
    interval = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    return {
        "slope_95_interval": interval,
        "draws": len(slopes),
        "resolved": bool(interval[0] < 0.0 and interval[1] < 0.0),
    }


def permutation_test(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Shuffle which target received which outcome, within graph and coupling.

    This preserves each cell's outcome distribution and destroys only the
    association with compatibility.
    """

    cells: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["graph_seed"], row.get("j_align"), row.get("epsilon"))
        cells.setdefault(key, []).append(row)
    x = np.asarray([r["compatibility"] for r in rows], dtype=float)
    y = np.asarray([r["original_minus_flat_auc"] for r in rows], dtype=float)
    observed = ordinary_least_squares(x, y)[0]
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(draws)):
        shuffled = []
        for block in cells.values():
            outcomes = [r["original_minus_flat_auc"] for r in block]
            rng.shuffle(outcomes)
            shuffled.extend(
                {"compatibility": r["compatibility"], "original_minus_flat_auc": value}
                for r, value in zip(block, outcomes)
            )
        bx = np.asarray([r["compatibility"] for r in shuffled], dtype=float)
        by = np.asarray([r["original_minus_flat_auc"] for r in shuffled], dtype=float)
        if np.allclose(bx, bx[0]):
            continue
        null.append(ordinary_least_squares(bx, by)[0])
    if not null:
        return {"draws": 0}
    null_array = np.asarray(null, dtype=float)
    p_value = float((np.sum(null_array <= observed) + 1) / (null_array.size + 1))
    return {
        "observed_slope": observed,
        "one_sided_p": p_value,
        "null_slope_95_interval": [
            float(np.percentile(null_array, 2.5)), float(np.percentile(null_array, 97.5))
        ],
        "draws": int(null_array.size),
    }


def monotonicity(per_target: dict[str, Any]) -> dict[str, Any]:
    """Report whether target means actually decrease with compatibility."""

    ordered = sorted(per_target.items(), key=lambda item: item[1]["mean_compatibility"])
    violations = []
    for (left_name, left), (right_name, right) in zip(ordered, ordered[1:]):
        if right["mean_original_minus_flat_auc"] > left["mean_original_minus_flat_auc"]:
            violations.append({
                "increasing_pair": [left_name, right_name],
                "auc": [left["mean_original_minus_flat_auc"], right["mean_original_minus_flat_auc"]],
            })
    return {
        "order_by_compatibility": [name for name, _ in ordered],
        "violations": violations,
        "monotone_decreasing": not violations,
    }


def target_family(name: str) -> str:
    """Group correlated targets. Flip shells are one family, not many systems."""

    import re

    return re.sub(r"_\d+$", "", name)


def per_coupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Slope, mean response and sign sequence within each coupling.

    Averaging couplings can produce a clean ordering that exists in neither,
    so every pooled statement has to be checked against the strata.
    """

    out: dict[str, Any] = {}
    for coupling in sorted({row["j_align"] for row in rows}):
        subset = [row for row in rows if row["j_align"] == coupling]
        means: dict[str, dict[str, float]] = {}
        for name in sorted({row["target"] for row in subset}):
            block = [r for r in subset if r["target"] == name]
            values = np.asarray([r["original_minus_flat_auc"] for r in block], dtype=float)
            means[name] = {
                "mean_compatibility": float(np.mean([r["compatibility"] for r in block])),
                "mean_original_minus_flat_auc": float(values.mean()),
                "sem": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan"),
            }
        ordered = sorted(means.items(), key=lambda item: item[1]["mean_compatibility"])
        x = np.asarray([r["compatibility"] for r in subset], dtype=float)
        y = np.asarray([r["original_minus_flat_auc"] for r in subset], dtype=float)
        slope = ordinary_least_squares(x, y)[0] if not np.allclose(x, x[0]) else float("nan")
        out[f"{coupling:g}"] = {
            "records": len(subset),
            "slope": slope,
            "mean_response": float(y.mean()),
            "sign_sequence": "".join(
                "+" if block["mean_original_minus_flat_auc"] > 0 else "-" for _, block in ordered
            ),
            "order_by_compatibility": [name for name, _ in ordered],
            "per_target": means,
        }
    sequences = {key: value["sign_sequence"] for key, value in out.items()}
    responses = {key: value["mean_response"] for key, value in out.items()}
    reversing = []
    if len(out) > 1:
        names = sorted({name for value in out.values() for name in value["per_target"]})
        for name in names:
            signs = {
                np.sign(value["per_target"][name]["mean_original_minus_flat_auc"])
                for value in out.values() if name in value["per_target"]
            }
            if len(signs) > 1:
                reversing.append(name)
    return {
        "by_coupling": out,
        "sign_sequences": sequences,
        "mean_response_by_coupling": responses,
        "targets_reversing_sign": sorted(reversing),
        "strata_agree": len(set(sequences.values())) == 1,
        "baseline_reverses": len({np.sign(v) for v in responses.values()}) > 1 if responses else False,
    }


def family_bootstrap(rows: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Resample target families rather than targets.

    Seven flip shells drawn from one construction are not seven independent
    physical systems, so resampling them individually understates the spread.
    """

    seeds = sorted({row["graph_seed"] for row in rows})
    families = sorted({target_family(row["target"]) for row in rows})
    index: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["graph_seed"], target_family(row["target"])), []).append(row)
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(int(draws)):
        picked_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        picked_families = rng.choice(families, size=len(families), replace=True)
        sample = [row for s in picked_seeds for f in picked_families for row in index.get((s, f), [])]
        if len(sample) < 4:
            continue
        x = np.asarray([r["compatibility"] for r in sample], dtype=float)
        y = np.asarray([r["original_minus_flat_auc"] for r in sample], dtype=float)
        if np.allclose(x, x[0]):
            continue
        slopes.append(ordinary_least_squares(x, y)[0])
    if not slopes:
        return {"draws": 0, "resolved": False}
    interval = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    return {
        "families": families,
        "slope_95_interval": interval,
        "draws": len(slopes),
        "resolved": bool(interval[0] < 0.0 and interval[1] < 0.0),
    }


def sign_separation(per_target: dict[str, Any], resolved_only: bool = False) -> dict[str, Any]:
    """Can one compatibility threshold separate the sign of the response?

    This is rank based, so it needs neither a linear fit nor monotone
    magnitudes. It is the sharpest distribution-free form of the hypothesis:
    targets more compatible with the flattened structure lose retention,
    targets less compatible gain it.
    """

    from math import comb

    ordered = sorted(per_target.items(), key=lambda item: item[1]["mean_compatibility"])
    if resolved_only:
        ordered = [
            item for item in ordered
            if not math.isnan(item[1].get("sem", float("nan")))
            and abs(item[1]["mean_original_minus_flat_auc"]) > item[1]["sem"]
        ]
    if len(ordered) < 3:
        return {"perfect_separation": False, "targets": len(ordered),
                "reason": "too few targets with a sign resolved from zero"}
    signs = [1 if block["mean_original_minus_flat_auc"] > 0 else -1 for _, block in ordered]
    positives = sum(1 for value in signs if value > 0)
    perfect = all(value > 0 for value in signs[:positives]) and all(
        value < 0 for value in signs[positives:]
    )
    total = len(signs)
    negatives = total - positives
    probability = (1.0 / comb(total, negatives)) if 0 < negatives < total else None
    boundary = None
    if perfect and 0 < positives < total:
        boundary = {
            "below": ordered[positives - 1][0],
            "below_compatibility": ordered[positives - 1][1]["mean_compatibility"],
            "above": ordered[positives][0],
            "above_compatibility": ordered[positives][1]["mean_compatibility"],
        }
    return {
        "order_by_compatibility": [name for name, _ in ordered],
        "sign_sequence": "".join("+" if value > 0 else "-" for value in signs),
        "perfect_separation": bool(perfect),
        "targets": total,
        "positive_targets": positives,
        "probability_under_random_signs": probability,
        "threshold_between": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260906)
    parser.add_argument("--crosslink-k", type=int)
    parser.add_argument("--crosslink-weight", type=float)
    parser.add_argument("--domain-angle-step", type=float)
    parser.add_argument("--random-target-count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--selection",
        default="prespecified",
        choices=("prespecified", "all"),
        help="Landscape-selected targets are chosen using the very landscape under test.",
    )
    args = parser.parse_args()

    archive = json.loads(args.archive.read_text(encoding="utf-8"))
    parameters = dict(archive["parameters"])
    records = archive["records"]

    # The archive does not record every construction parameter. Fall back to the
    # driver's defaults, and report what was assumed so a mismatch is visible
    # rather than silent.
    defaults = {
        "crosslink_k": 2,
        "crosslink_weight": 0.18,
        "domain_angle_step": np.pi / 4.0,
        "random_target_count": 1,
        "seed": 20260819,
    }
    assumed = {}
    for key, value in defaults.items():
        override = getattr(args, key, None)
        if override is not None:
            parameters[key] = override
        elif key not in parameters:
            parameters[key] = value
            assumed[key] = value

    labels, axes, blocks_per_side = block_labels(parameters["n"], parameters["cluster_size"])
    domain_count = blocks_per_side**2

    rows: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    for offset in range(int(parameters["seed_count"])):
        graph_seed = int(parameters["graph_seed"]) + offset
        _, src, tgt, original_phi, weights, _ = make_graph(
            parameters["n"],
            graph_mode="mosaic",
            graph_seed=graph_seed,
            cluster_size=parameters["cluster_size"],
            crosslink_k=parameters["crosslink_k"],
            crosslink_weight=parameters["crosslink_weight"],
            patch_angle_step=parameters["domain_angle_step"],
        )
        for j_align in parameters["j_values"]:
            for epsilon in parameters["epsilon_values"]:
                g_capillary = float(j_align) * float(epsilon)
                intervention = build_frame_intervention(
                    src=src, tgt=tgt, phi=original_phi, weights=weights,
                    labels=labels, axes=axes, j_align=float(j_align),
                    g_capillary=g_capillary, domain_count=domain_count,
                )
                catalog = target_catalog(
                    axes=axes, labels=labels, blocks_per_side=blocks_per_side,
                    original_couplings=intervention.original_couplings,
                    flat_couplings=intervention.realized_couplings,
                    random_target_count=int(parameters["random_target_count"]),
                    seed=int(parameters["seed"]) + 10007 * offset,
                    flip_shell_sizes=parameters.get("flip_shell_sizes", []),
                )
                for entry in catalog:
                    if args.selection == "prespecified" and entry["selection"] != "prespecified":
                        continue
                    measured = [
                        row for row in records
                        if row["graph_seed"] == graph_seed
                        and row["target"] == entry["name"]
                        and np.isclose(row["j_align"], float(j_align))
                        and np.isclose(row["epsilon"], float(epsilon))
                    ]
                    if not measured:
                        unmatched.append({"graph_seed": graph_seed, "target": entry["name"],
                                          "j_align": float(j_align), "epsilon": float(epsilon)})
                        continue
                    value = compatibility(
                        entry["angles"], src=src, tgt=tgt,
                        original_phi=original_phi, flat_phi=intervention.flat_phi,
                        weights=weights, j_align=float(j_align), g_capillary=g_capillary,
                    )
                    for row in measured:
                        rows.append({
                            "graph_seed": graph_seed,
                            "target": entry["name"],
                            "selection": entry["selection"],
                            "j_align": float(j_align),
                            "epsilon": float(epsilon),
                            "compatibility": value,
                            "original_minus_flat_auc": float(row["original_minus_flat_auc"]),
                        })

    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} joined records; cannot regress")

    archived = {row["target"] for row in records}
    reconstructed = {row["target"] for row in rows}
    dropped = sorted(archived - reconstructed - {"original_ground_state", "holonomy_created_stable_state"})
    if dropped:
        raise SystemExit(
            "the reconstructed catalogue does not contain these archived targets, so their "
            f"records would be silently ignored: {dropped}\n"
            "The catalogue is rebuilt from the archived parameters; a target family present "
            "in the archive but absent here means a construction parameter is missing or "
            "not passed through."
        )

    x = np.asarray([row["compatibility"] for row in rows], dtype=float)
    y = np.asarray([row["original_minus_flat_auc"] for row in rows], dtype=float)
    slope, intercept = ordinary_least_squares(x, y)
    pearson = float(np.corrcoef(x, y)[0, 1])
    order_x = np.argsort(np.argsort(x)).astype(float)
    order_y = np.argsort(np.argsort(y)).astype(float)
    spearman = float(np.corrcoef(order_x, order_y)[0, 1])
    predicted_sign = np.sign(-x)
    observed_sign = np.sign(y)
    agreement = float(np.mean(predicted_sign[predicted_sign != 0] == observed_sign[predicted_sign != 0]))
    bootstrap = graph_bootstrap(rows, args.bootstrap_draws, args.bootstrap_seed)

    interval = bootstrap["slope_95_interval"]
    resolved = bool(interval is not None and interval[0] < 0.0 and interval[1] < 0.0)
    within = within_target_fit(rows, args.bootstrap_draws, args.bootstrap_seed + 1)
    distinct_targets = len({row["target"] for row in rows})
    aggregated = aggregate_couplings(rows)
    agg_x = np.asarray([r["compatibility"] for r in aggregated], dtype=float)
    agg_y = np.asarray([r["original_minus_flat_auc"] for r in aggregated], dtype=float)
    agg_slope, _ = ordinary_least_squares(agg_x, agg_y)
    hierarchical = hierarchical_bootstrap(aggregated, args.bootstrap_draws, args.bootstrap_seed + 2)
    permutation = permutation_test(rows, args.bootstrap_draws, args.bootstrap_seed + 3)

    per_target: dict[str, Any] = {}
    for name in sorted({row["target"] for row in rows}):
        subset = [row for row in rows if row["target"] == name]
        values = np.asarray([r["original_minus_flat_auc"] for r in subset], dtype=float)
        per_target[name] = {
            "records": len(subset),
            "family": target_family(name),
            "mean_compatibility": float(np.mean([r["compatibility"] for r in subset])),
            "mean_original_minus_flat_auc": float(values.mean()),
            "sem": float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else float("nan"),
        }

    separation = sign_separation(per_target)
    separation_resolved = sign_separation(per_target, resolved_only=True)
    strata = per_coupling(rows)
    families = family_bootstrap(rows, args.bootstrap_draws, args.bootstrap_seed + 4)

    report = {
        "hypothesis": (
            "Loop holonomy moves retention from targets compatible with the flattened "
            "structure to targets incompatible with it, so the measured retention "
            "difference falls with target-flat compatibility."
        ),
        "selection_filter": args.selection,
        "records": len(rows),
        "graphs": len(sorted({row["graph_seed"] for row in rows})),
        "slope": slope,
        "intercept": intercept,
        "pearson": pearson,
        "spearman": spearman,
        "sign_rule_agreement": agreement,
        "graph_bootstrap": bootstrap,
        "negative_slope_resolved": resolved,
        "within_target": within,
        "coupling_aggregated": {
            "records": len(aggregated),
            "slope": agg_slope,
            "note": "J conditions averaged within graph and target; they are couplings, not replicates.",
        },
        "hierarchical_bootstrap_graphs_and_targets": hierarchical,
        "permutation_test": permutation,
        "monotonicity": monotonicity(per_target),
        "sign_separation": separation,
        "sign_separation_resolved_targets_only": separation_resolved,
        "per_coupling": strata,
        "family_bootstrap": families,
        "predictor_is_energy_advantage": (
            "compatibility = [E_original(target) - E_flat(target)] / (g * sum|w|); a negative "
            "slope is consistent with energetic target selection and does not isolate holonomy."
        ),
        "generalizes_over_targets": bool(hierarchical.get("resolved")),
        "within_target_covariate": bool(within.get("resolved")),
        "distinct_targets": distinct_targets,
        "pooled_fit_is_between_target_contrast": bool(resolved and not within.get("resolved")),
        "per_target": per_target,
        "assumed_parameters_absent_from_archive": assumed,
        "unmatched_catalog_entries": unmatched,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "holonomy_target_compatibility.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{'target':<32} {'n':>3} {'flat compatibility':>19} {'orig-flat AUC':>14}")
    print("-" * 72)
    for name, block in sorted(per_target.items(), key=lambda item: item[1]["mean_compatibility"]):
        print(f"{name:<32} {block['records']:>3} {block['mean_compatibility']:>19.5f} "
              f"{block['mean_original_minus_flat_auc']:>14.5f}")
    print()
    print(f"slope {slope:+.5f}   pearson {pearson:+.3f}   spearman {spearman:+.3f}   "
          f"sign agreement {agreement:.1%}")
    if interval is not None:
        print(f"graph-bootstrap 95% slope interval [{interval[0]:+.5f}, {interval[1]:+.5f}]"
              f"   {'RESOLVED negative' if resolved else 'not resolved'}")
    mono = monotonicity(per_target)
    if not mono["monotone_decreasing"]:
        print("target means are NOT monotone in compatibility; increasing pairs:")
        for item in mono["violations"]:
            print(f"    {item['increasing_pair'][0]} -> {item['increasing_pair'][1]}: "
                  f"{item['auc'][0]:+.5f} -> {item['auc'][1]:+.5f}")
    hb = hierarchical.get("slope_95_interval")
    if hb:
        print(f"hierarchical bootstrap (graphs x targets) 95% [{hb[0]:+.5f}, {hb[1]:+.5f}]"
              f"   {'resolved' if hierarchical['resolved'] else 'NOT resolved'}")
    if permutation.get("draws"):
        print(f"permutation test (targets shuffled within graph and coupling): "
              f"one-sided p = {permutation['one_sided_p']:.4f}")
    print(f"J-aggregated slope {agg_slope:+.5f} over {len(aggregated)} graph-target cells")
    wi = within.get("slope_95_interval")
    print(f"within-target slope {within.get('slope', float('nan')):+.5f}   "
          f"pearson {within.get('pearson', float('nan')):+.3f}"
          + (f"   95% [{wi[0]:+.5f}, {wi[1]:+.5f}]" if wi else ""))
    fb = families.get("slope_95_interval")
    if fb:
        print(f"family bootstrap (graphs x target families) 95% [{fb[0]:+.5f}, {fb[1]:+.5f}]"
              f"   {'resolved' if families['resolved'] else 'NOT resolved'}")

    print()
    print("PER COUPLING (pooling couplings can create an ordering present in neither)")
    for key, block in strata["by_coupling"].items():
        print(f"  J={key:<5} slope {block['slope']:+.5f}   mean response "
              f"{block['mean_response']:+.6f}   signs {block['sign_sequence']}")
    if strata["baseline_reverses"]:
        print("  The mean response REVERSES SIGN between couplings: the flattened network")
        print("  retains more at one coupling and less at the other.")
    if strata["targets_reversing_sign"]:
        print(f"  {len(strata['targets_reversing_sign'])} of {distinct_targets} targets reverse "
              f"sign between couplings.")

    print()
    if separation["perfect_separation"] and strata["strata_agree"]:
        edge = separation["threshold_between"]
        print(f"sign separation: {separation['sign_sequence']}  one threshold splits all "
              f"{separation['targets']} targets")
        print(f"    between {edge['below']} ({edge['below_compatibility']:+.5f}) and "
              f"{edge['above']} ({edge['above_compatibility']:+.5f}); "
              f"p = {separation['probability_under_random_signs']:.5f}")
    elif separation["perfect_separation"]:
        print("sign separation on coupling-averaged means is WITHHELD: the per-coupling sign")
        print("sequences disagree, so a pooled threshold is an averaging artifact.")
    if separation_resolved.get("perfect_separation"):
        print(f"  restricted to targets whose mean exceeds its standard error: "
              f"{separation_resolved['sign_sequence']}, "
              f"p = {separation_resolved['probability_under_random_signs']:.5f}")
    elif "reason" in separation_resolved:
        print(f"  restricted to targets resolved from zero: {separation_resolved['reason']}")

    print()
    print("VERDICT")
    print("  The predictor is the target's energy advantage in the flattened network, so a")
    print("  negative slope is consistent with ordinary energetic target selection. This")
    print("  analysis cannot separate that from a topological memory mechanism.")
    if strata["baseline_reverses"] or not strata["strata_agree"]:
        print("  Coupling dependence dominates: the sign of the frustrated-minus-flat response")
        print("  changes with J, so model the response as a(J) + b(J) * dE rather than as one")
        print("  threshold in normalized compatibility.")
    if hierarchical.get("resolved") and families.get("resolved"):
        print("  The slope survives resampling graphs, targets and target families, so target")
        print("  energy does organize which patterns are preferred within each coupling.")
    elif hierarchical.get("resolved"):
        print("  The slope survives resampling targets but NOT target families; correlated")
        print("  members of one construction are not independent systems.")
    if not within.get("resolved"):
        print("  Not established within target: compatibility is carried by target identity,")
        print("  so it is not demonstrated as a covariate.")
    print("  Decisive test: prepare the identical configuration in both networks, remove the")
    print("  field, use paired noise histories, scan J/D_r densely, and hold target energy and")
    print("  local torque fixed while changing only cycle frustration.")
    if assumed:
        print(f"assumed (absent from archive): {assumed}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} catalogue entries had no archived record")


if __name__ == "__main__":
    main()
