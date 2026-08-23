#!/usr/bin/env python3
"""Audit the frozen quantitative claims in the capillary-rotor PRL package.

The numerical checks use only publication-scale artifacts. Provenance and
language gates are reported separately because a numerical match cannot prove
that every raw file needed to reproduce a derived figure has been archived.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from review_rotating_colloids_capillary_sparse_attention import (
    audit_regime_taxonomy,
    audit_spatial_range,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "discoveries/theory_experiment_interface/rotating_colloids_hyperion"
GPU = DATA / "rotating_colloids_capillary_pair_prl_gpu"
OUT = DATA / "rotating_colloids_capillary_pair_prl_claim_audit"
MAIN_TEX = ROOT / "tex/rotating_colloids/rotating_colloids_prl_capillary.tex"
SUPPLEMENT_TEX = ROOT / "tex/rotating_colloids/rotating_colloids_prl_capillary_supplement.tex"
FIGURES = ROOT / "tex/rotating_colloids/capillary_prl_figures"

DENSE = GPU / "dense_map_n20/capillary_pair_scan.jsonl"
REGIMES = FIGURES / "capillary_regime_report.json"
CONTROLS = GPU / "matched_controls_n32/capillary_pair_scan.jsonl"
INTERNAL = DATA / "rotating_colloids_capillary_pair_prl_internal/capillary_internal_correlations.json"
SPIN = DATA / "rotating_colloids_spin_glass_prl_gpu/analysis/spin_glass_finite_size_report.json"
SPIN_SCAN = DATA / "rotating_colloids_spin_glass_prl_gpu"
ACTIVATED = FIGURES / "activated_memory_figure_report.json"
DISORDER_RETENTION = DATA / "rotating_colloids_disorder_retention_summary.json"
DISORDER_RETENTION_RAW = DATA / "rotating_colloids_disorder_retention_protocols"
HOLONOMY = DATA / "holonomy_memory_intervention/holonomy_memory_intervention_beta1_replication.json"
ORDER_N144 = DATA / "rotating_colloids_operation_order_memory_n12/operation_order_memory.jsonl"
ORDER_N256 = DATA / "rotating_colloids_operation_order_memory_n16/operation_order_memory.jsonl"
ORDER_FRACTION_N144 = DATA / "rotating_colloids_operation_order_memory_fraction_n12/operation_order_memory.jsonl"
RELAXED_EXCHANGE = DATA / "relaxed_exchange_order_minimal.json"
SUBMISSION_VALIDATIONS = DATA / "rotating_colloids_submission_validations"
ORDER_INDEPENDENT = (
    SUBMISSION_VALIDATIONS / "independent_order/operation_order_memory.jsonl"
)
ORDER_INDEPENDENT_REPORT = (
    SUBMISSION_VALIDATIONS
    / "independent_order/independent_noise_order_report.json"
)

# Frozen expectations for the endpoint-overlap separation at lambda = 0.9.
SPLIT_SEPARATION_SIGMA = 40.470438807650005
WRITE_SEPARATION_SIGMA = 28.607053040819057
SIZE_PATHS = {
    n * n: GPU / f"finite_size_n{n}/capillary_pair_scan.jsonl"
    for n in (12, 16, 24, 32, 48)
}
DYNAMICS_PATHS = [
    GPU / f"dynamics_seed_{seed}/capillary_pair_protocols.json"
    for seed in (17, 29, 43)
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unique_rows_by_key(rows_with_sources: Iterable[tuple[dict[str, Any], Path]]) -> tuple[list[dict[str, Any]], int]:
    """Collapse byte-equivalent copies while rejecting conflicting records."""

    by_key: dict[str, str] = {}
    source_by_key: dict[str, Path] = {}
    duplicates = 0
    for row, source in rows_with_sources:
        key = row.get("key")
        if not key:
            raise ValueError(f"activated-memory row without stable key in {source}")
        frozen = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key in by_key:
            if frozen != by_key[key]:
                raise ValueError(f"conflicting rows for {key}: {source_by_key[key]} and {source}")
            duplicates += 1
            continue
        by_key[key] = frozen
        source_by_key[key] = source
    return [json.loads(by_key[key]) for key in sorted(by_key)], duplicates


def sample_mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def sample_mean_sem(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size < 2:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def add_close(
    checks: list[dict[str, Any]],
    label: str,
    actual: float,
    expected: float,
    tolerance: float,
) -> None:
    error = abs(float(actual) - float(expected))
    checks.append(
        {
            "claim": label,
            "actual": float(actual),
            "expected": float(expected),
            "absolute_error": error,
            "tolerance": float(tolerance),
            "passed": bool(error <= tolerance),
        }
    )


def add_exact(checks: list[dict[str, Any]], label: str, actual: Any, expected: Any) -> None:
    checks.append(
        {
            "claim": label,
            "actual": actual,
            "expected": expected,
            "passed": bool(actual == expected),
        }
    )


def audit_independent_order(
    checks: list[dict[str, Any]], report_path: Path
) -> None:
    """Audit independent-noise sequence claims when the cluster report exists."""

    if not report_path.exists():
        add_exact(checks, "independent-noise order report present", False, True)
        return
    report = read_json(report_path)
    add_exact(checks, "independent-noise order report present", True, True)
    add_exact(checks, "independent-noise order rows", report["row_count"], 50)
    add_close(
        checks,
        "independent-noise h=8 contested-partitioned contrast",
        report["highest_field_contested_minus_partitioned"]["mean"],
        0.621257760240627,
        1e-12,
    )
    add_close(
        checks,
        "independent-noise h=8 contrast SEM",
        report["highest_field_contested_minus_partitioned"]["sem"],
        0.019976817317675182,
        1e-12,
    )
    for mode, expected in (
        (
            "contested",
            {
                "readout": (0.6335929224283922, 0.01979755507788101),
                "accuracy": (1.0, 0.0),
                "d_prime": (12.545756466746118, 0.9719137815444875),
            },
        ),
        (
            "partitioned",
            {
                "readout": (0.01233516218776521, 0.0026702140517260073),
                "accuracy": (0.55, 0.012499999999999999),
                "d_prime": (0.30705569373101216, 0.05237423499319883),
            },
        ),
    ):
        summary = report["summaries"][f"8:{mode}"]
        for metric, report_key in (
            ("readout", "terminal_order_readout"),
            ("accuracy", "decode_accuracy_zero_threshold"),
            ("d_prime", "decode_d_prime"),
        ):
            expected_mean, expected_sem = expected[metric]
            add_close(
                checks,
                f"independent-noise h=8 {mode} {metric} mean",
                summary[report_key]["mean"],
                expected_mean,
                1e-12,
            )
            add_close(
                checks,
                f"independent-noise h=8 {mode} {metric} SEM",
                summary[report_key]["sem"],
                expected_sem,
                1e-12,
            )


def first_sample_below(time: Sequence[float], values: Sequence[float], threshold: float) -> float:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    indices = np.flatnonzero(y <= threshold)
    if not indices.size:
        return float("nan")
    return float(t[indices[0]])


def metric(row: dict[str, Any], name: str) -> float:
    if name == "replica_overlap_magnitude":
        return float(row["replica_overlap"]["magnitude_mean"])
    return float(row[name])


def supplemental_figure_map(supplement: Path, letter_text: str) -> dict[str, Any]:
    """Cross-check the Letter's hard-coded S-numbers against the Supplement.

    The two documents compile separately, so the Letter refers to Supplemental
    figures by literal number. Inserting a figure renumbers everything after
    it without any LaTeX warning.
    """

    import re

    source = supplement.read_text(encoding="utf-8")
    labels = re.findall(r"\\label\{(fig:supp[^}]*)\}", source)
    numbering = {f"S{index}": label for index, label in enumerate(labels, start=1)}
    referenced = sorted({int(item) for item in re.findall(r"Figs?\.~S(\d+)", letter_text)})
    referenced += [
        int(item)
        for item in re.findall(r"Figs\.~S\d+ and S(\d+)", letter_text)
        if int(item) not in referenced
    ]
    out_of_range = [number for number in sorted(set(referenced)) if number > len(labels)]

    romans = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    table_labels = re.findall(r"\\label\{(tab:supp[^}]*)\}", source)
    table_numbering = {
        f"S{romans[index]}": label for index, label in enumerate(table_labels) if index < len(romans)
    }
    table_references = sorted(set(re.findall(r"Table~S([IVX]+)", letter_text)))
    unresolved_tables = [item for item in table_references if f"S{item}" not in table_numbering]
    return {
        "supplemental_figure_count": len(labels),
        "numbering": numbering,
        "letter_references": sorted(set(referenced)),
        "out_of_range_references": out_of_range,
        "supplemental_table_count": len(table_labels),
        "table_numbering": table_numbering,
        "letter_table_references": table_references,
        "unresolved_table_references": unresolved_tables,
        "all_references_resolve": not out_of_range and not unresolved_tables,
    }


def angular_localization_bits(resultant: float) -> float:
    """Minimum angular localization relative to uniform, in bits.

    Only the mean resultant R = <cos 2 dtheta> is archived. The maximum-entropy
    density on the circle of the doubled angle at fixed R is von Mises with
    concentration kappa solving I1/I0 = R. Maximizing the entropy minimizes its
    KL divergence from the uniform angular error distribution. This quantity is
    not a channel capacity or a mutual information.
    """

    from scipy.optimize import brentq
    from scipy.special import i0, i1

    if resultant <= 1e-12:
        return 0.0
    kappa = brentq(lambda k: i1(k) / i0(k) - resultant, 1e-12, 700.0)
    return float((kappa * (i1(kappa) / i0(kappa)) - math.log(i0(kappa))) / math.log(2.0))


def paired_aging_increment(dynamics: list[dict[str, Any]], lag: float) -> list[float]:
    """Per-graph change in the two-time correlation between the extreme t_w."""

    increments = []
    for run in dynamics:
        curves = run["aging"]["curves"]
        early = float(np.interp(lag, curves[0]["lag_time"], curves[0]["correlation"]))
        late = float(np.interp(lag, curves[-1]["lag_time"], curves[-1]["correlation"]))
        increments.append(late - early)
    return increments


def overlap_correlation_lengths(root: Path) -> dict[str, float]:
    """Graph-averaged and single-graph extremes of xi_L over the spin-glass ray."""

    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/spin_glass_scan.jsonl")):
        rows.extend(read_jsonl(path))
    cells: dict[tuple[float, int], list[float]] = {}
    for row in rows:
        key = (float(row["lambda"]), int(row["node_count"]))
        cells.setdefault(key, []).append(float(row["overlap"]["xi_L"]))
    averaged = [float(np.mean(values)) for values in cells.values()]
    return {
        "cells": len(cells),
        "graph_averaged_max": max(averaged),
        "graph_averaged_min": min(averaged),
        "single_graph_max": max(float(row["overlap"]["xi_L"]) for row in rows),
    }


def reproduce_activated_report(paths: list[Path], report: dict[str, Any]) -> dict[str, Any]:
    """Recompute the derived Fig. 4 report directly from the raw JSONL shards.

    A numerical match against the frozen derived report is not by itself
    evidence that the deposit can regenerate the figure. This gate closes that
    gap: it rebuilds the reported series from the raw rows and reports the
    largest relative deviation.
    """

    raw_rows = [(row, path) for path in paths for row in read_jsonl(path)]
    rows, duplicate_rows = unique_rows_by_key(raw_rows)
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(float(row["lambda"]), []).append(row)
    lambdas = sorted(groups)
    series = {
        f"{protocol}_{key}": [
            float(
                np.mean(
                    # Endpoint overlap, matching the Fig. 4(b) measure. The
                    # finite-window integral it replaced was bounded above by
                    # T_obs and so saturated at large lambda.
                    [float(row["protocols"][protocol][key]["final"]) for row in groups[lam]]
                )
            )
            for lam in lambdas
        ]
        for protocol in ("physical", "no_capillary")
        for key in ("split_summary", "release_summary")
    }
    worst = 0.0
    mismatched: list[str] = []
    for name, values in series.items():
        report_series = report.get("endpoint_overlap", report.get("integral_times", {}))
        expected = report_series.get(name, {}).get("mean")
        if expected is None or len(expected) != len(values):
            mismatched.append(name)
            worst = float("inf")
            continue
        deviation = np.abs(np.asarray(values) - np.asarray(expected, dtype=float))
        relative = deviation / np.maximum(np.abs(np.asarray(expected, dtype=float)), 1e-12)
        worst = max(worst, float(relative.max()))
    return {
        "raw_rows": len(rows),
        "raw_rows_seen": len(raw_rows),
        "identical_duplicate_rows_ignored": duplicate_rows,
        "raw_lambdas": lambdas,
        "raw_graphs_per_lambda": {f"{lam:g}": len(groups[lam]) for lam in lambdas},
        "row_count_matches_report": len(rows) == int(report["rows"]),
        "lambdas_match_report": lambdas == [float(value) for value in report["lambdas"]],
        "max_relative_deviation": worst,
        "series_absent_from_report": mismatched,
        "raw_reproduces_derived_report": (
            not mismatched
            and len(rows) == int(report["rows"])
            and lambdas == [float(value) for value in report["lambdas"]]
            and worst <= 1e-9
        ),
    }


def validate_disorder_retention_raw(root: Path) -> dict[str, Any]:
    """Verify the graph/size coverage behind the positional-disorder claim."""

    required = {
        (576, 0.05): 3,
        (576, 0.08): 5,
        (576, 0.11): 5,
        (576, 0.16): 5,
        (576, 0.28): 5,
        (1024, 0.11): 5,
        (1024, 0.16): 5,
    }
    cells: dict[tuple[int, float], set[int]] = {}
    files = sorted(root.glob("**/capillary_pair_protocols.json")) if root.exists() else []
    for path in files:
        try:
            graph = read_json(path)["model"]["graph"]
            key = (int(graph["node_count"]), round(float(graph["disorder"]), 8))
            cells.setdefault(key, set()).add(int(graph["seed"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    missing = {
        f"N={node_count},sigma={sigma:g}": {"required": count, "found": len(cells.get((node_count, sigma), set()))}
        for (node_count, sigma), count in required.items()
        if len(cells.get((node_count, sigma), set())) < count
    }
    return {
        "root": str(root),
        "protocol_files": [str(path) for path in files],
        "cells": {
            f"N={node_count},sigma={sigma:g}": sorted(seeds)
            for (node_count, sigma), seeds in sorted(cells.items())
        },
        "missing_cells": missing,
        "complete": not missing,
    }
def power_exponent(grouped: dict[int, list[dict[str, Any]]], name: str) -> float:
    node_counts = np.asarray(sorted(grouped), dtype=float)
    means = np.asarray(
        [np.mean([metric(row, name) for row in grouped[int(count)]]) for count in node_counts],
        dtype=float,
    )
    return float(np.polyfit(np.log(node_counts), np.log(np.maximum(np.abs(means), 1e-12)), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    checks: list[dict[str, Any]] = []

    # 1-20: finite-size regime taxonomy and its sensitivity to sampled graphs.
    dense_rows = read_jsonl(DENSE)
    regime_report = read_json(REGIMES)
    taxonomy = audit_regime_taxonomy(dense_rows, bootstraps=0, rng=np.random.default_rng(20260731))
    hidden = next(item for item in regime_report["regimes"] if item["name"] == "hidden mixed memory")
    add_exact(checks, "dense-map raw rows", len(dense_rows), 1323)
    add_exact(checks, "dense-map parameter cells", regime_report["parameter_cells"], 441)
    add_exact(checks, "dense-map graph realizations per cell", regime_report["graph_counts"], [3])
    add_exact(checks, "observable-space cluster count", regime_report["cluster_count"], 4)
    add_close(checks, "k=4 silhouette", regime_report["silhouette_scores"]["4"], 0.45132861086701526, 1e-10)
    add_close(checks, "k=5 silhouette", regime_report["silhouette_scores"]["5"], 0.44383614032389285, 1e-10)
    add_close(checks, "k=4 minus k=5 silhouette", taxonomy["k4_minus_k5_silhouette"], 0.00749247054312241, 1e-10)
    add_exact(checks, "k=4 maximizes tested silhouette", taxonomy["best_k"], 4)
    add_close(checks, "minimum initialization ARI", regime_report["initialization_ARI_min"], 1.0, 1e-12)
    add_exact(checks, "exact ordered graph-bootstrap replicates", taxonomy["bootstrap_replicates"], 27)
    add_exact(checks, "graph bootstraps selecting k=4", taxonomy["bootstrap_best_k_counts"].get("4", 0), 27)
    add_close(checks, "mean graph-bootstrap k=4 ARI", taxonomy["bootstrap_k4_ARI"]["mean"], 0.9719545926398437, 1e-10)
    leave_one_min = min(item["k4_ARI_against_full"] for item in taxonomy["leave_one_graph_seed_out"].values())
    add_close(checks, "minimum leave-one-graph k=4 ARI", leave_one_min, 0.9549815240499724, 1e-10)
    add_close(
        checks,
        "qEA-window feature correlation",
        taxonomy["feature_correlation"]["q_EA_mean"]["window_autocorrelation"],
        0.9864454839124128,
        1e-10,
    )
    add_exact(checks, "hidden mixed-memory cell count", hidden["cell_count"], 169)
    for name, expected in (
        ("S_mean", 0.11167048986572914),
        ("C2_mean", 0.6377075320675238),
        ("G2_mean", 0.4139681178113454),
        ("q_EA_mean", 0.6277204179214186),
        ("window_autocorrelation", 0.5192665961508657),
    ):
        add_close(checks, f"hidden-regime centroid {name}", hidden["centroid"][name], expected, 1e-10)

    # 21-33: selected state and matched controls at N=1024.
    controls = read_jsonl(CONTROLS)
    physical = [row for row in controls if row["control"] == "physical"]
    regular = [row for row in controls if row["control"] == "regular"]
    shuffled = [row for row in controls if row["control"] == "shuffled_frames"]
    g0_rows = [row for row in read_jsonl(SIZE_PATHS[1024]) if math.isclose(float(row["g_capillary"]), 0.0)]
    add_exact(checks, "selected-state graph realizations", len(physical), 5)
    for name, expected_mean, expected_std in (
        ("S_mean", 0.06651090671126134, 0.0037994814386568053),
        ("C2_mean", 0.6274020148540538, 0.008832466433874473),
        ("G2_mean", 0.44687331852602463, 0.010988593643146045),
        ("q_EA_mean", 0.6581022977828981, 0.006612176975813657),
    ):
        mean, std = sample_mean_std(row[name] for row in physical)
        add_close(checks, f"selected {name} mean", mean, expected_mean, 1e-10)
        add_close(checks, f"selected {name} graph SD", std, expected_std, 1e-10)
    add_close(checks, "g=0 G2 mean", np.mean([row["G2_mean"] for row in g0_rows]), 9.76097570651146e-06, 1e-12)
    add_close(checks, "g=0 qEA mean", np.mean([row["q_EA_mean"] for row in g0_rows]), 0.022473914389653748, 1e-12)
    add_close(checks, "regular-lattice S mean", np.mean([row["S_mean"] for row in regular]), 0.9446932259554665, 1e-12)
    add_close(checks, "shuffled-frame qEA mean", np.mean([row["q_EA_mean"] for row in shuffled]), 0.7481652575234572, 1e-12)

    # 34-45: five-size scaling at J=4, g=5.
    grouped: dict[int, list[dict[str, Any]]] = {}
    for node_count, path in SIZE_PATHS.items():
        grouped[node_count] = [
            row
            for row in read_jsonl(path)
            if row.get("control", "physical") == "physical" and math.isclose(float(row["g_capillary"]), 5.0)
        ]
    add_exact(checks, "finite-size node counts", sorted(grouped), [144, 256, 576, 1024, 2304])
    for node_count, expected in (
        (144, 0.15081520399241163),
        (256, 0.12484551011225595),
        (576, 0.08833536698042886),
        (1024, 0.06653026091137756),
        (2304, 0.039573839721890665),
    ):
        add_close(checks, f"N={node_count} global S", np.mean([row["S_mean"] for row in grouped[node_count]]), expected, 1e-12)
    for name, expected in (
        ("S_mean", -0.47872714735269567),
        ("replica_overlap_magnitude", -0.46563918288493533),
        ("q_EA_mean", 0.011185205858108928),
        ("C2_mean", -0.003849330803576246),
        ("G2_mean", 0.0022603009973430218),
    ):
        add_close(checks, f"finite-size exponent {name}", power_exponent(grouped, name), expected, 1e-12)
    add_close(checks, "N=2304 qEA", np.mean([row["q_EA_mean"] for row in grouped[2304]]), 0.6464701948066554, 1e-12)

    # 46-57: long-time inheritance, write-release, controls, and waiting-time dependence.
    dynamics = [read_json(path) for path in DYNAMICS_PATHS]
    add_exact(checks, "long-dynamics graph realizations", len(dynamics), 3)
    split_end = [float(run["split_replica"]["overlap_mean"][-1]) for run in dynamics]
    mean, std = sample_mean_std(split_end)
    add_close(checks, "split endpoint mean", mean, 0.461863513540163, 1e-12)
    add_close(checks, "split endpoint graph SD", std, 0.010285602535420461, 1e-12)
    split_cross = [
        first_sample_below(run["no_capillary_split_replica"]["time"], run["no_capillary_split_replica"]["overlap_mean"], 0.2)
        for run in dynamics
    ]
    mean, std = sample_mean_std(split_cross)
    add_close(checks, "g=0 split first-sample crossing mean", mean, 2.6666666666666665, 1e-12)
    add_close(checks, "g=0 split first-sample crossing SD", std, 0.2886751345948129, 1e-12)
    write_end = [float(run["write_release"]["release_overlap"][-1]) for run in dynamics]
    mean, std = sample_mean_std(write_end)
    add_close(checks, "write-release endpoint mean", mean, 0.4405082199085002, 1e-12)
    add_close(checks, "write-release endpoint graph SD", std, 0.020567678837819712, 1e-12)
    write_cross = []
    for run in dynamics:
        block = run["no_capillary_write_release"]
        release_time = np.asarray(block["release_time"], dtype=float)
        write_cross.append(first_sample_below(release_time - release_time[0], block["release_overlap"], 0.2))
    mean, std = sample_mean_std(write_cross)
    add_close(checks, "g=0 write first-sample crossing mean", mean, 4.333333333333333, 1e-12)
    add_close(checks, "g=0 write first-sample crossing SD", std, 0.28867513459481287, 1e-12)
    for index, expected in enumerate((0.675956353756127, 0.6949997695954956, 0.7012603658741137)):
        values = []
        for run in dynamics:
            curve = run["aging"]["curves"][index]
            values.append(float(np.interp(30.0, curve["lag_time"], curve["correlation"])))
        add_close(checks, f"waiting-time curve {index + 1} at lag 30", np.mean(values), expected, 1e-12)

    # 58-62: real-space range of the local correlations.
    spatial = audit_spatial_range(INTERNAL)
    aggregate = read_json(INTERNAL)["aggregate"]
    add_exact(checks, "spatial-correlation graph realizations", spatial["graph_count"], 3)
    add_exact(checks, "spatial samples per graph", aggregate["sample_count_per_graph"], [960, 960, 960])
    add_close(checks, "first-shell connected C2", spatial["first_shell_relative_connected"], 0.5569559272775115, 1e-12)
    add_close(checks, "first-shell bond-frame G2", spatial["first_shell_bond_frame"], 0.5045874679449952, 1e-12)
    add_close(checks, "operational correlation range r/a", spatial["estimated_correlation_range_r_over_a"], math.sqrt(3.0), 1e-12)

    # 63-66: independent equilibrium-replica discriminant.
    spin = read_json(SPIN)
    add_exact(checks, "spin-glass scan rows", spin["rows"], 200)
    add_exact(checks, "spin-glass node counts", spin["node_counts"], [144, 256, 576, 1024, 2304])
    add_exact(checks, "spin-glass disorder realizations per point", spin["disorder_realizations_per_point"], 5)
    add_exact(checks, "spin-glass finding", spin["finding"], "no_equilibrium_spin_glass_crossing_on_scanned_ray")

    # 67-70: coupling-dependent endpoint retained overlap.
    activated = read_json(ACTIVATED)
    add_exact(checks, "activated-memory summary rows", activated["rows"], 40)
    add_exact(checks, "activated-memory graphs per coupling", sorted(set(activated["graphs_per_lambda"].values())), [5])
    lambda_index = activated["lambdas"].index(0.9)
    values = activated.get("endpoint_overlap", activated.get("integral_times", {}))

    # Retention is asserted as a SEPARATION from the g = 0 control, not as a
    # ratio.  The control endpoint is statistically zero at every lambda, so a
    # ratio against it is unstable: recomputing the old ratio checks on the
    # endpoint overlap returns -169 and -55, sign-flipped by a denominator
    # consistent with zero.  The separation in units of the combined standard
    # error is well defined and monotonic in lambda.
    def separation(prefix: str) -> float:
        mp = values[f"physical_{prefix}"]["mean"][lambda_index]
        sp = values[f"physical_{prefix}"]["sem"][lambda_index]
        mz = values[f"no_capillary_{prefix}"]["mean"][lambda_index]
        sz = values[f"no_capillary_{prefix}"]["sem"][lambda_index]
        return (mp - mz) / ((sp ** 2 + sz ** 2) ** 0.5)

    add_close(checks, "lambda=0.9 split endpoint separation (sigma)",
              separation("split_summary"), SPLIT_SEPARATION_SIGMA, 1e-9)
    add_close(checks, "lambda=0.9 write endpoint separation (sigma)",
              separation("release_summary"), WRITE_SEPARATION_SIGMA, 1e-9)

    # 71-73: waiting-time increment resolved graph by graph.
    increments = paired_aging_increment(dynamics, 30.0)
    mean, std = sample_mean_std(increments)
    add_close(checks, "paired aging increment mean", mean, 0.02530401211798668, 1e-12)
    add_close(checks, "paired aging increment graph SD", std, 0.0018613104549769866, 1e-12)
    add_exact(checks, "graphs with positive aging increment", sum(1 for item in increments if item > 0), 3)

    # 74-77: angular localization relative to uniform, bounded by the measured overlaps.
    for label, values, expected in (
        ("write field applied", [float(run["write_release"]["write_overlap"][-1]) for run in dynamics], 1.4811116497946106),
        ("written state released", [float(run["write_release"]["release_overlap"][-1]) for run in dynamics], 0.29580233998316197),
        ("split replicas", split_end, 0.32661562528973526),
        ("g=0 written state", [float(run["no_capillary_write_release"]["release_overlap"][-1]) for run in dynamics], 6.572039757639629e-05),
    ):
        bits = [angular_localization_bits(max(value, 0.0)) for value in values]
        add_close(checks, f"angular localization bits per rotor, {label}", float(np.mean(bits)), expected, 1e-10)

    # 78: unnormalized overlap correlation length on the scanned ray.
    lengths = overlap_correlation_lengths(SPIN_SCAN)
    add_close(checks, "max graph-averaged overlap length xi_L", lengths["graph_averaged_max"], 1.4893358730245692, 1e-10)

    # 79-82: the g=0 control is more ordered, not less, on the same graphs.
    for name, expected_mean, expected_std in (
        ("S_mean", 0.1311428920457058, 0.010123705266531027),
        ("C2_mean", 0.8128899338103832, 0.002853034994366132),
    ):
        mean, std = sample_mean_std(row[name] for row in g0_rows)
        add_close(checks, f"g=0 {name}", mean, expected_mean, 1e-12)
        add_close(checks, f"g=0 {name} graph SD", std, expected_std, 1e-12)

    # 83-88: the positional-disorder retention maximum quoted in the Letter.
    disorder = read_json(DISORDER_RETENTION)
    n1024 = {float(row["sigma_over_a"]): row for row in disorder["n1024"]}
    add_exact(checks, "N=1024 disorder-retention amplitudes", sorted(n1024), [0.11, 0.16])
    add_exact(checks, "N=1024 disorder-retention graph seeds", n1024[0.11]["graph_seeds"], [17, 29, 43, 71, 97])
    add_close(checks, "N=1024 connected retention at sigma=0.11", n1024[0.11]["mean"], 0.5143, 1e-12)
    add_close(checks, "N=1024 connected retention at sigma=0.16", n1024[0.16]["mean"], 0.45442, 1e-12)
    add_close(checks, "N=1024 paired retention difference", disorder["comparison"]["mean_difference_0p11_minus_0p16"], 0.05988, 1e-12)
    add_close(checks, "N=1024 paired retention p value", disorder["comparison"]["paired_p_two_sided"], 0.0046120469, 1e-12)

    # 89-94: matched loop intervention quoted in the Letter.
    holonomy = read_json(HOLONOMY)
    add_exact(checks, "holonomy intervention graph count", holonomy["parameters"]["seed_count"], 24)
    add_close(checks, "holonomy one-flip stable-state excess", holonomy["landscape"]["mean_stable_state_excess"], 5.583333333333333, 1e-12)
    add_close(checks, "holonomy stable-state CI lower", holonomy["landscape"]["graph_bootstrap_95_interval"][0], 2.6666666666666665, 1e-12)
    add_close(checks, "holonomy stable-state CI upper", holonomy["landscape"]["graph_bootstrap_95_interval"][1], 8.833333333333334, 1e-12)
    add_exact(checks, "holonomy generic dynamic-memory decision", holonomy["dynamic_memory"]["generic_pattern_decision"], "not_passed")
    dynamic_ci = holonomy["dynamic_memory"]["graph_bootstrap_95_interval"]
    add_exact(checks, "holonomy generic dynamic-memory CI contains zero", bool(dynamic_ci[0] <= 0.0 <= dynamic_ci[1]), True)

    # 95-110: signed AB/BA readout and the relaxation-quotient control.
    order_n144 = read_jsonl(ORDER_N144)
    order_n256 = read_jsonl(ORDER_N256)
    order_fraction = read_jsonl(ORDER_FRACTION_N144)
    add_exact(checks, "N=144 order-memory raw rows", len(order_n144), 30)
    for mode, expected_mean, expected_sem in (
        ("contested", 0.5985411324151126, 0.001476314098046082),
        ("partitioned", 0.02653158374959344, 0.010361239611684695),
    ):
        values = [
            float(row["terminal_order_readout"])
            for row in order_n144
            if math.isclose(float(row["field"]), 8.0) and row["mode"] == mode
        ]
        mean, sem = sample_mean_sem(values)
        add_exact(checks, f"N=144 h=8 {mode} graph count", len(values), 3)
        add_close(checks, f"N=144 h=8 {mode} readout mean", mean, expected_mean, 1e-12)
        add_close(checks, f"N=144 h=8 {mode} readout SEM", sem, expected_sem, 1e-12)
    for mode, expected in (
        ("contested", 0.5780788838797862),
        ("partitioned", 0.0014965786804022453),
    ):
        values = [float(row["terminal_order_readout"]) for row in order_n256 if row["mode"] == mode]
        add_close(checks, f"N=256 h=8 {mode} readout mean", np.mean(values), expected, 1e-12)
    full_values = [
        float(row["terminal_order_readout"])
        for row in order_fraction
        if math.isclose(float(row["contest_fraction_requested"]), 1.0)
    ]
    full_mean, full_sem = sample_mean_sem(full_values)
    add_close(checks, "N=144 full-support readout mean", full_mean, 0.6669145621655357, 1e-12)
    add_close(checks, "N=144 full-support readout SEM", full_sem, 0.02238124394420527, 1e-12)

    audit_independent_order(checks, ORDER_INDEPENDENT_REPORT)

    relaxed = read_json(RELAXED_EXCHANGE)["bracket_control"]
    add_close(checks, "partial-support reduced-model bracket", relaxed["partial_k4"]["bracket_norm"], 8.892175983509397, 1e-12)
    add_close(checks, "full-support reduced-model bracket", relaxed["full_k8"]["bracket_norm"], 11.69703197342181, 1e-12)
    add_close(checks, "partial-support retained basin separation", relaxed["partial_k4"]["separation_retained"], 1.2999999999999952, 1e-12)
    add_close(checks, "full-support retained basin separation", relaxed["full_k8"]["separation_retained"], 0.0, 1e-12)
    add_exact(checks, "partial-support transmission outside write support", relaxed["partial_k4"]["retained_outside_common_support"], 4)

    activated_raw = sorted(DATA.glob("rotating_colloids_activated_memory_prl_gpu/**/activated_memory_scan.jsonl"))
    required = [
        DENSE, REGIMES, CONTROLS, INTERNAL, SPIN, ACTIVATED,
        DISORDER_RETENTION, HOLONOMY, ORDER_N144, ORDER_N256,
        ORDER_FRACTION_N144, RELAXED_EXCHANGE, ORDER_INDEPENDENT,
        ORDER_INDEPENDENT_REPORT, *SIZE_PATHS.values(), *DYNAMICS_PATHS,
    ]
    text = MAIN_TEX.read_text(encoding="utf-8")
    language_gates = {
        "no_permanent_memory_claim": "permanent memory" not in text.lower(),
        "no_equilibrium_glass_claim": "we establish an equilibrium glass" not in text.lower(),
        "constructed_graph_not_called_emergent": "grey neighbour network is an emergent cage" not in text.lower(),
        "finite_window_metric_not_called_lifetime": "integral retention time" not in text.lower(),
    }
    disorder_raw = validate_disorder_retention_raw(DISORDER_RETENTION_RAW)
    provenance = {
        "required_local_artifacts_present": all(path.exists() for path in required),
        "missing_required_local_artifacts": [str(path) for path in required if not path.exists()],
        "activated_memory_raw_jsonl_present": bool(activated_raw),
        "activated_memory_raw_jsonl": [str(path) for path in activated_raw],
        "language_gates": language_gates,
        "all_language_gates_passed": all(language_gates.values()),
        "supplemental_figure_references": supplemental_figure_map(SUPPLEMENT_TEX, text),
        "disorder_retention_raw": disorder_raw,
        "disorder_retention_raw_n1024_available": disorder_raw["complete"],
        "disorder_retention_provenance_note": disorder.get("provenance", {}).get("description"),
    }
    if activated_raw:
        provenance["activated_memory_reproduction"] = reproduce_activated_report(activated_raw, activated)
    else:
        provenance["activated_memory_reproduction"] = {
            "raw_reproduces_derived_report": False,
            "reason": "raw activated_memory_scan.jsonl shards are absent on this host",
        }
    # Fig. 4(c) is quoted in the Letter but was not recorded by earlier report
    # versions. Check it whenever the extended report is available.
    windows = activated.get("window_statistics", {}).get("longest_window")
    if windows:
        provenance["activated_memory_window_statistics"] = {
            "longest_window": windows,
            "recorded_for_all_lambdas": sorted(float(key) for key in windows)
            == [float(value) for value in activated["lambdas"]],
        }
    else:
        provenance["activated_memory_window_statistics"] = {
            "recorded_for_all_lambdas": False,
            "reason": "report predates the panel (c) window statistics; rebuild Fig. 4 to record them",
        }

    validation_manifest_path = SUBMISSION_VALIDATIONS / "validation_manifest.json"
    if validation_manifest_path.exists():
        validation_manifest = read_json(validation_manifest_path)
        provenance["submission_validation_archive"] = {
            "manifest": str(validation_manifest_path),
            "complete": bool(validation_manifest.get("complete"))
            and int(validation_manifest.get("timestep_rows", 0)) == 15
            and int(validation_manifest.get("mobile_cage_rows", 0)) == 25
            and int(validation_manifest.get("independent_order_rows", 0)) == 50
            and ORDER_INDEPENDENT.exists()
            and ORDER_INDEPENDENT_REPORT.exists(),
            "timestep_rows": int(validation_manifest.get("timestep_rows", 0)),
            "mobile_cage_rows": int(validation_manifest.get("mobile_cage_rows", 0)),
            "independent_order_rows": int(
                validation_manifest.get("independent_order_rows", 0)
            ),
        }
    else:
        provenance["submission_validation_archive"] = {
            "manifest": str(validation_manifest_path),
            "complete": False,
            "reason": "completed cluster validation directory is absent on this host",
        }

    provenance["all_publication_raw_provenance_passed"] = bool(
        provenance["activated_memory_reproduction"].get("raw_reproduces_derived_report", False)
        and provenance["disorder_retention_raw_n1024_available"]
        and provenance["submission_validation_archive"]["complete"]
    )

    passed = sum(bool(item["passed"]) for item in checks)
    report = {
        "audit_contract": "capillary_prl_publication_scale_v3",
        "quantitative_checks": checks,
        "checks": checks,
        "quantitative_checks_passed": passed,
        "quantitative_checks_total": len(checks),
        "all_checks_passed": passed == len(checks),
        "provenance": provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "capillary_pair_prl_claim_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Capillary-pair PRL claim audit",
        "",
        f"- Quantitative checks passed: `{passed}/{len(checks)}`",
        f"- All quantitative checks passed: `{report['all_checks_passed']}`",
        f"- Required local artifacts present: `{provenance['required_local_artifacts_present']}`",
        f"- Activated-memory raw JSONL present locally: `{provenance['activated_memory_raw_jsonl_present']}`",
        f"- Raw activated-memory shards reproduce the derived report: "
        f"`{provenance['activated_memory_reproduction']['raw_reproduces_derived_report']}`",
        f"- Fig. 4(c) window statistics recorded: "
        f"`{provenance['activated_memory_window_statistics']['recorded_for_all_lambdas']}`",
        f"- Language gates passed: `{provenance['all_language_gates_passed']}`",
        "",
        "The numerical contract covers the publication-scale regime map, matched controls, five-size scaling, long dynamics, spatial correlations, equilibrium-replica discriminant, coupling-dependent endpoint overlap, the disorder-retention maximum, the matched loop intervention, and both common- and independent-noise AB/BA release readouts.",
        "",
        "## Failed quantitative checks",
        "",
    ]
    failed = [item for item in checks if not item["passed"]]
    lines.extend([f"- `{item['claim']}`" for item in failed] or ["- None."])
    lines.extend(
        [
            "",
            "## Provenance gates",
            "",
            "The raw activated-memory JSONL is generated on the GPU cluster and must be included in the Zenodo deposit. Identical duplicate records are ignored by stable row key, while conflicting duplicates fail the audit.",
            "",
            "The N=1024 disorder-retention values are currently transcribed from cluster output. The deposit is not publication-complete until their raw protocol trajectories are present and regenerate the summary.",
            "",
            "The completed submission-validation directory must include 15 time-step rows, 25 mobile-cage rows, and 50 independent-noise sequence rows before the independent-noise Letter claims are publication-complete.",
        ]
    )
    (output_dir / "capillary_pair_prl_claim_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "quantitative_checks": f"{passed}/{len(checks)}",
                "all_checks_passed": report["all_checks_passed"],
                "activated_raw_present": provenance["activated_memory_raw_jsonl_present"],
                "activated_raw_reproduces_report": provenance["activated_memory_reproduction"][
                    "raw_reproduces_derived_report"
                ],
                "all_publication_raw_provenance_passed": provenance["all_publication_raw_provenance_passed"],
                "language_gates_passed": provenance["all_language_gates_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
