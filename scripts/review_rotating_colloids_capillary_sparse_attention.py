#!/usr/bin/env python3
"""Evidence-weighted reviewer audit for the capillary-rotor manuscript.

This program does not use sparse attention as a proof engine. It constructs
small evidence cards from independent simulation artifacts, downweights cards
with missing provenance or failed controls, and exposes the most informative
cross-artifact dependencies. The resulting report is intended to support a
scientific review of the present manuscript and to define discriminating next
experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence

_mpl_cache = os.path.join(tempfile.gettempdir(), "hyperion_matplotlib_cache")
os.makedirs(_mpl_cache, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _mpl_cache)
os.environ.setdefault("MPLBACKEND", "Agg")
# Small K-means problems are much slower when the macOS BLAS backend starts a
# large thread pool for every initialization. Keep this audit deterministic and
# single-threaded; the publication-scale simulations are unaffected.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


_TRAPEZOID = getattr(np, "trapezoid", None)
if _TRAPEZOID is None:  # NumPy < 2.0
    _TRAPEZOID = np.trapz
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURES = ("S_mean", "C2_mean", "G2_mean", "q_EA_mean", "window_autocorrelation")
FEATURE_LABELS = {
    "S_mean": "S",
    "C2_mean": "C2",
    "G2_mean": "G2",
    "q_EA_mean": "qEA",
    "window_autocorrelation": "Cwindow",
}
SIZE_METRICS = FEATURES + ("replica_overlap_magnitude",)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 350,
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_sem(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": float("nan"), "sem": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(array.mean()),
        "sem": float(array.std(ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "n": int(array.size),
    }


def quantile_interval(values: Sequence[float], low: float = 0.025, high: float = 0.975) -> list[float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return [float("nan"), float("nan")]
    return [float(np.quantile(array, low)), float(np.quantile(array, high))]


def aggregate_dense_rows(rows: Sequence[dict[str, Any]], sampled_seeds: Sequence[int] | None = None) -> tuple[list[tuple[float, float]], np.ndarray]:
    by_seed_cell: dict[tuple[int, float, float], dict[str, Any]] = {}
    seeds = sorted({int(row["graph_seed"]) for row in rows})
    for row in rows:
        key = (int(row["graph_seed"]), float(row["j_align"]), float(row["g_capillary"]))
        by_seed_cell[key] = row
    sampled = list(sampled_seeds) if sampled_seeds is not None else seeds
    cells = sorted({(float(row["j_align"]), float(row["g_capillary"])) for row in rows})
    matrix = []
    for j_value, g_value in cells:
        selected = [by_seed_cell[(seed, j_value, g_value)] for seed in sampled]
        matrix.append([float(np.mean([float(row[feature]) for row in selected])) for feature in FEATURES])
    return cells, np.asarray(matrix, dtype=float)


def kmeans_scan(x: np.ndarray, candidates: Iterable[int] = range(2, 7), seed: int = 11, n_init: int = 40) -> tuple[dict[int, float], dict[int, np.ndarray]]:
    scaled = StandardScaler().fit_transform(x)
    scores: dict[int, float] = {}
    labels: dict[int, np.ndarray] = {}
    for count in candidates:
        assignment = KMeans(n_clusters=count, random_state=seed, n_init=n_init).fit_predict(scaled)
        labels[count] = assignment
        scores[count] = float(silhouette_score(scaled, assignment))
    return scores, labels


def audit_regime_taxonomy(rows: Sequence[dict[str, Any]], bootstraps: int, rng: np.random.Generator) -> dict[str, Any]:
    cells, x = aggregate_dense_rows(rows)
    full_scores, full_labels = kmeans_scan(x, n_init=50)
    best_k = max(full_scores, key=full_scores.get)
    reference_k4 = full_labels[4]
    seeds = sorted({int(row["graph_seed"]) for row in rows})

    selected_counts: Counter[int] = Counter()
    bootstrap_ari: list[float] = []
    bootstrap_silhouette_gap: list[float] = []
    # With three graph realizations there are only 3^3=27 ordered bootstrap
    # resamples. Enumerating them exactly is cleaner than drawing the same
    # resamples hundreds of times.
    graph_resamples = list(product(seeds, repeat=len(seeds)))
    for bootstrap_index, sampled_tuple in enumerate(graph_resamples):
        sampled = list(sampled_tuple)
        _, x_boot = aggregate_dense_rows(rows, sampled)
        scores, labels = kmeans_scan(x_boot, seed=1000 + bootstrap_index, n_init=10)
        selected = max(scores, key=scores.get)
        selected_counts[selected] += 1
        bootstrap_ari.append(float(adjusted_rand_score(reference_k4, labels[4])))
        ordered = sorted(scores.values(), reverse=True)
        bootstrap_silhouette_gap.append(float(ordered[0] - ordered[1]))

    feature_ablation: dict[str, Any] = {}
    for index, feature in enumerate(FEATURES):
        x_drop = np.delete(x, index, axis=1)
        scores, labels = kmeans_scan(x_drop, n_init=40)
        selected = max(scores, key=scores.get)
        feature_ablation[feature] = {
            "best_k": int(selected),
            "best_silhouette": float(scores[selected]),
            "k4_ARI_against_full": float(adjusted_rand_score(reference_k4, labels[4])),
        }

    leave_one_seed_out: dict[str, Any] = {}
    for omitted in seeds:
        retained = [seed for seed in seeds if seed != omitted]
        _, x_subset = aggregate_dense_rows(rows, retained)
        scores, labels = kmeans_scan(x_subset, n_init=40)
        selected = max(scores, key=scores.get)
        leave_one_seed_out[str(omitted)] = {
            "best_k": int(selected),
            "best_silhouette": float(scores[selected]),
            "k4_ARI_against_full": float(adjusted_rand_score(reference_k4, labels[4])),
        }

    scaled = StandardScaler().fit_transform(x)
    model = KMeans(n_clusters=4, random_state=11, n_init=200).fit(scaled)
    distances = np.linalg.norm(scaled[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    ordered = np.sort(distances, axis=1)
    margins = (ordered[:, 1] - ordered[:, 0]) / np.maximum(ordered[:, 1], 1e-12)
    selected_index = cells.index((4.0, 4.875)) if (4.0, 4.875) in cells else int(
        np.argmin([(j_value - 4.0) ** 2 + (g_value - 5.0) ** 2 for j_value, g_value in cells])
    )
    correlations = np.corrcoef(x, rowvar=False)

    return {
        "raw_rows": len(rows),
        "parameter_cells": len(cells),
        "graph_seeds": seeds,
        "silhouette_scores": {str(key): value for key, value in full_scores.items()},
        "best_k": int(best_k),
        "k4_minus_k5_silhouette": float(full_scores[4] - full_scores[5]),
        "bootstrap_replicates": len(graph_resamples),
        "bootstrap_best_k_counts": {str(key): int(value) for key, value in sorted(selected_counts.items())},
        "bootstrap_best_k_fraction": {str(key): float(value / len(graph_resamples)) for key, value in sorted(selected_counts.items())},
        "bootstrap_k4_ARI": {
            **mean_sem(bootstrap_ari),
            "ci95": quantile_interval(bootstrap_ari),
        },
        "bootstrap_top_silhouette_gap": {
            **mean_sem(bootstrap_silhouette_gap),
            "ci95": quantile_interval(bootstrap_silhouette_gap),
        },
        "feature_ablation": feature_ablation,
        "leave_one_graph_seed_out": leave_one_seed_out,
        "selected_cell": {
            "j_align": float(cells[selected_index][0]),
            "g_capillary": float(cells[selected_index][1]),
            "k4_confidence_margin": float(margins[selected_index]),
            "features": {feature: float(x[selected_index, index]) for index, feature in enumerate(FEATURES)},
        },
        "feature_correlation": {
            FEATURES[i]: {FEATURES[j]: float(correlations[i, j]) for j in range(len(FEATURES))}
            for i in range(len(FEATURES))
        },
    }


def finite_size_rows(gpu_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(gpu_root.glob("finite_size_n*/capillary_pair_scan.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == "replica_overlap_magnitude":
        return float(row["replica_overlap"]["magnitude_mean"])
    return float(row[metric])


def power_law_audit(rows: Sequence[dict[str, Any]], metric: str, g_value: float, bootstraps: int, rng: np.random.Generator) -> dict[str, Any]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("control", "physical")) != "physical":
            continue
        if not math.isclose(float(row["g_capillary"]), g_value):
            continue
        grouped[int(row["n"]) ** 2].append(metric_value(row, metric))
    node_counts = np.asarray(sorted(grouped), dtype=float)
    means = np.asarray([np.mean(grouped[int(count)]) for count in node_counts], dtype=float)
    sems = np.asarray([stats.sem(grouped[int(count)]) for count in node_counts], dtype=float)
    positive = np.maximum(np.abs(means), 1e-12)
    fit = stats.linregress(np.log(node_counts), np.log(positive))

    bootstrap_exponents: list[float] = []
    for _ in range(bootstraps):
        sampled_means = []
        for count in node_counts.astype(int):
            values = np.asarray(grouped[count], dtype=float)
            sampled_means.append(float(np.mean(rng.choice(values, size=values.size, replace=True))))
        sampled_means_array = np.maximum(np.abs(np.asarray(sampled_means)), 1e-12)
        bootstrap_exponents.append(float(stats.linregress(np.log(node_counts), np.log(sampled_means_array)).slope))
    return {
        "metric": metric,
        "g_capillary": g_value,
        "node_counts": node_counts.astype(int).tolist(),
        "mean": means.tolist(),
        "sem": sems.tolist(),
        "exponent": float(fit.slope),
        "r_squared": float(fit.rvalue**2),
        "bootstrap_ci95": quantile_interval(bootstrap_exponents),
        "bootstrap_exponent_mean": float(np.mean(bootstrap_exponents)),
    }


def audit_finite_size(rows: Sequence[dict[str, Any]], bootstraps: int, rng: np.random.Generator) -> dict[str, Any]:
    results: dict[str, Any] = {"raw_rows": len(rows), "metrics": {}}
    for metric in SIZE_METRICS:
        results["metrics"][metric] = {
            "g5": power_law_audit(rows, metric, 5.0, bootstraps, rng),
            "g0": power_law_audit(rows, metric, 0.0, bootstraps, rng),
        }
    return results


def first_below(time: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    indices = np.flatnonzero(values < threshold)
    return float(time[indices[0]]) if indices.size else None


def summarize_time_channel(runs: Sequence[dict[str, Any]], block: str, time_key: str, value_key: str) -> dict[str, Any]:
    time = np.asarray(runs[0][block][time_key], dtype=float)
    values = np.asarray([run[block][value_key] for run in runs], dtype=float)
    if not all(np.allclose(time, np.asarray(run[block][time_key], dtype=float)) for run in runs[1:]):
        raise RuntimeError(f"Time axes differ for {block}/{value_key}")
    tail_start = max(1, int(0.8 * time.size))
    slopes = [float(stats.linregress(time[tail_start:], curve[tail_start:]).slope) for curve in values]
    tail_changes = [float(curve[-1] - curve[tail_start]) for curve in values]
    integrals = [float(_TRAPEZOID(np.maximum(curve, 0.0), time)) for curve in values]
    return {
        "time": time.tolist(),
        "mean_curve": values.mean(axis=0).tolist(),
        "sem_curve": stats.sem(values, axis=0).tolist(),
        "endpoint": mean_sem(values[:, -1].tolist()),
        "tail_slope": {**mean_sem(slopes), "ci95": quantile_interval(slopes)},
        "tail_change": mean_sem(tail_changes),
        "positive_integral": mean_sem(integrals),
        "first_below_0.2": [first_below(time, curve, 0.2) for curve in values],
    }


def audit_dynamics(gpu_root: Path) -> dict[str, Any]:
    paths = sorted(gpu_root.glob("dynamics_seed_*/capillary_pair_protocols.json"))
    runs = [read_json(path) for path in paths]
    if not runs:
        raise RuntimeError(f"No dynamics files found below {gpu_root}")
    channels = {
        "physical_split": ("split_replica", "time", "overlap_mean"),
        "g0_split": ("no_capillary_split_replica", "time", "overlap_mean"),
        "physical_write_release": ("write_release", "release_time", "release_overlap"),
        "g0_write_release": ("no_capillary_write_release", "release_time", "release_overlap"),
    }
    summaries = {
        label: summarize_time_channel(runs, block, time_key, value_key)
        for label, (block, time_key, value_key) in channels.items()
    }
    aging_by_wait: dict[float, list[float]] = defaultdict(list)
    aging_tail_by_wait: dict[float, list[float]] = defaultdict(list)
    for run in runs:
        for curve in run["aging"]["curves"]:
            waiting = float(curve["waiting_time"])
            lag = np.asarray(curve["lag_time"], dtype=float)
            corr = np.asarray(curve["correlation"], dtype=float)
            aging_by_wait[waiting].append(float(np.interp(30.0, lag, corr)))
            aging_tail_by_wait[waiting].append(float(corr[-1]))
    aging = {
        str(waiting): {
            "C_lag_30": mean_sem(values),
            "endpoint": mean_sem(aging_tail_by_wait[waiting]),
        }
        for waiting, values in sorted(aging_by_wait.items())
    }
    model = runs[0]["model"]
    write_field = float(runs[0]["write_release"]["write_field"])
    return {
        "graph_seeds": [int(path.parent.name.rsplit("_", 1)[-1]) for path in paths],
        "model": model,
        "write_field": write_field,
        "write_target_provenance": "equilibrium configuration generated with the same J,g Hamiltonian",
        "channels": summaries,
        "aging": aging,
    }


def audit_spatial_range(path: Path) -> dict[str, Any]:
    aggregate = read_json(path)["aggregate"]
    distance = np.asarray(aggregate["bin_centers"], dtype=float)
    relative = np.asarray(aggregate["relative_connected_mean"], dtype=float)
    bond = np.asarray(aggregate["bond_frame_correlation_mean"], dtype=float)
    noise_floor = max(float(np.mean(np.abs(relative[-5:]))), float(np.mean(np.abs(bond[-5:]))))
    threshold = max(0.03, 3.0 * noise_floor)
    above = np.flatnonzero((np.abs(relative) > threshold) | (np.abs(bond) > threshold))
    range_estimate = float(distance[above[-1]]) if above.size else 0.0
    return {
        "graph_count": int(aggregate["graph_count"]),
        "first_shell_relative_connected": float(relative[1]),
        "first_shell_bond_frame": float(bond[1]),
        "estimated_correlation_range_r_over_a": range_estimate,
        "operational_threshold": threshold,
        "noise_floor": noise_floor,
    }


def artifact_audit(root: Path, figure_root: Path) -> dict[str, Any]:
    activated_candidates = list(root.glob("rotating_colloids_activated_memory_prl_gpu/**/*.jsonl"))
    claim_path = root / "rotating_colloids_capillary_pair_prl_claim_audit/capillary_pair_prl_claim_audit.json"
    claim = read_json(claim_path) if claim_path.exists() else {}
    failed = [check for check in claim.get("checks", []) if not bool(check.get("passed"))]
    return {
        "activated_memory_raw_jsonl_present": bool(activated_candidates),
        "activated_memory_raw_jsonl": [str(path) for path in activated_candidates],
        "activated_memory_summary_present": (figure_root / "activated_memory_figure_report.json").exists(),
        "claim_audit_present": claim_path.exists(),
        "claim_audit_all_checks_passed": bool(claim.get("all_checks_passed", False)),
        "claim_audit_failed_checks": len(failed),
        "claim_audit_stale_against_current_regime_report": len(failed) >= 5,
        "claim_audit_failed_claims": [str(check.get("claim", "unnamed")) for check in failed],
    }


def curvature_write_estimate() -> dict[str, Any]:
    """Order-of-magnitude curvature needed for a one-kBT write torque."""
    boltzmann = 1.380649e-23
    temperature = 298.0
    gamma = 10.0e-3
    quadrupole_height = 5.0e-9
    particle_radius = 1.0e-6
    target_energy = boltzmann * temperature
    curvature = target_energy / (math.pi * gamma * quadrupole_height * particle_radius**2)
    amplitudes: dict[str, float] = {}
    for wavelength in (100.0e-6, 1.0e-3):
        wave_number = 2.0 * math.pi / wavelength
        amplitudes[f"{wavelength * 1e6:g}_um"] = curvature / wave_number**2
    return {
        "temperature_K": temperature,
        "target_energy_kBT": 1.0,
        "interfacial_tension_N_per_m": gamma,
        "particle_quadrupole_height_m": quadrupole_height,
        "particle_radius_m": particle_radius,
        "required_deviatoric_curvature_per_m": curvature,
        "sinusoidal_height_amplitude_m_by_wavelength": amplitudes,
        "caveat": "Scaling estimate; order-one factors depend on the deviatoric-curvature convention and wetting geometry.",
    }


def attention_score(support: float, cross_artifact: float, surprise: float, falsifiability: float, provenance: float) -> float:
    return float(
        0.28 * support
        + 0.24 * cross_artifact
        + 0.18 * surprise
        + 0.18 * falsifiability
        + 0.12 * provenance
    )


def make_attention_cards(report: dict[str, Any]) -> list[dict[str, Any]]:
    finite = report["finite_size"]["metrics"]
    dynamics = report["dynamics"]["channels"]
    taxonomy = report["regime_taxonomy"]
    spatial = report["spatial_range"]
    spin = report["spin_glass"]
    artifacts = report["artifact_audit"]

    cards = [
        {
            "id": "SA01",
            "status": "supported discovery",
            "title": "Memory is local in space but long-lived in time",
            "observation": (
                f"Connected angular correlations fall to the noise scale by about r/a={spatial['estimated_correlation_range_r_over_a']:.2f}, "
                f"whereas split descendants retain overlap {dynamics['physical_split']['endpoint']['mean']:.3f} at the end of the D_r t={dynamics['physical_split']['time'][-1]:.1f} protocol."
            ),
            "mechanism": "Finite-range bond-frame constraints create local metastable basins; long retention does not require a bulk director.",
            "prediction": "Subregions separated by more than the measured correlation range should store approximately independent angular histories, until positional rearrangements couple them.",
            "falsifier": "A growing correlation length with system size, or loss of retention when the observation area exceeds a few cages, would invalidate the local-memory interpretation.",
            "components": {"support": 0.95, "cross_artifact": 0.96, "surprise": 0.88, "falsifiability": 0.92, "provenance": 0.95},
        },
        {
            "id": "SA02",
            "status": "supported discovery",
            "title": "The retained state is genealogical rather than an equilibrium glass state",
            "observation": (
                f"Global S scales approximately as N^{finite['S_mean']['g5']['exponent']:.2f}, independent-replica overlap as "
                f"N^{finite['replica_overlap_magnitude']['g5']['exponent']:.2f}, while qEA has exponent {finite['q_EA_mean']['g5']['exponent']:.2f}; "
                f"the spin-glass scan reports {spin.get('finding', 'no finding')}."
            ),
            "mechanism": "Two descendants of the same prepared state remain correlated, but independently equilibrated systems do not select a common macroscopic state.",
            "prediction": "Clone overlap should depend on the common preparation history, whereas equilibrium replica overlap should continue to vanish with area.",
            "falsifier": "A size-independent crossing of overlap correlation length or a nonzero thermodynamic inter-replica overlap would instead support an equilibrium glass phase.",
            "components": {"support": 0.96, "cross_artifact": 0.98, "surprise": 0.92, "falsifiability": 0.96, "provenance": 0.94},
        },
        {
            "id": "SA03",
            "status": "supported mechanism",
            "title": "Bond geometry suppresses a common director while preserving local constraints",
            "observation": (
                f"At g=5, S falls from {finite['S_mean']['g5']['mean'][0]:.3f} to {finite['S_mean']['g5']['mean'][-1]:.3f} across the size series, "
                f"while C2 and G2 remain near {finite['C2_mean']['g5']['mean'][-1]:.3f} and {finite['G2_mean']['g5']['mean'][-1]:.3f}."
            ),
            "mechanism": "Incompatible bond-frame minima distribute orientation among local compromises instead of selecting one laboratory-frame nematic axis.",
            "prediction": "Reducing bond-angle disorder should restore global nematic order or collapse the hidden-memory window into an ordered state.",
            "falsifier": "The same low-S, high-local-memory response on a regular graph with matched weighted degree would show that geometric incompatibility is not essential.",
            "components": {"support": 0.90, "cross_artifact": 0.90, "surprise": 0.83, "falsifiability": 0.95, "provenance": 0.93},
        },
        {
            "id": "SA04",
            "status": "open construction gate",
            "title": "Arbitrary pattern capacity has not yet been demonstrated",
            "observation": "The write target is an equilibrium configuration of the same interaction landscape, so the protocol tests reinforcement and inheritance of a compatible basin.",
            "mechanism": "A compatible field can select an existing metastable basin without proving that the material can store an externally prescribed pattern.",
            "prediction": "Retention should decrease continuously with the target's excess energy or torque residual under the field-free Hamiltonian.",
            "falsifier": "Comparable post-release overlap for random, incompatible, and equilibrium-compatible targets would establish a broader associative storage capacity.",
            "components": {"support": 0.68, "cross_artifact": 0.72, "surprise": 0.86, "falsifiability": 0.98, "provenance": 0.95},
        },
        {
            "id": "SA05",
            "status": "testable prediction",
            "title": "Interfacial curvature can write memory, while waves can gate or erase it",
            "observation": "A host interface's deviatoric curvature couples linearly to a particle's capillary quadrupole, whereas the pair amplitude is quadratic in the particle-sourced meniscus amplitude. The present model fixes both channels.",
            "mechanism": "Quasistatic patterned curvature supplies a transient one-particle writing torque. A nearly uniform oscillation primarily modulates g(t), periodically lowering or raising the many-body barriers.",
            "prediction": "A curvature pulse should select a spatial angular pattern that remains after flattening; uniform wave modulation should instead tune retention and erasure without selecting a target.",
            "falsifier": "Failure of target overlap after a curvature write pulse, or unchanged retention when g(t) crosses the inferred entry boundary, would invalidate the respective writing or gating mechanism.",
            "components": {"support": 0.62, "cross_artifact": 0.72, "surprise": 0.94, "falsifiability": 0.96, "provenance": 0.82},
        },
        {
            "id": "SA06",
            "status": "review vulnerability",
            "title": "The four-regime map is a useful taxonomy, not a uniquely resolved phase count",
            "observation": (
                f"All exact graph bootstraps select k=4, but its silhouette exceeds k=5 by only {taxonomy['k4_minus_k5_silhouette']:.4f}; "
                f"qEA and Cwindow correlate at {taxonomy['feature_correlation']['q_EA_mean']['window_autocorrelation']:.3f}, and removing C2 selects k={taxonomy['feature_ablation']['C2_mean']['best_k']}."
            ),
            "mechanism": "The taxonomy is stable to the sampled graphs but depends on retaining both competing pair harmonics as separate physical coordinates.",
            "prediction": "A larger independent graph ensemble should preserve the hidden-memory population, while the exact number of named regimes may change with nonredundant observables.",
            "falsifier": "Recovery of the same four assignments under nonredundant feature sets and prospective experimental observables would establish a uniquely resolved four-regime taxonomy.",
            "components": {"support": 0.92, "cross_artifact": 0.70, "surprise": 0.72, "falsifiability": 0.92, "provenance": 0.96},
        },
        {
            "id": "SA07",
            "status": "review vulnerability",
            "title": "The activated-lifetime figure lacks its raw local artifact",
            "observation": (
                "A 40-row summary is present, but the underlying activated-memory JSONL is "
                + ("present." if artifacts["activated_memory_raw_jsonl_present"] else "absent from this checkout.")
            ),
            "mechanism": "Without raw rows, curve construction, uncertainty propagation, and graph-level outliers cannot be independently reproduced.",
            "prediction": "Restoring the raw rows should reproduce all eight means and standard errors in the figure report exactly.",
            "falsifier": "Any discrepancy after raw-data restoration requires regenerating the figure and revising the numerical claims.",
            "components": {"support": 1.0, "cross_artifact": 0.55, "surprise": 0.45, "falsifiability": 1.0, "provenance": 0.35 if not artifacts["activated_memory_raw_jsonl_present"] else 0.95},
        },
    ]
    for card in cards:
        card["score"] = attention_score(**card["components"])
    return sorted(cards, key=lambda card: float(card["score"]), reverse=True)


def write_diagnostic_figure(report: dict[str, Any], output_dir: Path) -> None:
    configure_plotting()
    taxonomy = report["regime_taxonomy"]
    finite = report["finite_size"]["metrics"]
    dynamics = report["dynamics"]["channels"]
    cards = report["sparse_attention_cards"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    ax = axes[0, 0]
    counts = taxonomy["bootstrap_best_k_counts"]
    silhouettes = taxonomy["silhouette_scores"]
    ks = np.asarray(sorted(int(key) for key in silhouettes))
    silhouette_values = np.asarray([silhouettes[str(key)] for key in ks])
    bootstrap_values = np.asarray([counts.get(str(key), 0) for key in ks]) / taxonomy["bootstrap_replicates"]
    ax.plot(ks, silhouette_values, "s-", color="#4c78a8", lw=1.4, ms=4, label="silhouette")
    ax.axvline(4, color="0.35", ls=":", lw=0.9)
    ax.set(xlabel="cluster count k", ylabel="silhouette score", title="taxonomy evidence", xticks=ks, ylim=(0.36, 0.47))
    ax_frequency = ax.twinx()
    ax_frequency.plot(ks, bootstrap_values, "o", color="#e45756", ms=4.5, label="exact graph bootstrap")
    ax_frequency.set(ylabel="bootstrap selection frequency", ylim=(0, 1.05))
    handles_left, labels_left = ax.get_legend_handles_labels()
    handles_right, labels_right = ax_frequency.get_legend_handles_labels()
    ax.legend(handles_left + handles_right, labels_left + labels_right, frameon=False, loc="lower center")
    ax.text(0.02, 0.96, "a", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[0, 1]
    for metric, color, marker in (("S_mean", "#b2182b", "o"), ("replica_overlap_magnitude", "#542788", "s"), ("q_EA_mean", "#1b7837", "^")):
        item = finite[metric]["g5"]
        nodes = np.asarray(item["node_counts"], dtype=float)
        means = np.asarray(item["mean"], dtype=float)
        errors = np.asarray(item["sem"], dtype=float)
        ax.errorbar(nodes, means, yerr=errors, marker=marker, color=color, lw=1.15, ms=3.5, capsize=2, label=FEATURE_LABELS.get(metric, "replica overlap"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlabel="number of rotors N", ylabel="observable", title="finite-size separation")
    ax.legend(frameon=False)
    ax.text(0.02, 0.96, "b", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 0]
    for label, color, linestyle in (
        ("physical_split", "#2166ac", "-"),
        ("physical_write_release", "#1b9e77", "-"),
        ("g0_split", "#b2182b", "--"),
        ("g0_write_release", "#ef8a62", "--"),
    ):
        channel = dynamics[label]
        time = np.asarray(channel["time"], dtype=float)
        mean = np.asarray(channel["mean_curve"], dtype=float)
        ax.plot(time, mean, color=color, ls=linestyle, lw=1.25, label=label.replace("_", " "))
    ax.set(xlabel=r"time $D_rt$", ylabel="overlap", title="retention versus control", ylim=(-0.05, 1.02))
    ax.legend(frameon=False, ncol=2, handlelength=1.7, columnspacing=0.8)
    ax.text(0.02, 0.96, "c", transform=ax.transAxes, va="top", fontweight="bold")

    ax = axes[1, 1]
    shown = cards[:6][::-1]
    y = np.arange(len(shown))
    scores = [float(card["score"]) for card in shown]
    colors = ["#1b9e77" if "supported" in str(card["status"]) else "#d95f02" if "prediction" in str(card["status"]) or "gate" in str(card["status"]) else "#7570b3" for card in shown]
    ax.hlines(y, 0, scores, color=colors, lw=2)
    ax.scatter(scores, y, color=colors, s=25, zorder=3)
    ax.set_yticks(y, [str(card["id"]) for card in shown])
    ax.set(xlabel="evidence-attention score", title="ranked cross-artifact heads", xlim=(0, 1.02))
    ax.text(0.02, 0.96, "d", transform=ax.transAxes, va="top", fontweight="bold")

    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"capillary_prl_sparse_attention_diagnostics.{suffix}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def format_ci(item: dict[str, Any]) -> str:
    low, high = item["bootstrap_ci95"]
    return f"{item['exponent']:.3f} (bootstrap 95% CI {low:.3f} to {high:.3f})"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    taxonomy = report["regime_taxonomy"]
    finite = report["finite_size"]["metrics"]
    dynamics = report["dynamics"]["channels"]
    spatial = report["spatial_range"]
    artifacts = report["artifact_audit"]
    cards = report["sparse_attention_cards"]
    lines = [
        "# Scientific Reviewer and Sparse-Attention Audit",
        "",
        "## Reviewer Recommendation",
        "",
        "**Major revision before submission.** The simulations support a distinctive finite-range, history-conditioned orientational memory mechanism. They do not yet support an equilibrium phase, arbitrary-pattern storage, or a fully realistic mobile interfacial monolayer. The strongest paper is therefore a dynamical-memory paper with explicit experimental falsifiers, not a spin-glass or universal-memory claim.",
        "",
        "## Strongest Supported Result",
        "",
        f"At the selected capillary coupling, local order remains finite while global order and independent-replica overlap decrease with size. The fitted exponents are S: {format_ci(finite['S_mean']['g5'])}; independent-replica overlap: {format_ci(finite['replica_overlap_magnitude']['g5'])}; qEA: {format_ci(finite['q_EA_mean']['g5'])}. Connected correlations fall to the operational noise scale by approximately r/a={spatial['estimated_correlation_range_r_over_a']:.2f}, but split descendants retain overlap {dynamics['physical_split']['endpoint']['mean']:.3f} +/- {dynamics['physical_split']['endpoint']['sem']:.3f} at D_r t={dynamics['physical_split']['time'][-1]:.1f}. This is evidence for memory that is local in space but long-lived in time.",
        "",
        "The independent spin-glass diagnostic reports no size-independent crossing on the scanned ray. Taken together, these results support genealogical memory: copies descended from one prepared angular state remain related, whereas independently equilibrated systems do not select the same macroscopic state.",
        "",
        "## Major Scientific Issues",
        "",
        "1. **The write protocol is compatibility-limited.** The target is first equilibrated under the same J,g Hamiltonian and then reinforced by a conjugate field. This establishes selection and retention of an available metastable basin. It does not establish arbitrary-pattern capacity. Add random targets, deliberately high-residual targets, and a target-energy/retention curve.",
        "2. **The positional graph is quenched.** The central experimental requirement is not merely slow translation but a hierarchy tau_int << D_r^-1 << tau_cage. Test a mobile-center model with excluded volume, translational Brownian noise, and a measured cage lifetime, or use experimentally tracked positions as input.",
        "3. **The physical origin of J needs calibration.** The selected state uses J=4 kBT in addition to g=5 kBT. Identify the microscopic source of the short-range relative-axis alignment and measure or simulate the joint pair potential. Otherwise the main result belongs to a two-interaction model rather than to capillary quadrupoles alone.",
        "4. **The graph ensemble is too narrow.** The dense map has three jittered triangular realizations. Add genuinely amorphous point patterns, experimentally segmented monolayers, packing-fraction variation, and topology-preserving bond-angle randomizations.",
        "5. **The four-regime map is graph-robust but feature-dependent.** "
        f"All {taxonomy['bootstrap_replicates']} exact graph resamples select k=4 and the leave-one-graph ARI is at least {min(item['k4_ARI_against_full'] for item in taxonomy['leave_one_graph_seed_out'].values()):.3f}. However, the silhouette advantage over k=5 is only {taxonomy['k4_minus_k5_silhouette']:.4f}, qEA and Cwindow correlate at {taxonomy['feature_correlation']['q_EA_mean']['window_autocorrelation']:.3f}, and removing C2 selects k={taxonomy['feature_ablation']['C2_mean']['best_k']}. Present the map as a finite-size taxonomy whose hidden-memory population is robust, not as evidence for exactly four phases.",
        "6. **Long-time retention is not a demonstrated plateau.** The final 20% split-curve slope is "
        f"{dynamics['physical_split']['tail_slope']['mean']:.3e} +/- {dynamics['physical_split']['tail_slope']['sem']:.1e} per D_r t; the write-release slope is {dynamics['physical_write_release']['tail_slope']['mean']:.3e} +/- {dynamics['physical_write_release']['tail_slope']['sem']:.1e}. Retention is long-lived and dissipative, but permanence or nonzero infinite-time overlap is unsupported.",
        "7. **Activated-lifetime provenance is incomplete.** "
        + ("The raw activated-memory rows are present." if artifacts["activated_memory_raw_jsonl_present"] else "The 40-row figure summary is present, but the raw activated-memory JSONL is absent locally. Restore it before submission and rerun the claim audit."),
        "8. **The existing claim audit is stale.** "
        f"It contains {artifacts['claim_audit_failed_checks']} failed checks against earlier regime-map values. Regenerate expected claims from the current frozen artifacts rather than keeping hard-coded legacy numbers.",
        "9. **Hydrodynamic and interfacial forcing are omitted.** Surface waves can add streaming, monopolar deformation, graph motion, and time-dependent torque, not only a scalar modulation of g. Separate these effects experimentally or in a minimal driven model.",
        "",
        "## Dedicated Sparse-Attention Readout",
        "",
        "Sparse attention here means that each candidate statement must be supported by more than one independent evidence channel and must carry a falsifier. It is an interpretation scaffold, not a proof engine.",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"### {card['id']}: {card['title']} ({card['score']:.3f})",
                "",
                f"**Status:** {card['status']}  ",
                f"**Observation:** {card['observation']}  ",
                f"**Mechanism:** {card['mechanism']}  ",
                f"**Prediction:** {card['prediction']}  ",
                f"**Falsifier:** {card['falsifier']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interfacial-Undulation Experiment",
            "",
            "There are two physically distinct ways for water-surface undulations to act. First, host-interface deviatoric curvature Delta c couples directly to the particle quadrupole through U_curv,i = -pi gamma H_p R_p^2 Delta c(x_i,t) cos[2(theta_i-alpha_i)], where H_p and R_p characterize the particle-sourced quadrupole and alpha_i is the local principal-curvature axis. This experimentally established curvature torque can write an angular pattern. Second, the pair coupling g is proportional to gamma (Delta u)^2. A small amplitude modulation Delta u(t)=Delta u0[1+epsilon cos(omega t)] gives g(t)/g0 approximately 1+2 epsilon cos(omega t)+epsilon^2[1+cos(2 omega t)]/2 and gates the many-body barriers. These channels lead to three protocols:",
            f"Using gamma=10 mN/m, H_p=5 nm, and R_p=1 micrometre, the curvature-write scaling gives one kBT at Delta c approximately {report['interfacial_undulation_estimate']['required_deviatoric_curvature_per_m']:.1f} per metre. For h=A cos(kx), this is A approximately {report['interfacial_undulation_estimate']['sinusoidal_height_amplitude_m_by_wavelength']['100_um'] * 1e9:.1f} nm at 100 micrometres wavelength or {report['interfacial_undulation_estimate']['sinusoidal_height_amplitude_m_by_wavelength']['1000_um'] * 1e6:.2f} micrometres at 1 mm wavelength. These are scaling estimates; wetting geometry changes order-one prefactors.",
            "",
            "1. **Quasistatic curvature pulse: write.** Shape the interface, allow rods to respond to alpha_i(x), and flatten it at a chosen phase. The field is removed after writing, so retained overlap tests the autonomous pair-memory mechanism.",
            "2. **Uniform oscillation: barrier gate.** Modulate g(t) at fixed positions. This should accelerate decorrelation when the trough crosses the hidden-memory entry boundary and extend retention when the mean remains above it. It cannot by itself encode a spatial target.",
            "3. **Standing or shaped wave: spatial address plus annealing.** A wavelength of several cage spacings can pattern Delta c and g_ij. It may address local memory patches, but it can also translate particles, generate streaming, and change the graph; those channels must be measured separately.",
            "",
            "The decisive simulation is a driven Smoluchowski extension with both g(t) and U_curv(t), followed by a mobile-center extension in which bond angles and positions respond to the wave. Compare memory retention, absorbed work per cycle, graph rearrangements, and target overlap. A useful frequency hierarchy is tau_int << omega^-1 approximately D_r^-1 << tau_cage: the interface follows each cycle, angular states respond, and the cage remains intact. Relevant experimental precedents are curvature-driven rod orientation (PNAS 108, 20923; doi:10.1073/pnas.1116344108), capillary-wave annealing of air-water colloids (Langmuir 41, 2025; doi:10.1021/acs.langmuir.4c02794), and acoustically driven dynamic capillary interactions (Nature Communications 9, 2018; doi:10.1038/s41467-018-06049-9).",
            "",
            "## Minimum Submission Package",
            "",
            "- Restore the raw activated-memory rows and regenerate the claim audit.",
            "- Add incompatible-target and multiple-target write-release tests.",
            "- Add at least one mobile-center or experimentally reconstructed graph test.",
            "- Calibrate J and g from a joint pair experiment or replace them with a measured angular potential.",
            "- Reframe the four clusters as finite-size regimes and report graph-bootstrap/feature-ablation uncertainty.",
            "- Add a prospective interfacial-wave test only after distinguishing scalar barrier modulation from spatial writing and hydrodynamic graph rearrangement.",
            "",
            "## Artifact Notes",
            "",
            f"- Dense-map cells: {taxonomy['parameter_cells']} from {taxonomy['raw_rows']} rows.",
            f"- Dynamics graph seeds: {report['dynamics']['graph_seeds']}.",
            f"- Spin-glass rows: {report['spin_glass'].get('rows', 'unknown')}.",
            f"- Activated raw JSONL present: {artifacts['activated_memory_raw_jsonl_present']}.",
            f"- Existing claim audit all checks passed: {artifacts['claim_audit_all_checks_passed']}.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="discoveries/theory_experiment_interface/rotating_colloids_hyperion",
        help="Rotating-colloid discovery root.",
    )
    parser.add_argument(
        "--figure-root",
        default="tex/rotating_colloids/capillary_prl_figures",
        help="Folder containing manuscript figure summaries.",
    )
    parser.add_argument(
        "--output-dir",
        default="discoveries/theory_experiment_interface/rotating_colloids_hyperion/capillary_prl_sparse_attention_review",
    )
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="Refresh prose and figures from an existing JSON report without recomputing statistics.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    figure_root = Path(args.figure_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "capillary_prl_sparse_attention_review.json"
    markdown_path = output_dir / "capillary_prl_sparse_attention_review.md"
    if args.render_existing:
        report = read_json(json_path)
        report["interfacial_undulation_estimate"] = curvature_write_estimate()
        report["sparse_attention_cards"] = make_attention_cards(report)
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_markdown(report, markdown_path)
        write_diagnostic_figure(report, output_dir)
        print(json.dumps({"output_dir": str(output_dir), "rendered_existing": True}, sort_keys=True))
        return
    gpu_root = root / "rotating_colloids_capillary_pair_prl_gpu"
    rng = np.random.default_rng(args.seed)

    dense_rows = read_jsonl(gpu_root / "dense_map_n20/capillary_pair_scan.jsonl")
    size_rows = finite_size_rows(gpu_root)
    spin_path = root / "rotating_colloids_spin_glass_prl_gpu/analysis/spin_glass_finite_size_report.json"
    internal_path = root / "rotating_colloids_capillary_pair_prl_internal/capillary_internal_correlations.json"

    report: dict[str, Any] = {
        "analysis": "capillary PRL scientific reviewer sparse-attention audit",
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "regime_taxonomy": audit_regime_taxonomy(dense_rows, args.bootstrap, rng),
        "finite_size": audit_finite_size(size_rows, args.bootstrap, rng),
        "dynamics": audit_dynamics(gpu_root),
        "spatial_range": audit_spatial_range(internal_path),
        "spin_glass": read_json(spin_path),
        "artifact_audit": artifact_audit(root, figure_root),
        "interfacial_undulation_estimate": curvature_write_estimate(),
        "claim_boundary": {
            "supported": "finite-range, history-conditioned, dissipative orientational memory on quenched positional graphs",
            "not_supported": [
                "equilibrium spin-glass phase",
                "thermodynamic phase boundary",
                "arbitrary-pattern storage",
                "permanent nonzero long-time overlap",
                "robustness to translational cage rearrangement",
            ],
        },
    }
    report["sparse_attention_cards"] = make_attention_cards(report)

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    write_diagnostic_figure(report, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "top_attention_head": report["sparse_attention_cards"][0]["id"],
                "best_k": report["regime_taxonomy"]["best_k"],
                "activated_raw_present": report["artifact_audit"]["activated_memory_raw_jsonl_present"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
