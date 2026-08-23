import math

import numpy as np
import torch

from scripts.classify_rotating_colloids_capillary_regimes import (
    build_feature_ablation_report,
)
from scripts.test_rotating_colloids_operation_order_memory import compare_orders
from scripts.rotating_colloids_capillary_pair import make_caged_graph
from scripts.validate_rotating_colloids_mobile_cage import mobile_energy
from scripts.validate_rotating_colloids_timestep import build_report
from scripts.merge_rotating_colloids_submission_validations import (
    independent_order_report,
    mobile_report,
)


def test_feature_ablation_report_keeps_persistence_coordinates_separate():
    rng = np.random.default_rng(7)
    centers = np.asarray(
        [
            [0.05, 0.05, 0.05, 0.05, 0.05],
            [0.10, 0.85, 0.10, 0.65, 0.62],
            [0.10, 0.10, 0.85, 0.68, 0.66],
            [0.10, 0.60, 0.55, 0.72, 0.70],
        ]
    )
    values = np.vstack([center + 0.015 * rng.normal(size=(20, 5)) for center in centers])
    labels = np.repeat(np.arange(4), 20)
    names = [
        "hidden mixed memory" if label == 3 else f"regime {label}"
        for label in labels
    ]
    report = build_feature_ablation_report(
        values,
        full_scaled=values,
        full_labels=labels,
        full_regime_names=names,
        requested=4,
    )
    assert set(report["variants"]) == {
        "drop_q_EA",
        "drop_C_window",
        "merge_persistence_coordinates",
        "pca_whitened",
    }
    assert report["q_EA_C_window_correlation"] > 0.9
    assert report["variants"]["drop_q_EA"]["cluster_count"] == 4


def test_mobile_energy_has_finite_angular_and_positional_gradients():
    graph = make_caged_graph(
        4,
        disorder=0.08,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    theta = torch.full((2, 16), 0.3, requires_grad=True)
    positions = torch.as_tensor(
        np.repeat(graph.positions[None, :, :], 2, axis=0), dtype=torch.float32
    ).requires_grad_(True)
    reference = torch.as_tensor(graph.positions[None, :, :], dtype=torch.float32)
    box = torch.as_tensor(graph.box[None, None, :], dtype=torch.float32)
    src = torch.as_tensor(graph.src, dtype=torch.long)
    tgt = torch.as_tensor(graph.tgt, dtype=torch.long)
    energy = mobile_energy(
        theta,
        positions,
        reference,
        box,
        src,
        tgt,
        src,
        tgt,
        r0=float(graph.metadata["nearest_spacing_median"]),
        j_align=4.0,
        g_capillary=5.0,
        cage_stiffness=20.0,
        alignment_range=1.35,
        alignment_decay=0.20,
        core_diameter=0.55,
        core_strength=10.0,
    )
    gradients = torch.autograd.grad(energy.sum(), (theta, positions))
    assert all(torch.isfinite(value).all() for value in gradients)


def test_independent_noise_order_protocol_emits_decoding_statistics():
    graph = make_caged_graph(
        4,
        disorder=0.08,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    rng = np.random.default_rng(13)
    initial = rng.uniform(0.0, math.pi, size=(4, 16)).astype(np.float32)
    axis = rng.uniform(0.0, math.pi, size=16).astype(np.float32)
    result = compare_orders(
        graph,
        initial,
        axis,
        mode="contested",
        field=3.0,
        pulse_steps=3,
        release_steps=4,
        stride=1,
        j_align=0.4,
        g_capillary=0.5,
        dt=0.01,
        seed=99,
        noise_mode="independent",
        device=torch.device("cpu"),
    )
    assert result["noise_mode"] == "independent"
    assert len(result["terminal_QA_AB"]) == 4
    assert 0.0 <= result["terminal_decode_accuracy_zero_threshold"] <= 1.0
    assert math.isfinite(result["terminal_decode_d_prime"])


def test_timestep_report_uses_finest_step_as_reference():
    rows = []
    for dt, value in ((0.01, 0.4), (0.005, 0.5)):
        for graph_seed in (17, 29):
            rows.append(
                {
                    "dt": dt,
                    "graph_seed": graph_seed,
                    "S_mean": value,
                    "C2_mean": value,
                    "G2_mean": value,
                    "q_EA_mean": value,
                    "window_autocorrelation": value,
                    "split_endpoint": value,
                    "written_endpoint": value,
                }
            )
    report = build_report(rows)
    assert report["finest_dt"] == 0.005
    assert report["summaries"][-1]["absolute_difference_from_finest"]["S_mean"] == 0.0
    assert math.isclose(
        report["summaries"][0]["paired_difference_from_finest"]["S_mean"]["mean"],
        -0.1,
        abs_tol=1e-12,
    )


def test_mobile_report_discloses_soft_core_penetration():
    rows = []
    for seed, minimum in ((17, 0.40), (29, 0.42)):
        rows.append({
            "graph_seed": seed,
            "cage_stiffness": 1000.0,
            "split_endpoint": 0.55,
            "rms_displacement_endpoint": 0.06,
            "edge_jaccard_mean": 0.94,
            "initial_edges_retained_mean": 0.98,
            "minimum_separation_mean": minimum,
            "position_force_clip_fraction": 0.0,
        })
    report = mobile_report(rows, core_diameter=0.55)
    assert report["minimum_tested_passing_stiffness"] == 1000.0
    assert report["summaries"][0]["soft_core_penetration_observed"] is True


def test_independent_order_report_includes_decode_contrast():
    rows = []
    for field in (1.0, 8.0):
        for mode, readout, accuracy in (
            ("partitioned", 0.01, 0.51),
            ("contested", 0.60 if field == 8.0 else 0.02, 0.98 if field == 8.0 else 0.52),
        ):
            for seed in (17, 29):
                rows.append({
                    "graph_seed": seed,
                    "field": field,
                    "mode": mode,
                    "contest_fraction_requested": 0.25,
                    "noise_mode": "independent",
                    "terminal_order_readout": readout,
                    "terminal_decode_accuracy_zero_threshold": accuracy,
                    "terminal_decode_d_prime": 3.0 if accuracy > 0.9 else 0.1,
                })
    report = independent_order_report(rows)
    assert report["row_count"] == 8
    assert report["highest_field"] == 8.0
    assert report["highest_field_contested_minus_partitioned"]["mean"] > 0.5
