import math

import numpy as np

import torch

from scripts.rotating_colloids_capillary_pair import (
    _graph_tensors,
    _torque,
    make_caged_graph,
    replica_overlap,
    simulate_ensemble,
)
from scripts.test_rotating_colloids_operation_order_memory import masks


def test_capillary_bond_frame_harmonic_matches_relative_angle_form():
    theta_i = 0.37
    theta_j = 1.14
    phi_ij = -0.42
    relative_i = theta_i - phi_ij
    relative_j = theta_j - phi_ij
    pair_form = math.cos(2.0 * (theta_i + theta_j - 2.0 * phi_ij))
    capillary_form = math.cos(2.0 * relative_i + 2.0 * relative_j)
    assert math.isclose(pair_form, capillary_form, rel_tol=0.0, abs_tol=1e-14)


def test_capillary_weights_follow_inverse_fourth_power():
    graph = make_caged_graph(
        8,
        disorder=0.0,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    r0 = float(graph.metadata["nearest_spacing_median"])
    expected = np.clip((r0 / graph.distance) ** 4, 0.0, 6.0)
    np.testing.assert_allclose(graph.capillary_weight, expected, rtol=1e-13, atol=1e-13)
    assert int(graph.metadata["edge_count"]) > 0
    assert float(graph.metadata["minimum_separation"]) > 0.5


def test_independent_random_replica_overlap_has_finite_size_floor_only():
    rng = np.random.default_rng(123)
    theta = rng.uniform(0.0, math.pi, size=(64, 1024))
    overlap = replica_overlap(theta)
    # The magnitude has a Rayleigh finite-N floor of order N^-1/2.
    assert abs(float(overlap["signed_mean"])) < 0.01
    assert float(overlap["magnitude_mean"]) < 0.06


def test_write_weight_localizes_the_external_torque():
    graph = make_caged_graph(
        4,
        disorder=0.0,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    theta = torch.zeros((1, 16), dtype=torch.float32)
    axis = torch.full_like(theta, math.pi / 4.0)
    weight = torch.zeros_like(theta)
    weight[:, :5] = 1.0
    torque = _torque(
        theta,
        _graph_tensors(graph, torch.device("cpu")),
        j_align=0.0,
        g_capillary=0.0,
        write_axis=axis,
        write_field=2.0,
        write_weight=weight,
    )
    np.testing.assert_allclose(torque[0, :5].numpy(), 4.0, atol=1e-6)
    np.testing.assert_allclose(torque[0, 5:].numpy(), 0.0, atol=1e-7)


def test_simulator_exposes_terminal_state_for_chained_protocols():
    graph = make_caged_graph(
        4,
        disorder=0.16,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    result = simulate_ensemble(
        graph,
        j_align=4.0,
        g_capillary=5.0,
        replicas=2,
        burn_in_steps=2,
        sample_steps=3,
        sample_stride=2,
        dt=0.0025,
        seed=23,
        device=torch.device("cpu"),
    )
    terminal = np.asarray(result["state_after_steps"])
    assert terminal.shape == (2, 16)
    assert np.isfinite(terminal).all()


def test_partitioned_write_supports_share_no_particles_or_direct_edges():
    for seed in (17, 29, 43):
        graph = make_caged_graph(
            12,
            disorder=0.16,
            cutoff=2.6,
            alignment_range=1.35,
            alignment_decay=0.20,
            seed=seed,
        )
        first, second = masks(graph.positions, float(graph.box[0]), "partitioned")
        assert not np.any((first > 0) & (second > 0))
        direct = (
            ((first[graph.src] > 0) & (second[graph.tgt] > 0))
            | ((second[graph.src] > 0) & (first[graph.tgt] > 0))
        )
        assert not np.any(direct)


def test_contested_support_contrast_vanishes_at_both_uniform_limits():
    graph = make_caged_graph(
        12,
        disorder=0.16,
        cutoff=2.6,
        alignment_range=1.35,
        alignment_decay=0.20,
        seed=17,
    )
    contrasts = []
    for fraction in (0.0, 0.25, 1.0):
        first, second = masks(
            graph.positions,
            float(graph.box[0]),
            "contested",
            contest_fraction=fraction,
        )
        np.testing.assert_array_equal(first, second)
        contrasts.append(float((first * second).var()))
    assert contrasts[0] == 0.0
    assert contrasts[1] > 0.0
    assert contrasts[2] == 0.0
