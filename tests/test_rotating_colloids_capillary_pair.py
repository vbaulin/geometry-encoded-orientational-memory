import math

import numpy as np

from scripts.rotating_colloids_capillary_pair import make_caged_graph, replica_overlap


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

