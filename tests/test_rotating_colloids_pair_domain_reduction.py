import math
import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_rotating_colloids_pair_domain_reduction import (  # noqa: E402
    block_labels,
    induced_couplings,
)
from rotating_colloids_hyperion_case import make_graph  # noqa: E402


def test_domain_reduction_matches_cross_edge_rotor_energy() -> None:
    n = 32
    cluster_size = 8
    epsilon = 1.614286
    _, src, tgt, phi, weights, _ = make_graph(
        n,
        graph_mode="mosaic",
        cluster_size=cluster_size,
        crosslink_k=2,
        crosslink_weight=0.18,
        patch_angle_step=math.pi / 4.0,
    )
    labels, axes, blocks_per_side = block_labels(n, cluster_size)
    cross = labels[src] != labels[tgt]
    couplings, _, _ = induced_couplings(
        src=src,
        tgt=tgt,
        phi=phi,
        weights=weights,
        labels=labels,
        axes=axes,
        epsilon=epsilon,
    )

    rng = np.random.default_rng(7)
    for _ in range(32):
        spins = rng.choice((-1, 1), size=blocks_per_side**2)
        theta = axes + (spins[labels] < 0) * (math.pi / 2.0)
        rotor_energy = -np.sum(
            weights[cross]
            * (
                np.cos(2.0 * (theta[src[cross]] - theta[tgt[cross]]))
                + epsilon
                * np.cos(
                    2.0
                    * (
                        theta[src[cross]]
                        + theta[tgt[cross]]
                        - 2.0 * phi[cross]
                    )
                )
            )
        )
        reduced_energy = -sum(
            coupling * spins[block_i] * spins[block_j]
            for (block_i, block_j), coupling in couplings.items()
        )
        assert math.isclose(rotor_energy, reduced_energy, rel_tol=0.0, abs_tol=1e-12)


def test_reported_graph_has_frustrated_domain_loops_and_metastable_states() -> None:
    from analyze_rotating_colloids_pair_domain_reduction import (  # noqa: E402
        enumerate_landscape,
        plaquette_frustration,
    )

    n = 32
    cluster_size = 8
    _, src, tgt, phi, weights, _ = make_graph(
        n,
        graph_mode="mosaic",
        graph_seed=12345,
        cluster_size=cluster_size,
        crosslink_k=2,
        crosslink_weight=0.18,
        patch_angle_step=math.pi / 4.0,
    )
    labels, axes, blocks_per_side = block_labels(n, cluster_size)
    couplings, _, _ = induced_couplings(
        src=src,
        tgt=tgt,
        phi=phi,
        weights=weights,
        labels=labels,
        axes=axes,
        epsilon=1.614286,
    )
    plaquettes = plaquette_frustration(couplings, blocks_per_side)
    landscape = enumerate_landscape(couplings, blocks_per_side**2)

    assert plaquettes["frustrated_plaquettes"] == 3
    assert landscape["single_domain_flip_stable_states"] == 66
    assert landscape["ground_degeneracy"] == 2

