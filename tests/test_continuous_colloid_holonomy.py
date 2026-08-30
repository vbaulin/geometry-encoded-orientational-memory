import math
import unittest

import numpy as np

from discovery.continuous_colloid_holonomy import (
    build_frame_intervention,
    gauge_equivalent_phases,
    induced_domain_couplings,
    pair_energy,
    simulate_write_release,
)


class ContinuousColloidHolonomyTests(unittest.TestCase):
    def setUp(self):
        self.src = np.asarray([0, 2, 0, 1, 2], dtype=np.int64)
        self.tgt = np.asarray([1, 3, 2, 3, 3], dtype=np.int64)
        self.phi = np.asarray([0.0, 0.0, 0.17, 0.41, 0.73])
        self.weights = np.asarray([1.0, 1.0, 0.4, 0.35, 0.25])
        self.labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        self.axes = np.asarray([0.0, 0.0, math.pi / 4.0, math.pi / 4.0])

    def test_frame_intervention_changes_only_cross_domain_frames(self):
        intervention = build_frame_intervention(
            src=self.src,
            tgt=self.tgt,
            phi=self.phi,
            weights=self.weights,
            labels=self.labels,
            axes=self.axes,
            j_align=1.0,
            g_capillary=2.5,
            domain_count=2,
        )
        self.assertTrue(np.allclose(intervention.frame_shift[:2], 0.0))
        self.assertTrue(np.all(np.abs(intervention.frame_shift[2:]) <= math.pi / 4.0 + 1e-12))
        recalculated = induced_domain_couplings(
            src=self.src,
            tgt=self.tgt,
            phi=intervention.flat_phi,
            weights=self.weights,
            labels=self.labels,
            axes=self.axes,
            j_align=1.0,
            g_capillary=2.5,
        )
        self.assertEqual(set(recalculated), set(intervention.realized_couplings))
        for edge in recalculated:
            self.assertAlmostEqual(recalculated[edge], intervention.realized_couplings[edge], places=10)

    def test_frame_intervention_flattens_a_frustrated_cycle_at_fixed_magnitude(self):
        src = np.asarray([0, 1, 2, 3], dtype=np.int64)
        tgt = np.asarray([1, 2, 3, 0], dtype=np.int64)
        phi = np.asarray([math.pi / 4.0, 0.0, 0.0, 0.0])
        weights = np.ones(4)
        labels = np.arange(4, dtype=np.int64)
        axes = np.zeros(4)
        intervention = build_frame_intervention(
            src=src,
            tgt=tgt,
            phi=phi,
            weights=weights,
            labels=labels,
            axes=axes,
            j_align=0.2,
            g_capillary=2.0,
            domain_count=4,
        )
        self.assertEqual(intervention.negative_flux_original, 1)
        self.assertEqual(intervention.negative_flux_realized, 0)
        self.assertLess(intervention.magnitude_relative_l1, 1e-12)
        self.assertLess(intervention.magnitude_relative_max, 1e-12)

    def test_gauge_phase_transform_preserves_energy(self):
        gauge = np.asarray([1, -1], dtype=np.int8)
        chi, transformed_phi, eta = gauge_equivalent_phases(
            src=self.src,
            tgt=self.tgt,
            phi=self.phi,
            labels=self.labels,
            gauge=gauge,
        )
        rng = np.random.default_rng(17)
        theta = rng.uniform(0.0, math.pi, size=(8, 4))
        original = pair_energy(
            theta,
            src=self.src,
            tgt=self.tgt,
            phi=self.phi,
            weights=self.weights,
            j_align=1.3,
            g_capillary=0.8,
        )
        transformed = pair_energy(
            theta + eta[None, :],
            src=self.src,
            tgt=self.tgt,
            phi=transformed_phi,
            weights=self.weights,
            j_align=1.3,
            g_capillary=0.8,
            chi=chi,
        )
        np.testing.assert_allclose(original, transformed, atol=1e-11, rtol=0.0)

    def test_gauge_phase_transform_preserves_write_release_curve(self):
        gauge = np.asarray([1, -1], dtype=np.int8)
        chi, transformed_phi, eta = gauge_equivalent_phases(
            src=self.src,
            tgt=self.tgt,
            phi=self.phi,
            labels=self.labels,
            gauge=gauge,
        )
        rng = np.random.default_rng(23)
        initial = rng.uniform(0.0, math.pi, size=(4, 4))
        target = np.asarray([0.1, 0.1, 0.8, 0.8])
        common = dict(
            src=self.src,
            tgt=self.tgt,
            weights=self.weights,
            j_align=0.7,
            g_capillary=0.9,
            replicas=4,
            write_steps=8,
            release_steps=12,
            stride=3,
            dt=0.002,
            write_field=2.0,
            noise_seed=31,
        )
        original = simulate_write_release(
            **common,
            phi=self.phi,
            target=target,
            initial_theta=initial,
        )
        transformed = simulate_write_release(
            **common,
            phi=transformed_phi,
            chi=chi,
            target=target + eta,
            initial_theta=initial + eta[None, :],
        )
        np.testing.assert_allclose(
            original["overlap_curve"], transformed["overlap_curve"], atol=1e-10, rtol=0.0
        )
        self.assertAlmostEqual(original["q_ea"], transformed["q_ea"], places=10)


if __name__ == "__main__":
    unittest.main()
