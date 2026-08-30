from __future__ import annotations

import numpy as np

from discovery.colloid_holonomy_memory import (
    apply_gauge,
    build_intervention,
    exact_landscape,
    fundamental_cycle_fluxes,
    paired_memory_test,
    paired_metastable_retention_test,
)


def frustrated_square() -> dict[tuple[int, int], float]:
    return {
        (0, 1): 1.0,
        (1, 2): 2.0,
        (2, 3): 1.5,
        (0, 3): -0.75,
    }


def test_cycle_product_is_gauge_invariant() -> None:
    couplings = frustrated_square()
    transformed = apply_gauge(couplings, np.asarray([1, -1, 1, -1]))
    assert fundamental_cycle_fluxes(couplings, 4) == (-1,)
    assert fundamental_cycle_fluxes(transformed, 4) == (-1,)


def test_flat_intervention_changes_only_sign_holonomy() -> None:
    intervention = build_intervention(frustrated_square(), 4, seed=7)
    assert intervention.original_fluxes == (-1,)
    assert intervention.flat_fluxes == (1,)
    assert {
        edge: abs(value) for edge, value in intervention.original.items()
    } == {edge: abs(value) for edge, value in intervention.flat.items()}


def test_gauge_equivalent_landscapes_are_identical() -> None:
    intervention = build_intervention(frustrated_square(), 4, seed=13)
    original = exact_landscape(intervention.original, 4)
    control = exact_landscape(intervention.gauge_equivalent, 4)
    assert original == control


def test_common_random_numbers_make_dynamic_gauge_control_exact() -> None:
    intervention = build_intervention(frustrated_square(), 4, seed=17)
    result = paired_memory_test(
        intervention,
        4,
        replicas=48,
        beta=2.0,
        write_field=2.0,
        write_sweeps=12,
        release_sweeps=20,
        seed=23,
    )
    assert result["gauge_auc_residual"] < 1e-14
    assert result["original"]["release_overlap"] == result["gauge_equivalent_control"]["release_overlap"]


def test_metastable_retention_uses_states_created_by_holonomy() -> None:
    intervention = build_intervention(frustrated_square(), 4, seed=29)
    result = paired_metastable_retention_test(
        intervention,
        4,
        beta=2.0,
        release_sweeps=20,
        repeats_per_state=4,
        seed=31,
    )
    if result["decision"] == "measured":
        assert result["additional_stable_states"] > 0
        assert result["gauge_auc_residual"] < 1e-14
    else:
        assert result["reason"]
