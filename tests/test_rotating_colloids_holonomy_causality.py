import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "test_rotating_colloids_holonomy_causality.py"
)
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("_rotating_colloids_holonomy_causality_impl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

count_one_flip_stable_states = MODULE.count_one_flip_stable_states
construct_holonomy_variants = MODULE.construct_holonomy_variants
cycle_bits = MODULE.cycle_bits
exact_controls = MODULE.exact_controls
make_schedule = MODULE.make_schedule
simulate_variants = MODULE.simulate_variants
solve_gf2 = MODULE.solve_gf2
transformed_schedule = MODULE.transformed_schedule


def square_couplings() -> dict[tuple[int, int], float]:
    return {
        (0, 1): 0.8,
        (0, 2): 1.1,
        (1, 3): -0.9,
        (2, 3): 1.3,
    }


def test_gf2_solver_recovers_requested_cycle_parity() -> None:
    matrix = np.asarray([[1, 1, 1, 1]], dtype=np.uint8)
    for rhs in (np.asarray([0], dtype=np.uint8), np.asarray([1], dtype=np.uint8)):
        solution = solve_gf2(matrix, rhs)
        assert np.array_equal((matrix @ solution) & 1, rhs)


def test_holonomy_intervention_preserves_local_unsigned_graph() -> None:
    result = construct_holonomy_variants(square_couplings(), 2, doses=[0, 1])
    assert [row["dose"] for row in result["variants"]] == [0, 1]
    for row in result["variants"]:
        assert int(np.sum(cycle_bits(row["sign_bits"], result["cycle_matrix"]))) == row["dose"]
        assert row["match"]["coupling_magnitudes_max_error"] == 0.0
        assert row["match"]["degree_sequence_max_error"] == 0.0
        assert row["match"]["unsigned_spectrum_max_error"] == 0.0


def test_gauge_transform_leaves_stochastic_memory_trajectory_exact() -> None:
    result = construct_holonomy_variants(square_couplings(), 2, doses=[0, 1])
    matrix = result["variants"][1]["matrix"][None, :, :]
    target = np.ones(4, dtype=np.int8)
    schedule = make_schedule(nodes=4, replicas=24, write_sweeps=20, release_sweeps=40, seed=9)
    original = simulate_variants(
        matrix,
        schedule,
        target=target,
        interaction_scale=1.7,
        write_field=0.8,
    )
    gauge = np.asarray([1, -1, 1, -1], dtype=np.int8)
    transformed_matrix = gauge[None, :, None] * matrix * gauge[None, None, :]
    transformed = simulate_variants(
        transformed_matrix,
        transformed_schedule(schedule, gauge),
        target=target * gauge,
        interaction_scale=1.7,
        write_field=0.8,
    )
    assert np.array_equal(original["overlap_curve"], transformed["overlap_curve"])
    assert np.array_equal(original["q_EA"], transformed["q_EA"])


def test_zero_interaction_and_inversion_controls_are_exact() -> None:
    result = construct_holonomy_variants(square_couplings(), 2, doses=[0, 1])
    matrices = np.stack([row["matrix"] for row in result["variants"]])
    target = np.ones(4, dtype=np.int8)
    schedule = make_schedule(nodes=4, replicas=16, write_sweeps=10, release_sweeps=20, seed=11)
    controls = exact_controls(
        matrices,
        schedule,
        target,
        interaction_scale=2.0,
        write_field=1.0,
        seed=13,
    )
    assert controls["zero_interaction_pass"]
    assert controls["gauge_equivalence_pass"]

    baseline = result["baseline_bits"]
    for row in result["variants"]:
        intervention = baseline ^ row["sign_bits"]
        assert np.array_equal(row["sign_bits"] ^ intervention, baseline)


def test_exact_landscape_capacity_is_gauge_invariant() -> None:
    result = construct_holonomy_variants(square_couplings(), 2, doses=[0, 1])
    matrix = result["variants"][1]["matrix"]
    gauge = np.asarray([1, -1, -1, 1], dtype=np.int8)
    transformed = gauge[:, None] * matrix * gauge[None, :]
    assert count_one_flip_stable_states(matrix) == count_one_flip_stable_states(transformed)
