import numpy as np
import torch

from scripts.rotating_colloids_capillary_pair import make_caged_graph
from scripts.rotating_colloids_matched_preparation import coefficients, evolve, overlaps, run_case


def graph():
    return make_caged_graph(4, disorder=0.16, cutoff=2.6, alignment_range=1.35,
                            alignment_decay=0.2, seed=17)


def test_connected_overlap_subtracts_both_complex_directors():
    theta = np.full((2, 1, 3, 16), np.pi / 4)
    target = np.zeros(16)
    result = overlaps(theta, target, target)
    np.testing.assert_allclose(result["Q_connected"], 0, atol=1e-14)
    assert not np.allclose(result["Q"] - result["S"] ** 2, result["Q_connected"])


def test_rigid_rotation_preserves_aligned_pattern_overlap():
    target = np.linspace(0, np.pi, 16, endpoint=False)
    result = overlaps((target + 0.6)[None, None, None], target, target)
    np.testing.assert_allclose(result["Q"], np.cos(1.2), atol=1e-14)
    np.testing.assert_allclose(result["Q_rotation_aligned"], 1, atol=1e-14)
    np.testing.assert_allclose(result["Q_connected_rotation_aligned"], 1, atol=1e-14)


def test_equal_weight_control_preserves_edge_sum():
    a, b = coefficients(graph(), 4, 5)
    np.testing.assert_allclose(a[2], a[0] + b[0])
    np.testing.assert_array_equal(b[1:], 0)


def test_checkpoint_restart_is_identical_and_samples_include_endpoints(tmp_path):
    torch.set_num_threads(1)
    g = graph()
    a, b = coefficients(g, 4, 5)
    initial = np.repeat(np.random.default_rng(123).uniform(0, np.pi, (1, 2, 16)), 3, axis=0)
    kwargs = dict(dt=0.0025, steps=12, seed=57, device=torch.device("cpu"), samples=5, checkpoint_steps=4)
    direct = evolve(g, initial, a, b, path=tmp_path / "direct.npz", **kwargs)
    stopped = evolve(g, initial, a, b, path=tmp_path / "resumed.npz", stop_after=5, **kwargs)
    assert stopped is None
    resumed = evolve(g, initial, a, b, path=tmp_path / "resumed.npz", **kwargs)
    np.testing.assert_array_equal(direct["theta"], resumed["theta"])
    np.testing.assert_array_equal(direct["theta"][0], initial.astype(np.float32))
    np.testing.assert_array_equal(direct["theta"][-1], direct["terminal_theta"])
    assert direct["time"][0] == 0
    assert direct["time"][-1] == 12 * 0.0025


def test_case_uses_same_release_start_for_all_matched_arms(tmp_path):
    torch.set_num_threads(1)
    config = {"schema": 1, "n": 4, "graph_seed": 17, "disorder": 0.16,
              "coupling_scale": 1.0, "target": "relaxed", "replicas": 2,
              "dt": 0.0025, "equilibration_steps": 3, "write_steps": 4,
              "release_steps": 6, "write_field": 1.5}
    run_case(config, tmp_path, torch.device("cpu"), 3)
    with np.load(tmp_path / "preparation.npz") as saved:
        states = saved["release_initial"]
    np.testing.assert_array_equal(states[0], states[1])
    np.testing.assert_array_equal(states[0], states[2])
    with np.load(tmp_path / "release.npz") as saved:
        np.testing.assert_array_equal(saved["theta"][0], states)
        assert not np.array_equal(saved["theta"][-1, 0], saved["theta"][-1, 1])
def test_connected_overlap_obeys_director_bound():
    import numpy as np
    from scripts.rotating_colloids_matched_preparation import overlaps

    rng = np.random.default_rng(27)
    for _ in range(40):
        theta = rng.uniform(0, np.pi, (4, 2, 19))
        target = rng.uniform(0, np.pi, 19)
        result = overlaps(theta, target, target)
        assert np.all(result["Q_connected"] >= result["Q"] - result["S"] - 1e-14)
        assert np.all(result["Q_connected"] <= result["Q"] + result["S"] + 1e-14)


def test_legacy_disorder_subtraction_requires_explicit_choice():
    import pytest
    from scripts.analyze_rotating_colloids_disorder_protocols import summarize

    write = {"release_time": [0, 1], "release_S": [0.3, 0.3],
             "release_overlap": [0.8, 0.5], "write_overlap": [0.8], "release_G2": [0.4, 0.4]}
    split = {"time": [0, 1], "overlap_mean": [1, 0.6]}
    run = {"write_release": write, "split_replica": split,
           "no_capillary_write_release": write, "no_capillary_split_replica": split}
    with pytest.raises(ValueError, match="Exact connected overlap"):
        summarize(run, 0.1)
    assert summarize(run, 0.1, legacy_director_subtraction=True)["connected_write_end"] == pytest.approx(0.41)
    write["release_connected_overlap"] = [0.7, 0.37]
    split["overlap_connected_mean"] = [0.8, 0.42]
    assert summarize(run, 0.1)["connected_write_end"] == pytest.approx(0.37)
    assert summarize(run, 0.1)["connected_split_end"] == pytest.approx(0.42)
