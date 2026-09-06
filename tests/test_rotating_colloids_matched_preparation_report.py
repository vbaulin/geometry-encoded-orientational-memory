import numpy as np
import pytest

from scripts.plot_rotating_colloids_matched_preparation import (
    endpoint_overlaps, graph_statistics, load_cases,
)
from scripts.build_rotating_colloids_release import copy_source, release_file_entries


def test_graph_sem_uses_graph_means_not_thermal_replica_count():
    mean, sem = graph_statistics([[1, 2], [3, 4], [5, 6]])
    np.testing.assert_allclose(mean, [3, 4])
    np.testing.assert_allclose(sem, [2 / np.sqrt(3)] * 2)
    with pytest.raises(ValueError):
        graph_statistics([[1, 2]])
    with pytest.raises(ValueError):
        graph_statistics([[1, 2], [np.nan, 4]])


def test_endpoint_connection_removes_common_director():
    target = np.full(13, 0.1)
    theta = np.full((5, 48, 13), 0.3)
    result = endpoint_overlaps(theta, target, target)
    np.testing.assert_allclose(result["Q"], np.cos(0.4))
    np.testing.assert_allclose(result["Q_connected"], 0, atol=1e-14)


def test_balanced_rotated_pattern_has_real_and_aligned_overlaps():
    target = np.linspace(0, np.pi, 16, endpoint=False)
    result = endpoint_overlaps((target + 0.2)[None, None, :], target, target)
    np.testing.assert_allclose(result["Q_connected"], np.cos(0.4))
    np.testing.assert_allclose(result["Q_rotation_aligned"], 1)


def test_empty_publication_grid_is_not_a_complete_run(tmp_path):
    with pytest.raises(ValueError, match="Incomplete case grid"):
        load_cases(tmp_path)


def test_release_excludes_media_manuscript_and_runtime_locks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("release.npz", "summary.json", "manifest.json", "figure.pdf",
                 "figure.png", "paper.tex", ".run.lock", "release.partial.npz"):
        (source / name).write_bytes(b"test")
    destination = tmp_path / "deposit"
    copy_source(source, destination)
    assert {p.name for p in destination.iterdir()} == {"release.npz", "summary.json", "manifest.json"}


def test_per_case_manifests_are_hashed_but_root_manifest_is_not(tmp_path):
    import hashlib

    nested = tmp_path / "raw" / "case"
    nested.mkdir(parents=True)
    for name in ("manifest.json", "SHA256SUMS"):
        (tmp_path / name).write_bytes(b"root bookkeeping")
        (nested / name).write_bytes(b"case provenance")
    entries = release_file_entries(tmp_path)
    assert {row["path"] for row in entries} == {"raw/case/manifest.json", "raw/case/SHA256SUMS"}
    assert all(row["sha256"] == hashlib.sha256(b"case provenance").hexdigest() for row in entries)
