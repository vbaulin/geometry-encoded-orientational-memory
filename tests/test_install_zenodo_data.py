import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_rotating_colloids_release import DATASETS, DERIVED_FILES
from scripts.install_zenodo_data import DIRECTORIES, FILES, install_archive


def make_archive(root: Path) -> Path:
    archive = root / "archive"
    for name in DIRECTORIES:
        folder = archive / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "sample.json").write_text("{}\n")
    for name in FILES:
        path = archive / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    entries = [
        {"path": str(path.relative_to(archive)), "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in archive.rglob("*") if path.is_file()
    ]
    (archive / "manifest.json").write_text(json.dumps(
        {"schema_version": 2, "complete": True, "files": entries}))
    return archive


def test_installer_covers_all_release_destinations():
    mappings = {**DIRECTORIES, **FILES}
    for source, destination in {**DATASETS, **DERIVED_FILES}.items():
        if mappings.get(source) == destination:
            continue
        # A derived copy can already be present within an installed raw tree.
        assert any(destination.is_relative_to(parent) for parent in DIRECTORIES.values()), source


@pytest.mark.parametrize("mode", ["symlink", "copy"])
def test_install_complete_archive(tmp_path, mode):
    archive = make_archive(tmp_path)
    repository = tmp_path / "repository"
    installed = install_archive(archive, repository, mode, False)
    assert len(installed) == len(DIRECTORIES) + len(FILES)
    for source, destination in {**DIRECTORIES, **FILES}.items():
        path = repository / destination
        assert path.exists()
        assert path.is_symlink() == (mode == "symlink" and source.startswith("raw/"))
        if path.is_file():
            assert path.read_bytes() == (archive / source).read_bytes()


def test_incomplete_source_plan_leaves_repository_untouched(tmp_path):
    archive = make_archive(tmp_path)
    source = next(iter(FILES))
    (archive / source).unlink()
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["files"] = [entry for entry in manifest["files"] if entry["path"] != source]
    (archive / "manifest.json").write_text(json.dumps(manifest))
    repository = tmp_path / "repository"
    with pytest.raises(FileNotFoundError, match="archive sources are missing"):
        install_archive(archive, repository, "symlink", False)
    assert not repository.exists()


def test_changed_data_is_rejected_before_installation(tmp_path):
    archive = make_archive(tmp_path)
    (archive / next(iter(FILES))).write_text("[]\n")
    repository = tmp_path / "repository"
    with pytest.raises(ValueError, match="checksum or size mismatch"):
        install_archive(archive, repository, "copy", False)
    assert not repository.exists()


def test_force_can_reinstall_links_without_removing_archive(tmp_path):
    archive = make_archive(tmp_path)
    repository = tmp_path / "repository"
    first = install_archive(archive, repository, "symlink", False)
    second = install_archive(archive, repository, "symlink", True)
    assert first == second
    assert (archive / next(iter(FILES))).is_file()


def test_derived_reports_can_change_without_modifying_deposit(tmp_path):
    archive = make_archive(tmp_path)
    repository = tmp_path / "repository"
    install_archive(archive, repository, "symlink", False)
    for source, destination in FILES.items():
        original = (archive / source).read_bytes()
        (repository / destination).write_text('{"regenerated": true}\n')
        assert (archive / source).read_bytes() == original
    source = "derived/quantitative_claim_audit"
    (repository / DIRECTORIES[source] / "sample.json").write_text('{"regenerated": true}\n')
    assert (archive / source / "sample.json").read_text() == "{}\n"


def test_existing_destination_does_not_cause_partial_installation(tmp_path):
    archive = make_archive(tmp_path)
    repository = tmp_path / "repository"
    existing = repository / list(FILES.values())[-1]
    existing.parent.mkdir(parents=True)
    existing.write_text("user result\n")
    with pytest.raises(FileExistsError, match="destinations exist"):
        install_archive(archive, repository, "copy", False)
    assert existing.read_text() == "user result\n"
    assert not (repository / next(iter(DIRECTORIES.values()))).exists()
