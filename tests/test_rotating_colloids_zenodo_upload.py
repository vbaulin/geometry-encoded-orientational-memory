from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.upload_rotating_colloids_zenodo import (
    build_archive,
    create_new_version,
    remote_matches,
    upload_assets,
    verify_release,
)


def make_release(root: Path, *, complete: bool = True) -> Path:
    release = root / "release"
    release.mkdir()
    data = release / "raw" / "value.jsonl"
    data.parent.mkdir()
    data.write_text('{"value": 1}\n', encoding="utf-8")
    for name in (
        "README.md",
        "DATA_DICTIONARY.md",
        "SHA256SUMS",
        "zenodo_metadata.json",
    ):
        (release / name).write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "complete": complete,
        "file_count": 1,
        "missing_required_sources": [] if complete else ["raw/test"],
        "files": [{"path": "raw/value.jsonl", "bytes": data.stat().st_size}],
    }
    (release / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    return release


def test_release_must_be_complete(tmp_path: Path) -> None:
    release = make_release(tmp_path, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        verify_release(release)


def test_legacy_complete_manifest_is_rejected(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predates"):
        verify_release(release)


def test_archive_and_upload_assets_preserve_release(tmp_path: Path) -> None:
    release = make_release(tmp_path)
    verify_release(release)
    archive = build_archive(release, tmp_path / "release.tar.gz", rebuild=False)
    assets = upload_assets(release, archive)
    assert archive.is_file()
    assert [path.name for path in assets] == [
        "release.tar.gz",
        "README.md",
        "DATA_DICTIONARY.md",
        "manifest.json",
        "SHA256SUMS",
        "zenodo_metadata.json",
    ]


def test_remote_match_requires_size_and_md5(tmp_path: Path) -> None:
    path = tmp_path / "data.tar.gz"
    path.write_bytes(b"zenodo-data")
    checksum = hashlib.md5(path.read_bytes()).hexdigest()
    assert remote_matches(
        path,
        {"filesize": path.stat().st_size, "checksum": f"md5:{checksum}"},
    )
    assert not remote_matches(
        path,
        {"filesize": path.stat().st_size + 1, "checksum": f"md5:{checksum}"},
    )


def test_create_new_version_follows_latest_draft_and_rewrites_state(
    tmp_path: Path,
) -> None:
    class Response:
        ok = True
        status_code = 201
        text = ""

        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def json(self) -> dict:
            return self.payload

    class Session:
        def post(self, url: str, timeout: tuple[int, int]) -> Response:
            assert url.endswith("/22173161/actions/newversion")
            return Response(
                {"links": {"latest_draft": "https://zenodo.test/deposit/22180000"}}
            )

        def get(self, url: str, timeout: tuple[int, int]) -> Response:
            assert url == "https://zenodo.test/deposit/22180000"
            return Response(
                {
                    "id": 22180000,
                    "conceptrecid": "22170000",
                    "links": {
                        "self": url,
                        "html": "https://zenodo.test/deposit/22180000",
                    },
                }
            )

    state_path = tmp_path / "state.json"
    state, draft = create_new_version(
        Session(),
        "https://zenodo.org/api",
        state_path,
        {"deposition_id": 22173161, "concept_record_id": "22170000"},
        {
            "id": 22173161,
            "record_id": 22173161,
            "doi": "10.5281/zenodo.22173161",
            "links": {},
        },
    )
    assert draft["id"] == 22180000
    assert state["deposition_id"] == 22180000
    assert state["previous_record_id"] == 22173161
    assert state["previous_doi"] == "10.5281/zenodo.22173161"
    assert json.loads(state_path.read_text(encoding="utf-8")) == state
