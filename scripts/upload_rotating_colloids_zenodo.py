#!/usr/bin/env python3
"""Create, resume, verify, and optionally publish a Zenodo data deposition.

The release directory is first checked against its manifest. It is then packed
into one tar.gz file so the internal directory hierarchy is preserved and the
Zenodo record contains fewer than 100 files. A state file records the deposit
ID and bucket URL immediately after creation. Rerunning the same command uses
that draft and skips remote files whose size and MD5 checksum already match.

Publishing is deliberately separate: add ``--publish`` only after inspecting
the draft URL printed by the upload command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any, Iterable
from urllib.parse import quote


DEFAULT_API = "https://zenodo.org/api"
DEFAULT_RELEASE = Path("release/zenodo_geometry_encoded_orientational_memory")
DOCUMENT_FILES = (
    "README.md",
    "DATA_DICTIONARY.md",
    "manifest.json",
    "SHA256SUMS",
    "zenodo_metadata.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"").strip("'")
        if name and name not in os.environ:
            os.environ[name] = value


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(release_dir: Path) -> dict[str, Any]:
    manifest_path = release_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing release manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if int(manifest.get("schema_version", 0)) < 2:
        raise ValueError(
            "release manifest predates the mandatory submission-validation "
            "contract; rebuild it with the current release builder"
        )
    if not manifest.get("complete"):
        missing = manifest.get("missing_required_sources", [])
        raise ValueError(f"release manifest is incomplete: {missing}")
    forbidden = [
        item["path"]
        for item in manifest.get("files", [])
        if Path(str(item["path"])).suffix.lower()
        in {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
    ]
    if forbidden:
        raise ValueError(f"publication media present in data release: {forbidden}")
    missing_files = [
        item["path"]
        for item in manifest.get("files", [])
        if not (release_dir / str(item["path"])).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"manifest files absent from release: {missing_files}")
    return manifest


def build_archive(release_dir: Path, archive: Path, *, rebuild: bool) -> Path:
    if archive.exists() and not rebuild:
        return archive
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix(archive.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz", compresslevel=6) as handle:
        handle.add(release_dir, arcname=release_dir.name, recursive=True)
    temporary.replace(archive)
    return archive


def upload_assets(release_dir: Path, archive: Path) -> list[Path]:
    documents = [release_dir / name for name in DOCUMENT_FILES]
    missing = [str(path) for path in documents if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release documents missing: {missing}")
    return [archive, *documents]


def normalized_remote_files(deposition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = {}
    for item in deposition.get("files", []):
        name = item.get("filename") or item.get("key")
        if name:
            files[str(name)] = item
    return files


def remote_matches(path: Path, item: dict[str, Any]) -> bool:
    remote_size = item.get("filesize", item.get("size"))
    if remote_size is None or int(remote_size) != path.stat().st_size:
        return False
    checksum = str(item.get("checksum", ""))
    if checksum.startswith("md5:"):
        checksum = checksum.split(":", 1)[1]
    return bool(checksum) and checksum.lower() == file_digest(path, "md5")


def request_json(response: Any, action: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"body": response.text[:2000]}
    if not response.ok:
        raise RuntimeError(
            f"Zenodo {action} failed with HTTP {response.status_code}: {payload}"
        )
    return payload


def create_or_resume(
    session: Any,
    api_url: str,
    state_path: Path,
    *,
    create: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state_path.exists():
        state = read_json(state_path)
        self_url = state["links"]["self"]
        deposition = request_json(
            session.get(self_url, timeout=(30, 180)), "read draft"
        )
        return state, deposition
    if not create:
        raise FileNotFoundError(
            f"no upload state at {state_path}; pass --create to make a new draft"
        )
    response = session.post(
        f"{api_url.rstrip('/')}/deposit/depositions",
        json={},
        timeout=(30, 180),
    )
    deposition = request_json(response, "create draft")
    state = {
        "api_url": api_url.rstrip("/"),
        "deposition_id": deposition["id"],
        "concept_record_id": deposition.get("conceptrecid"),
        "links": deposition["links"],
        "published": False,
    }
    write_json(state_path, state)
    return state, deposition


def update_metadata(session: Any, deposition: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    response = session.put(
        deposition["links"]["self"],
        json={"metadata": metadata},
        timeout=(30, 180),
    )
    return request_json(response, "update metadata")


def upload_files(
    session: Any,
    deposition: dict[str, Any],
    assets: Iterable[Path],
) -> dict[str, Any]:
    remote = normalized_remote_files(deposition)
    bucket = deposition["links"]["bucket"].rstrip("/")
    for path in assets:
        existing = remote.get(path.name)
        if existing and remote_matches(path, existing):
            print(json.dumps({"event": "upload_skip", "file": path.name}))
            continue
        print(
            json.dumps(
                {
                    "event": "upload_start",
                    "file": path.name,
                    "bytes": path.stat().st_size,
                }
            ),
            flush=True,
        )
        with path.open("rb") as handle:
            response = session.put(
                f"{bucket}/{quote(path.name)}",
                data=handle,
                timeout=(30, 6 * 60 * 60),
            )
        item = request_json(response, f"upload {path.name}")
        if not remote_matches(path, item):
            raise RuntimeError(f"remote checksum or size mismatch for {path.name}")
        print(json.dumps({"event": "upload_complete", "file": path.name}))
    return request_json(
        session.get(deposition["links"]["self"], timeout=(30, 180)),
        "verify draft",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-url", default=DEFAULT_API)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--rebuild-archive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    release_dir = args.release_dir.resolve()
    try:
        manifest = verify_release(release_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    archive = (
        args.archive.resolve()
        if args.archive
        else release_dir.parent / f"{release_dir.name}.tar.gz"
    )
    state_path = (
        args.state_file.resolve()
        if args.state_file
        else release_dir.parent / f".{release_dir.name}.zenodo-state.json"
    )
    build_archive(release_dir, archive, rebuild=args.rebuild_archive)
    assets = upload_assets(release_dir, archive)
    summary = {
        "release_complete": True,
        "manifest_files": int(manifest.get("file_count", len(manifest.get("files", [])))),
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": file_digest(archive, "sha256"),
        "upload_files": [path.name for path in assets],
        "state_file": str(state_path),
        "api_url": args.api_url.rstrip("/"),
    }
    if args.dry_run:
        print(json.dumps({"event": "dry_run", **summary}, indent=2, sort_keys=True))
        return 0

    load_env(args.env_file)
    token = os.environ.get("ZENODO_KEY", "").strip()
    if not token:
        raise RuntimeError(
            f"ZENODO_KEY is not set and was not found in {args.env_file}"
        )
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("install requests>=2.28 before uploading") from exc

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    state, deposition = create_or_resume(
        session, args.api_url, state_path, create=args.create
    )
    metadata = read_json(release_dir / "zenodo_metadata.json")
    deposition = update_metadata(session, deposition, metadata)
    deposition = upload_files(session, deposition, assets)
    state.update(
        {
            "links": deposition["links"],
            "draft_html": deposition["links"].get("html"),
            "prereserved_doi": deposition.get("metadata", {})
            .get("prereserve_doi", {})
            .get("doi"),
            "archive_sha256": summary["archive_sha256"],
            "uploaded_files": sorted(normalized_remote_files(deposition)),
        }
    )
    write_json(state_path, state)

    if args.publish:
        response = session.post(deposition["links"]["publish"], timeout=(30, 300))
        deposition = request_json(response, "publish draft")
        state.update(
            {
                "published": True,
                "record_id": deposition.get("id"),
                "doi": deposition.get("doi"),
                "record_html": deposition.get("links", {}).get("html"),
            }
        )
        write_json(state_path, state)
        print(
            json.dumps(
                {
                    "event": "published",
                    "record_id": state.get("record_id"),
                    "doi": state.get("doi"),
                    "url": state.get("record_html"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "event": "draft_ready",
                    "deposition_id": state["deposition_id"],
                    "doi": state.get("prereserved_doi"),
                    "url": state.get("draft_html"),
                    "state_file": str(state_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
