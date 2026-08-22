import json
from pathlib import Path

import pytest

from scripts.build_rotating_colloids_release import (
    copy_source,
    copy_unique_activated_shards,
    disorder_retention_source,
)
from scripts.plot_rotating_colloids_activated_memory_prl import load_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_activated_rows_deduplicate_exact_copies_and_reject_conflicts(tmp_path: Path) -> None:
    row = {"key": "n=24|graph=17|lambda=0.3", "lambda": 0.3, "value": 1}
    write_jsonl(tmp_path / "a" / "activated_memory_scan.jsonl", [row])
    write_jsonl(tmp_path / "nested" / "a" / "activated_memory_scan.jsonl", [row])

    rows, provenance = load_rows(tmp_path)
    assert rows == [row]
    assert provenance["raw_rows_seen"] == 2
    assert provenance["identical_duplicate_rows_ignored"] == 1

    conflicting = dict(row, value=2)
    write_jsonl(tmp_path / "b" / "activated_memory_scan.jsonl", [conflicting])
    with pytest.raises(SystemExit, match="conflicting activated-memory rows"):
        load_rows(tmp_path)


def test_zenodo_copy_excludes_manuscript_media(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "raw.json").write_text("{}\n", encoding="utf-8")
    (source / "figure.pdf").write_bytes(b"not a real pdf")
    (source / "figure.png").write_bytes(b"not a real png")

    destination = tmp_path / "destination"
    copy_source(source, destination)
    assert (destination / "raw.json").exists()
    assert not (destination / "figure.pdf").exists()
    assert not (destination / "figure.png").exists()


def test_unique_activated_staging_ignores_hash_identical_shards(tmp_path: Path) -> None:
    row = {"key": "n=24|graph=17|lambda=0.3", "lambda": 0.3}
    first = tmp_path / "source" / "seeds_17" / "activated_memory_scan.jsonl"
    second = tmp_path / "source" / "nested" / "seeds_17" / "activated_memory_scan.jsonl"
    write_jsonl(first, [row])
    write_jsonl(second, [row])

    staged = copy_unique_activated_shards([first, second], tmp_path / "archive")
    assert len(list((tmp_path / "archive").glob("**/activated_memory_scan.jsonl"))) == 1
    assert sum("duplicate_of" in item for item in staged) == 1


def test_disorder_retention_completeness_requires_every_claimed_cell(tmp_path: Path) -> None:
    requirements = {
        (576, 0.05): 3,
        (576, 0.08): 5,
        (576, 0.11): 5,
        (576, 0.16): 5,
        (576, 0.28): 5,
        (1024, 0.11): 5,
        (1024, 0.16): 5,
    }
    for (node_count, sigma), count in requirements.items():
        for seed in range(count):
            path = tmp_path / f"N{node_count}_s{sigma:g}_g{seed}" / "capillary_pair_protocols.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "model": {
                            "graph": {
                                "node_count": node_count,
                                "disorder": sigma,
                                "seed": seed,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

    _, files, validation = disorder_retention_source(tmp_path, tmp_path)
    assert len(files) == sum(requirements.values())
    assert validation["complete"] is True

    files[0].unlink()
    _, _, validation = disorder_retention_source(tmp_path, tmp_path)
    assert validation["complete"] is False
    assert validation["missing_cells"]
