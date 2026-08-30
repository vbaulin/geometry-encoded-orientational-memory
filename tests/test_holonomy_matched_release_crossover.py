import json

from scripts.analyze_holonomy_matched_release_crossover import per_coupling, read_rows
from scripts.test_holonomy_matched_release_crossover import make_parser, run


def test_matched_release_driver_uses_identical_preparation(tmp_path):
    args = make_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--n",
            "4",
            "--cluster-size",
            "2",
            "--graph-seeds",
            "17",
            "--beta-j-values",
            "0.8",
            "--random-target-count",
            "0",
            "--flip-shell-sizes",
            "",
            "--replicas",
            "2",
            "--release-steps",
            "2",
            "--stride",
            "1",
            "--dt",
            "0.01",
        ]
    )

    manifest = run(args)
    rows, duplicates = read_rows(tmp_path)

    assert manifest["target_count_per_cell"] == 3
    assert len(rows) == 3
    assert duplicates == 0
    assert max(abs(row["initial_overlap_arm_difference"]) for row in rows) < 1e-12
    assert {row["preparation"] for row in rows} == {
        "identical_target_plus_gaussian_perturbation"
    }
    assert all(row["paired_noise"] for row in rows)

    blocks = per_coupling(rows, draws=20, seed=9)
    assert list(blocks) == ["0.8"]
    assert blocks["0.8"]["graphs"] == 1

    # A restart must skip every completed stable key.
    second = run(args)
    assert second["new_rows"] == 0
    assert len(
        [
            json.loads(line)
            for line in (tmp_path / "matched_release_scan.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    ) == 3
