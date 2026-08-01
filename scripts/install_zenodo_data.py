#!/usr/bin/env python3
"""Install an extracted Zenodo data deposit into the repository layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


DATA_ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")
DIRECTORIES = {
    "raw/capillary_pair_publication_runs": DATA_ROOT / "rotating_colloids_capillary_pair_prl_gpu",
    "raw/capillary_internal_correlations": DATA_ROOT / "rotating_colloids_capillary_pair_prl_internal",
    "raw/equilibrium_replica_discriminant": DATA_ROOT / "rotating_colloids_spin_glass_prl_gpu",
    "raw/activated_memory": DATA_ROOT / "rotating_colloids_activated_memory_prl_gpu",
    "raw/grooved/uniform_scan_n16": DATA_ROOT / "rotating_colloids_grooved_uniform_scan_n16",
    "raw/grooved/uniform_memory_zoom_n16": DATA_ROOT / "rotating_colloids_grooved_uniform_memory_zoom_n16",
    "raw/grooved/uniform_finite_size": DATA_ROOT / "rotating_colloids_grooved_uniform_finite_size",
    "raw/grooved/long_range_disorder": DATA_ROOT / "rotating_colloids_grooved_longrange_disorder",
    "raw/grooved/triangular_frustrated_n16": DATA_ROOT / "rotating_colloids_grooved_triangular_frustrated_n16",
    "raw/grooved/mosaic_hidden_search_n32": DATA_ROOT / "rotating_colloids_grooved_mosaic_hidden_search_n32",
    "raw/grooved/protocols_n16_validation": DATA_ROOT / "rotating_colloids_grooved_protocols_n16_validation",
    "raw/grooved/protocols_quick_mosaic": DATA_ROOT / "rotating_colloids_grooved_protocols_quick_mosaic",
    "derived/quantitative_claim_audit": DATA_ROOT / "rotating_colloids_capillary_pair_prl_claim_audit",
}
FILES = {
    "derived/figures/capillary_regime_report.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/capillary_regime_report.json")
    ),
    "derived/figures/activated_memory_figure_report.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/activated_memory_figure_report.json")
    ),
    "derived/figures/groove_evidence_summary.json": (
        Path("tex/rotating_colloids/grooved_prl_figures/groove_evidence_summary.json")
    ),
}


def install(source: Path, destination: Path, mode: str, force: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(f"destination exists: {destination}")
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Extracted Zenodo data directory")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    repository = args.repo_root.resolve()
    manifest_path = archive / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing archive manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise SystemExit(
            "archive manifest is incomplete: "
            + ", ".join(manifest.get("missing_required_sources", []))
        )

    installed = []
    for source_name, destination_name in {**DIRECTORIES, **FILES}.items():
        source = archive / source_name
        if not source.exists():
            raise SystemExit(f"archive source is missing: {source}")
        destination = repository / destination_name
        install(source, destination, args.mode, args.force)
        installed.append(str(destination.relative_to(repository)))
    print(json.dumps({"mode": args.mode, "installed": installed}, indent=2))


if __name__ == "__main__":
    main()
