#!/usr/bin/env python3
"""Install an extracted Zenodo data deposit into the repository layout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

try:
    from .upload_rotating_colloids_zenodo import verify_release
except ImportError:
    from upload_rotating_colloids_zenodo import verify_release


DATA_ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")
DIRECTORIES = {
    "raw/capillary_pair_publication_runs": DATA_ROOT / "rotating_colloids_capillary_pair_prl_gpu",
    "raw/capillary_internal_correlations": DATA_ROOT / "rotating_colloids_capillary_pair_prl_internal",
    "raw/equilibrium_replica_discriminant": DATA_ROOT / "rotating_colloids_spin_glass_prl_gpu",
    "raw/activated_memory": DATA_ROOT / "rotating_colloids_activated_memory_prl_gpu",
    "raw/disorder_retention": DATA_ROOT / "rotating_colloids_disorder_retention_protocols",
    "raw/order_of_operations/N144": DATA_ROOT / "rotating_colloids_operation_order_memory_n12",
    "raw/order_of_operations/N256": DATA_ROOT / "rotating_colloids_operation_order_memory_n16",
    "raw/order_of_operations/support_fraction_N144": DATA_ROOT
    / "rotating_colloids_operation_order_memory_fraction_n12",
    "raw/submission_validations": DATA_ROOT / "rotating_colloids_submission_validations",
    "raw/holonomy_matched_release_crossover": DATA_ROOT / "holonomy_matched_release_crossover",
    "raw/matched_preparation_N1024": DATA_ROOT / "rotating_colloids_matched_preparation_prl",
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
    "derived/figures/matched_preparation_report.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/figS_matched_preparation.json")
    ),
    "derived/figures/capillary_regime_report.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/capillary_regime_report.json")
    ),
    "derived/figures/activated_memory_figure_report.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/activated_memory_figure_report.json")
    ),
    "derived/figures/capillary_prl_figure_summary.json": (
        Path("tex/rotating_colloids/capillary_prl_figures/capillary_prl_figure_summary.json")
    ),
    "derived/figures/groove_evidence_summary.json": (
        Path("tex/rotating_colloids/grooved_prl_figures/groove_evidence_summary.json")
    ),
    "derived/disorder/rotating_colloids_disorder_retention_summary.json": (
        DATA_ROOT / "rotating_colloids_disorder_retention_summary.json"
    ),
    "derived/holonomy/holonomy_causality.json": (
        DATA_ROOT / "holonomy_causality/holonomy_causality.json"
    ),
    "derived/holonomy/continuous_holonomy_memory.json": (
        DATA_ROOT / "continuous_holonomy_memory/continuous_holonomy_memory.json"
    ),
    "derived/holonomy/holonomy_memory_intervention_beta1_replication.json": (
        DATA_ROOT / "holonomy_memory_intervention/holonomy_memory_intervention_beta1_replication.json"
    ),
    "derived/order_of_operations/relaxed_exchange_order_minimal.json": (
        DATA_ROOT / "relaxed_exchange_order_minimal.json"
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


def install_archive(archive: Path, repository: Path, mode: str, force: bool) -> list[str]:
    archive, repository = archive.resolve(), repository.resolve()
    verify_release(archive)
    mappings = {**DIRECTORIES, **FILES}
    missing = [name for name in DIRECTORIES if not (archive / name).is_dir()]
    missing.extend(name for name in FILES if not (archive / name).is_file())
    if missing:
        raise FileNotFoundError("archive sources are missing: " + ", ".join(missing))
    destinations = [repository / name for name in mappings.values()]
    # Check the complete plan before creating links or replacing user-selected data.
    conflicts = [str(path) for path in destinations if path.exists() or path.is_symlink()]
    if conflicts and not force:
        raise FileExistsError("destinations exist: " + ", ".join(conflicts))
    for source_name, destination_name in mappings.items():
        source = archive / source_name
        destination = repository / destination_name
        if not destination.is_symlink() and source.resolve().is_relative_to(destination.resolve()):
            raise ValueError(f"installation would replace its own archive source: {source}")
    installed = []
    for source_name, destination_name in mappings.items():
        destination = repository / destination_name
        # Analysis rewrites derived reports; keep the deposited originals intact.
        effective_mode = "copy" if source_name.startswith("derived/") else mode
        install(archive / source_name, destination, effective_mode, force)
        installed.append(str(destination.relative_to(repository)))
    return installed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="Extracted Zenodo data directory")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        installed = install_archive(args.archive, args.repo_root, args.mode, args.force)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"mode": args.mode, "installed": installed}, indent=2))


if __name__ == "__main__":
    main()
