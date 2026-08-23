#!/usr/bin/env python3
"""Assemble the data deposit for Geometry-Encoded Hidden Orientational Memory.

Large simulation records belong in the Zenodo deposit. The companion GitHub
repository contains code, manuscript sources, and small derived reports only.
The builder refuses to declare a complete deposit when cluster-generated raw
records behind a manuscript claim are absent. Publication figures and PDFs
belong with the manuscript/code release and are deliberately excluded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any


REPOSITORY_NAME = "geometry-encoded-orientational-memory"
DATA_ROOT = Path("discoveries/theory_experiment_interface/rotating_colloids_hyperion")

DATASETS = {
    "raw/capillary_pair_publication_runs": DATA_ROOT / "rotating_colloids_capillary_pair_prl_gpu",
    "raw/capillary_internal_correlations": DATA_ROOT / "rotating_colloids_capillary_pair_prl_internal",
    "raw/equilibrium_replica_discriminant": DATA_ROOT / "rotating_colloids_spin_glass_prl_gpu",
    "raw/order_of_operations/N144": DATA_ROOT / "rotating_colloids_operation_order_memory_n12",
    "raw/order_of_operations/N256": DATA_ROOT / "rotating_colloids_operation_order_memory_n16",
    "raw/order_of_operations/support_fraction_N144": DATA_ROOT
    / "rotating_colloids_operation_order_memory_fraction_n12",
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

# Prospective submission tests are copied when a cluster run has produced
# them. Their absence does not invalidate the present finite-time evidence.
OPTIONAL_DATASETS = {
    "raw/submission_validations": DATA_ROOT / "rotating_colloids_submission_validations",
}

DERIVED_FILES = {
    "derived/figures/capillary_regime_report.json": Path(
        "tex/rotating_colloids/capillary_prl_figures/capillary_regime_report.json"
    ),
    "derived/figures/capillary_prl_figure_summary.json": Path(
        "tex/rotating_colloids/capillary_prl_figures/capillary_prl_figure_summary.json"
    ),
    "derived/figures/activated_memory_figure_report.json": Path(
        "tex/rotating_colloids/capillary_prl_figures/activated_memory_figure_report.json"
    ),
    "derived/figures/groove_evidence_summary.json": Path(
        "tex/rotating_colloids/grooved_prl_figures/groove_evidence_summary.json"
    ),
    "derived/figures/capillary_regime_feature_ablation.json": DATA_ROOT
    / "rotating_colloids_capillary_pair_prl_gpu/phase_diagram/capillary_regime_feature_ablation.json",
    "derived/disorder/rotating_colloids_disorder_retention_summary.json": DATA_ROOT
    / "rotating_colloids_disorder_retention_summary.json",
    "derived/holonomy/holonomy_causality.json": DATA_ROOT / "holonomy_causality/holonomy_causality.json",
    "derived/holonomy/continuous_holonomy_memory.json": DATA_ROOT
    / "continuous_holonomy_memory/continuous_holonomy_memory.json",
    "derived/holonomy/holonomy_memory_intervention_beta1_replication.json": DATA_ROOT
    / "holonomy_memory_intervention/holonomy_memory_intervention_beta1_replication.json",
    "derived/order_of_operations/relaxed_exchange_order_minimal.json": DATA_ROOT
    / "relaxed_exchange_order_minimal.json",
}

EXCLUDED_ARCHIVE_PATTERNS = (
    "__pycache__", "*.pyc", ".DS_Store", "*.pdf", "*.png", "*.jpg", "*.jpeg", "*.svg"
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(*EXCLUDED_ARCHIVE_PATTERNS),
        )
    else:
        if source.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".svg"}:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def activated_source(root: Path, source: Path | None) -> tuple[Path | None, list[Path]]:
    candidate = source
    if candidate is None:
        candidate = root / DATA_ROOT / "rotating_colloids_activated_memory_prl_gpu"
    elif not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if candidate.is_file():
        files = [candidate] if candidate.name == "activated_memory_scan.jsonl" else []
        return (candidate if files else None), files
    if candidate.is_dir():
        files = sorted(candidate.glob("**/activated_memory_scan.jsonl"))
        return (candidate if files else None), files
    return None, []


def resolved_input(root: Path, supplied: Path | None, default_relative: Path) -> Path:
    candidate = supplied if supplied is not None else root / default_relative
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def disorder_retention_source(root: Path, source: Path | None) -> tuple[Path, list[Path], dict[str, Any]]:
    candidate = resolved_input(
        root,
        source,
        DATA_ROOT / "rotating_colloids_disorder_retention_protocols",
    )
    files = sorted(candidate.glob("**/capillary_pair_protocols.json")) if candidate.is_dir() else []
    cells: dict[tuple[int, float], set[int]] = {}
    for path in files:
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))["model"]["graph"]
            key = (int(graph["node_count"]), round(float(graph["disorder"]), 8))
            cells.setdefault(key, set()).add(int(graph["seed"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    required = {
        (576, 0.05): 3,
        (576, 0.08): 5,
        (576, 0.11): 5,
        (576, 0.16): 5,
        (576, 0.28): 5,
        (1024, 0.11): 5,
        (1024, 0.16): 5,
    }
    missing_cells = {
        f"N={node_count},sigma={sigma:g}": {"required": count, "found": len(cells.get((node_count, sigma), set()))}
        for (node_count, sigma), count in required.items()
        if len(cells.get((node_count, sigma), set())) < count
    }
    return candidate, files, {
        "cells": {
            f"N={node_count},sigma={sigma:g}": sorted(seeds)
            for (node_count, sigma), seeds in sorted(cells.items())
        },
        "missing_cells": missing_cells,
        "complete": not missing_cells,
    }


def copy_unique_activated_shards(files: list[Path], destination: Path) -> list[dict[str, Any]]:
    """Stage unique raw shards without copying an accidentally nested project tree."""

    destination.mkdir(parents=True, exist_ok=True)
    seen_hashes: dict[str, Path] = {}
    staged: list[dict[str, Any]] = []
    for path in files:
        digest = sha256(path)
        if digest in seen_hashes:
            staged.append({"source": str(path), "duplicate_of": str(seen_hashes[digest]), "sha256": digest})
            continue
        seen_hashes[digest] = path
        shard_name = path.parent.name
        target_dir = destination / shard_name
        suffix = 2
        while target_dir.exists():
            target_dir = destination / f"{shard_name}_{suffix}"
            suffix += 1
        target_dir.mkdir(parents=True)
        shutil.copy2(path, target_dir / path.name)
        manifest = path.parent / "activated_memory_manifest.json"
        if manifest.exists():
            shutil.copy2(manifest, target_dir / manifest.name)
        staged.append({"source": str(path), "destination": str(target_dir), "sha256": digest})
    return staged


def copy_disorder_protocols(files: list[Path], destination: Path) -> list[dict[str, Any]]:
    """Stage each disorder protocol with its adjacent run metadata."""

    staged: list[dict[str, Any]] = []
    for path in files:
        run = json.loads(path.read_text(encoding="utf-8"))
        graph = run["model"]["graph"]
        node_count = int(graph["node_count"])
        sigma = float(graph["disorder"])
        seed = int(graph["seed"])
        target = destination / f"N{node_count}_sigma{sigma:g}_seed{seed}"
        if target.exists():
            raise RuntimeError(f"duplicate disorder protocol destination: {target}")
        target.mkdir(parents=True)
        copied = []
        for name in ("capillary_pair_protocols.json", "capillary_pair_run_summary.json", "run_manifest.json"):
            source = path.parent / name
            if source.exists():
                shutil.copy2(source, target / name)
                copied.append(name)
        staged.append(
            {
                "source": str(path.parent),
                "destination": str(target),
                "node_count": node_count,
                "sigma_over_a": sigma,
                "graph_seed": seed,
                "files": copied,
            }
        )
    return staged


def build_readme(complete: bool, missing: list[str]) -> str:
    status = "complete" if complete else "staging: one or more required sources are missing"
    missing_text = "None." if not missing else "\n".join(f"- `{item}`" for item in missing)
    return f"""# Geometry-Encoded Hidden Orientational Memory: data deposit

Archive status: **{status}**

This deposit contains the raw JSON/JSONL simulation records, run manifests,
publication-scale controls, and derived numerical reports used for the
manuscript *Geometry-Encoded Hidden Orientational Memory*. Source code is
maintained separately at
https://github.com/vbaulin/{REPOSITORY_NAME}.

## Contents

- `raw/capillary_pair_publication_runs/`: 441-cell state map, matched controls,
  five-size scaling, and three long-dynamics realizations.
- `raw/activated_memory/`: eight-coupling, five-graph retention scan used for
  Fig. 4. Each GPU shard contributes an `activated_memory_scan.jsonl` file.
- `raw/equilibrium_replica_discriminant/`: independent-replica finite-size scan.
- `raw/order_of_operations/`: pulse-amplitude, size, and support-fraction tests
  for the signed AB/BA readout after field removal.
- `raw/capillary_internal_correlations/`: real-space correlation analysis.
- `raw/grooved/`: programmable easy-axis realization reported in the Supplement.
- `raw/disorder_retention/`: write--release trajectories behind the positional-
  disorder optimum at `N=576` and `N=1024`.
- `raw/submission_validations/`: time-step, mobile-cage, and independent-noise
  tests, when generated by the prospective cluster run.
- `derived/`: figure summaries, loop-intervention reports, and the exact
  numerical audit.
- `manifest.json` and `SHA256SUMS`: file-level provenance and integrity checks.

## Missing required sources

{missing_text}

The archive must not be uploaded as the final Zenodo record while this list is
nonempty. Run the builder on the GPU cluster with `--activated-input` pointing
to the activated-memory result directory and `--disorder-retention-input`
pointing to the parent of the disorder protocol runs. Alternatively, install
both sources in the project discovery tree and rerun without
`--allow-incomplete`.

## Units and conventions

Angles are apolar and therefore periodic modulo pi. Energies are reported in
units of `k_B T`; times are reported as `D_r t`; distances are reduced by the
undistorted cage spacing `a` or the median nearest-neighbour separation `r_0`,
as specified by each run manifest. JSONL files contain one independent
parameter/graph record per line.

## Licenses

Data are prepared for release under CC BY 4.0. Code is released separately
under the MIT License. A Zenodo DOI must be inserted in the manuscript before
submission.
"""


def build_dictionary() -> str:
    return """# Data dictionary

## Core observables

| Field | Meaning |
|---|---|
| `S_mean` | Magnitude of the global apolar nematic order parameter. |
| `C2_mean` | Weighted relative-angle correlation of short-range neighbours. |
| `G2_mean` | Weighted bond-frame quadrupolar correlation. |
| `q_EA_mean` | Finite-window single-trajectory angular persistence statistic. |
| `window_autocorrelation` | Endpoint angular autocorrelation over the sampled window. |
| `replica_overlap` | Overlap of independently initialized or split replicas, as identified by the containing protocol. |
| `final` | Endpoint retained overlap `Q(T_obs)` at the common observation time. |
| `positive_integral_time` | Auxiliary finite-window area `integral max(Q(t),0) dt / abs(Q(0))`; it is not an intrinsic relaxation lifetime and is not the Fig. 4(b) ordinate. |

## Independent axes and identifiers

| Field | Meaning |
|---|---|
| `j_align` or `J` | Short-range relative-angle coupling in units of `k_B T`. |
| `g_capillary` or `g` | Bond-frame quadrupolar coupling at `r_0`, in units of `k_B T`. |
| `lambda` | Scale factor in `(J,g)=lambda(4,5) k_B T`. |
| `graph_seed` | Quenched positional-disorder realization. |
| `seed` | Thermal-noise or run seed. |
| `node_count` | Number of rotors. |
| `time` | Reduced rotational time `D_r t`. |

Exact command-line parameters, sample counts, time steps, and cutoffs are
stored in each `run_manifest.json` or run summary. The manuscript and
Supplement define all reported observables mathematically.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release/zenodo_geometry_encoded_orientational_memory"),
    )
    parser.add_argument("--activated-input", type=Path)
    parser.add_argument("--disorder-retention-input", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    source_activated, activated_files = activated_source(root, args.activated_input)
    disorder_source, disorder_files, disorder_validation = disorder_retention_source(
        root, args.disorder_retention_input
    )

    missing = [str(path) for path in DATASETS.values() if not (root / path).exists()]
    missing.extend(str(path) for path in DERIVED_FILES.values() if not (root / path).exists())
    if not activated_files:
        missing.append("activated_memory_scan.jsonl shard(s) (cluster-generated raw source for Fig. 4)")
    if not disorder_validation["complete"]:
        missing.append(
            "disorder-retention protocol trajectories: "
            + json.dumps(disorder_validation["missing_cells"], sort_keys=True)
        )
    if missing and not args.allow_incomplete:
        raise SystemExit("release is incomplete:\n" + "\n".join(f"- {item}" for item in missing))

    if output.exists():
        if not args.clean:
            raise SystemExit(f"output already exists: {output}; pass --clean to rebuild")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copied_sources: list[dict[str, Any]] = []
    for destination, relative_source in DATASETS.items():
        source = root / relative_source
        if not source.exists():
            continue
        copy_source(source, output / destination)
        copied_sources.append({"source": str(relative_source), "destination": destination})
    for destination, relative_source in OPTIONAL_DATASETS.items():
        source = root / relative_source
        if not source.exists():
            continue
        copy_source(source, output / destination)
        copied_sources.append(
            {"source": str(relative_source), "destination": destination, "optional": True}
        )
    for destination, relative_source in DERIVED_FILES.items():
        source = root / relative_source
        if not source.exists():
            continue
        copy_source(source, output / destination)
        copied_sources.append({"source": str(relative_source), "destination": destination})
    if source_activated is not None:
        destination = "raw/activated_memory"
        staged_shards = copy_unique_activated_shards(activated_files, output / destination)
        copied_sources.append(
            {
                "source": str(source_activated),
                "destination": destination,
                "activated_memory_jsonl_shards": [str(path) for path in activated_files],
                "staged_shards": staged_shards,
            }
        )
    if disorder_files:
        destination = "raw/disorder_retention"
        staged_protocols = copy_disorder_protocols(disorder_files, output / destination)
        copied_sources.append(
            {
                "source": str(disorder_source),
                "destination": destination,
                "protocol_files": [str(path) for path in disorder_files],
                "staged_protocols": staged_protocols,
                "validation": disorder_validation,
            }
        )

    complete = not missing
    (output / "README.md").write_text(build_readme(complete, missing), encoding="utf-8")
    (output / "DATA_DICTIONARY.md").write_text(build_dictionary(), encoding="utf-8")
    metadata = {
        "title": "Geometry-Encoded Hidden Orientational Memory: simulation data",
        "upload_type": "dataset",
        "publication_date": date.today().isoformat(),
        "creators": [{"name": "Baulin, Vladimir A.", "affiliation": "Synthetix Institute"}],
        "description": (
            "Raw Brownian-rotor simulation records and derived numerical reports for a study of "
            "preparation-dependent orientational retention generated by short-range alignment and "
            "bond-frame quadrupolar interactions on quenched disordered graphs."
        ),
        "keywords": [
            "colloids",
            "orientational memory",
            "Brownian rotors",
            "capillary quadrupoles",
            "geometrical frustration",
        ],
        "license": "cc-by-4.0",
        "related_identifiers": [
            {
                "identifier": f"https://github.com/vbaulin/{REPOSITORY_NAME}",
                "relation": "isSupplementedBy",
                "resource_type": "software",
            }
        ],
    }
    (output / "zenodo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name in {"manifest.json", "SHA256SUMS"}:
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    forbidden_media = [
        item["path"] for item in files
        if Path(item["path"]).suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".svg"}
    ]
    if forbidden_media:
        raise RuntimeError("manuscript media leaked into the data archive: " + ", ".join(forbidden_media))
    manifest = {
        "archive": "geometry_encoded_orientational_memory_data",
        "complete": complete,
        "missing_required_sources": missing,
        "copied_sources": copied_sources,
        "disorder_retention_validation": disorder_validation,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in files),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "complete": complete,
                "file_count": len(files),
                "total_bytes": manifest["total_bytes"],
                "missing": missing,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
