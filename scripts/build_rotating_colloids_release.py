#!/usr/bin/env python3
"""Assemble the data deposit for Geometry-Encoded Hidden Orientational Memory.

Large simulation records belong in the Zenodo deposit. The companion GitHub
repository contains code, manuscript sources, and small derived reports only.
The builder refuses to declare a complete deposit when the cluster-generated
activated-memory JSONL shard set is absent.
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
}

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
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
        )
    else:
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
- `raw/capillary_internal_correlations/`: real-space correlation analysis.
- `raw/grooved/`: programmable easy-axis realization reported in the Supplement.
- `derived/`: figure summaries and the exact 82-check numerical audit.
- `manifest.json` and `SHA256SUMS`: file-level provenance and integrity checks.

## Missing required sources

{missing_text}

The archive must not be uploaded as the final Zenodo record while this list is
nonempty. Run the builder on the GPU cluster with `--activated-input` pointing
to the activated-memory result directory, or copy that directory into the
project discovery tree and rerun without `--allow-incomplete`.

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
| `positive_integral_time` | Finite-window area `integral max(Q(t),0) dt / abs(Q(0))`; it is not an intrinsic relaxation lifetime. |

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
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    source_activated, activated_files = activated_source(root, args.activated_input)

    missing = [str(path) for path in DATASETS.values() if not (root / path).exists()]
    missing.extend(str(path) for path in DERIVED_FILES.values() if not (root / path).exists())
    if not activated_files:
        missing.append("activated_memory_scan.jsonl shard(s) (cluster-generated raw source for Fig. 4)")
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
    for destination, relative_source in DERIVED_FILES.items():
        source = root / relative_source
        if not source.exists():
            continue
        copy_source(source, output / destination)
        copied_sources.append({"source": str(relative_source), "destination": destination})
    if source_activated is not None:
        destination = "raw/activated_memory"
        if source_activated.is_dir():
            copy_source(source_activated, output / destination)
        else:
            copy_source(source_activated, output / destination / source_activated.name)
        copied_sources.append(
            {
                "source": str(source_activated),
                "destination": destination,
                "activated_memory_jsonl_shards": [str(path) for path in activated_files],
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
    manifest = {
        "archive": "geometry_encoded_orientational_memory_data",
        "complete": complete,
        "missing_required_sources": missing,
        "copied_sources": copied_sources,
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
