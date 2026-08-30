# Geometry-Encoded Hidden Orientational Memory

This public repository contains the Brownian-rotor simulation, analysis,
plotting, audit code, and unit tests associated with *Geometry-Encoded Hidden
Orientational Memory*. The model combines short-range relative-angle
alignment with a bond-frame quadrupolar interaction on a quenched disordered
neighbour graph.

The release is deliberately separated by artifact type. GitHub is the
software repository. Zenodo is the numerical data archive and contains raw
trajectories, derived reports, manifests, and checksums, but no source code,
figures, or compiled manuscripts. arXiv is the authoritative manuscript and
Supplemental Material archive. TeX sources retained here serve only the
software audit and reproducible figure workflow.

The numerical data release is available at
[doi:10.5281/zenodo.22174752](https://doi.org/10.5281/zenodo.22174752).

## Evidence represented here

The publication-scale results distinguish preparation-dependent dynamical
retention from equilibrium replica ordering. In the selected finite-size
regime, global nematic order and independently equilibrated replica overlap
decrease approximately as `N^-1/2`, while local pair correlations and
finite-window persistence remain finite. Split descendants of one prepared
state retain overlap to the longest simulated time. These statements are
checked by `scripts/audit_rotating_colloids_capillary_prl.py` against 126
frozen numerical expectations.

The audit separates numerical checks from provenance. A `126/126` numerical
result does not make the release complete unless both the raw
`activated_memory_scan.jsonl` shards used for Fig. 4 and the positional-
disorder write--release trajectories are installed from the data deposit. The
identical-start loop-flattening scan cited in the Supplemental Material is a
third required raw source.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

GPU simulations require a CUDA-compatible PyTorch build appropriate for the
host driver. Install it separately following the PyTorch instructions; do
not replace it with the CPU wheel for production runs.

## Install the data deposit

Download and extract the
[Zenodo data archive](https://doi.org/10.5281/zenodo.22174752), then map it
into the directory layout expected by the scripts:

```bash
python scripts/install_zenodo_data.py /path/to/zenodo_geometry_encoded_orientational_memory
```

The installer rejects an incomplete archive. It uses relative symbolic links
by default; pass `--mode copy` when links are unsuitable.

## Reproduce the numerical audit

```bash
MPLCONFIGDIR=/tmp/orientational-memory-mpl \
python -B scripts/audit_rotating_colloids_capillary_prl.py
```

Expected result: `126/126` quantitative checks, all language gates passed, and
all raw-provenance gates true after data installation.

## Rebuild figures

```bash
python -B scripts/classify_rotating_colloids_capillary_regimes.py \
  --input discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl_gpu/dense_map_n20/capillary_pair_scan.jsonl \
  --output-dir tex/rotating_colloids/capillary_prl_figures

python -B scripts/plot_rotating_colloids_capillary_prl.py

python -B scripts/plot_rotating_colloids_activated_memory_prl.py \
  --input-dir discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_activated_memory_prl_gpu \
  --output-dir tex/rotating_colloids/capillary_prl_figures
```

The Fig. 4(b) ordinate is the endpoint retained overlap `Q(T_obs)` at the
common observation time. Physical and `g=0` branches are compared by their
difference in units of combined graph-level SEM, not by a ratio to a control
whose mean is statistically zero. The auxiliary finite-window area remains in
the raw record but is not the plotted statistic.

The Fig. 4 builder writes `activated_memory_figure_report.json` next to the
figure, including the panel (a) retention surface and the panel (c)
observation-window statistics. When a previous report is present it also
writes `activated_memory_report_delta.json`. A nonzero `max_relative_change`
there means the numbers quoted in the Letter and the frozen expectations in
`scripts/audit_rotating_colloids_capillary_prl.py` have to be updated
together. Keep a copy of the previous report before rebuilding.

The positional-disorder summary figure is rebuilt with:

```bash
python -B scripts/plot_rotating_colloids_disorder_retention.py \
  --input discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_disorder_retention_summary.json \
  --output build/disorder_retention
```

## Publication-scale GPU runs

The production shell drivers are:

- `scripts/run_rotating_colloids_capillary_pair_prl_gpu.sh`
- `scripts/run_rotating_colloids_activated_memory_prl_gpu.sh`
- `scripts/run_rotating_colloids_spin_glass_prl_4gpu.sh`
  (`scripts/run_rotating_colloids_spin_glass_prl_gpu.sh` for one device)
- `scripts/run_rotating_colloids_groove_protocols_gpu.sh`
- `scripts/run_rotating_colloids_prl_submission_validations_4gpu.sh`
- `scripts/run_rotating_colloids_disorder_retention_4gpu.sh`
- `scripts/run_rotating_colloids_order_memory_publication.sh`
- `scripts/run_holonomy_matched_release_crossover_4gpu.sh`

Each simulation output is append-only and resumes completed parameter points.
The shell scripts document the exact replica counts, graph seeds, step counts,
and CUDA rank partitioning.

The activated-memory driver shards its five quenched graphs over whatever
CUDA devices are visible, so the same command works on a single RTX 3090 and
on a four-GPU node. Override the split with `GPUS=0,1`, restrict the graphs
with `GRAPH_SEEDS`, and pass `SKIP_FIGURE=1` to run the scan alone:

```bash
nohup bash scripts/run_rotating_colloids_activated_memory_prl_gpu.sh \
  > activated_memory_gpu.log 2>&1 < /dev/null &
```

The submission-validation driver runs three discriminating tests: weak
time-step convergence at equal physical duration, retention with mobile but
caged particle centres, and two-pulse decoding with independent thermal
noise. It uses four GPUs concurrently and resumes append-only records:

```bash
nohup bash scripts/run_rotating_colloids_prl_submission_validations_4gpu.sh \
  > rotating_colloids_prl_submission_validations.log 2>&1 < /dev/null &
```

To reconstruct the 33 disorder-retention trajectories required by the data
deposit, run:

```bash
nohup bash scripts/run_rotating_colloids_disorder_retention_4gpu.sh \
  > rotating_colloids_disorder_retention_4gpu.log 2>&1 < /dev/null &
```

Complete graph/disorder cells are JSON-validated and skipped on restart.

The operation-order evidence is compact and was generated on CPU. This
resumable runner reconstructs the N=144 amplitude scan, N=256 size check,
support-fraction control, and reduced relaxation model:

```bash
DEVICE=cpu bash scripts/run_rotating_colloids_order_memory_publication.sh
```

The loop-intervention evidence can be regenerated independently with
`scripts/test_colloid_holonomy_memory.py`,
`scripts/test_rotating_colloids_holonomy_causality.py`, and
`scripts/test_continuous_colloid_holonomy_memory.py`. Their focused unit tests
verify gauge invariance, matched unsigned couplings, stochastic controls, and
the matched-release analysis.

The matched-release driver starts the original and loop-flattened networks
from identical angles and applies paired Brownian noise. It isolates release
from arm-dependent writing and resumes by graph, coupling and target key:

```bash
nohup bash scripts/run_holonomy_matched_release_crossover_4gpu.sh \
  > holonomy_matched_release_crossover.log 2>&1 < /dev/null &
```

## Manuscript

The citable manuscript and Supplemental Material will be distributed through
arXiv. The TeX sources under `tex/rotating_colloids/` are retained here only
so the audit can verify figure numbering, language gates, and reported values.
They are not the archival manuscript release.

## Data release

The data-only release is archived at
[doi:10.5281/zenodo.22174752](https://doi.org/10.5281/zenodo.22174752).
`scripts/build_rotating_colloids_release.py` reconstructs the deposit,
computes SHA-256 hashes, and refuses a complete build when required raw
trajectories are absent. The matched-release crossover archive is mandatory
because its result is reported in the Supplemental Material. Source code,
generated figures, TeX files, and compiled PDFs are excluded from the Zenodo
archive.

## License

Code is licensed under the MIT License. The separate data deposit is prepared
under CC BY 4.0.
