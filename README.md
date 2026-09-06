# Geometry-Encoded Hidden Orientational Memory

This public repository contains the Brownian-rotor simulation, analysis,
plotting, audit code, and unit tests associated with *Geometry-Encoded Hidden
Orientational Memory*. The model combines short-range relative-angle
alignment with a bond-frame quadrupolar interaction on a quenched disordered
neighbour graph.

The release is deliberately separated by artifact type. GitHub is the
software repository. Zenodo is the numerical data archive and contains raw
trajectories, derived reports, manifests, and checksums, but no source code,
figures, or compiled manuscripts. The manuscript and Supplemental Material,
including their TeX sources, will be submitted separately to arXiv. They are
not included in this repository.

The current numerical data release is
[doi:10.5281/zenodo.22544787](https://doi.org/10.5281/zenodo.22544787),
published on 6 September 2026. It includes the full N=1024 matched-preparation
comparison: 405 checksum-listed files in a 4.14 GB archive. The
[all-versions DOI](https://doi.org/10.5281/zenodo.22173160) follows subsequent
data releases; use the version-specific DOI to reproduce these results.

## Evidence represented here

The publication-scale results distinguish preparation-dependent dynamical
retention from equilibrium replica ordering. In the selected finite-size
regime, global nematic order and independently equilibrated replica overlap
decrease approximately as `N^-1/2`, while local pair correlations and
finite-window persistence remain finite. Split descendants of one prepared
state retain overlap to the longest simulated time. These statements are
checked by `scripts/audit_rotating_colloids_capillary_prl.py` against 129
frozen numerical expectations.

The audit separates numerical checks from provenance. A `129/129` numerical
result does not make the release complete unless both the raw
`activated_memory_scan.jsonl` shards used for Fig. 4 and the positional-
disorder write--release trajectories are installed from the data deposit. The
identical-start loop-flattening scan cited in the Supplemental Material is a
third required raw source.

## Retention from the same prepared pattern

The completed N=1024 comparison starts three field-free systems from exactly
the same angles. The capillary model retains the local angular pattern,
whereas removing the capillary interaction or replacing it with additional
relative-angle alignment loses the pattern. All three use the same positional
graph and target. The equal-weight control preserves the sum of positive
pair coefficients, not the target energy, torque, or local curvature.

At `D_r t = 625`, the unrotated target gives:

| Positional disorder `sigma/a` | Capillary connected overlap | No capillary term | Equal-weight relative alignment |
| --- | --- | --- | --- |
| 0.11 | 0.5022 +/- 0.0326 | 0.0003 +/- 0.0013 | 0.0021 +/- 0.0015 |
| 0.16 | 0.4729 +/- 0.0129 | -0.0036 +/- 0.0046 | -0.0002 +/- 0.0006 |

Errors are SEM across five graphs, after averaging 48 thermal replicas per
graph. Global nematic order in the capillary model is approximately 0.075.
Connected overlap subtracts the product of the instantaneous and target
complex directors for each replica; it is not the older `Q - mean(S)^2`
diagnostic. Allowing a rigid rotation does not recover the lost control
patterns. The second target is a global quarter-turn of the first, testing a
symmetry-related pattern rather than an independent family of stored images.

The simulator saves angular trajectories and random-number states and can
resume an interrupted stage. The independent figure script reconstructs
endpoint observables from the saved angles. See the
[run and analysis instructions](docs/ROTATING_COLLOIDS_MATCHED_PREPARATION_RUN.md)
for the complete protocol and Supplemental Fig. S10 and Table SVII.

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
[Zenodo data archive](https://doi.org/10.5281/zenodo.22544787), then map it
into the directory layout expected by the scripts:

```bash
python scripts/install_zenodo_data.py /path/to/zenodo_geometry_encoded_orientational_memory
```

The installer verifies every manifest file's size and SHA256 before making
changes and checks all required paths together. It uses relative symbolic
links for raw data by default; pass `--mode copy` when links are unsuitable.
Derived reports are always copied so rebuilding them cannot change the
deposited originals. Existing
destinations are preserved unless `--force` is explicitly supplied.

## Reproduce the numerical audit

```bash
MPLCONFIGDIR=/tmp/orientational-memory-mpl \
python -B scripts/audit_rotating_colloids_capillary_prl.py
```

Expected result: `129/129` quantitative checks and all raw-provenance gates
true after data installation. No manuscript files are required. Optional
language and figure-reference checks can be requested with
`--manuscript-dir /path/to/separate/arxiv/sources`.

## Rebuild figures

Figures and reports are generated locally, not tracked in Git. The scripts
retain their original output directory names under `tex/`; those directories
contain no manuscript sources in this repository.

```bash
python -B scripts/classify_rotating_colloids_capillary_regimes.py \
  --input discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_capillary_pair_prl_gpu/dense_map_n20/capillary_pair_scan.jsonl \
  --output-dir tex/rotating_colloids/capillary_prl_figures

python -B scripts/plot_rotating_colloids_capillary_prl.py

python -B scripts/plot_rotating_colloids_activated_memory_prl.py \
  --input-dir discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_activated_memory_prl_gpu \
  --output-dir tex/rotating_colloids/capillary_prl_figures

python -B scripts/plot_rotating_colloids_matched_preparation.py \
  --input-dir discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_matched_preparation_prl \
  --output tex/rotating_colloids/capillary_prl_figures/figS_matched_preparation
```

The Fig. 4(b) ordinate is the endpoint retained overlap `Q(T_obs)` at the
common observation time. Physical and `g=0` branches are compared by their
difference in units of combined graph-level SEM, not by a ratio to a control
whose mean is statistically zero. The auxiliary finite-window area remains in
the raw record but is not the plotted statistic.

The Fig. 4 builder writes `activated_memory_figure_report.json` next to the
figure, including the panel (a) retention surface, panel (c) half-overlap
crossing times, and the supplementary observation-window statistics. When a previous report is present it also
writes `activated_memory_report_delta.json`. A nonzero `max_relative_change`
there means the numbers quoted in the Letter and the frozen expectations in
`scripts/audit_rotating_colloids_capillary_prl.py` have to be updated
together. Keep a copy of the previous report before rebuilding.

Panel (c) distinguishes observed half-overlap crossings from trajectories
that have not crossed by the final recorded time. The supplementary
`figS_persistence_windows` plot shows the observation-window dependence of
the finite-window persistence statistic.

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
- `scripts/run_rotating_colloids_matched_preparation_4gpu.sh`

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

The separate N=1024 matched-preparation run uses five graphs, two positional
disorders, two symmetry-related targets, and five release conditions with
independent thermal noise. Its publication defaults are frozen in the driver:

```bash
GPUS=0,1,2,3 nohup bash scripts/run_rotating_colloids_matched_preparation_4gpu.sh \
  > matched_preparation.log 2>&1 < /dev/null &
```

Do not mix this comparison with the loop-flattening scan: the Hamiltonians,
target preparations, and noise pairing differ. Neither intervention holds
target energy, torque, and local curvature fixed while changing only topology.

## Manuscript

The manuscript and Supplemental Material will be distributed separately
through arXiv. Neither TeX sources nor manuscript PDFs are included here.

## Data release

The data-only release is archived at
[doi:10.5281/zenodo.22544787](https://doi.org/10.5281/zenodo.22544787).
`scripts/build_rotating_colloids_release.py` reconstructs the deposit,
computes SHA-256 hashes, and refuses a complete build when required raw
trajectories are absent. The matched-release crossover archive is mandatory
because its result is reported in the Supplemental Material. Source code,
generated figures, TeX files, and compiled PDFs are excluded from the Zenodo
archive.

The full matched-preparation trajectories and per-case provenance manifests
are also required by the current release builder. Old disorder trajectories
without stored complex directors can reproduce only the explicitly labelled
legacy subtraction, using `--legacy-director-subtraction` with
`scripts/analyze_rotating_colloids_disorder_protocols.py`. They cannot be
retrofitted into exact connected overlaps from `Q` and `S` alone.

## License

Code is licensed under the MIT License. The separate data deposit is prepared
under CC BY 4.0.
