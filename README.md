# Geometry-Encoded Hidden Orientational Memory

This repository contains the Brownian-rotor simulation, analysis, plotting,
and audit code for the manuscript *Geometry-Encoded Hidden Orientational
Memory*. The model combines short-range relative-angle alignment with a
bond-frame quadrupolar interaction on a quenched disordered neighbour graph.

The repository is private during manuscript preparation. Raw publication
records are kept outside GitHub in a Zenodo data deposit. The repository
contains manuscript sources, small derived reports, and unit tests; it
deliberately excludes generated figures, compiled PDFs, and large JSONL scans.
Figures are rebuilt from the separately archived data.

## Evidence represented here

The publication-scale results distinguish preparation-dependent dynamical
retention from equilibrium replica ordering. In the selected finite-size
regime, global nematic order and independently equilibrated replica overlap
decrease approximately as `N^-1/2`, while local pair correlations and
finite-window persistence remain finite. Split descendants of one prepared
state retain overlap to the longest simulated time. These statements are
checked by `scripts/audit_rotating_colloids_capillary_prl.py` against 94
frozen numerical expectations.

The audit separates numerical checks from provenance. A `94/94` numerical
result does not make the release complete unless both the raw
`activated_memory_scan.jsonl` shards used for Fig. 4 and the positional-
disorder write--release trajectories are installed from the data deposit.

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

Download and extract the Zenodo archive, then map it into the directory
layout expected by the scripts:

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

Expected result: `94/94` quantitative checks, all language gates passed, and
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
python -B scripts/plot_rotating_colloids_disorder_retention.py
```

## Publication-scale GPU runs

The production shell drivers are:

- `scripts/run_rotating_colloids_capillary_pair_prl_gpu.sh`
- `scripts/run_rotating_colloids_activated_memory_prl_gpu.sh`
- `scripts/run_rotating_colloids_spin_glass_prl_4gpu.sh`
  (`scripts/run_rotating_colloids_spin_glass_prl_gpu.sh` for one device)
- `scripts/run_rotating_colloids_groove_protocols_gpu.sh`

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

## Manuscript length

`scripts/count_prl_length.py tex/rotating_colloids/rotating_colloids_prl_capillary.tex`
reports the APS core word-equivalent count against the 3750-word PRL limit.
The current manuscript is estimated at 4624 word equivalents and therefore
still requires a core/End-Matter restructuring, shortening, or waiver.

## Manuscript

The Letter and Supplemental Material are in `tex/rotating_colloids/`. Compile
from that directory with `latexmk -pdf` or an equivalent RevTeX-capable TeX
Live installation.

## Data release

The Zenodo DOI is pending. `scripts/build_rotating_colloids_release.py`
assembles the data folder, computes SHA-256 hashes, and refuses a complete
build when either the activated-memory raw shard set or the disorder-retention
protocol trajectories are absent. Generated figures and compiled PDFs are
excluded from the data archive.

## License

Code is licensed under the MIT License. The separate data deposit is prepared
under CC BY 4.0.
