# Matched-Preparation Colloidal Memory

## Physical Question

Does the capillary interaction preserve a prepared local angular pattern when
the comparison starts from exactly the same angles? Is its advantage still
present when a relative-alignment control has the same sum of positive bond
coefficients? Does the pattern disappear, or merely rotate as a whole?

The existing production curves compare separately prepared targets. This run
adds matched release; it does not replace those curves or change the model.

## Cluster Command

This repository contains all required scripts; KnowledgeParser is not needed.
Install the repository requirements and a CUDA-enabled PyTorch build compatible
with the GPU driver. The production run used four RTX 3090 GPUs; the driver
also accepts a shorter GPU list.

From the repository root, run:

```bash
GPUS=0,1,2,3 nohup bash scripts/run_rotating_colloids_matched_preparation_4gpu.sh \
  > matched_preparation.log 2>&1 < /dev/null &
```

The defaults are N=1024, five graph seeds, two disorder amplitudes (0.11 and
0.16), two symmetry-related targets, 48 replicas, dt=0.0025, 50,000 target
relaxation steps, 50,000 write steps, and 250,000 field-free release steps.
There are 20 graph/disorder/target cases. Each release advances five arms:

1. Physical capillary Hamiltonian from its written state.
2. g=0 Hamiltonian from exactly that physical written state.
3. Relative alignment with coefficient J*w_minus+g*w_plus on each edge, from
   exactly that physical written state.
4. g=0 Hamiltonian from its own state written with the same target and field.
5. Equal-weight relative alignment from its own state written with the same
   target and field.

All arms have independent Brownian noise. The two targets are related by a
common quarter-turn, an exact symmetry, rather than two independent target
families. The equal-weight control is not energy-, torque-, or Hessian-matched
at the target. This is not a topology-only intervention.

The shell script assigns graph seeds across four GPUs. Allow at least 10 GB
of free output space for angular trajectories and temporary checkpoints.

## Benchmark And Resume

A short full-size GPU benchmark, separate from production output:

```bash
CUDA_VISIBLE_DEVICES=0 python -B scripts/rotating_colloids_matched_preparation.py \
  --device cuda --output-dir matched_preparation_benchmark \
  --sizes 32 --graph-seeds 17 --disorders 0.11 --targets relaxed \
  --replicas 48 --equilibration-steps 1000 --write-steps 1000 \
  --release-steps 4000 --checkpoint-steps 1000
```

Use the logged stage times to estimate the full run on the actual GPU.
The release dominates. A CPU pilot cannot provide a reliable 3090 ETA.

Rerun the identical production command to resume. Each stage saves angles,
its random-number state, and recorded samples every 10,000 steps. A completed
case is reused only with matching code and configuration hashes. Changing
parameters or code requires a new output directory; do not delete checkpoints
to force incompatible runs together. One lock protects each case from duplicate
writers. CUDA floating-point reductions need not be bitwise deterministic
across devices; exact checkpoint continuation is unit-tested on CPU.

## Output Files

Default output root:

`discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_matched_preparation_prl`

Preserve that complete directory, not only the merged report. Each case includes:

- `manifest.json`: model parameters and source hashes.
- `preparation.npz`: target, unrelated target, graph, weights, release angles.
- `write.npz` and `release.npz`: angular trajectories and terminal states.
- `observables.npz`: laboratory, exact connected, rotation-aligned overlaps,
  complex directors, and unrelated-target overlap for every replica.
- `spatial_final.npz`: equal-time spatial correlations on the same trajectories.
- `summary.json`: endpoint measurements per replica.

The merged `matched_preparation_report.json` keeps graph-level uncertainties
and separates sizes, disorders, targets, and controls. It does not pool
symmetry-related targets as independent graphs.

## Current Evidence

The full N=1024 run is complete: 20 cases, five release conditions per case,
48 replicas, five independent graph seeds, and final reduced time 625.
The saved initial angles agree exactly across the three matched conditions.
All terminal arrays are finite and the required raw files are present.

For the unrotated target, exact connected overlaps at the endpoint are:

| sigma/a | Capillary | g=0, same start | Equal-weight relative alignment, same start |
|---|---|---|---|
| 0.11 | 0.5022 +/- 0.0326 | 0.0003 +/- 0.0013 | 0.0021 +/- 0.0015 |
| 0.16 | 0.4729 +/- 0.0129 | -0.0036 +/- 0.0046 | -0.0002 +/- 0.0006 |

Errors are SEM across graphs, after averaging thermal replicas within each
graph. Physical global order is 0.075 and 0.076, respectively. The quarter-turn
target gives comparable retention; it is a symmetry-related construction,
not an independent pattern family. The two controls also lose the connected
pattern when written separately. This establishes an interaction-form effect
on retention without attributing it solely to loop topology.

Regenerate the supplementary figure and verify its numerical sources with:

```bash
MPLCONFIGDIR=/tmp/colloid-mpl python -B scripts/plot_rotating_colloids_matched_preparation.py \
  --input-dir discoveries/theory_experiment_interface/rotating_colloids_hyperion/rotating_colloids_matched_preparation_prl \
  --output tex/rotating_colloids/capillary_prl_figures/figS_matched_preparation
```

The published Zenodo version 10.5281/zenodo.22174752 contains the original
production data. On 6 September 2026, the complete update was uploaded from
the cluster and published with author approval as
[record 22544787](https://zenodo.org/records/22544787), DOI
10.5281/zenodo.22544787. It contains 405 checksum-listed files, including the
full matched-preparation run. The previous published version is unchanged.
The public API confirms the new record, archive size, and checksum.

All six uploaded assets were independently checked against Zenodo's recorded
sizes and MD5 checksums. The data archive is 4,136,218,708 bytes, with SHA256
`deaf12e4142e17df4d9c84623a78d0277b01f80791542e03a28969c6ed00c28f`.
No manuscript, figure, or source-code files are included in the data deposit.
