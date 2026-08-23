# Capillary-pair PRL claim audit

- Quantitative checks passed: `110/111`
- All quantitative checks passed: `False`
- Required local artifacts present: `False`
- Activated-memory raw JSONL present locally: `True`
- Raw activated-memory shards reproduce the derived report: `True`
- Fig. 4(c) window statistics recorded: `True`
- Language gates passed: `True`

The numerical contract covers the publication-scale regime map, matched controls, five-size scaling, long dynamics, spatial correlations, equilibrium-replica discriminant, coupling-dependent endpoint overlap, the disorder-retention maximum, the matched loop intervention, and both common- and independent-noise AB/BA release readouts.

## Failed quantitative checks

- `independent-noise order report present`

## Provenance gates

The raw activated-memory JSONL is generated on the GPU cluster and must be included in the Zenodo deposit. Identical duplicate records are ignored by stable row key, while conflicting duplicates fail the audit.

The N=1024 disorder-retention values are currently transcribed from cluster output. The deposit is not publication-complete until their raw protocol trajectories are present and regenerate the summary.

The completed submission-validation directory must include 15 time-step rows, 25 mobile-cage rows, and 50 independent-noise sequence rows before the independent-noise Letter claims are publication-complete.
