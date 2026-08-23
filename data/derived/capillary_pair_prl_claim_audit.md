# Capillary-pair PRL claim audit

- Quantitative checks passed: `110/110`
- All quantitative checks passed: `True`
- Required local artifacts present: `True`
- Activated-memory raw JSONL present locally: `True`
- Raw activated-memory shards reproduce the derived report: `True`
- Fig. 4(c) window statistics recorded: `True`
- Language gates passed: `True`

The numerical contract covers the publication-scale regime map, matched controls, five-size scaling, long dynamics, spatial correlations, equilibrium-replica discriminant, coupling-dependent endpoint overlap, the disorder-retention maximum, the matched loop intervention, and the signed AB/BA release readout.

## Failed quantitative checks

- None.

## Provenance gates

The raw activated-memory JSONL is generated on the GPU cluster and must be included in the Zenodo deposit. Identical duplicate records are ignored by stable row key, while conflicting duplicates fail the audit.

The N=1024 disorder-retention values are currently transcribed from cluster output. The deposit is not publication-complete until their raw protocol trajectories are present and regenerate the summary.
