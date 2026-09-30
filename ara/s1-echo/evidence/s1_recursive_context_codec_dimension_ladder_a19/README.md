# A19 Evidence Index

## Execution

- host: `io.grepcode.cn`
- smoke task: `639`
- formal task: `640`
- post-hoc baseline audit task: `641`
- remote plot attempt: `642` (failed because `matplotlib` was absent; retained)
- formal seed: `20260930`
- split seed: `20260924`
- formal dimensions: `2,4,8,16`
- budget: 200 full epochs per dimension

## Inputs

- source count field: A14 1M-line context counts
- shape: `512 x 1024`
- fit pairs: `29,460,566`
- development pairs: `7,366,000`
- sealed-test pairs: `3,670,609`
- counts SHA-256:
  `c4a2a0aad79a4357cbbd6e1591a8a31a873dfa2918fb62b8bda2d5925917f864`

## Result

All registered arms completed with finite metrics and exact reload equality.
The registered mechanical gate passed. `d=2` stayed uniform; `d=4/8/16`
learned progressively lower NLL, with a monotone READ-depth curve.

The post-hoc audit shows that the best recursive arm (`d=16`, NLL `5.572473`)
does not beat the no-input global context prior (NLL `5.555696`). Root
shuffling damages `d=16` by `0.068362`, about `5.03%` of its gain over uniform.
The evidence therefore supports a trainable recursive address decoder with a
weak token-conditional channel, not a competitive invertible context codec.

## Files

- `formal_seed20260930/material.json`: frozen inputs and configuration
- `formal_seed20260930/summary.json`: registered gates and dimension summary
- `formal_seed20260930/dimension_*.json`: trajectories, depth curves, controls,
  state diagnostics, and checkpoint hashes
- `formal_seed20260930/dimension_*.pt`: small reload-audit checkpoints
- `formal_seed20260930/posthoc_baselines.json`: explicit global-prior audit
- `formal_seed20260930/a19_dimension_ladder.png`: locally rendered result figure
- `smoke_seed20260930/`: preflight evidence
- `taskd_logs/`: complete taskd logs for tasks 639 through 642

The figure was generated from the fetched JSON evidence with the plotting
script's Pillow fallback. No training output was recomputed locally.

## Claim Status

`S1-RECURSIVE-CONTEXT-CODEC-A19-C01`: **supported at its registered mechanical
gate, but insufficient as a practical codec result**.

Not proved: sentence generation, translation, semantic grounding, exact
reversibility, learned context topology, or superiority to flat/PCA baselines.
