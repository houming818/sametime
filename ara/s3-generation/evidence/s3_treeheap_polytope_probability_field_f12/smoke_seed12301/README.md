# F12 TreeHeap polytope probability-field audit

- Claim: `S3-TREEHEAP-POLYTOPE-PROBABILITY-FIELD-F12`
- Gates: `{"O0_exact_origin": true, "O1_finite_gradient_and_energy": true, "O2_frozen_checkpoint": true, "P1_multi_direction_cells": true, "P2_grouped_finite_step_advantage": true, "P3_scalar_subspace_insufficient": true}`
- Multi-direction cells: `[{"protocol_depth": 5, "merge_depth": 0}, {"protocol_depth": 5, "merge_depth": 1}, {"protocol_depth": 5, "merge_depth": 2}, {"protocol_depth": 6, "merge_depth": 0}, {"protocol_depth": 6, "merge_depth": 1}, {"protocol_depth": 6, "merge_depth": 2}, {"protocol_depth": 6, "merge_depth": 3}, {"protocol_depth": 7, "merge_depth": 0}, {"protocol_depth": 7, "merge_depth": 1}, {"protocol_depth": 7, "merge_depth": 2}, {"protocol_depth": 7, "merge_depth": 3}, {"protocol_depth": 7, "merge_depth": 4}]`
- Grouped-advantage depths: `[5, 6, 7]`
- Global scalar energy fraction: `0.30029559`
- Runtime seconds: `21.757`

## Evidence audit

- Pre-registration commit: `d34bd5f`
- taskd: `404`, status `done`, exit code `0`
- WMT examples: `32` rows x `3` protocol depths = `96` example records
- Native/zero origin: exact (`max logits delta = 0`, `max NLL delta = 0`)
- Frozen checkpoint: exact
- GPU guard: `270 W`; observed max `165.86 W`, `57 C`, `1242 MiB`
- `summary.json`, `contract.json`, `examples.csv`, `run.log`, and GPU samples parsed successfully

The logic document was updated with results after the run. Its pre-run SHA-256 in
`inputs.sha256` therefore refers to the version fixed by commit `d34bd5f`.
