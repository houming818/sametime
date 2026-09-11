# F13 learned TreeHeap polytope router smoke

- Claim: `S3-TREEHEAP-LEARNED-POLYTOPE-ROUTER-F13`
- Gates: `{"O0_matched_exact_origin": true, "O1_finite_gradient_and_energy": true, "O2_frozen_and_reload": true, "P1_group_differentiation": true, "P2_heldout_nll_advantage": false, "P3_generation_health": true}`
- Comparison: `{"grouped_minus_scalar_valid_nll": 0.007453106252726549, "grouped_minus_scalar_test_nll": 0.007646687825521248, "grouped_minus_scalar_bleu4_median": 3.8631607955625418, "grouped_u_group_std_mean": 0.036203107330948114, "reload_test_nll_max_delta": 0.0}`
- Runtime seconds: `442.927`

## Evidence audit

- Pre-registration commit: `c7360d7`
- taskd: `405`, status `done`, exit code `0`
- Both arms: `32,080` parameters, matched state, `300` batches steps
- Step-zero native NLL delta: `0` for both arms
- Frozen 106M checkpoint: exact; theta reload NLL delta: `0`
- GPU guard: `270 W`; observed max `176.54 W`, `67 C`, `1502 MiB`
- P1 and P3 passed; P2 failed because grouped test NLL was `0.007647` above scalarized

The BLEU gain is a smoke signal from 16 generated rows and is concentrated at
protocol depth 6. It requires a larger read-only evaluation before promotion.
