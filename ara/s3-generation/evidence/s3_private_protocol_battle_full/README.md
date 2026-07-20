# TreeHeap Private Protocol Battle

```json
{
  "status": "partial",
  "aggregate": {
    "flat": {
      "nll_mean": 6.040144935437881,
      "nll_min": 6.0319869227017495,
      "nll_max": 6.045468151886887,
      "bleu4_mean": 5.352978997251923,
      "parameters": 27421377,
      "seconds_mean": 111.60624853769939
    },
    "h1": {
      "nll_mean": 6.1231115792555455,
      "nll_min": 6.117974015821166,
      "nll_max": 6.130276967287991,
      "bleu4_mean": 4.97188773471109,
      "parameters": 27619714,
      "seconds_mean": 405.4004827340444
    },
    "h2": {
      "nll_mean": 6.134092140087395,
      "nll_min": 6.125751353112188,
      "nll_max": 6.1407046960515705,
      "bleu4_mean": 5.289227101774938,
      "parameters": 27435395,
      "seconds_mean": 775.635891755422
    },
    "h4": {
      "nll_mean": 6.193396942777385,
      "nll_min": 6.178971576285389,
      "nll_max": 6.2058246484912765,
      "bleu4_mean": 5.0853287877723705,
      "parameters": 27343237,
      "seconds_mean": 1552.2178662618
    }
  },
  "intervention_damage_nll": {
    "source_shuffle": 1.9532312881927298,
    "root_shuffle": 2.111321826765166,
    "detail_shuffle": [
      0.020621557013667413,
      0.04417532593584461,
      0.10029096529757364,
      0.2331515203909662,
      0.25766269407331155,
      0.018702367224442185
    ],
    "pair_break": [
      0.5219997708085495,
      0.48107638353076165,
      0.4278312779460425,
      0.5057577812572482,
      0.21182493736252006,
      -6.637258147801361e-05
    ],
    "head_ablate": [
      0.06380271534877746,
      0.08744848453071707,
      0.05543180230404943,
      0.07001432753439829
    ]
  },
  "gates": {
    "P1_trainable_all_heads": true,
    "P2_h4_beats_h1": false,
    "P2_all_heads_help": true,
    "P3_source_root_causal": true,
    "P3_details_pairs_causal": true,
    "P4_seed_private_or_shared": true,
    "P5_h4_beats_flat": false
  }
}
```

## Execution note

The training process finished at `2026-07-20T02:01:23+08:00` with
`exit_code=0`.  The enclosing systemd unit later reported exit code `23`
because its first non-root rsync to NAS was denied; the original error is kept
in `stderr.log`.  The complete directory, including all three checkpoints, was
subsequently copied with sudo and verified at
`/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_battle_full/`.
