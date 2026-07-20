# Small Transformer benchmark

```json
{
  "claim": "S3-PRIVATE-PROTOCOL-TF-C02",
  "predict": "P-S3-PRIVATE-PROTOCOL-TF-02",
  "status": "competitive",
  "smoke": false,
  "host": "io",
  "seconds": 460.63783407211304,
  "config": {
    "baseline_summary": "ara/s3-generation/evidence/s3_private_protocol_battle_full/summary.json",
    "data": "/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv",
    "spm_model": "/home/nio/datasets/wmt_massive/sp_bpe_massive.model",
    "evidence_dir": "ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full",
    "recipes": [
      "same_recipe",
      "standard_recipe"
    ],
    "seeds": [
      71901,
      71902,
      71903
    ],
    "source_col": 1,
    "target_col": 0,
    "train_samples": 30000,
    "valid_samples": 2000,
    "test_samples": 2000,
    "max_scan": 300000,
    "min_len": 8,
    "max_len": 32,
    "batch_size": 64,
    "num_workers": 2,
    "device": "cuda",
    "tf_dim": 256,
    "tf_heads": 4,
    "tf_encoder_layers": 2,
    "tf_decoder_layers": 2,
    "tf_feedforward": 512,
    "tf_dropout": 0.1,
    "max_positions": 128,
    "same_epochs": 4,
    "standard_epochs": 8,
    "same_lr": 0.002,
    "standard_lr": 0.0005,
    "warmup_fraction": 0.1,
    "label_smoothing": 0.1,
    "smoke": false
  },
  "sampling": {
    "scanned": 300000,
    "eligible": 173313,
    "selected": 34000,
    "sampling": "deterministic_reservoir_then_shuffle"
  },
  "transformer": {
    "same_recipe": {
      "nll_mean": 6.442279684384492,
      "nll_stdev": 0.00623350375697195,
      "bleu4_mean": 2.9422168718848067,
      "bleu4_stdev": 0.15293674251121353,
      "parameters": 27278337,
      "seconds_mean": 52.17865435282389
    },
    "standard_recipe": {
      "nll_mean": 6.533036939035269,
      "nll_stdev": 0.004260404419119122,
      "bleu4_mean": 2.8941157559766975,
      "bleu4_stdev": 0.1916264468049261,
      "parameters": 27278337,
      "seconds_mean": 100.82245723406474
    }
  },
  "registered_baselines": {
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
  "treeheap_h1_minus_standard_transformer_nll": -0.40992535977972366,
  "parameter_gap_fraction_vs_h1": 0.012359903509500496,
  "boundary": "Small matched Transformer benchmark; not an industry-top or global-optimum result."
}
```

## Execution note

The formal run completed on `io` at `2026-07-20T11:13:09+08:00` with
`exit_code=0`.  The six checkpoints and all text evidence were verified under
`/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_transformer_benchmark_full/`.
