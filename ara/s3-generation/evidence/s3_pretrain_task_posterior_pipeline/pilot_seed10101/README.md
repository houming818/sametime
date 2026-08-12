# C10 Pretrain-to-Task Posterior Pipeline

Mode: `pilot`

Claim: `S3-TREEHEAP-PRETRAIN-POSTERIOR-C10`

Host: `io`

Seed: `10101`

## Result summary

- Pretraining: 100,000,768 target pieces, best valid NLL `5.680394`.
- Matched WMT task: PT test NLL `5.403696`, SC test NLL `6.291975`.
- Token BLEU4: PT `5.065443`, SC `1.171101`.
- Posterior: native JS `0.609404`, unigram JS `0.641004`.
- Native READ matched forced-leaf READ to `7e-8` NLL.
- Path-shape audit rejected a single linked path, but uniform leaf pooling cost
  only `0.027111` NLL.
- Observer-resolution thresholds prune numerical tails but do not repair the
  dominant leaf-resolution collapse.

This directory contains structured summaries and audits only. Large reloadable
checkpoints remain on `io` and are identified by SHA-256 in the summaries.
