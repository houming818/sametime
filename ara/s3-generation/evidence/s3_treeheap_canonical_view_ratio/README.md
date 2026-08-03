# S3-TREEHEAP-CANONICAL-VIEW-C05 Evidence

Status: preregistered; smoke and one-seed screening pending.

This directory records the matched continuation-training experiment defined in
`../../logic/treeheap_canonical_view_ratio.md`.

Registered one-seed screening:

```text
initial checkpoint: taskd 89 checkpoint_best.pt
ratios:             0.0, 0.2, 0.4, 0.6
view seed:          9101
continuation rows:  same 300,000 raw-line interval per arm
learning rate:      2e-4
optimizer:          cloned taskd 89 optimizer state
primary metric:     native Butterfly validation NLL
```

Every source-target row appears once per arm. Canonical and Butterfly subsets
share one token-normalized CE loss and one optimizer step. The run does not
duplicate examples or add view labels to the loss.

Expected generated artifacts:

```text
startup.json
p000_seed9101/summary.json
p020_seed9101/summary.json
p040_seed9101/summary.json
p060_seed9101/summary.json
summary.json
stdout.log
```

The single-seed screen cannot upgrade the claim. A positive screen only selects
a candidate ratio for a later multi-seed confirmation.
