# S3-TREEHEAP-CANONICAL-VIEW-C05 Evidence

Status: completed; registered one-seed screening negative.

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

The runner may reuse an arm only when its completed `summary.json` matches the
registered claim, seed, ratio, source interval, and training length. This lets a
failed later arm resume without retraining an already completed independent arm.

## Result

Taskd 96 exposed and preserved a subgroup-padding failure after completing the
`p=0` arm. Commit `6dac477` trimmed each split view subgroup to its own real
source width. Taskd 98 verified the failing mixed-view path, and taskd 99 resumed
the completed control and finished all registered arms.

```text
p=0.0 native NLL: 3.2710   cross-view JS: 0.2378
p=0.2 native NLL: 3.2826   cross-view JS: 0.1011
p=0.4 native NLL: 3.2939   cross-view JS: 0.0910
p=0.6 native NLL: 3.3090   cross-view JS: 0.0838
```

The primary prediction was not confirmed because `p=0` remained best on native
NLL. Canonical mixing did produce a strong, monotonic cross-view consistency
effect without removing source dependence. See `summary.json`, the per-arm
Dreams, and the three `taskd-*.log` files for the complete record.
