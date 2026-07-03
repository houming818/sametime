# S1 Local Flip Echo

Created: 2026-07-03
Owner: Codex Review
Stage: S1 Echo

## Claim

`S1-ECHO-LOCAL-FLIP-C01`

The more realistic S1 echo perturbation is local same-algebra TreeHeap flip,
not full-sentence flip:

```text
sentence
-> choose a local span
-> build a local TreeHeap for that span
-> Flip(span_root, full_depth)
-> keep the rest of the sentence unchanged
-> learn inverse recovery
```

This tests whether S1 echo can recover from local phrase-like reorderings while
preserving the rest of the sentence.

## Predict

`P-S1-ECHO-LOCAL01`

If the local TreeHeap flip route is meaningful:

1. Hard local TreeHeap flip applied twice should restore every sentence exactly.
2. A learned inverse route conditioned on span length and span start should
   recover held-out sentences with high exact/token/edit scores.
3. Recovery should be reported by sentence length, span length, and span
   position.
4. Evidence must include readable examples.

## Boundary

This proof still gives the model the perturbed span start/length. Therefore it
does not prove automatic `node/depth` discovery. It is the realistic successor
to full-tree flip smoke proof.

## Experiment

Script:

```text
ara/s1-echo/src/s1_local_flip_echo_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_local_flip_echo_probe/
```

Host:

```text
io.grepcode.cn
```

Dataset:

```text
WMT17 English side
samples = 20,000
train/test/OOD = 16,000 / 2,000 / 2,000
sentence length = 8..32
vocab = 8192
span length = 2..8
```

Perturbation:

```text
target_sentence
-> choose span metadata (start, length)
-> WriteLeaves(span)
-> Flip(span_root, full_depth)
-> observed_sentence
```

Recovery:

```text
observed_sentence
-> fixed random token codebook leaf state
-> learned span-conditioned inverse route
-> canonical state
-> nearest-codebook token readout
```

The fixed codebook is intentional: the proof should test structural recovery,
not whether a decoder can memorize token IDs.

## Evidence Summary

```text
hard_local_treeheap_closure_exact = 1.0000

learned_ood_exact                 = 1.0000
learned_ood_token_acc             = 1.0000
learned_ood_edit_similarity       = 1.0000

no_inverse_ood_exact              = 0.0010
no_inverse_ood_token_acc          = 0.7698
no_inverse_ood_edit_similarity    = 0.7376
```

All span lengths `2..8` and all span-position buckets
`front/middle/back` reached OOD exact `1.0000` in this run.

## Interpretation

Supported:

```text
Given a local span address and span depth, TreeHeap same-algebra local flip can
be inverted by a learned route on held-out real WMT English sentences.
```

Not proved:

```text
automatic span discovery
automatic depth discovery
semantic understanding
translation
WMT quality improvement
```

Next claim should remove the span metadata:

```text
learn P(node, depth | H, context)
```
