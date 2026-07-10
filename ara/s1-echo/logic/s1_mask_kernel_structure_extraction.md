# S1 Mask Kernel Structure Extraction

Date: 2026-07-10
Status: executed / weak positive
Related claim: `S1-MASK-KERNEL-C01`

## Problem

SPR-048 proved that scalar loss can write finite rules into a parameter
TreeHeap forest. The next question is different:

```text
Can a kernel convolve over a masked string / masked TreeHeap and output a
probability bucket of plausible fillers?
```

This is a structure-extraction question, not a token-echo question.

## Claim

```text
S1-MASK-KERNEL-C01

TreeHeap mask kernel can convolve over a masked string / masked TreeHeap state
and output a probability bucket of structurally plausible fillers, rather than
only reconstructing an observed token.
```

## Data Hypothesis

The corpus contains repeated context-object observations:

```text
I ate rice
I ate noodles
I cooked rice
```

The held-out question is:

```text
I cooked [MASK] -> noodles
```

A pure pair-memory model has not seen `cooked + noodles`. A structure-aware
model may infer it because:

```text
rice and noodles share the object slot for ate;
rice also appears in the object slot for cooked;
therefore noodles may transfer to cooked.
```

The model is not given `food`, `drink`, or `place` labels during training.
Those labels are used only after training to audit top-k bucket purity.

## Minimal Model

Write the masked sentence into a complete 4-leaf TreeHeap:

```text
          root
        /      \
      left     right
     /   \     /   \
    I   verb  MASK  PAD
```

The TreeHeap model uses shared bottom-up compose:

```text
H_parent = Compose_Theta(H_left, H_right)
```

and a mask convolution/read state:

```text
S_mask = K_Theta(H_root, H_right, H_mask_leaf)
logits = S_mask @ object_embedding^T
```

The output is:

```text
P(object | masked_tree)
```

## Baselines

```text
pair_memory:
  count-based distribution for each verb/context

BoW MLP:
  bag-of-input-token neural baseline

flat sequence MLP:
  position-aware neural baseline without TreeHeap compose

shuffled TreeHeap:
  same TreeHeap model, but target objects shuffled in training
```

## Predict

If the claim is supported:

```text
P1. TreeHeap held-out MRR > pair memory held-out MRR.
P2. TreeHeap held-out MRR > shuffled TreeHeap held-out MRR.
P3. TreeHeap top-k bucket purity > shuffled TreeHeap bucket purity.
P4. TreeHeap output remains a useful probability bucket, not only a single-token echo.
```

If BoW/flat baselines match or beat TreeHeap, the result is only weak/mixed:
the data may not yet require TreeHeap structure, or the kernel may not be using
substructure strongly enough.

## Falsification

Downgrade or reject if:

```text
1. shuffled corpus matches TreeHeap;
2. pair memory matches held-out rank;
3. top-k candidates do not form the expected structure bucket;
4. output always collapses to one token;
5. BoW/flat baselines dominate under the same data and parameter budget.
```

## Result

Executed on `io.grepcode.cn` with CUDA:

```text
decision = weak positive / baseline-contested
pilot_pass = true

treeheap heldout MRR/top5/purity/entropy = 0.2833 / 1.0000 / 0.8667 / 0.6998
pair     heldout MRR/top5/purity         = 0.1200 / 0.0000 / 0.4000
bow      heldout MRR/top5/purity         = 0.2833 / 1.0000 / 0.8667
flat     heldout MRR/top5/purity         = 0.1976 / 0.5000 / 0.6000
shuffled heldout MRR/top5/purity         = 0.2532 / 0.6667 / 0.4667
```

Interpretation:

```text
positive:
  TreeHeap beats pair memory and shuffled control.
  TreeHeap top-k forms the intended structure bucket better than shuffled.

limitation:
  BoW MLP matches TreeHeap on this controlled corpus.
  Therefore this is not evidence of a TreeHeap-specific advantage yet.
```

The current toy corpus mostly asks for verb-context bucket prediction. A BoW
model can solve much of that. The next proof must require substructure, path,
or local span information that BoW cannot cheaply use.

## Evidence

The proof should save:

```text
summary.json
trace.jsonl
topk_examples.json
README.md
```

## Boundary

This is not WMT translation and not proof of natural-language understanding.
It is a controlled S1 proof that asks whether a masked TreeHeap kernel can
extract reusable local structure from observations.
