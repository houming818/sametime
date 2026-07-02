# S1 Sentence Flip Echo

Created: 2026-07-02
Owner: Codex Review
Stage: S1 Echo

## Claim

`S1-ECHO-SENT-C01`

On real short English sentences, TreeHeap sentence echo should be evaluated as a
same-algebra perturbation/recovery task:

```text
canonical sentence
-> WriteLeaves
-> TreeHeap Flip(root, full_depth)
-> learned inverse TreeHeap route
-> canonical state
-> shared token decoder
-> canonical sentence
```

The perturbation must be produced by the TreeHeap `Flip(node, depth)` operator,
not by an external array reverse.

## Predict

`P-S1-ECHO-SENT01`

If the current S1 echo direction has real evidence rather than only empty
theory:

1. Hard TreeHeap flip closure should restore every sentence exactly:

```text
DecodeLeaves(Flip(Flip(WriteLeaves(x)))) = x
```

2. A learned inverse route should recover a high percentage of real held-out
sentences, with metrics reported by sentence length.
3. A no-inverse baseline should fail on the same disturbed input.
4. Evidence must include readable sentence examples:

```text
observed -> restored -> target
```

## Experiment

Dataset:

```text
WMT17 English side, whitespace tokenized
length 3..32
20,000 sentence samples
train/test/ood = 16,000 / 2,000 / 2,000
```

Perturbation:

```text
H = TreeHeapWrite(sentence)
H_observed = Flip(H, root, full_depth)
```

For a balanced TreeHeap with arbitrary number of leaves, recursive full-depth
flip reverses the leaf order, but the operation is defined on the TreeHeap:

```text
flip(node):
  node.left, node.right = flip(node.right), flip(node.left)
```

It is not defined as `array[::-1]`.

Learned recovery:

```text
observed leaf embeddings
-> learned length-conditioned inverse route
-> canonical state
-> shared decoder
```

Loss:

```text
CE(decoded_tokens, canonical_tokens)
+ state_loss(canonical_state, canonical_leaf_embeddings)
+ route_entropy
```

Metrics:

```text
exact_match
token_acc
edit_similarity
by_length_exact
by_length_token_acc
by_length_edit_similarity
readable examples
```

## Falsification

Downgrade or reject if:

```text
hard TreeHeap flip closure is not exactly 1.0;
learned inverse is not materially better than no-inverse baseline;
sentence exact collapses with length even on short held-out sentences;
evidence lacks readable observed/restored/target examples;
the perturbation is implemented as external array reverse rather than TreeHeap flip.
```

## Boundary

This does not prove:

```text
translation
semantic understanding
automatic local node/depth discovery
unsupervised natural trigger learning
```

It only tests sentence-level same-algebra flip echo.
