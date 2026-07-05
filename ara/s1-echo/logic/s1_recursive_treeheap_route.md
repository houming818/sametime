# S1 Recursive TreeHeap Route

Claim: `S1-RECURSIVE-ROUTE-C01`

Predict: `P-S1-RECURSIVE-ROUTE01`

Date: 2026-07-05

## Why This Exists

The previous S1 echo blogs and claims used the word "route" too loosely.

Some experiments used:

```python
route_logits[length, out_pos, in_pos]
state = route @ leaf
```

That is a learnable flat route matrix.  It can recover a sequence, but it is not
TreeHeap routing.

A real TreeHeap route must start at a heap node and move through heap addresses:

```text
i = 1
while not stop:
  S_i = [arr[i], arr[2i], arr[2i+1]]
  action = K_theta(q, S_i, address_i)
  action in {stop, left, right}
  if left:  i = 2i
  if right: i = 2i + 1
```

This proof rebuilds the S1 flip-echo route with that stricter definition.

## Claim

TreeHeap can implement S1 mirror recovery as a recursive `stop/left/right`
reader over heap addresses, not as a flat `L x L` route matrix.

## Predict

If the claim is true:

1. hard TreeHeap mirror algebra should recover OOD sentences exactly;
2. a learned recursive route kernel should recover OOD sentences exactly or
   near-exactly;
3. the evidence should contain actual `stop/left/right` action traces;
4. the old length-indexed flat route matrix should fail on unseen lengths,
   because it has one independent table per length.

## Experiment

Script:

```text
ara/s1-echo/src/s1_recursive_treeheap_route_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_recursive_treeheap_route_probe/
```

Data:

```text
/mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv
```

Setup:

```text
samples:        20,000 WMT-massive English sentences
heap max_len:   32 leaves
train lengths:  3..24
OOD lengths:    25..32
```

For a canonical sentence:

```text
[w0, w1, w2, ...]
```

we write tokens into heap leaves.  Then `mirror(root)` flips the whole heap.
To recover canonical token position `p`, the reader must walk from `arr[1]` to
the mirrored leaf address:

```text
target_leaf = max_len - 1 - p
```

The learned TreeHeap reader does not receive a direct `out_pos -> in_pos`
matrix.  It receives local address features and must repeatedly choose:

```text
left / right / stop
```

The flat baseline is the old style:

```text
route_logits[length, out_pos, in_pos]
```

It is trained on lengths `<=24` and evaluated on unseen lengths `25..32`.

## Results

OOD metrics:

| metric | value |
|---|---:|
| hard TreeHeap oracle exact | `1.0000` |
| learned recursive route exact | `1.0000` |
| learned recursive token acc | `1.0000` |
| flat length-matrix exact | `0.0000` |
| flat length-matrix token acc | `0.0097` |

Pass checks:

```json
{
  "oracle_ood_exact": true,
  "recursive_ood_exact_ge_0_99": true,
  "recursive_uses_step_actions": true,
  "flat_length_matrix_fails_unseen_lengths": true
}
```

Example recursive actions for `pos=0` under a 32-leaf mirrored heap:

```json
{
  "pos": 0,
  "target_leaf": 31,
  "actions": [2, 2, 2, 2, 2, 0],
  "node": 63
}
```

Here `2` means `right` and `0` means `stop`.  The reader starts at `arr[1]`,
walks right five times, reaches leaf node `63`, then stops.  This is a heap
path, not a matrix row.

## Interpretation

This repairs the mechanism boundary:

```text
flat route matrix:
  useful baseline
  not TreeHeap

recursive TreeHeap route:
  arr[i]
  arr[2i]
  arr[2i+1]
  stop/left/right
  action trace
```

The proof supports a narrow TreeHeap route claim.  It does not rescue the
over-written parts of SPR-041 / SPR-043 / SPR-044.  Those older numeric results
remain useful only as mechanism-limited evidence.

## Limits

This proof does not show:

- translation;
- semantic grounding;
- unsupervised discovery of which span should be flipped;
- natural-language trigger learning;
- advantage over a flat shared route baseline.

The route target position is supervised.  The next proof must remove or weaken
that supervision and compare against stronger flat/shared/pointer baselines.

