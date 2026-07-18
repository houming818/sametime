# Full-corpus decoder causal audit

Claim ID: `S3-FULL-DECODER-CAUSAL-C01`

Status before experiment: **open / preregistered**

## Question

The full-corpus model can lower teacher-forced NLL and produce Chinese-shaped
text. That does not by itself show what the decoder reads. The decoder may:

1. ignore source memory and rely on its target prefix;
2. read source token content as an unordered bag;
3. read states produced by TreeHeap FOLD at coarser resolutions.

This audit freezes one checkpoint and intervenes on the memory presented to the
same decoder. No parameter is updated.

## Current decoder

For decoder query `q` and memory nodes `m_i`, attention is

$$a_i = \operatorname{softmax}_i(q^T m_i / \sqrt d)$$

$$c = \sum_i a_i m_i$$

The GRU then receives the previous target token and `c`.

For any permutation `pi` of the memory nodes, permuting the mask with the
nodes leaves `c` unchanged. Therefore the decoder is exactly permutation
invariant at a fixed frontier:

$$\operatorname{Decoder}(\pi(M),\pi(mask),p)=\operatorname{Decoder}(M,mask,p)$$

This is a deductive property of the implementation, not an empirical guess.
At the leaf frontier it means the decoder cannot directly read left/right
addresses or token order. Structure can still matter at a coarser frontier,
because FOLD changes node values before attention reads them.

## Interventions

The same held-out samples and gold target prefixes are used in every condition.

| Condition | Change | Meaning |
|---|---|---|
| `native` | correct state | reference |
| `zero` | replace every state vector by zero | remove encoder information |
| `sample_swap` | give each target another sample's state | test source conditioning |
| `node_reverse` | reverse nodes and mask within each sample | exact decoder permutation test |
| `source_sibling_swap` | swap adjacent valid source tokens before FOLD | test left/right pairing |
| `source_half_swap` | exchange valid source halves before FOLD | test larger subheap placement |

The audit runs at root, a middle frontier, and leaf. It also reports all native
frontier NLL values from root to leaf.

## Metrics

For intervention `x` at depth `d`:

$$\Delta\operatorname{NLL}_{x,d}=\operatorname{NLL}_{x,d}-\operatorname{NLL}_{native,d}$$

It also records the maximum absolute logit difference between `native` and
`node_reverse`.

## Predictions

`P1 state causality`

At leaf, `sample_swap` or `zero` increases NLL by at least `0.20`. If not, the
decoder is predominantly a target-prefix language model at this checkpoint.

`P2 fixed-frontier permutation invariance`

At every tested frontier, reversing memory nodes and their mask changes logits
by at most `1e-5` and NLL by at most `1e-6`. Failure means the implementation
does not match the stated attention equation.

`P3 FOLD structure sensitivity`

Sibling or half swaps have materially more effect at root/middle than at leaf:

$$\max(\Delta NLL_{swap,root},\Delta NLL_{swap,middle})-\Delta NLL_{swap,leaf}\ge 0.02$$

At leaf the source swap only permutes the same token embeddings, so its expected
damage is approximately zero. A positive coarse-frontier margin shows that
FOLD encodes the changed grouping into state values. It does not by itself show
that the grouping is linguistically correct or better than a flat baseline.

## Decision table

| Result | Interpretation |
|---|---|
| P1 fails | decoder-only shortcut dominates |
| P1 passes, P3 fails | source content is used, TreeHeap grouping is not causal |
| P1 and P3 pass | content and FOLD grouping are both causal at different frontiers |
| P2 fails | audit or implementation bug; do not interpret P3 |

## Boundaries

This is a frozen single-checkpoint mechanism audit. It cannot establish model
quality, semantic correctness, TreeHeap superiority, consciousness, or world
knowledge. Teacher forcing is deliberately held constant to isolate source
memory causality; free-generation quality remains a separate gate.

## Result: step 160K frozen checkpoint

Host: `io`, CPU only, six threads. The pilot used 8 held-out samples and the
confirmation used 64. The confirmation completed in 255.9 seconds while the
300K GPU training process continued at approximately 21K target tokens/second.

### Confirmation metrics

| Frontier | Native NLL | Zero damage | Wrong-sample damage | Sibling-swap damage | Half-swap damage |
|---|---:|---:|---:|---:|---:|
| root, depth 0 | 5.2319 | +0.3154 | +0.7669 | +0.00834 | +0.00381 |
| middle, depth 3 | 5.1910 | +0.3563 | +0.7448 | +0.00642 | +0.00681 |
| leaf, depth 6 | 5.1321 | +0.4152 | +0.8084 | approximately 0 | approximately 0 |

`P1` passed. Replacing the state with another sample increased NLL by about
`0.75-0.81`, and zeroing it cost `0.32-0.42`. The decoder is not merely using
the gold target prefix; source content is strongly causal.

`P2` missed its preregistered `1e-5` maximum-logit tolerance. Reversing nodes
changed NLL by no more than `3.3e-7`, but different floating-point reduction
orders produced maximum individual-logit differences up to `3.62e-4`. The
real-arithmetic permutation-invariance derivation still applies, and the NLL
result confirms functional invariance at measured precision. The numeric gate
remains marked failed rather than being relaxed after seeing the result.

`P3` failed. The largest 64-sample root/middle structural damage was only
`0.00834`, below the registered `0.02` margin. The 8-sample pilot had appeared
close to the gate (`0.01978`), but the larger audit reduced rather than
strengthened that signal.

### Decision

Status: **partial support / source conditioning supported / current topology
use not supported**.

The word "decoder shortcut" must now be used precisely:

- a decoder-only or target-prefix-only shortcut is rejected;
- a content-memory shortcut is supported;
- meaningful use of the current FOLD topology in the normal full-corpus path is
  not supported by this checkpoint.

The clean-leaf path is an unordered memory set by construction. Coarse FOLD
states differ numerically after source regrouping, but the decoder's measured
benefit from that difference is too small to pass the causal gate. The next
architecture change should constrain normal READ to selected root/subheap
frontiers or add an address-sensitive READ kernel. More data alone does not
repair this decoder protocol.

Evidence:

- `evidence/s3_full_decoder_causal_audit/` (8-sample pilot)
- `evidence/s3_full_decoder_causal_audit_64/` (64-sample confirmation)
