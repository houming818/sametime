# Addressed recursive lifting depth audit

Claim ID: `S3-TREE-LIFT-RECURSIVE-C01`

Status: **contract phase supported / language depth audit open**

## Correction inherited from S3-DECODER-DEPTH-GROWTH-C01

The previous depth-growth pilot recursively produced level tensors but passed
each complete level to an unconditional set reader. Parent-child edges were not
part of READ. It therefore tested learned multiresolution pooling, not TreeHeap
recursive decoding. Its metrics remain valid for that mechanism but do not
falsify a TreeHeap depth claim.

This audit reuses the already supported lifting implementation in
`s2_adaptive_lifting_wmt.py` and the parent-to-child active-mass recurrence in
`s2_lifting_pump_wmt.py`. It does not introduce a new algebra merely to obtain a
positive result.

## Concept contract

For a binary node with anchor child `a` and predicted child `b`, the shared
lifting kernel computes

$$
r=b-P_\theta(a),\qquad p=a+U_\phi(r).
$$

The inverse computes

$$
a=p-U_\phi(r),\qquad b=r+P_\theta(a).
$$

The encoder must recursively feed each level's parent states into the next
FOLD. The decoder must carry active route mass from each parent only to that
parent's registered children. A tensor layout is allowed as an efficient heap
storage representation; losing the parent-to-child index relation is not.

The implementation must preserve the following conceptual fields, explicitly
or by a documented heap-index formula:

```text
address, parent_address, child_addresses, depth, leaf_span,
coarse_state, addressed_detail_residual
```

## Main question

After one model is trained on real Chinese text with unrestricted recursive
READ, does freezing it and increasing a hard recursive depth cap reveal a
source-causal, address-causal rate-distortion curve?

A depth cap is not a list prefix. At a capped node the route kernel is forced to
STOP. At an uncapped node its expand mass is distributed only to registered
children:

$$
m_{i\to j}=m_i(1-p_{stop,i})p(j\mid i),\qquad j\in children(i).
$$

The conservation audit requires

$$
m_i=m_i p_{stop,i}+\sum_{j\in children(i)}m_{i\to j}
$$

up to floating-point error.

## Data and training

- source: 128 BPE tokens from `/home/nio/datasets/pretrain`;
- target: the following 32 BPE tokens plus EOS;
- tokenizer: `/home/nio/datasets/wmt_massive/sp_bpe_massive.model`;
- one adaptive learned-update lifting encoder;
- one probabilistic recursive decoder;
- one shared predictor/update/route parameter set at every depth;
- train the native unrestricted route only;
- use identical document stream, batches, initialization seed, optimizer, and
  update count for recursive, flat, and target-only controls;
- pilot: one seed; replication: three seeds only after mechanism gates pass.

Training does not randomly assign the same complete target to independent
depth arrays. Depth is an evaluation intervention on a frozen recursive model.

## Deterministic contract proof

Before language training, an 8-leaf synthetic heap must record every recursive
FOLD and UNFOLD edge.

1. Full residual roundtrip state MSE below `1e-10`.
2. Every non-root node has exactly one parent.
3. Every internal node's registered children match the heap index rule.
4. STOP plus child route mass conserves incoming mass within `1e-6`.
5. Swapping two addressed residuals at one depth changes only the two affected
   subheaps in the deterministic inverse.

Failure of this phase invalidates the implementation. Language metrics must not
be interpreted.

### Contract result (2026-07-19)

The deterministic probe passed on `io` CPU:

- full lifting roundtrip MSE: `3.2341e-15`;
- maximum route-mass conservation error across depth caps: `5.9605e-8`;
- swapping two finest residual addresses changed the intended four-leaf span by
  MSE `1.4019` while outside-span MSE remained `2.7198e-15`;
- complete addresses, unique parents, heap child rules, span partitions, and
  finite-value gates all passed.

This supports only the algebra/address contract. The real-text source, depth,
and outline predictions below remain open.

## Frozen evaluation matrix

For each depth cap `d=0..D`, report:

- teacher-forced future NLL and perplexity;
- free-generation nonempty and unique-output rates;
- source-shuffle damage;
- root-shuffle damage;
- within-depth residual-address swap damage;
- pre-FOLD sibling-pair break damage;
- route mass by depth and mass-conservation error;
- active nodes, READ calls, and estimated FLOPs.

Also report `target_only`, `flat_seq`, and full recursive READ. Flat may win; the
claim concerns mechanism identity before superiority.

## Pilot predictions

`P0 contract`: all five deterministic contract gates pass.

`P1 source-causal root`: depth-0 root beats target-only by at least `0.05` NLL,
and source shuffling raises depth-0 NLL by at least `0.10`.

`P2 recursive growth`: full recursive READ beats forced-root by at least `0.05`
NLL, and at least two successive depth-cap increments reduce NLL by `0.01` or
more.

`P3 address causality`: swapping residual addresses while preserving their
multiset raises full recursive NLL by at least `0.02`.

`P4 pairing causality`: breaking pre-FOLD sibling pairs raises NLL by at least
`0.02` at two or more FOLD depths.

`P5 closure`: trained lifting FOLD/UNFOLD state MSE remains below `1e-10`.

`P6 route conservation`: maximum route-mass conservation error is below
`1e-6`, and at least two depths receive mean STOP mass of `0.05` or more.

`P7 fair streams`: all model controls consume byte-identical training batch
IDs. A hash of the first 1,024 batch IDs is recorded for each control and must
match.

No threshold is registered for beating flat in the pilot. That comparison is
reported, not hidden.

## Interpretation

- P0 failure: code does not implement the claimed algebra.
- P0 pass, P1 failure: lifting is valid but the root is not a useful source
  outline for this task.
- P1/P2 pass, P3/P4 fail: multiresolution content is useful, but TreeHeap
  addresses and grouping are not established.
- P0--P6 pass: pilot support for a recursive, source- and address-causal
  coarse-to-detail protocol.
- Flat wins: no architecture superiority claim, even if mechanism gates pass.

## Boundaries

This experiment does not require internal states to translate into human-readable
summaries. It does not prove syntax, world knowledge, consciousness, optimal
branching factor, compression efficiency, or superiority over Transformer.
