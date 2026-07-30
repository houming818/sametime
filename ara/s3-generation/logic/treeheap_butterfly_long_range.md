# TreeHeap Butterfly Long-Range Communication

Status: preregistered, awaiting evidence

Date: 2026-07-30

Owners: Houming818 and Codex Review

## 1. Problem

A fixed adjacent binary TreeHeap gives nearby leaves a short communication path,
but leaves in different large subheaps meet only at a coarse ancestor. For an
eight-leaf tree, leaf `0` and leaf `7` communicate through six tree edges. An
exact FOLD/UNFOLD contract preserves data, but it does not guarantee that a
finite-dimensional root explicitly preserves every long-range relation.

Dense all-node attention would shorten the path, but it would also replace the
TreeHeap READ protocol with a flat `N x N` comparison. Circular shifts avoid
one boundary but introduce a false last-to-first adjacency and do not reduce
every pair distance.

This experiment tests a third construction: a fixed-width XOR Butterfly layer
built from local two-node kernels over TreeHeap binary addresses.

## 2. Construction

Let `N = 2^D` and let leaf `i` have a `D`-bit address. At stage `s`, leaf `i`
communicates with:

```text
partner_s(i) = i XOR 2^s
```

Each stage contains `N/2` disjoint pairs. After `D` stages, information from
any address can reach every address without a dense pairwise matrix.

The deterministic algebra contract uses the orthogonal two-node kernel:

```text
c = (a + b) / sqrt(2)
d = (a - b) / sqrt(2)
```

The learned routing probe uses the same XOR pairing schedule. A shared
query-conditioned value decides whether each stage preserves or exchanges its
pair. There is no STOP gate. Every stage runs; an unused stage may learn a
near-identity value.

## 3. Claim

### S3-TREEHEAP-BUTTERFLY-LONGRANGE-C01

A fixed-capacity XOR Butterfly extension composed from local two-node kernels
can give TreeHeap leaves an all-address receptive field in `log2(N)` stages,
while preserving an exact orthogonal information path. On an
address-conditioned long-range retrieval task, it should outperform a
same-depth adjacent-only kernel and a finite root-bottleneck tree, including on
held-out address combinations and a longer unseen width, without constructing
an `N x N` attention matrix.

This is a communication-mechanism claim. It is not a language, semantic,
translation-quality, compression-ratio, or Transformer-superiority claim.

## 4. Predictions

### Deductive contract

For widths `8, 16, 32, 64`:

1. Forward followed by inverse Butterfly has MSE at most `1e-10`.
2. Relative L2 energy error is at most `1e-6`.
3. The influence relation covers all `N x N` input-output address pairs after
   exactly `log2(N)` local stages.
4. The orthogonal path preserves gradient norm within `[0.999, 1.001]`.
5. Pair operations equal `(N / 2) * log2(N)` and no rank-4 `B x N x N x D`
   attention tensor is constructed.

### Inductive routing probe

Use random token arrays and a query mask `q`. The target at output address `i`
is:

```text
target[i] = source[i XOR q]
```

Train at width 32 only on masks whose binary Hamming weight is one or two.
Evaluate on unseen width-32 masks with Hamming weight at least three, including
the maximal-distance mask `31`, and then evaluate without retraining at width
64, including mask `63`.

Across three registered seeds:

1. Butterfly held-out width-32 token accuracy is at least `0.95`.
2. Butterfly maximal-distance accuracy is at least `0.90`.
3. Butterfly unseen-width-64 accuracy is at least `0.90`.
4. Butterfly beats adjacent-only and root-bottleneck controls by at least
   `0.25` absolute accuracy on held-out width-32 masks.
5. The learned stage value is query-sensitive: changing a query bit changes
   the corresponding stage's exchange probability by at least `0.50`.

## 5. Controls

| Arm | Structure | Purpose |
|---|---|---|
| `butterfly` | XOR partner changes at every binary address bit | proposed sparse long-range mechanism |
| `adjacent_only` | repeats only `(0,1), (2,3), ...` pairing | tests whether depth alone is sufficient |
| `root_bottleneck` | recursively folds all leaves to one finite vector, then predicts all addresses | tests the coarse shared-root bottleneck |

All arms use the same token vocabulary, training examples, optimizer updates,
token-level cross entropy, and output vocabulary. Parameter counts and runtime
are reported; exact equality is not claimed where the topology requires
different modules.

## 6. Falsification

Reject the full claim if any deductive contract fails, or if the learned
Butterfly misses the registered long-range/OOD gates in at least two of three
seeds. Downgrade it to an algebra-only result if exact reachability succeeds but
the learned routing probe does not.

Even if every gate passes, do not infer:

- that WMT or dialogue quality improves;
- that binary XOR addresses match linguistic dependency structure;
- that Butterfly is more compute-efficient than a tuned Transformer;
- that private semantic protocol emergence has occurred.

## 7. Planned artifacts

```text
src/s3_treeheap_butterfly_long_range.py
evidence/s3_treeheap_butterfly_long_range/
  command.sh
  trace.jsonl
  summary.json
  README.md
```

The first run is deliberately synthetic because it isolates address distance.
Only after the mechanism survives this falsification should the same layer be
inserted into the frozen WMT TreeHeap encoder/decoder for a matched ablation.

