# Residual TreeHeap Forest Pretraining

Date: 2026-07-13
Status: rejected as written / structural side-result supported
Claim: `S3-RESIDUAL-FOREST-C01`
Predict: `P-S3-RESIDUAL-FOREST-01`

## Question

Can a TreeHeap grow into a trainable encoder/decoder by preallocating several
parameter TreeHeap kernels and preserving one residual state stream through
recursive folds?

This experiment does not ask whether adding parameters always helps. It asks a
narrower question:

> Under the same data, parameter count, optimizer, and TreeHeap addresses, does
> the native residual update preserve useful state and gradient information
> better than replacing the state at every fold?

## Mathematical Objects

The experiment keeps three objects separate:

```text
H       runtime TreeHeap state produced by one text block
Theta   long-term learnable parameter TreeHeap forest
Q       the next-token query answered by the root decoder
```

For a node `i`, a local kernel reads the three-node subheap:

```text
S_i = [H[i], H[left(i)], H[right(i)]]
```

Each head `h` owns a three-slot parameter TreeHeap:

```text
Theta_h = [Theta_h,root, Theta_h,left, Theta_h,right]
```

The same `Theta_h` is convolved at every valid address. It is therefore a
shared TreeHeap operator, not a separate matrix for every sentence position.
The implementation uses a low-rank nonlinear form:

```math
K_h(S_i) = W^{out}_h\,\mathrm{GELU}(W^{root}_hH[i]
            + W^{left}_hH[2i] + W^{right}_hH[2i+1] + b_h).
```

A learned gate makes a probability bucket over heads:

```math
g_i = \mathrm{softmax}(G_\Theta(S_i)),
\qquad
\Delta H[i] = \sum_h g_{i,h} K_h(S_i).
```

The residual model performs the TreeHeap-native write:

```math
H_{t+1}[i] = \mathrm{Norm}(H_t[i] \oplus \alpha\Delta H_t[i]),
```

where `oplus` is vector addition in `arr[i]` while the TreeHeap address and
child rules remain fixed. The matched no-residual control performs:

```math
H_{t+1}[i] = \mathrm{Norm}(\Delta H_t[i]).
```

The identity term in the residual model gives the local Jacobian

```math
\frac{\partial H_{t+1}}{\partial H_t}
= I + \alpha\frac{\partial \Delta H_t}{\partial H_t}.
```

This does not guarantee semantics. It creates a path along which an existing
state and its gradient can survive repeated recursive convolution.

## Encoder And Decoder

The encoder is deliberately narrow:

```text
64 token ids
  -> token vectors written to ordered TreeHeap leaves
  -> shared kernel forest recursively folds adjacent subheaps
  -> one root H_root
```

The decoder sees only `H_root`:

```text
H_root -> linear probability bucket over the next token
```

There is no leaf-attention bypass. If the root loses the input information, the
decoder cannot recover it from the original array.

## Data Protocol

The full run uses all readable documents from the raw Chinese pretraining
sources:

```text
news2016zh train
wiki_zh shards
webtext2019zh train
```

All sources are first tokenized once with the existing 16K SentencePiece model.
The source corpus and derived non-overlapping blocks remain on `io`'s local
disk under `/home/nio/datasets`; NAS is used only for a final evidence backup.
Blocks are fixed `65 x uint16` rows:

```text
first 64 ids -> encoder input
last id      -> next-token target Q
```

The manifest records source files, sizes, block counts, tokenizer identity, and
shard checksums. Every compared model reads the exact same rows.

## Proof Stages

### Stage A: mechanism gate

Use a bounded subset and train:

```text
residual_forest     4 heads, residual update
noresidual_forest   4 heads, state replacement
single_residual     1 head, residual update
```

Residual and no-residual models have exactly the same trainable parameters.

Measure:

```text
validation NLL / perplexity / top-1 / top-5
root representation variance and mean pairwise cosine
embedding and early-kernel gradient norms
head gate entropy and utilization
head-ablation NLL increase
left/right address shuffle NLL increase
```

### Stage B: full-corpus evidence

If Stage A is numerically stable, continue the residual and no-residual models
over every prepared training shard. Save checkpoints and validation results at
fixed step intervals. A long run may reject the hypothesis; duration is not a
pass criterion.

## Predictions

If the claim is correct:

```text
P1 residual_forest valid NLL < noresidual_forest valid NLL.
P2 its leaf-embedding gradient does not vanish relative to the root decoder.
P3 destroying left/right addresses increases NLL, showing use of tree order.
P4 removing at least one learned head increases NLL, and heads are not identical.
P5 root variance stays non-zero; the model does not map every block to one state.
P6 the trends persist, rather than disappear, on the full-corpus run.
```

The single-head run is diagnostic. More heads are useful only if they improve
held-out prediction and head interventions show non-redundant use.

## Decision Gates

Support as a mechanism pilot only if, across at least three seeds:

```text
residual NLL improvement over matched no-residual >= 0.02
address-shuffle NLL increase                        >= 0.02
maximum useful-head ablation NLL increase           >= 0.01
root feature variance                               > 1e-4
all metrics are finite and reproducible
```

Upgrade to full-corpus support only if the first two effects remain positive at
the end of the complete shard manifest.

## Falsification

Reject or redesign if:

```text
the no-residual model matches the residual model across seeds;
address destruction leaves NLL unchanged;
all heads collapse to interchangeable outputs;
root states collapse while the decoder exploits a hidden leaf bypass;
the result depends on different data or parameter budgets;
the full-corpus trend contradicts the bounded mechanism proof.
```

## What This Will Not Prove

Even a passing result does not prove consciousness, a world model, WMT
superiority, automatic semantic categories, or dynamic parameter growth. It
would establish a smaller architectural fact: a preallocated TreeHeap parameter
forest plus a native residual state stream is trainable on real text and uses
its recursive addresses measurably.

## Result (2026-07-14)

The complete run consumed all `38,251,247` prepared blocks from `7,564,966`
documents in one source pass:

```text
news blocks   28,052,019
wiki blocks    4,375,165
web blocks     5,824,063
elapsed        28,581 seconds (7.94 hours)
```

The main residual claim failed:

| model | parameters | valid NLL | top-1 | top-5 | address-destroy delta NLL |
|---|---:|---:|---:|---:|---:|
| four-head residual | 6,310,341 | NaN | 0.0010 | 0.0036 | NaN |
| four-head no residual | 6,310,341 | 6.2365 | 0.1196 | 0.2329 | +2.6252 |
| one-head residual | 6,197,874 | 6.4350 | 0.1038 | 0.2109 | +1.9113 |

The four-head residual trace first records a non-finite loss at step `62,400`.
Its final checkpoint has non-finite values in all eleven parameter tensors,
including `residual_scale`. Because only the latest checkpoint was retained,
the evidence cannot identify which tensor became non-finite first. The
one-head residual model remained finite and learned `residual_scale=0.004142`,
almost closing its residual branch. Therefore the supported conclusion is:

> An unconstrained learned global residual scale combined with the current
> four-head delta is not stable over a full corpus. This implementation is
> rejected; TreeHeap residual learning in general is not disproved.

The no-residual result is a separate structural positive:

```text
root variance                       0.05843
mean head gate                     [0.123, 0.152, 0.268, 0.457]
head gate entropy                   1.255 nats (max ln(4)=1.386)
mean pairwise head cosine           0.428
head-ablation delta NLL             +1.185, +2.467, +1.157, +3.485
left/right address destruction      +2.625 NLL
```

This supports `S3-TREEHEAP-ROOT-COMPRESS-C01`: the finite no-residual model
uses recursive address pairing and all four kernel heads. It does not establish
that the heads represent human-readable features, nor that the model beats a
matched Transformer/MLP.

## Next Residual Gate

Do not rerun the same residual equation unchanged. A valid repair experiment
must isolate numerical stability using:

```text
bounded alpha = alpha_max * sigmoid(a)
FP32 residual accumulation under mixed-precision training
per-head delta normalization before gated summation
non-finite detection that stops and preserves the last finite checkpoint
periodic immutable checkpoints around the first divergence region
```

The repaired residual variant must beat the already finite no-residual model;
merely avoiding NaN is not enough.
