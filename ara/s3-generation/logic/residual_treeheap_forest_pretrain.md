# Residual TreeHeap Forest Pretraining

Date: 2026-07-13
Status: designed / proof queued
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
