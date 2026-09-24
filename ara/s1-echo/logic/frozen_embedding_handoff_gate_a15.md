# A15: Frozen TreeHeap embedding handoff gate

## Status

Preregistered before the formal run.

## Question

A11-A14 established that a TreeHeap probability-residual field can be fitted,
saved, reloaded, and scaled without routing collapse. A15 does not continue to
search for the lowest embedding NLL. It asks the next engineering question:

> Can a frozen TreeHeap coordinate table be consumed by a small downstream
> decoder under a fixed budget, and does the downstream model causally use the
> installed coordinates?

This is a handoff gate, not a claim that the coordinates are an optimal language
embedding or that they already solve TreeHeap FOLD/READ.

## Frozen candidates

Two previously sealed checkpoints are tested:

1. `A14-1M-D3`: the 1,000,000-line, depth-3 A14 endpoint.
2. `A13-200K-D5`: the independently trained 200,000-line, depth-5 A13 endpoint.

For token `i`, the handoff vector is constructed only from frozen routing state:

```text
z_i = normalize([signed_path_i, tanh(standardized_route_margin_i), leaf_one_hot_i])
```

No downstream label is used to construct `z_i`.

## Downstream tasks

### Token READ

A linear decoder receives one frozen coordinate and predicts its token identity.
The clean and Gaussian-noise accuracies measure whether the coordinate is a
usable, robust interface. This task is deliberately elementary.

### Short-sequence Echo

Real held-out WMT English sentences after the A14 sealed region are tokenized.
Only target-vocabulary tokens are retained in order, producing length 3-8
sequences. A one-layer bidirectional GRU receives the coordinate sequence and
predicts every original token. Token accuracy and exact-sequence accuracy are
reported.

### Masked center READ

The center coordinate is replaced by zero and the same class of sequence model
predicts the missing token from the remaining sequence. This is the semantic
handoff probe: unlike Echo, token identity cannot be copied from the center.

## Matched controls

Every arm has the same coordinate width and downstream architecture/budget.

- `treeheap_native`: frozen TreeHeap coordinates.
- `frequency_shuffle`: TreeHeap rows permuted only within frequency bins.
- `random_fixed`: frozen Gaussian coordinates, row-normalized like TreeHeap.
- `context_projection`: frozen random projection of the checkpoint's empirical
  context-probability rows into the same width.
- `learned_embedding`: ordinary trainable lookup initialized randomly.

For a decoder trained on `treeheap_native`, evaluation also replaces its table
with `frequency_shuffle` and `zero`. These are interventions on one trained
decoder, not separately optimized alternatives.

## Fixed contract

- Three seeds: `20260930, 20260931, 20260932`.
- Same WMT file, SentencePiece model, extracted sequences, batches, step budget,
  decoder architecture, and initial downstream weights within a seed.
- Embedding candidates and controls use the checkpoint's native handoff width.
- Frozen arms cannot update their coordinate table.
- The learned baseline may update only its lookup plus the same decoder.
- Formal data starts after valid English line 1,100,000, outside the A14
  1,000,000-line fit and 100,000-line sealed test regions.
- All checkpoint, data-region, tokenizer, and output hashes are recorded.

## Preregistered gates

The candidate passes the mechanical handoff gate when all three seeds satisfy:

1. all coordinates and losses are finite;
2. the frozen table hash is unchanged after training;
3. native Token READ clean accuracy is at least `0.95`;
4. native Echo token accuracy is at least `0.80`;
5. replacing the native table by zero lowers Echo token accuracy by at least
   `0.50` absolute;
6. replacing the native table by frequency-shuffled rows lowers Echo token
   accuracy by at least `0.25` absolute.

The semantic handoff gate is evaluated separately on seed medians:

7. native masked-center top-1 accuracy exceeds `random_fixed` by at least
   `0.01` absolute; and
8. native masked-center top-1 accuracy exceeds `frequency_shuffle` by at least
   `0.01` absolute.

Failure of gates 7-8 means the coordinates are mechanically readable but do not
yet demonstrate useful corpus geometry. It does not erase A11-A14.

`learned_embedding` and `context_projection` are references, not pass/fail
targets. No run is stopped because a quality curve is non-monotonic. Automatic
stopping is allowed only for non-finite values, CUDA/OOM/Xid failure, corrupted
artifacts, or a violated fixed contract.

## Claim

`S1-F-HANDOFF-A15-C01` remains open until the formal evidence is complete.

