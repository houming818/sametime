# C14: recursive autoregressive target TreeHeap

Status: partial mechanism support; generation claim rejected in controlled smoke

## Historical correction

Earlier WMT systems generated usable translations because the decoder predicted
`P(y_t | y_<t, H_x)`: a GRU carried the target prefix and issued a new recursive
query into the source TreeHeap at every token. C13 instead predicted 128 target
leaves in parallel from source state alone. Its rank-1/rank-2 comma/full-stop
collapse therefore does not contradict the earlier translation result.

## Claim

`S3-TARGET-TREE-AUTOREGRESSIVE-C14`:

> A fixed-capacity target TreeHeap grown from Zero by repeated
> `READ -> token collapse -> WRITE -> path FOLD` can replace the GRU target
> hidden state in WMT generation. The same lifting coordinates should carry
> the source and target states, while source and target-history interventions
> remain causal and greedy output avoids the C13 conditional collapse.

## Algorithm

At step zero:

\[
H_y^{(0)}=Zero.
\]

For each target token:

\[
q_t=READ_{path}(H_y^{(t)}),
\qquad c_t=READ_{src}(q_t,H_x),
\]

\[
p_t=softmax(D(q_t,c_t)),
\qquad z_t=collapse(p_t),
\]

\[
H_y^{(t+1)}=FOLD_{path}(WRITE(H_y^{(t)},t,E(z_t))).
\]

Only the newly written leaf-to-root path is recomputed. The target state has a
fixed maximum capacity and allocates no external nodes. Training writes the
gold token (teacher forcing); greedy inference writes the model token. There is
no GRU, LSTM, Transformer, or separate flat recurrent hidden state.

## Controlled smoke

- real aligned WMT English-to-Chinese rows;
- one shared token embedding and lifting predictor for source/target heaps;
- source root-first recursive probabilistic READ;
- target query reads root and the recursively addressed most-recent path;
- compare native evaluation with shuffled source, zero target history,
  root-only target history, and source-root-only read;
- report NLL, token BLEU-4, nonempty outputs, repetition, closure and examples.

## Predictions

1. Held-out NLL improves by at least `0.50` from initialization and stays finite.
2. Source shuffle increases NLL by at least `0.20`.
3. Zeroing target history increases NLL by at least `0.20`.
4. Removing the non-root target path increases NLL by at least `0.02`.
5. Forcing source READ to root increases NLL by at least `0.02`.
6. Greedy token BLEU-4 exceeds `1.0`, nonempty rate exceeds `0.95`, and maximum
   repeated-token run is below the output length.
7. Source FOLD/UNFOLD state MSE remains below `1e-10`.

## Falsification

Reject the target-TreeHeap protocol if teacher-forced NLL falls while target
history/path interventions are neutral, or greedy generation still collapses
to one source-invariant token. Passing is a mechanism PoC, not a claim of
translation superiority over the historical GRU/flat/Transformer baselines.

## Result (io task 61, 2026-07-29)

The controlled run used 5,000 aligned WMT pairs, 500 validation pairs and 500
test pairs. The model had 10,309,507 parameters and trained for five epochs on
io's constrained RTX 3090. Evidence is stored in
`../evidence/s3_stone1_c14_target_tree_autoregressive/summary.json`.

| Measurement | Prediction | Result | Gate |
|---|---:|---:|---|
| Initial to test NLL | improvement >= 0.50 | `9.6855 -> 6.9601` (`-2.7254`) | pass |
| Shuffled-source damage | >= 0.20 | `+0.0979` | fail |
| Zero-target-history damage | >= 0.20 | `+4.3857` | pass |
| Target-root-only damage | >= 0.02 | `+0.7718` | pass |
| Source-root-only damage | >= 0.02 | `+0.5325` | pass |
| Greedy BLEU-4 | > 1.0 | `0.2150` | fail |
| Nonempty rate | > 0.95 | `1.0` | pass |
| Maximum repeated-token run | below output capacity | `4` of `32` | pass |
| Source closure MSE | < 1e-10 | `1.08e-14` | pass |

Training NLL fell every epoch, but validation NLL was best at epoch 3
(`6.7596`) and then rose to `6.9157`. The five-epoch checkpoint therefore
already shows overfitting at this data scale.

### What the experiment supports

The target history is not a decorative input. Replacing the growing target
TreeHeap with Zero costs `4.3857` NLL. Restricting target READ to its root costs
another `0.7718`, so information outside the root and on the addressed
leaf-to-root path is causally used. This is the first controlled evidence in
this branch that a target-side TreeHeap state can carry the autoregressive
history without a GRU/LSTM hidden state.

The source hierarchy is also used: forcing source READ to stop at root costs
`0.5325` NLL. Exact source FOLD/UNFOLD closure remains numerical to about
`1e-14` MSE.

### What failed

The strong claim is rejected. BLEU-4 is only `0.2150`; outputs are short,
template-like fragments such as `此外,美国在美国的...`, not usable translations.
Only 18.4% of test rows produce unique strings.

Source identity is too weak: shuffling whole source sentences costs only
`0.0979` NLL, below the declared `0.20` gate. In addition, the learned source
route puts effectively all mass on the deepest level (`1.0` at depth 5).
The recursive loop executes, but its probability container has collapsed to a
fixed deepest-read policy instead of learning when to stop at different
resolutions.

### Decision

Retain the narrow claim that incremental `WRITE -> path FOLD -> READ` can serve
as a causal autoregressive target state. Do not claim that it has replaced the
historical GRU as a translation decoder, learned a variable-depth source
protocol, or avoided conditional generation collapse.

The next experiment must address two observed failures rather than merely add
training time: preserve and evaluate the best validation checkpoint, and make
source-conditioned generation/stop-depth identifiable. A larger run is not
justified until shuffled-source damage and output diversity improve in the
controlled setting.
