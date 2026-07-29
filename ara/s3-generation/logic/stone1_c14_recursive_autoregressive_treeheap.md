# C14: recursive autoregressive target TreeHeap

Status: registered; experiment pending

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

