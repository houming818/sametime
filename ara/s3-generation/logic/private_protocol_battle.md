# TreeHeap Private Protocol Battle

Date: 2026-07-19
Result date: 2026-07-20
Status: partial
Claim: `S3-PRIVATE-PROTOCOL-BATTLE-C01`
Predict: `P-S3-PRIVATE-PROTOCOL-BATTLE-01`

## Question

Earlier ARA evidence already supports a narrow existence statement: raw-token
surface loss can train shared recursive TreeHeap operators into an
address-sensitive continuous encoder/decoder protocol.  The remaining question
is operational, not semantic:

> Can several private TreeHeap lifting heads, trained only by the final real
> seq2seq loss, combine into a protocol that is more effective than one head
> and a matched flat sequence baseline?

No head receives a syntax, route, depth, summary, semantic, or compression
label.  No claim is made about what a root, detail, depth, or head means to a
human observer.

## Existing Foundation

The proof begins from registered evidence rather than reopening old gates:

1. `M0-OPCODEC-C01`: a learned short TreeHeap operator program transfers to
   unseen addresses in a controlled number-law world.
2. `S3-TREEHEAP-CODEC-C01`: continuous WRITE/FOLD/DETAIL/UNFOLD/READ reaches
   token top-1 `0.9955/0.9915`; shifting addressed details lowers it to
   `0.0024/0.0014`.
3. `C7-008`: the lifting pump closes recursively, receives task gradients, and
   exposes causal root/detail/pairing channels, although its original strong
   root-echo gate failed.
4. `S3-TREE-LIFT-RECURSIVE-C01`: one frozen WMT checkpoint improves from NLL
   `13.8100` at depth cap 0 to `4.6335` at full recursive depth.

These results establish a trainable recursive medium.  They do not establish a
competitive multi-head protocol.

## Model Contract

For total protocol width `D` and `M` heads, each head receives width `D/M`:

```math
H_m = E_{\theta_m}(X), \qquad m=1,\ldots,M.
```

Every encoder uses the same addressed lifting topology but independent WRITE,
PREDICT, and UPDATE parameters.  The decoder recursively reads each private
TreeHeap and concatenates the returned contexts:

```math
C_t = [R_{\phi_1}(q_t,H_1);\ldots;R_{\phi_M}(q_t,H_M)].
```

Only the final teacher-forced translation cross-entropy is optimized:

```math
L=-\sum_t \log P(y_t\mid X,y_{<t}).
```

The decoder has no source-token, leaf-array, or encoder-attention bypass.  It
can access source information only through the recursively reconstructed
TreeHeap levels.

## Stage A: Overnight Gate

Use real WMT massive English-to-Chinese pairs and one fixed sampled data stream.
Train these variants at three initialization seeds:

```text
flat_seq
treeheap_h1
treeheap_h2
treeheap_h4
```

The total TreeHeap protocol width remains fixed across head counts.  Parameter
counts, elapsed time, NLL, token BLEU-4, exact generation, and route mass are
recorded.  This controls the easiest “more heads only means more width” error,
although exact parameter/FLOP equality is reported rather than assumed.

The first `treeheap_h4` seed receives registered interventions:

```text
source shuffle
root shuffle
detail shuffle at every depth
pre-FOLD pair break at every depth
ablate each head separately
```

Three independently trained four-head pairs `(E1,D1)`, `(E2,D2)`, `(E3,D3)`
are then crossed as `Ei -> Dj`.  Original and crossed NLL are recorded.  A
cross-pair failure is evidence that encoder and decoder formed seed-specific
compatible coordinates; cross-pair success instead suggests a reproducible
shared coordinate system and is not to be rewritten as failure.

## Predictions

`P1 Trainability`

- every run has finite gradients and non-empty held-out generation;
- every TreeHeap head receives a non-zero encoder gradient.

`P2 Multi-head composition`

- mean four-head test NLL is at least `0.02` lower than mean one-head NLL;
- the complete four-head model is better than every single-head ablation in
  the audited seed.

`P3 Structural causality`

- source and root shuffle each increase NLL by at least `0.05`;
- at least three detail depths and three pair depths increase NLL by `0.02`;
- at least three of four head ablations increase NLL by `0.01`.

`P4 Private pairing`

- median crossed-pair NLL damage is at least `0.10`, or crossed pairs remain
  within `0.02` and are explicitly classified as a shared protocol rather than
  a seed-private protocol.

`P5 Competitive result`

- mean four-head NLL is at least `0.02` lower than mean flat NLL under the
  registered data and total-width budget.

## Decision

- `supported`: P1-P5 pass across the registered seeds.
- `partial`: the protocol is trainable and structurally causal, but multi-head
  or flat superiority fails.
- `not_supported`: TreeHeap structure is not causal or training is unstable.

Stage A does not settle comparison with a parameter/FLOP-matched Transformer.
That is Stage B and should use the winner from this gate; adding it to the first
overnight matrix would mix architecture selection with final benchmarking.

## Registered Run Result

The formal run completed on `io` in `8762.91` seconds using three seeds,
30,000 training pairs, 2,000 validation pairs, and 2,000 test pairs.  All four
variants used about 27.3--27.6 million parameters.  Mean test results were:

| Variant | NLL (lower is better) | BLEU-4 (higher is better) | Mean train time |
|---|---:|---:|---:|
| flat | 6.0401 | 5.3530 | 111.6 s |
| TreeHeap h1 | 6.1231 | 4.9719 | 405.4 s |
| TreeHeap h2 | 6.1341 | 5.2892 | 775.6 s |
| TreeHeap h4 | 6.1934 | 5.0853 | 1552.2 s |

The registered gates resolved as follows:

| Gate | Result | Evidence |
|---|---|---|
| P1 trainability | pass | all runs were finite and every TreeHeap head received gradient |
| P2 four heads beat one head | fail | h4 NLL `6.1934` was worse than h1 NLL `6.1231` |
| P2 every h4 head helps | pass | individual ablation damage was `0.0554`--`0.0874` NLL |
| P3 structural causality | pass | source/root shuffle damage was `+1.9532/+2.1113`; registered detail and pair counts passed |
| P4 private pairing | pass | cross-seed encoder/decoder damage was `+2.2507`--`+4.2958` NLL |
| P5 h4 beats flat | fail | h4 was `+0.1533` NLL worse than flat |

This is a `partial` result.  It supports the existence of a learned,
structurally causal, seed-private TreeHeap encoder/decoder protocol.  It rejects
the registered prediction that splitting the fixed protocol width into four
heads improves this task, and it provides no competitive advantage over the
matched flat baseline.  The next experiment must explain or remove the
multi-head bottleneck before a Transformer battle is justified.

Evidence is stored in
`ara/s3-generation/evidence/s3_private_protocol_battle_full/`; the three
checkpoints are archived on `io` under
`/mnt/nas/ara/s3-generation/evidence/s3_private_protocol_battle_full/`.

## Falsification Boundary

Do not infer semantic heads, human-readable roots, world knowledge,
consciousness, finite-bit compression, or universal TreeHeap superiority from
this run.  Do not add auxiliary head losses after seeing the result.  A lower
score than flat is a valid result and must remain in evidence.
