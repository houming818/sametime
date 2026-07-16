# S2 Adaptive Alternating Lifting on WMT Massive

## Claim

`S2-ADAPTIVE-LIFT-WMT-C01`:

> A strictly reversible lifting TreeHeap whose update is learned from data and
> whose prediction orientation alternates by depth should form a better S2
> source state than the fixed, permanently left-anchored pump. On the same
> WMT-massive split and decoder, it should lower held-out translation NLL while
> preserving numerical closure and causal multiresolution READ.

This is a comparative S2 claim. It does not claim that learned update kernels,
alternating orientation, or lifting schemes are new mathematics. The project
contribution is the combination, preregistered comparison, implementation, and
evidence.

## Old Pump

At every depth the old pump predicts the right child from the left child and
uses a fixed update:

\[
D=R-P_\theta(L),
\qquad
U=L+\frac12D.
\]

Its inverse is:

\[
L=U-\frac12D,
\qquad
R=D+P_\theta(L).
\]

The construction is closed, but it imposes two inductive biases without data
support: the left side is always the reference, and every scale uploads exactly
one half of the detail.

## New Pump

The new update kernel starts at the old rule but can move under translation
gradients:

\[
A_\phi(D)=\frac12D+\frac12\tanh M_\phi(D).
\]

The final layer of \(M_\phi\) is initialized to zero, so training starts from
the old \(\frac12D\) pump rather than from a different random coordinate
system.

At even depths, orientation is left-to-right:

\[
D=R-P_\theta(L),
\qquad
U=L+A_\phi(D).
\]

The inverse is:

\[
L=U-A_\phi(D),
\qquad
R=D+P_\theta(L).
\]

At odd depths, orientation is mirrored:

\[
D=L-P_\theta(R),
\qquad
U=R+A_\phi(D),
\]

with inverse:

\[
R=U-A_\phi(D),
\qquad
L=D+P_\theta(R).
\]

Neither \(P_\theta\) nor \(A_\phi\) must be invertible. Encoder and decoder
only need to reuse the same kernels. Therefore closure is structural and must
remain true for every learned parameter value.

## Models and Attribution

The 30K ablation trains:

1. `flat_seq`: ordinary sequence encoder and attention;
2. `old_recursive`: fixed update, left anchor at every depth;
3. `learned_update`: learned \(A_\phi\), left anchor at every depth;
4. `alternate_fixed`: fixed \(\frac12D\), alternating orientation;
5. `adaptive_alternate`: learned \(A_\phi\), alternating orientation.

The 200K scale run trains `flat_seq`, `old_recursive`, and the winning new
pump. All models use the same tokenizer, rows, target decoder family, best-valid
checkpoint rule, batch policy, and evaluation code.

## Data

- corpus: `/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv`;
- raw size: 14,170,275 pairs, Chinese in column 0 and English in column 1;
- tokenizer: `sp_bpe_massive.model`, vocabulary 32,000;
- direction: English to Chinese;
- sampling: deterministic reservoir over the registered scan window, followed
  by a seeded shuffle before train/valid/test split;
- length: 8 to 32 pieces plus EOS, represented in a 64-leaf TreeHeap.

## Predict

`P-S2-ADAPTIVE-LIFT-WMT-01`:

### Attribution pilot: 30K

1. every pump has FOLD/UNFOLD MSE below `1e-10` and finite gradients;
2. at least one learned or alternating variant improves over `old_recursive`
   by `0.03` held-out NLL;
3. `adaptive_alternate` is no worse than the better single change by more than
   `0.03` NLL;
4. the learned update departs from its initialization: held-out update delta
   RMS is at least `1e-3`.

### Scale proof: 200K

5. the selected new pump improves over `old_recursive` by at least `0.05` test
   NLL;
6. the new pump closes at least 25% of the old pump's NLL gap to `flat_seq`;
7. complete source shuffle raises new-pump NLL by at least `0.50`;
8. root shuffle and at least three detail depths each raise NLL by `0.05`;
9. recursive pair breaking raises NLL by `0.05` at three or more depths;
10. native READ leaves at least `0.05` stop mass at two resolutions and no more
    than `0.90` at leaves;
11. generated output is finite and non-empty. Token BLEU and examples are
    reported but are not pass gates.

## Falsification

- If learned update is neutral, fixed half-detail upload is sufficient at this
  scale.
- If alternation is neutral or harmful, permanent left anchoring was not the
  active blocker.
- If closure fails, the implementation is wrong; translation quality cannot
  rescue the claim.
- If the new pump improves NLL but source/detail/pair interventions are neutral,
  the gain is not evidence of better TreeHeap use.
- If flat sequence remains better, TreeHeap quality superiority remains
  rejected even if the new pump beats the old pump.
- A 200K result does not establish full-corpus convergence, production BLEU,
  compression, sparse compute, world knowledge, or consciousness.

## Evidence Targets

- `evidence/s2_adaptive_lifting_wmt_ablation/`
- `evidence/s2_adaptive_lifting_wmt_200k/`
