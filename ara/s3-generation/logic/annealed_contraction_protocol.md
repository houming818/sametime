# Annealed Contraction Protocol

Status: preregistered overnight proof

Claim IDs:

- `S3-ANNEAL-TOY-C01`
- `S3-ANNEAL-REAL-C01`

Origin: Houming818, 2026-07-16

## Core Hypothesis

A complete input should enter TreeHeap before any information is removed.
Recursive FOLD then acts as a learned lossy contraction when observed at a
single frontier: low levels retain surface detail, while increasingly narrow
frontiers should retain the information that removes the most generation
loss. Addressed details remain available in the complete `H_state`, but only
information carried by a frontier survives at that resolution.

Human-readable illustration:

```text
L0: 夏天的凉风吹过一望无际的青青草原
L1: 夏天的凉风吹过草原
L2: 风吹过草原
```

The internal states are not required to decode literally to these strings.
The formal claim is predictive sufficiency under decreasing capacity, not a
gold syntax or summarization claim.

## Algebra

The experiment keeps the supported learned-update lifting operation:

\[
D = R-P_\theta(L),\qquad U=L+A_\phi(D).
\]

The full state `(root, all addressed D)` remains reversible. A depth-`d`
frontier contains only the reconstructed nodes at that resolution and has
`2^d` vectors. Capacity therefore changes from one root vector to 64 leaf
vectors without replacing the input by MASK.

Training samples one readable frontier per batch. The decoder predicts a
future span from the complete past:

\[
\mathcal L_{future}
=-\sum_t\log p_\Theta(y_t\mid F_d(x),y_{<t}).
\]

No summary, syntax, core-token, mask, route, depth, or category label enters
this loss. Depth is supplied only as an address/resolution coordinate.

## Proof A: Controlled Predictive Core

Generate complete 16-token observations from two independent factors:

- a three-token event core `C`;
- thirteen nuisance/detail tokens `N`.

The future three-token sequence `Y` is a deterministic compositional function
of `C` and independent of `N` given `C`:

\[
P(Y\mid C,N)=P(Y\mid C).
\]

Core positions and token identities are not passed to the model. They are
retained by the data generator only for post-training interventions.

### Toy Predict

- `T1`: root future exact accuracy is at least `0.90` on held-out core
  combinations.
- `T2`: shuffling core tokens across samples damages root NLL by at least
  `0.50` more than shuffling an equal number of nuisance tokens.
- `T3`: a linear probe trained after model training predicts core identity
  from root at least `0.30` better than nuisance identity.
- `T4`: root performance beats equal-width mean-pool and flat bottleneck
  controls by at least `0.05` exact accuracy on held-out combinations.
- `T5`: at least two seeds pass T1-T3.

Passing this proof means only that contraction can preserve a
data-defined predictive core without a core label in the task loss.

## Proof B: Real-Corpus Frontier Annealing

Use complete Chinese text blocks from news2016zh, Wikipedia, and webtext2019zh.
The first 64 SentencePiece tokens are input; the following 16 are the future
generation target. There is no MASK.

Train identical learned-update lifting models under three curricula:

1. `leaf_only`: always decode from 64 leaf states;
2. `uniform`: sample one of the seven frontiers uniformly;
3. `annealed`: begin leaf-biased, pass through uniform, and finish root-biased
   while retaining a 25% uniform exploration floor.

All variants receive the same number of updates, corpus stream, optimizer,
parameter count, and future-span cross-entropy.

### Real Predict

- `R1`: all variants remain finite and reduce held-out NLL.
- `R2`: annealed root NLL beats uniform root NLL by at least `0.05`.
- `R3`: annealed root NLL is no more than `1.00` worse than annealed leaf NLL;
  the 64:1 frontier contraction retains useful future information.
- `R4`: root source shuffle damages NLL by at least `0.20`.
- `R5`: pre-FOLD adjacent left/right swap damages root NLL by at least `0.05`.
- `R6`: root greedy output is at least 75% non-empty, has adjacent repetition
  at most `0.40`, and has at least 10% unique outputs.
- `R7`: frontier NLL ordered from root to leaves has Spearman correlation at
  most `-0.70` with depth, giving an empirical rate-distortion ordering rather
  than an arbitrary set of unrelated states.
- `R8`: FOLD/UNFOLD closure remains below `1e-10` MSE.

## Decision Boundary

The full annealed-contraction hypothesis is supported only if Proof A shows
core-over-nuisance selection and Proof B shows sample-specific root use. A
smooth depth curve alone is insufficient. If the real model predicts from
details/leaves but root shuffle remains cheap, retain only the generic
multi-resolution generation observation. If mean pooling or a flat bottleneck
matches the toy, do not claim a TreeHeap-specific effect.

This experiment does not establish consciousness, human-readable summaries,
unique semantics, WMT superiority, or computational compression. It tests
whether the proposed training pressure can make useful information survive a
TreeHeap contraction.
