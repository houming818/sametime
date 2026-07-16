# Annealed Contraction Protocol

Status: completed; toy mechanism supported, real protocol partially supported

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

## Results (io, 2026-07-16)

### Proof A

The complete-input toy passed the cross-seed predictive-core gate. Two of
three seeds exceeded `0.99` root exact accuracy; the remaining seed reached
`0.89975`, just below the preregistered `0.90` threshold. In every seed,
shuffling the three predictive core tokens damaged NLL by more than `16`,
while shuffling matched nuisance tokens had approximately zero cost. Root
linear probes separated core identity from nuisance identity by more than
`0.94`.

This is evidence that future-only loss can make a narrow root retain the
data-defined predictive factor without receiving a core label. It is not yet
a TreeHeap-specific advantage: `T4` failed in all seeds. Mean pooling matched
or slightly beat TreeHeap in seed 72001, and TreeHeap's gains in the other two
seeds (`0.01525`, `0.04525` exact) remained below the preregistered `0.05`
margin.

### Proof B

All three real-corpus runs completed 20,000 updates. Leaf-only training made
the leaf useful (`5.3615` NLL) while allowing root NLL to deteriorate to
`35.8732`. Uniform depth training produced root NLL `5.7158`. The annealed
curriculum produced root NLL `5.6240`, a `0.0918` improvement over uniform,
while its root-to-leaf gap was only `0.0603`. Its seven-frontier NLL profile
had Spearman correlation `-0.75` with depth. `R1`, `R2`, `R3`, `R6`, `R7`,
and `R8` passed.

The structural part did not pass. Source shuffling damaged root NLL by only
`0.0852`, below the `0.20` gate, and swapping adjacent left/right siblings
before FOLD changed NLL by `-0.0011`. Therefore `R4` and `R5` failed. The
current root carries useful coarse predictive statistics and supports an
ordered multi-resolution readout, but it has not shown causal use of sample
identity or TreeHeap address order.

Evidence:

- `evidence/s3_annealed_contraction_toy/`
- `evidence/s3_annealed_frontier_pretrain/`
- checkpoints and SHA-256 pointers under
  `/mnt/nas/ara/s3-generation/evidence/s3_annealed_*`

The next experiment must target content- and address-sensitive composition,
not merely extend this curriculum. A valid successor should preserve the
annealed root gain while making source shuffle and pre-FOLD sibling swap
causal across multiple seeds.
