# Multiresolution Damage Repair

Status: preregistered

Claim ID: `S3-MULTIRES-REPAIR-C01`

Origin: Houming818 and Codex Review, 2026-07-17

## Question

Can an already encoded TreeHeap recover useful internal state after a root,
detail, or subheap loss? This is not token masking. Every source token first
participates in WRITE and FOLD. Damage is applied only to the resulting
`H_state`, after which a repair kernel must infer missing addressed residuals
from surviving scales.

## Frozen Base

Use the completed annealed checkpoint:

`/mnt/nas/ara/s3-generation/evidence/s3_annealed_frontier_pretrain/checkpoint_annealed.pt`

Its encoder, decoder, embeddings, predictor, update kernel, and resolution
embeddings remain frozen. The experiment trains only repair kernels. This
prevents the base model from moving information around after seeing the damage
protocol.

For a detail node at depth `d`, the clean lifting state supplies:

\[
D_d=R-P_\theta(L), \qquad U_d=L+A_\phi(D_d).
\]

The damage operator erases a contiguous set of addressed `D_d` nodes after
FOLD. The damaged TreeHeap is UNFOLDed once without repair and once after a
predicted residual is inserted.

## Repair Models

`parent_only` predicts a missing residual from its parent and depth. It tests
generic latent regression without horizontal or global TreeHeap context.

`cross_scale` receives the parent, root, surviving left/right residual
neighbors, depth, and normalized path coordinates:

\[
\hat D_d=G_\psi(U_d,H_{root},D_{left},D_{right},d,path).
\]

Both are shared across all six FOLD depths. Targets are the encoder's own clean
residuals. No syntax, category, summary, MASK token, or external language label
is provided. A wrong-address audit reverses path coordinates and rolls sibling
context while leaving the target unchanged.

## Damage Schedule

Training alternates between one-node erasure and contiguous 25% erasure at a
random depth. Evaluation covers every depth and three severities:

- one addressed detail node;
- contiguous 25%;
- contiguous 50%.

The same clean batches and damage masks are used for all repair controls.

## Metrics

For clean, damaged, and repaired states record:

- missing-detail normalized MSE;
- reconstructed-leaf MSE;
- affected-leaf nearest-embedding token accuracy;
- frozen future-decoder NLL.

For a loss `L`, repair fraction is:

\[
\rho_L=
\frac{L_{damaged}-L_{repaired}}
{L_{damaged}-L_{clean}+\epsilon}.
\]

Latent MSE has `L_clean=0`, so `rho = 1 - repaired_mse/damaged_mse`.

## Predicts and Gates

- `P1 damage`: zeroing addressed details must produce mean leaf MSE above
  `1e-4` and reduce affected token retrieval by at least `0.05`.
- `P2 repair`: cross-scale repair must recover at least `50%` of damaged leaf
  MSE on average.
- `P3 structure`: cross-scale repair fraction must beat parent-only by at
  least `0.05`.
- `P4 address`: wrong path/sibling context must increase repaired leaf MSE by
  at least `10%` over native cross-scale repair.
- `P5 language`: where damage raises future NLL by at least `0.01`, repair must
  recover at least `25%` of that NLL damage on average.
- `P6 graceful`: cross-scale repair must improve leaf MSE over zero fill at
  every registered severity, with finite metrics.
- `P7 frozen`: no frozen checkpoint tensor may change.

## Decision Boundary

Full support requires P1-P7. If latent repair succeeds but future NLL does not
move, retain only a mathematical state-repair claim. If parent-only matches
cross-scale or wrong addresses are cheap, do not attribute repair to TreeHeap
structure. If damage itself is cheap, the state was unused and no repair claim
is available. Exact token recovery is not required on ambiguous real language.

This proof does not establish consciousness, fractal identity, unique semantic
summaries, or superiority over a trained end-to-end Transformer repair model.
It asks whether the existing TreeHeap state supports measurable post-FOLD
damage recovery and whether cross-scale addressed context contributes.
