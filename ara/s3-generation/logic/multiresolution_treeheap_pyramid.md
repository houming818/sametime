# Multiresolution TreeHeap Pyramid

Date: 2026-07-14
Status: supported mechanism / three-seed 1M-block proof
Claim: `S3-TREEHEAP-PYRAMID-C01`
Predict: `P-S3-TREEHEAP-PYRAMID-01`

## Claim

A frozen TreeHeap root compressor can be extended into a multiresolution codec:
the root preserves coarse task-relevant information, while a bounded detail
code stored at each internal address progressively restores local information.
Increasing the detail budget should monotonically improve held-out
reconstruction without changing the frozen root model's next-token NLL.

## Existing Evidence Used As The Starting Point

The full-corpus `S3-TREEHEAP-ROOT-COMPRESS-C01` model provides the frozen
downsampling kernel:

```text
38,251,247 real-text blocks
64 input tokens -> one 192D root
valid NLL = 6.236525
address-destruction delta NLL = +2.625219
```

This establishes a useful root state and address dependence in one seed. It
does not establish progressive reconstruction or compression efficiency.

## Analysis And Synthesis Pair

At every internal node, the frozen encoder produces:

```math
P = D_\theta(L,R),
```

where `D_theta` is the trained three-slot TreeHeap fold kernel. A new detail
encoder and decoder learn:

```math
d = Q_\phi(L,R,P), \qquad d\in\mathbb{R}^k,
```

```math
(\hat L,\hat R) = U_\psi(P,d).
```

The stored code is one root plus one addressed detail vector for every
internal node. Decoding starts at the root and recursively reconstructs child
states using each node's detail code.

## Fixed Rate Budgets

For 64 leaves of dimension 192, the uncompressed activation budget is 12,288
floats. A complete binary tree has 63 internal nodes. The pyramid budget is:

```math
R(k)=192+63k.
```

The preregistered budgets are:

| Detail width `k` | Floats | Fraction of original |
|---:|---:|---:|
| 0 | 192 | 1.56% |
| 8 | 696 | 5.66% |
| 16 | 1,200 | 9.77% |
| 32 | 2,208 | 17.97% |
| 64 | 4,224 | 34.38% |

These are activation counts, not entropy-coded bit rates. No stronger
compression statement is allowed until quantization is measured.

## Training Protocol

1. Load and freeze the completed four-head no-residual root compressor.
2. Expose its left, right, and parent states at every fold level.
3. Train only `Q_phi` and `U_psi` on real pretraining blocks.
4. Keep the root next-token decoder frozen as a no-regression audit.
5. Run three seeds on one million blocks before any complete-corpus run.

The first proof does not jointly optimize next-token and reconstruction losses.
Freezing the root encoder prevents loss weighting from rewriting the already
measured global representation.

## Baselines And Controls

```text
root only (k=0)
fixed mean-pooling pyramid
fixed Haar-like average/difference pyramid
flat autoencoder with the same stored-float budget
random left/right pairing with the same learned Q/U
address destruction on the trained TreeHeap pyramid
full original leaves as a reconstruction upper bound
```

## Metrics

```text
frozen root next-token NLL
leaf-state reconstruction MSE
tied-codebook token top-1/top-5
sequence token accuracy and edit similarity
bag overlap and position accuracy
stored floats per token
rate-distortion curve
per-level detail ablation
```

## Predictions

```text
P1  abs(new root NLL - 6.236525) < 1e-5.
P2  Reconstruction MSE decreases monotonically for k=0,8,16,32,64.
P3  Spearman correlation between detail budget and reconstruction quality >= 0.9.
P4  k=64 reduces leaf reconstruction MSE by at least 50% versus root-only.
P5  Address destruction raises reconstruction MSE by at least 10% or lowers
    token accuracy by at least 5 percentage points.
P6  At one or more matched budgets, learned TreeHeap beats mean pooling and
    the flat autoencoder on held-out reconstruction.
P7  Coarse-level and fine-level detail ablations have measurably different
    effects on global bag content and local token/position recovery.
```

## Decision

Support the mechanism claim only if `P1` through `P5` pass across all three
seeds. `P6` is required before claiming a TreeHeap-specific rate-distortion
advantage. `P7` is required before claiming meaningful scale specialization.

## Falsification

Reject or narrow the claim if reconstruction is not monotonic, if nearly full
leaf capacity is required, if random pairing matches the trained addresses, if
the flat/mean controls dominate at equal rate, or if the frozen root NLL changes
because of an implementation bypass.

## Original Contribution Boundary

Multiresolution analysis, image pyramids, wavelets, autoencoders, and
rate-distortion theory are established prior work. The project contribution is
the TreeHeap-specific formulation: a recursively addressed root-plus-detail
codec built on the measured TreeHeap fold checkpoint, together with the claim,
controls, rate budgets, and falsification protocol above. This document does
not assert universal novelty or patent novelty.

## Result

The preregistered run completed on `io` with seeds `71401/71402/71403`. Each
seed trained on one million real-text blocks and was evaluated on 8,192 held-out
blocks. The frozen checkpoint fingerprint was identical before and after the
experiment, and audit-subset next-token NLL was exactly unchanged at
`6.2997973263`.

| `k` | TreeHeap floats | Flat floats | TreeHeap MSE | Flat MSE | Haar MSE | Token top-1 | Sequence exact |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 192 | 192 | 1.9334 | **1.9104** | 1.9110 | 6.50% | 0.00% |
| 8 | 696 | 768 | **0.8803** | 1.8416 | 1.8232 | 51.05% | 0.00% |
| 16 | 1,200 | 1,344 | **0.7622** | 1.7459 | 1.7471 | 76.84% | 0.00% |
| 32 | 2,208 | 2,304 | **0.6516** | 1.7047 | 1.5979 | 95.01% | 5.99% |
| 64 | 4,224 | 4,224 | **0.5526** | 1.6042 | 1.2881 | 99.64% | 82.68% |

Address-shifting the learned `k=64` detail codes raised MSE from `0.5526` to
`3.3147`. All three seeds had a strictly monotonic MSE curve and Spearman
correlation approximately `1.0`. The `k=64` MSE was 28.6% of root-only MSE,
passing the preregistered 50% reduction gate.

The matched-or-higher-rate flat attention codec had a failure tail at `k=64`
(`1.5245/1.3767/1.9113`), whereas TreeHeap was stable
(`0.5526/0.5499/0.5554`). The result therefore supports an advantage over the
implemented flat and fixed-Haar controls, but it is not a universal comparison
against all flat autoencoders or Transformers.

## Decision

`P1` through `P5` pass across all three seeds, so the multiresolution mechanism
claim is supported. `P6` passes for the implemented equal-rate controls and is
recorded as bounded positive evidence. Random-pairing training and per-level
detail ablations were not implemented in this run; therefore scale
specialization, optimal compression, and universal TreeHeap superiority remain
open.

Evidence: `../evidence/s3_multiresolution_treeheap_pyramid/main_v2/`.
