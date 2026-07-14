# Multiscale Subheap Mask and Upward Information Extraction

Date: 2026-07-14
Status: partial support / upward signal found / positive-depth trend rejected
Claim: `S3-TREEHEAP-MASK-C01`
Predict: `P-S3-TREEHEAP-MASK-01`

## Problem

The native codec echo experiment learned a strong addressed-detail protocol,
but its root was unnecessary. That result does not show that TreeHeap cannot
form useful upper states. Surface echo rewards the shortest local copy path, so
the optimizer has no reason to move recoverable information through ancestors.

Houming818 proposes a task-level correction: damage complete TreeHeap
substructures at several depths. As the damaged subtree grows, local details
become insufficient and the reconstruction gradient should move toward parent,
ancestor, and root states.

## Scope Of This Proof

This first proof is **compression-style masking**:

1. The encoder observes the complete 64-token block.
2. A complete aligned subheap is selected after encoding.
3. All addressed detail codes internal to that subheap are removed before
   decoding.
4. Cross-entropy is evaluated only on leaves inside the removed subheap.

It asks whether information observed at the leaves is extracted upward into
states outside the removed local detail region. It does not yet ask the model to
infer text that the encoder never observed. A later inference-style mask must
corrupt the input before encoding.

## TreeHeap Operators

Raw token writing:

```math
h_i=W_\alpha(x_i).
```

Shared bottom-up analysis convolution:

```math
p_i=F_\theta\left(\frac{h_{2i}+h_{2i+1}}{\sqrt 2}\right),
```

```math
d_i=Q_\phi\left(\frac{h_{2i}-h_{2i+1}}{\sqrt 2}\right).
```

Shared top-down synthesis convolution:

```math
(\hat h_{2i},\hat h_{2i+1})=U_\psi(\hat p_i,d_i).
```

For a selected aligned leaf span `M_d` of size `2^d`, every detail whose
receptive field lies completely inside that span is set to zero. Details above
the cut remain available. Therefore successful recovery must use the upper
TreeHeap band rather than a local detail copy.

## Compared Training Conditions

Two exactly matched models are trained from random initialization:

| Model | Training question |
|---|---|
| `echo` | Recover every token with every detail present. |
| `multiscale_mask` | Recover only a randomly selected subheap after its internal details are removed. |

The models have the same WRITE/FOLD/DETAIL/UNFOLD/READ architecture, parameter
count, data order, optimizer, and token loss. Only the task distribution differs.
This isolates the user's hypothesis that a deeper question changes where the
model stores useful information.

## Data And Smoke

```text
corpus              real Chinese pretraining blocks
tokenizer           existing 16,000-piece SentencePiece model
block length        64
mask span sizes     2 / 4 / 8 / 16 / 32 leaves
state/detail        128D / 32D continuous
training blocks     100,000 per model in smoke
seed                71421
host                io RTX 3090
```

Continuous states are intentional here. This claim concerns upward information
placement, not bit-rate compression.

## Predictions

Let `NLL_m(d)` be masked-leaf NLL for model `m` at span depth `d`. Let

```math
\Delta_{root,m}(d)=NLL_{m,root\_zero}(d)-NLL_{m,normal}(d).
```

The preregistered smoke gates are:

```text
P1  At span sizes 16 and 32, multiscale_mask masked NLL is at least 0.30
    lower than echo masked NLL.
P2  At span size 32, multiscale_mask root-zero raises masked NLL by at
    least 0.10.
P3  Root contribution has positive depth trend: Spearman(depth,
    Delta_root) >= 0.60.
P4  At span size 32, multiscale_mask root contribution exceeds echo root
    contribution by at least 0.05 NLL.
P5  Replacing the root with another sample damages span-32 recovery by at
    least 0.10 NLL.
P6  WRITE, FOLD, DETAIL, UNFOLD, and READ all receive finite non-zero
    gradients under masked-only loss.
```

P1 is the central gate. P2-P5 distinguish useful upward extraction from a
better local decoder or an unused root. P3 is a trend gate, not a requirement
that every adjacent depth be strictly monotonic.

## Decision Rules

- **Support** only if P1-P6 all pass in the smoke and reproduce across three
  seeds in a later main run.
- **Partial support** if multiscale masking improves deep-subheap recovery but
  root interventions fail. That would mean information moved above local
  details but remained in near-root detail bands rather than the root itself.
- **Reject this implementation** if the matched echo model performs equally,
  if root/upper interventions are neutral, or if only shallow masks improve.

A positive result proves task-induced upward information extraction in this
codec. It does not prove semantic categories, world knowledge, inference from
unseen content, useful bit compression, or superiority to Transformers.

## Smoke Result

The `io` smoke completed on 100,000 real Chinese blocks per matched model.
Ordinary echo reached `0.9950` token top-1 with intact details but collapsed
under subheap removal. Multiscale masking obtained masked token top-1
`0.1426/0.0869/0.0750/0.0642/0.0636` for spans `2/4/8/16/32`.

Root-zero NLL deltas were `0.843/0.499/0.330/0.287/0.273`; wrong-root deltas
were `0.335/0.347/0.353/0.350/0.361`. Root state is therefore causal and
sample-specific, but its contribution strictly decreased with mask depth.
Depth Spearman was `-1.0`, rejecting P3. P1, P2, P4, P5, and P6 passed.

The result is partial support: masking moved some information above local
details and made root interventions meaningful, but did not produce increasing
root reliance or high-quality deep-subheap recovery. Repeated-token outputs at
span 32 indicate a coarse-frequency collapse rather than structured subtree
reconstruction. Evidence: `evidence/s3_treeheap_multiscale_mask/smoke/`.

## Follow-up

The next experiment corrupts the input before encoding and predicts the missing
subheap from real context. That inference-style mask tests whether the extracted
upper states model corpus regularities rather than merely redundantly storing
observed tokens.
