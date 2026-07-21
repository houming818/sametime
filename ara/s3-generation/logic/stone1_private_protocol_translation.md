# STONE-1: Learned TreeHeap Private-Protocol Translation PoC

Date: 2026-07-21  
Status: preregistered  
Claim: `S3-STONE1-PRIVATE-PROTOCOL-C01`  
Predict: `P-S3-STONE1-PRIVATE-PROTOCOL-01`

## Engineering Objective

Deliver a real English-to-Chinese CLI whose checkpoint is trained on the
frozen WMT-massive evaluation platform established by SPR-065. The milestone
is complete only when output quality, TreeHeap structural causality, and CLI
engineering gates pass together.

The first product is translation rather than open-domain dialogue. Translation
already has a locked tokenizer, 1M nested train split, 2K validation/test split,
and matched Flat/Transformer reference values. Dialogue requires a separate
pretraining and instruction-data contract.

## Claim

With no rotation, syntax, depth, or orientation labels, a fixed-capacity
TreeHeap encoder and decoder can learn a private structural protocol from final
seq2seq NLL. A shared local kernel may choose the anchor direction of every
`root/left/right` subheap. If those choices are useful, the learned protocol
must beat identity-only and frozen-random controls, remain causally dependent
on TreeHeap addresses, and support non-teacher-forced generation.

Rotation is not a target. Local mirror/rotation-like behavior is audited only
after training as one possible internal consequence.

## TreeHeap Contract

For left/right states `L,R`, predictor `P`, update `U`, and hard local gate
`g in {0,1}`:

```text
g = 1: detail = R - P(L); parent = L + U(detail)
g = 0: detail = L - P(R); parent = R + U(detail)
```

The gate is emitted by a shared content-conditioned kernel with a
straight-through gradient. Its hard bit is stored beside the detail inside the
same fixed-capacity `H_state`. UNFOLD uses that bit:

```text
anchor    = parent - U(detail)
predicted = detail + P(anchor)
g = 1: (L,R) = (anchor,predicted)
g = 0: (L,R) = (predicted,anchor)
```

Therefore FOLD/UNFOLD remains exactly reversible while the protocol may learn
different local directions. No node is allocated during recursion.

The decoder cannot read the finest leaf level. It receives only root and
compressed levels. This closes the direct token/string bypass used by earlier
echo models.

## Experimental Arms

All arms contain the same gate-kernel parameters and the same recursive
decoder. Only use of the local direction kernel changes:

| Arm | Local structural behavior |
|---|---|
| `identity` | every valid pair uses the left anchor |
| `learned_structural` | shared content/depth kernel learns a hard anchor bit |
| `frozen_random` | fixed address/depth pattern chooses anchor bits |

Discovery/smoke may use smaller nested prefixes. Formal confirmation uses:

```text
train unique rows = 1,000,000
validation/test   = frozen 2,000 / 2,000
seeds             = 71901, 71902, 71903
batch             = 64
updates           = 15,625
heap width        = 64
leaf levels cut   = 1
```

## Numeric Predictions

### Product quality

```text
Q1 learned mean test NLL                    <= 3.90
Q2 learned mean token BLEU-4                >= 13.5
Q3 learned NLL standard deviation           <= 0.05
Q4 non-empty generation                     = 1.00
Q5 severe repetition rate                   <= 0.10
```

### TreeHeap existence

```text
S1 identity mean NLL - learned mean NLL      >= 0.05
S2 frozen-random mean NLL - learned mean NLL >= 0.10
S3 force-identity damage on learned model    >= 0.10 NLL
S4 force-random damage on learned model      >= 0.10 NLL
S5 left/right address-swap damage            >= 0.10 NLL
S6 FOLD/UNFOLD maximum closure error         < 1e-5
```

### Engineering

```text
E1 batch-1 greedy P50 on io RTX 3090         <= 1,000 ms
E2 peak allocated VRAM                       <= 4 GiB
E3 checkpoint size                           <= 300 MiB
E4 all gradients and metrics remain finite
E5 CLI loads the recorded checkpoint and translates without teacher forcing
```

## Decision

```text
all Q + S + E pass     -> supported / STONE-1 complete
Q + E pass, S fails    -> seq2seq demo only; not a TreeHeap PoC
S + E pass, Q fails    -> mechanism PoC only; not product-ready
otherwise              -> not supported under this recipe
```

Thresholds are engineering targets, not claims of state-of-the-art quality.
NLL comparisons are valid only on the frozen split and tokenizer.

## Falsification

The structural claim is rejected if learned, identity, and random arms are
equivalent; if leaf/string bypass is restored; or if address/gate interventions
do not damage the learned model. A lower NLL caused only by extra parameters is
not TreeHeap evidence.

The product milestone is rejected if teacher-forced NLL improves but greedy
generation remains empty, repetitive, or below the registered BLEU target.

## Boundary

A positive translation result does not establish open-domain dialogue, world
knowledge, human-readable internal semantics, automatic discovery of arbitrary
rotation formulae, or superiority over industry-scale Transformers. The local
kernel supplies a legal structural action space; training chooses actions but
does not invent new TreeHeap algebra in this experiment.

## Evidence Contract

Code: `../src/s3_stone1_private_protocol.py`  
CLI: `../src/treeheap_cli.py`  
Evidence: `../evidence/s3_stone1_private_protocol/`

The formal directory must contain the standard ARA report files plus checkpoint
SHA-256, inference latency, structure-intervention results, gate statistics,
generation examples, and a CLI smoke transcript.
