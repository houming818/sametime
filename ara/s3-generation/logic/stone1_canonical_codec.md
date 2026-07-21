# STONE-1 C02: Canonical 0.4/0.6 Codec

Date: 2026-07-22
Status: preregistered
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-PRIVATE-PROTOCOL-C02`
Predict: `P-S3-STONE1-PRIVATE-PROTOCOL-02`

## Why This Is Still STONE-1

C01 did not complete STONE-1. It rejected one recipe: a hard
straight-through kernel that learned whether each local pair used its left or
right child as anchor. The fixed identity arm was better on every seed, while
left/right address swapping remained strongly damaging.

C02 is a revised experiment inside the same milestone. It fixes handedness as
TreeHeap algebra and moves learning into the values carried by the recursive
codec. STONE-1 is complete only if the original quality, structural-causality,
closure, and CLI engineering goals pass together.

## Canonical Algebra

For every ordered pair `(L, R)`, define the fixed base transform:

```text
detail = R - L
parent = 0.4 L + 0.6 R
```

The inverse is:

```text
L = parent - 0.6 detail
R = parent + 0.4 detail
```

The transform is bilateral: both `parent` and `detail` depend on both inputs.
It is order-sensitive because swapping `L,R` negates `detail` and changes the
weighted parent. Its matrix and inverse are:

```text
[parent]   [ 0.4  0.6] [L]
[detail] = [-1.0  1.0] [R]

[L]   [1.0 -0.6] [parent]
[R] = [1.0  0.4] [detail]
```

The determinant is exactly `1.0`; no orientation gate or path-selection
network exists.

## Learned Residual Codec

C02 permits continuous residual learning around the canonical transform:

```text
P_theta(L) = L + delta_P_theta(L)
U_theta(d) = 0.6 d + delta_U_theta(d)

detail = R - P_theta(L)
parent = L + U_theta(detail)
```

The residual output layers are initialized to zero. At step zero, the learned
arm is therefore exactly the fixed `0.4/0.6` algebra. UNFOLD uses the same
functions:

```text
L = parent - U_theta(detail)
R = detail + P_theta(L)
```

This remains algebraically closed for arbitrary deterministic `P_theta` and
`U_theta`. Encoder and decoder are trained jointly by one translation
cross-entropy objective. Their continuous co-adaptation is the candidate
private protocol; only discrete left/right gate learning was removed.

## Experimental Arms

All arms use fixed handedness, the same embeddings, recursive decoder, data,
parameter shapes, optimizer, and update count.

| Arm | Codec |
|---|---|
| `canonical_algebraic` | fixed `P(L)=L`, `U(d)=0.6d` |
| `canonical_learned` | zero-initialized continuous residuals learn in `P/U` |
| `canonical_frozen` | a small fixed random residual codec; decoder may adapt but codec cannot learn |

The controls test different questions:

- algebraic: is the deterministic bilateral codec already sufficient?
- frozen: can the decoder adapt to any stable reversible codec?
- learned: does task gradient place useful information in the codec itself?

The experiment does not learn handedness and does not supervise syntax,
rotation, depth semantics, or human-readable summaries.

## Frozen Platform

Formal confirmation reuses the C01 platform exactly:

```text
source                 = WMT-massive en->zh TSV
train                  = 1,000,000 unique pairs
validation/test        = frozen 2,000 / 2,000
tokenizer              = frozen 32K SentencePiece model
seeds                  = 71901, 71902, 71903
batch                  = 64
updates per arm        = 15,625
heap width             = 64
finest leaf levels cut = 1
```

The same split hashes and no-overlap checks must be reproduced in evidence.

## Predictions

### Product quality

The STONE-1 product thresholds do not move:

```text
Q1 canonical-learned mean test NLL          <= 3.90
Q2 canonical-learned mean token BLEU-4      >= 13.5
Q3 canonical-learned NLL standard deviation <= 0.05
Q4 non-empty generation                     = 1.00
Q5 severe repetition rate                   <= 0.10
```

### Codec causality

```text
S1 algebraic mean NLL - learned mean NLL    >= 0.05
S2 frozen mean NLL - learned mean NLL       >= 0.10
S3 force-algebraic damage on learned model  >= 0.10 NLL
S4 left/right address-swap damage           >= 0.10 NLL
S5 full-state NLL gain over root-only       >= 0.50 NLL
S6 at least 4/5 added depth levels improve NLL
S7 FOLD/UNFOLD maximum closure error         < 1e-5
```

`S1/S2/S3` are the core private-protocol tests. `S4` retains C01's address
causality. `S5/S6` are positive depth-growth tests: instead of only deleting
structure, they ask whether exposing successively finer TreeHeap levels adds
usable held-out information.

### Engineering

```text
E1 batch-1 greedy P50 on io RTX 3090         <= 1,000 ms
E2 peak allocated VRAM                       <= 4 GiB
E3 checkpoint size                           <= 300 MiB
E4 all gradients and metrics remain finite
E5 CLI loads the learned checkpoint and generates without teacher forcing
```

## Interventions

The best-validation learned checkpoint is evaluated under:

1. native learned codec;
2. residuals disabled, forcing exact fixed `0.4/0.6` algebra;
3. left/right addresses swapped after UNFOLD;
4. root-only through all available compressed levels.

The learned residual parameter norm is recorded but is not evidence by itself.
Only held-out damage or improvement can support codec causality.

## Decision

```text
all Q + S + E pass     -> supported / STONE-1 complete
Q + E pass, S fails    -> seq2seq demo only
S + E pass, Q fails    -> codec mechanism PoC only
otherwise              -> C02 not supported; STONE-1 remains incomplete
```

No threshold may be changed after the formal result is visible.

## Falsification

C02 is falsified if the learned residual codec fails to beat fixed algebraic
and frozen controls; if disabling learned residuals does not damage held-out
NLL; if depth growth does not add information; if address causality disappears;
or if the learned arm misses the registered product and stability targets.

A negative result does not reject TreeHeap as a deterministic algebra. It says
that final translation loss did not make the residual `P/U` codec an
indispensable private protocol under this recipe.

## Evidence Contract

Code: `../src/s3_stone1_canonical_codec.py`
CLI: `../src/treeheap_canonical_cli.py`
Smoke evidence: `../evidence/s3_stone1_canonical_codec_smoke/`
Formal evidence: `../evidence/s3_stone1_canonical_codec/`

Evidence must include `command.sh`, `config.json`, dataset hashes,
`trace.jsonl`, per-run results, intervention/depth-growth results, closure,
checkpoint SHA-256, CLI transcript, `summary.json`, and `REPORT.md`.
