# STONE-1 C03: Capacity and Rate-Distortion Audit

Date: 2026-07-22
Status: preregistered
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-CAPACITY-RATE-DISTORTION-C03`
Predict: `P-S3-STONE1-CAPACITY-RATE-DISTORTION-03`

## Question

C02 established that the decoder uses TreeHeap codec values, left/right
addresses, and successively finer recursive levels. It did not meet product
quality or seed-stability gates. C03 asks whether this remaining distortion is
primarily a model-capacity limit, an update-count limit, or a protocol limit.

Parameters are treated as the finite storage budget of the learned private
protocol, not as an automatically beneficial scale knob. A larger model is
useful only if held-out distortion falls reproducibly while TreeHeap causal
signals remain present.

## Capacity Accounting

With vocabulary `V=32,001`, state width `D`, and decoder hidden width `H`, the
current implementation has:

```text
total parameters = 3VD + VH + V + 8HD + 3H^2 + 10D^2 + 19D + 6H + 1
```

The registered configurations are:

| Label | D | H | Parameters | FP32 parameter bytes |
|---|---:|---:|---:|---:|
| `base_28m` | 192 | 256 | 27,620,482 | 110,481,928 |
| `balanced_50m` | 320 | 512 | 50,267,778 | 201,071,112 |
| `balanced_92m` | 512 | 1024 | 91,931,906 | 367,727,624 |

At 28M, lexical embedding/output parameters account for `26,656,833`
parameters, the learned codec for `296,832`, and the remaining recursive
reader/decoder for `666,817`. Whole-model scaling therefore tests system
capacity; it must not be reported as codec-only scaling.

## Frozen Platform

Reuse C02 without changing data identity or structural rules:

```text
train / validation / test = 1,000,000 / 2,000 / 2,000
seeds                     = 71901, 71902, 71903
batch size                = 64
tokenizer                  = frozen 32K SentencePiece
heap width / leaf cut      = 64 / 1
codec                      = canonical learned 0.4/0.6 residual codec
optimizer / LR / clipping  = AdamW / 0.002 / 1.0
```

Training, validation, and test hashes must match C02. No historical checkpoint
is loaded; every arm starts from its registered seed.

## Stage A: Separate Capacity From More Training

Run two new arms across all three seeds:

| Arm | Parameters | Updates | Purpose |
|---|---:|---:|---|
| `base_28m_long` | 27.62M | 31,250 | test whether the old model only needed a second pass |
| `balanced_50m_equal` | 50.27M | 15,625 | test added capacity at the original exposure budget |

The completed C02 `base_28m` result at 15,625 updates is the frozen reference.
The new implementation must reproduce its dataset hashes and evaluation code.

The two new arms are also a rough compute-matched comparison. Parameter-update
products are `27.62M * 31,250 = 863.1B` and
`50.27M * 15,625 = 785.4B`; the longer 28M arm receives about `9.9%` more by
this coarse measure. This is not an exact FLOP count, but prevents the larger
model from receiving both more parameters and more update steps.

Record training-window NLL, validation NLL, test NLL, bits per token,
BLEU-4, seed variance, checkpoint bytes, peak VRAM, latency, and NLL improvement
per added million parameters. Also repeat force-algebraic, address-swap, and
positive depth-growth audits on the best 50M checkpoint.

## Stage A Predictions

Capacity evidence requires all of:

```text
C1  mean NLL(50M, 15,625) <= mean NLL(28M, 15,625) - 0.08
C2  mean BLEU(50M)        >= mean BLEU(28M) + 0.75
C3  NLL std(50M)          <= 0.08
C4  NLL(50M) <= NLL(28M-long) + 0.02 using half the updates
C5  force-algebraic damage >= 0.10 NLL
C6  address-swap damage    >= 0.10 NLL
C7  root-to-full gain       >= 0.50 NLL and at least 4/5 depths improve
C8  closure max             < 1e-5
C9  nonempty = 1.0 and severe repetition <= 0.10
```

STONE-1 product thresholds remain unchanged and are reported separately:

```text
Q1 mean test NLL <= 3.90
Q2 mean BLEU-4   >= 13.5
Q3 NLL std       <= 0.05
```

Passing capacity gates does not complete STONE-1 unless all product,
structural, and engineering gates pass together.

## Smoke and Stop Conditions

Before formal execution, run one seed for `500` updates per new arm. Continue
only if both runs have finite gradients, declining validation NLL, peak VRAM
below `10 GiB`, non-empty evidence logs, and no GPU reset. The io 3090 power
and frequency limits must not be changed.

The formal queue is serial. Forecast runtime is `10-13 h` for Stage A based on
C02 throughput; this estimate must be replaced by the measured 500-step rate
after smoke. Stop the queue on OOM, non-finite loss, GPU disappearance, or an
epoch estimate above `6 h` for one arm before investigating the implementation.

Learning rate remains fixed at `0.002` to isolate capacity under one training
recipe. A failure therefore rejects capacity scaling under this optimizer
contract, not every possible scale-specific optimizer.

## Conditional Stage B

Do not run 92M automatically. Run `balanced_92m` only if Stage A passes
`C1`, `C3`, and at least two of `C5-C7`. This avoids spending a long GPU run on
a direction that improved only one lucky seed or lost TreeHeap causality.

If Stage B opens, use the same three seeds and 15,625 updates. Its registered
prediction is a further mean NLL reduction of at least `0.05` over 50M without
increasing NLL standard deviation above `0.08`.

## Interpretation Table

| Observation | Decision |
|---|---|
| 28M-long improves as much as 50M | update budget, not capacity, was limiting |
| 50M improves train and held-out metrics across seeds | capacity limitation supported |
| training improves but held-out does not | overfit or data limitation |
| quality improves but structural interventions vanish | larger model bypassed TreeHeap protocol |
| neither arm improves | architecture/protocol bottleneck, stop scaling |

## Falsification Boundary

C03 does not claim that TreeHeap needs Qwen/Kimi-scale parameters, that larger
models are inherently better, or that parameter count equals stored knowledge.
It tests only whether the C02 system lies on a reproducible capacity-distortion
slope under one frozen WMT platform.

Planned evidence: `../evidence/s3_stone1_capacity_rate_distortion/`.
