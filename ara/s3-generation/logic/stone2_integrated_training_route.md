# STONE-2 Integrated Training Route

Date: 2026-08-23

Claim: `S3-STONE2-INTEGRATED-C03`

Status: integrated smoke executed; 8/9 gates passed, S7 failed, so long training
is paused.

## Target

Train one resumable TreeHeap base checkpoint from natural text, continue the
same architecture on bilingual translation and Chinese QA, and show that useful
generation causally depends on Butterfly communication, bounded recursive FOLD,
and multi-level `H_state`. Flat, root-only, leaf-only and fixed-output shortcuts
are disallowed.

```text
immutable data release
-> natural-text pretrain
-> matched task train
-> frozen posterior/generation proof
-> TreeHeap interventions
-> CLI, reload and release audit
```

## Candidate stack

```text
SentencePiece + distinct PAD/EOS/task slots
-> token WRITE
-> dynamic XOR Butterfly
-> zero-reference bounded reversible FOLD
-> root + addressed parent/detail levels
-> mandatory multi-level READ without learned STOP
-> recurrent Decoder
-> token probability container
-> autoregressive collapse
```

The integrated smoke combines C13 `ref_zero` with C11 `READ-only`. The
unsupported second `K_up`, K7 read adapter, BLANK curriculum, root-exclusive
Decoder, learned STOP and teacher distillation are excluded. Runtime state is a
TreeHeap; shared Kernel parameters remain tensors and parameter-memory topology
is outside the STONE-2 pass gate.

## Pipeline

- Pretrain: `NioText-ZH-Integrity-2985K-v1`, fresh next-span windows over
  source widths `4/8/16/32/64/128/256`.
- Translation: `NioClean-ZHEN-S098-7M-v2`, bidirectional, with immutable raw
  WMT evaluation and matched raw/selected controls.
- QA: matched `NioQA-ZH-S090/S095/S098-v1` ladder before selecting a task view.
- Medical QA remains a separate domain continuation.

The Chinese version records the complete numerical, structural, generation,
reload, stop and notification gates.

## Execution note

Task 292 passed closure, no-STOP, gradient, structural intervention,
non-fixed-generation and exact-reload checks, but failed the preregistered
per-depth S7 gate. Frozen task 293 found positive coarse/middle/fine Shapley
contributions summing to `0.1311` NLL benefit, with negative pair interactions.
This is evidence of distributed multiresolution contribution plus redundancy,
not a retroactive S7 pass. A successor energy/interference smoke is required.

Task 294 materialized and independently hash-verified the five core data
releases. Their registry root is
`75caafdc24058eb96a957fd680b41789843eb3726e4febb4a110b7c96b38be29`.
