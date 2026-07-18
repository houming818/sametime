# Full-Corpus Repair-Aware Seq2Seq

Status: running / 60K interim audit recorded

Claim ID: `S3-FULL-REPAIR-SEQ2SEQ-C01`

Origin: Houming818 and Codex Review, 2026-07-17

## Objective

Turn the supported post-FOLD parent-detail repair mechanism into a reusable
Chinese seq2seq checkpoint under real data pressure. Training reads only the
local io mirror under `/home/nio/datasets`; NAS is not on the training hot
path. NAS may receive a cold backup only after a local checkpoint is complete.

## Data

Stream complete local files rather than a fixed sample:

- news2016zh and Chinese Wikipedia: context-to-continuation;
- webtext2019zh: question/title-to-answer;
- BELLE 2M and 1M: instruction/input-to-output;
- baike2018qa: question/description-to-answer;
- translation2019zh and WMT massive: English-to-Chinese.

Every emitted example is counted by source in the checkpoint and evidence.
Malformed, empty, or replacement-character-heavy rows are rejected. Parquet
and legacy-corrupted medical files are excluded from this run because the io
runtime cannot currently verify them without adding an unregistered parser.
Official validation files are used where available. Wiki validation uses
files excluded from the training file list. BELLE has no separate validation
file, so a deterministic `row_index mod 1000` partition reserves one row for
validation and excludes it from training. WMT17 is held out from WMT massive.

## Model and Tasks

Warm-start the 34M-parameter annealed TreeHeap and the supported parent-only
repair kernel from local copies. Source length is 64 SentencePiece tokens and
target length is 64. Each batch samples one of three tasks:

1. `clean_leaf`: decode target from the complete leaf frontier;
2. `clean_multires`: decode target from a randomly selected TreeHeap frontier;
3. `repair_leaf`: erase addressed detail after complete FOLD, infer it from
   the direct parent with the shared repair kernel, UNFOLD, and decode target.

The loss is target cross-entropy plus a small normalized residual-repair loss
only on repair batches. Encoder, decoder, FOLD kernels, and repair kernel are
jointly optimized. This run tests existence and product utility, not
superiority over Transformer.

## Run

- single io RTX 3090 under the existing 270W service limit;
- local data and local atomic checkpoints;
- mixed precision only if the smoke test is finite;
- resumable optimizer/scaler/RNG/source counters;
- registered target: 300,000 updates, adjustable only before launch based on
  measured smoke throughput to keep the run within approximately 8-16 hours;
- validation and CLI examples at fixed intervals.

## Predicts

- `P1 finite`: no OOM, NaN, dropped GPU, or checkpoint corruption.
- `P2 data pressure`: every registered source contributes at least 1,000
  examples and total non-padding target tokens exceed 500M.
- `P3 learning`: held-out mixed teacher-forced NLL improves by at least `0.20`
  over the warm-start checkpoint.
- `P4 repair`: repaired validation NLL stays within `0.50` of clean-leaf NLL
  and recovers at least `50%` of matched damaged NLL.
- `P5 generation`: at least `90%` of held-out generations are non-empty,
  adjacent repetition is at most `0.35`, and unique-output fraction is at
  least `0.50`.
- `P6 conditioning`: source shuffle increases held-out target NLL by at least
  `0.20`; otherwise generation may be an unconditional language prior.
- `P7 resume`: a saved local checkpoint reloads with identical validation NLL
  within `1e-5` and can continue for at least one update.

## Decision Boundary

Passing P1-P7 yields a usable repair-aware TreeHeap seq2seq pretraining
checkpoint, not a proof of general intelligence or architectural superiority.
If continuation improves but QA/translation examples remain unrelated to the
source, retain only language continuation. If repair remains good but source
shuffle is cheap, do not call it conditional seq2seq. All failed runs and
partial checkpoints remain evidence.

## Runtime Incident: 60K Pause

The first launch was deliberately stopped after log step 64,900, retaining the
atomic step-60,000 checkpoint. Validation learning was healthy, but throughput
fell from roughly 23K to 1K valid target tokens per second. The cause was in
the input implementation rather than the model: continuation streams appended
a newly tokenized full document before draining the old buffer, then deleted
from the front of an ever-growing Python list. This created progressive
near-quadratic copying and unbounded unused token accumulation.

The registered repair replaces the list with a FIFO deque, drains buffered
tokens before reading another document, chunks very large raw documents before
SentencePiece encoding, and bounds pair-text encoding to a prefix safely wider
than the 64-token model input. Training resumes from step 60,000 with optimizer,
model, repair kernel, scaler, RNG, trace, and counters restored. The iterable
corpus cursor itself restarts, so this is a state-exact checkpoint resume but
not a byte-exact continuation of corpus order; the interruption is retained in
evidence and must be disclosed in the final result.

## Interim Audit at Step 60,000

| Metric | 10K | 60K | Direction |
|---|---:|---:|---|
| clean held-out NLL | 5.4325 | 5.0249 | improves monotonically |
| source-shuffle NLL damage | +0.5431 | +0.7924 | conditioning strengthens monotonically |
| damaged NLL | 6.1766 | 5.3098 | improves |
| repaired NLL | 5.4361 | 5.0297 | remains near clean |
| repair fraction | 0.995 | 0.983 | high and stable |

At 60K the stream had processed 143,542,378 non-padding target tokens, with
all eight sources contributing more than 143K examples. The result is positive
interim evidence for source-conditioned probability prediction and matched
parent-detail repair. It is not yet a usable generator: some instruction and
translation examples are relevant, but many QA/continuation samples repeat
phrases or answer with a generic prior.

The registered adjacent-token repetition metric is insufficient because it
does not detect repeated multi-token phrases such as repeated place names or
short clauses. P5 therefore cannot pass on the numeric gate alone. Final
adjudication must add 2-8-gram repetition/longest-cycle audits and retain human
examples.

Resume evaluates `initial` at the restored 60K state. That value may measure
60K-to-final improvement, but it cannot replace the claim's original warm-start
baseline. Final P3 must be recomputed post hoc by evaluating the frozen original
warm checkpoint on the identical held-out stream; both comparisons must be
reported.
