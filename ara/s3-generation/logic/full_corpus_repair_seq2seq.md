# Full-Corpus Repair-Aware Seq2Seq

Status: preregistered long run

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
