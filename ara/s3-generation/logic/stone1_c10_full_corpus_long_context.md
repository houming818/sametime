# STONE-1 C10: Full-Corpus Long-Context Growth

Date: 2026-07-26
Status: 128-token smoke supported / core-raw full pass running
Milestone: `STONE-1-LONG`
Claim: `S3-STONE1-FULL-CORPUS-LONG-C10`
Predict: `P-S3-STONE1-FULL-CORPUS-LONG-10`

## Question

STONE-1 was signed on one million unique WMT sentence pairs with source and
target lengths restricted to 8--32 SentencePiece tokens. The io host now holds
about 24 GiB of raw Chinese and bilingual corpora. C10 asks whether the same
TreeHeap family can grow to 128 effective tokens, learn from every eligible
local training record, and retain the short-sequence behavior and structural
causality established by C09.

This is not a claim that concatenated sentences become one grammatical
sentence. They form a long training sequence with explicit EOS boundaries.

## Data Contract

The corpus inventory must finish before training constants are frozen. Files
are classified as:

```text
raw continuation: news2016, wiki_zh, webtext2019, available Zhihu shards
parallel text:    WMT Massive, WMT17, translation2019
instruction/QA:   BELLE, baike2018qa, Chinese medical dialogue
excluded:         derived blocks, checkpoints, distillation caches,
                  hidden download caches and incomplete files
```

Original rows are assigned to train/validation/test before packing. Packing is
performed only inside a split. Every segment retains EOS. Parallel source and
target segments are packed in the same order.

The io inventory completed in 64.60 seconds. After excluding derived blocks,
checkpoints, tokenizers, distillation caches, hidden caches and incomplete
downloads, the currently eligible training set is:

| category | files | bytes | physical rows |
|---|---:|---:|---:|
| raw continuation | 1,280 | 15,363,453,793 | 7,595,600 |
| parallel | 3 | 3,840,256,287 | 19,431,709 |
| instruction/QA | 13 | 4,310,456,203 | 5,357,075 |
| total | 1,296 | 23,512,416,539 | 32,384,384 |

The row count is a lower bound because the four materialized Zhihu parquet
shards cannot be counted until a parquet runtime is installed. Zhihu shard
`00000` is absent, so C10 must not describe the local mirror as complete
Zhihu. Content filtering and packing happen after this physical inventory.

The first 128-token/256-leaf AMP smoke learned from initial validation NLL
12.8650 to final NLL 8.0385 in 300 steps. All eight recursive depths were
visible and shuffling any depth increased NLL. Peak allocated VRAM was only
1.33 GiB. The smoke is not accepted, however, because at least one FP16
gradient became non-finite before GradScaler skipped the update. Task 35 repeats
the same architecture and data contract in FP32 before any long run starts.

The FP32 repeat passed all preregistered smoke gates. Validation NLL fell from
13.0525 to 7.8967 in 300 steps, every recursive depth remained visible,
shuffling stored detail caused measurable damage, and peak allocated VRAM was
2.59 GiB. This closes the numerical smoke question, not the full-corpus claim.

A separate BF16 batch ladder measured the scaling envelope without changing
the model or 128-token contract:

| batch | effective token/s | peak VRAM | finite |
|---:|---:|---:|:---:|
| 8 | 540 | 1.82 GiB | yes |
| 16 | 1,121 | 2.85 GiB | yes |
| 32 | 2,299 | 5.11 GiB | yes |
| 64 | 4,494 | 9.45 GiB | yes |
| 96 | 6,608 | 13.98 GiB | yes |
| 128 | 8,798 | 18.37 GiB | yes |

Batch 96 is the formal-run choice. It leaves about 6 GiB of the 24 GiB device
unused for allocator variation and the known fragility of io's second-hand
3090. The already prepared raw corpus contains 38,251,247 blocks of 64 tokens,
or about 2.448 billion tokens. At the measured batch-96 rate, raw continuation
alone is approximately 4.3 device-days. Parallel and instruction stages make
the first exhaustive pass an approximately one-week job; this estimate must be
replaced by observed wall time once the deterministic full-pass iterator runs.

## Core-Raw Materialization And Launch

Task 42 rebuilt the raw continuation data with the frozen 32K tokenizer. It
visited all 1,276 available news/wiki/web files, inserted EOS after every
accepted document, packed short documents without erasing boundaries, and
created blocks of exactly 256 tokens:

```text
block[0:128]   = source
block[128:256] = target
```

The training split contains 11,016,813 blocks and 2,820,304,204 tokens in 111
shards. The validation split contains 280,391 blocks and 71,780,218 tokens in
three shards. Both manifests report `complete_source_pass=true`. Preparation
took 572.5 seconds including validation. Every shard carries SHA-256, and the
packer stores a file/line cursor plus the pending token tail for recovery.

This is the complete core-raw pass, not yet every locally eligible raw file.
Four materialized Zhihu parquet shards require a parquet runtime and are staged
for a later continuation on the same checkpoint; shard 00000 is absent from the
local mirror and remains a declared corpus gap.

Tasks 43 and 44 validated BF16 training and exact checkpoint continuation.
Task 44 resumed from step 10 and row 960 and reached step 20 and row 1,920.
The run remained finite, peaked at 13.89 GiB, and measured about 8,282 target
tokens/second. Task 45 then started the exhaustive batch-96 core-raw pass with
a 72-hour taskd window, 1,000-step atomic checkpoints and 2,000-step frozen
validation. Its initial wall-time estimate is about 47 hours.

The intended capacity ladder is:

| arm | effective tokens | TreeHeap leaves | recursive depths |
|---|---:|---:|---:|
| C09 baseline | 32 | 64 | 6 |
| bridge | 64 | 128 | 7 |
| target | 128 | 256 | 8 |

State width remains 320 and decoder hidden width remains 512 in the first
test. This isolates context capacity from parameter scaling. The total forced
depth-pressure budget remains 0.12, giving 0.02 per level at depth 6 and 0.015
per level at depth 8.

## Training Stages

1. Teacher-free continuation pretraining over raw Chinese documents.
2. Full-pass bilingual seq2seq training over every eligible parallel pair.
3. Instruction/QA continuation over every eligible prompt/response pair.
4. Frozen evaluations on short WMT, long WMT, raw continuation and dialogue.

The same TreeHeap checkpoint moves through all stages. No Qwen or other model
provides targets. The corpus itself supplies deterministic next-span or paired
target loss.

## Predictions

```text
P1 all declared train files are visited and recorded in a manifest
P2 no atomic source row crosses train/validation/test boundaries
P3 the 128-token model remains finite on one complete corpus pass
P4 long-bucket held-out NLL beats the untrained 128-token model by >= 0.50
P5 C09 short-bucket NLL degrades by no more than 0.20 after long training
P6 detail/path intervention increases long-bucket NLL by >= 0.10
P7 removing upper TreeHeap levels harms cross-sentence targets more than
   within-sentence local targets by >= 0.05 NLL
P8 checkpoint resume reproduces frozen outputs exactly
```

P7 is the positive depth-growth test. It does not assume in advance that root
means summary or that a particular depth has a human grammatical label.

## Falsification

- If longer training improves only leaf-visible decoding and structural
  interventions remain harmless, the result is a flat sequence model inside a
  TreeHeap container.
- If random sentence packing matches adjacent-document packing on cross-sentence
  evaluation, packing did not teach discourse structure.
- If 128-token quality improves only by losing the C09 short contract, C10 is
  a capacity trade rather than a strict extension.
- If the full pass cannot be resumed deterministically, no product or scaling
  claim is allowed.

## Evidence

```text
../evidence/s3_stone1_c10_corpus_inventory/
../evidence/s3_stone1_c10_long_smoke/
../evidence/s3_stone1_c10_long_smoke_fp32/
../evidence/s3_stone1_c10_batch_ladder/
../evidence/s3_stone1_c10_raw_pack/
../evidence/s3_stone1_c10_raw_train_smoke/
../evidence/s3_stone1_c10_full_corpus/
```
