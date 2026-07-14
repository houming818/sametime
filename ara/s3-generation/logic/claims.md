# S3 Claims Registry: Generation

Owner: Review Engineer
Writer: Codex
Created: 2026-07-10
Purpose: Track TreeHeap generation-layer claims.

## Claim Status Rules

- `verified`: evidence exists and includes falsification/baseline checks.
- `supported`: positive evidence exists, but baseline/falsification is incomplete.
- `weak positive`: positive evidence exists but strong baselines match or nearly match.
- `open`: plausible but not yet tested.
- `rejected`: tested and failed.

## Semantic Huffman Generation

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-SEM-HUFF-GEN-C01 | S3 should eventually train TreeHeap as a semantic Huffman generation code: encoder, tree code, kernels, and surface decoder jointly minimize reconstruction, masked prediction, route depth, reusable-substructure, and generation losses so that query-useful structures become short, shared, decodable internal subheaps that can generate surface text. | roadmap / blocked by S1 encoder evidence | Design: `semantic_huffman_generation.md`, 2026-07-10. DeepSeek review accepted: this is too large to treat as the next proof. It is retained as a roadmap claim, not an active evidence claim, until the minimal S1 encoder proof shows that TreeHeap can induce useful internal subheaps. | Reject or downgrade if the smaller frozen-encoder decoder gate fails, if BoW/flat baselines match on substructure-controlled generation tasks, if frequency Huffman matches learned semantic Huffman, if echo/readback collapses under compression pressure, if shorter routes do not correspond to reusable query/generation structure, if shuffled corpus preserves the effect, or if training depends on hand semantic labels. |

## Frozen Encoder Decoder Gate

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-FROZEN-DECODER-C01 | Given a frozen TreeHeap encoder that has already induced useful internal subheaps, an S3 decoder should be able to read those internal subheap states and generate a surface label or short surface text better than shuffled/internal-control baselines. | supported numeric / limited gate | `s3_frozen_decoder_gate_probe.py` on io, 2026-07-10, evidence `ara/s3-generation/evidence/s3_frozen_decoder_gate_probe/`. The decoder freezes S1 assignments from `s1_encoder_minimal_observer_probe_026`, builds a `prefix -> P(surface_label)` probability bucket, and performs leave-one-object-out readout. Structured vs shuffled: decoder_top1 `0.4398` vs `0.0903`, decoder_mrr `0.6736` vs `0.3297`, entropy `1.9378` vs `2.2394` bits. | Reject or keep limited if BoW/flat baselines match on same-bag/different-tree generation, if exact sentence generation fails, if the proof depends on gold labels/object IDs as inputs, or if it cannot be reproduced from saved S1 evidence. This gate proves frozen internal bucket readability, not WMT translation or open-ended generation. |

## Same-Bag Different-Tree Generation

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-SAMEBAG-GEN-C01 | If the same leaf sequence and the same bag of tokens can have two different TreeHeap structures with two different surface outputs, then a TreeHeap decoder that reads internal subheaps should generate the correct output better than BoW or flat sequence decoders that do not receive tree structure. | queued proof | Script: `s3_same_bag_tree_generation_probe.py`. Evidence target: `ara/s3-generation/evidence/s3_same_bag_tree_generation_probe/`. Controlled task: `[a,b,c]` with shape `((a b) c)` generates `PAIR a b`, while the same `[a,b,c]` with shape `(a (b c))` generates `PAIR b c`. | Reject if BoW or flat sequence baselines match TreeHeap on OOD exact generation without tree structure, if TreeHeap cannot generalize to held-out triples, if the task leaks shape through token identity/order, or if the result is promoted into WMT/natural-language generation without a real-data follow-up. |

## Task-Loss Structural Emergence

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-TREEHEAP-EMERGENCE-C01 | With only surface generation cross-entropy and no route, depth, merge, category, or compression supervision, a TreeHeap encoder-decoder can learn a non-trivial `stop/left/right` read policy whose held-out generation depends causally on the relevant internal subheap. | queued proof | Script: `s3_treeheap_emergence_probe.py`; design `treeheap_task_loss_emergence.md`; evidence target `evidence/s3_treeheap_emergence_probe/`. The controlled task keeps the same ordered leaves `[a,b,c]` but changes bracketing and the required generated pair. | Reject if TreeHeap does not beat structure-blind BoW/flat sequence controls on OOD triples, if root-only read or zeroing the selected internal subheap does not materially reduce its OOD generation, if its learned route does not prefer the actual internal child, or if the effect survives address/subheap destruction equally well. This does not claim natural-language emergence or a universal loss threshold. |

## Real WMT Seq2Seq Generation

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-WMT-SEQ2SEQ-C01 | A recursively composed TreeHeap encoder can be trained end-to-end from real WMT English text to generate Chinese SentencePiece sequences with only teacher-forced translation cross-entropy; its held-out generation, internal-node ablations, and comparison against BoW/flat-sequence encoders are measurable in one run. | queued feasibility proof | Script: `s3_wmt_treeheap_seq2seq.py`; design `wmt_treeheap_seq2seq.md`; evidence target `evidence/s3_wmt_treeheap_seq2seq/`. The encoder is trained from scratch on WMT17, not initialized from historical collapsed checkpoints. | Reject this feasibility claim if real WMT data cannot be loaded/tokenized, training does not lower held-out NLL versus initialization, greedy decoding is empty/degenerate, or the TreeHeap encoder cannot produce a reproducible comparison with BoW and flat-sequence controls. A lower score than the flat baseline is a valid result, not a reason to hide the run. |

## P0 World-Observation Pretraining

| ID | Claim | Status | Evidence | Falsification |
|---|---|---|---|---|
| S3-P0-WORLD-OBS-C01 | A TreeHeap encoder-decoder can be pretrained from scratch on real unlabeled Chinese text by predicting a held-out continuation span from a preceding TreeHeap context; gradients update shared global parameters, while each document creates only transient TreeHeap state. | running smoke | Script: `s3_treeheap_p0_pretrain.py`; design `p0_world_observation_pretrain.md`; evidence target `evidence/s3_p0_world_observation_smoke/`. P0 reads news, Wikipedia, and web text only, excluding QA/instruction/translation data. | Reject or keep open if the stream cannot be reproduced from manifest data, held-out continuation NLL does not decrease, generated continuations are degenerate, or a TreeHeap full/leaf/root audit gives no evidence that the recursive states are used. This is not a claim of complete world knowledge or QA ability. |

## Residual TreeHeap Forest Pretraining

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S3-RESIDUAL-FOREST-C01 | A preallocated multi-head parameter TreeHeap forest can be trained on real text through a root-only next-token objective, and the native residual update `H_next = H plus DeltaH` should preserve useful state and gradients better than a same-parameter no-residual TreeHeap while measurably using left/right addresses and more than one kernel head. | rejected as written / full corpus | `evidence/s3_residual_treeheap_forest/full/`, io, 2026-07-14. Complete pass: 38,251,247 blocks from news/wiki/web. Four-head residual first logged NaN at step 62,400 and ended non-finite, while the exactly parameter-matched no-residual model finished at valid NLL `6.2365`. Single-head residual remained finite at NLL `6.4350` and learned residual scale `0.004142`. The evidence rejects the unconstrained global residual-scale implementation, not all possible TreeHeap residuals. | Reopen only with bounded residual gain, FP32 residual accumulation, per-head normalization, non-finite guards, retained pre-divergence checkpoints, and a preregistered requirement to beat the finite no-residual control. |
| S3-TREEHEAP-ROOT-COMPRESS-C01 | A shared multi-head three-slot TreeHeap kernel can recursively compress a 64-token real-text block into one root state from which a decoder predicts the next token, with measurable causal dependence on left/right address pairing and individual kernel heads. | supported full-corpus / single-seed | Same full-corpus evidence. The finite four-head no-residual model reached valid NLL/top1/top5 `6.2365/0.1196/0.2329`; destroying address pairing increased NLL by `2.6252`; ablating heads 0..3 increased NLL by `1.1853/2.4668/1.1574/3.4853`; root variance `0.05843`; gate entropy `1.255` nats vs maximum `ln(4)=1.386`. Decoder receives only the root, with no leaf bypass. | Downgrade if reruns across seeds do not reproduce address/head intervention costs, if matched flat/Transformer/root-pooling baselines match the same compute and bottleneck, if retrained head-removal controls erase the ablation effect, or if less destructive address controls show no effect. This does not prove semantic heads, compression optimality, world knowledge, or architecture superiority. |

## Conditional Denoising Seq2Seq

| ID | Claim | Status | Evidence | Falsification |
|---|---|---|---|---|
| S3-DENOISE-SEQ2SEQ-C01 | Raw Chinese text can provide a strongly conditioned seq2seq objective by reconstructing a clean block or missing short span from a damaged block; this should reproduce the stable short generation seen on WMT. | rejected at 1K-step smoke / diagnostic retained | `s3_conditional_denoising_seq2seq.py`; `evidence/s3_conditional_denoising_smoke/`; `evidence/s3_conditional_gap_denoising_smoke/`. Full reconstruction: TreeHeap/Flat/BoW NLL `6.6111/5.7053/6.5707`. Short gap: `7.2434/7.1996/7.2371`. Exact was zero; TreeHeap full and leaf-only were tied in both runs. | Reopen only if a controlled scale-up produces readable held-out recovery and a causal internal-node gain over leaf-only plus matched flat/BoW controls. Current evidence does not support learned topology, persistent memory, world knowledge, or consciousness. |

## WMT Translation-Loss Learned Fold

| ID | Claim | Status | Evidence | Falsification |
|---|---|---|---|---|
| S3-WMT-LEARNED-FOLD-C01 | Real aligned translation loss can train a hard-forward, soft-backward adjacent-fold kernel so that source TreeHeap topology becomes data-selected rather than fixed by array indices, while retaining measurable Chinese sequence generation. | rejected at smoke / route non-causal | `s3_wmt_learned_fold_seq2seq.py`; `evidence/s3_wmt_learned_fold_smoke/`. Learned/fixed/flat/BoW NLL: `6.0448/6.0205/5.8792/5.8761`. Learned full and leaf-only were tied (`6.0448/6.0434`). Route audit: 490 unique routes in 500 samples and 100% changed after token shuffle, but the decoder did not use internal nodes. | Reopen only after removing the leaf-attention bypass and showing a full-vs-leaf causal gain, superiority to fixed/random routes, and route utility beyond input-dependent diversity. |

## WMT Fixed-Bandwidth Frontier

| ID | Claim | Status | Evidence | Falsification |
|---|---|---|---|---|
| S3-WMT-FRONTIER-C01 | Under the same decoder and fixed `K=4` source-memory budget, a translation-loss learned TreeHeap frontier should beat fixed-tree, random-tree, and flat four-vector frontiers; perturbing learned subheaps should measurably damage held-out translation. | main claim not supported / weak causal route signal | `s3_wmt_frontier_bottleneck.py`; `s3_wmt_frontier_intervention.py`; evidence `evidence/s3_wmt_frontier_smoke/`. Learned/fixed/random/flat NLL `6.5988/6.6012/6.7020/6.5080`. Same-checkpoint fixed/random route replacement increased NLL by `0.0097/0.0285`, below the `0.05` gate. | Reopen only after compose-state information preservation is redesigned and learned frontier beats flat `K=4` plus fixed/random controls across preregistered seeds. Current evidence does not establish a TreeHeap WMT advantage. |
