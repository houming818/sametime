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
