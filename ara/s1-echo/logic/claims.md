# S1 Claims Registry: SPR Echo

Owner: Review Engineer
Writer: Codex
Created: 2026-06-16
Updated: 2026-06-25
Purpose: Track SPR S1 claims, evidence, and falsification criteria.

## Claim Status Rules

- `verified`: evidence exists and includes a falsification or baseline check.
- `supported`: positive evidence exists, but baseline/falsification is incomplete.
- `open`: plausible but not yet tested.
- `rejected`: tested and failed.

## Capacity And Order

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-C01 | Decomposed TreeHeap routing has enough capacity to assign most WMT14 words to unique effective leaves. | verified | `spr_echo_proof.py`: `solo=41311/41429`, `solo%=99.7%` on io, 2026-06-16 | If solo rate drops below 95% under fixed seed and same WMT14 slice. |
| S1-C02 | Pure cyclic shift is order-colliding; sign alternation breaks the symmetry. | verified | `spr_hash_cyclic.py`: pure roll collision true, sign-alt separated true | If `A,B` and `B,A` remain equal after sign-alt on a non-degenerate vector set. |
| S1-C03 | Echo reconstruction can reach near-perfect BLEU without learned Transformer attention. | supported | `spr_echo_proof.py`: BLEU-4 `99.99` on io | If shuffled leaf labels or random remapping preserves BLEU, this only proves lookup capacity, not structure. |

## Semantic Routing

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-C10 | SPR paths can encode semantic distinctions when routing is conditioned on context. | supported | `spr_context_proof.py`: controlled polysemy context route acc `1.00`, shuffled acc `0.48`; old token-only route acc `0.43`. | Real-corpus contexts, random-hash baseline, or BoW baseline match/beat the context route. |
| S1-C11 | Current S1 token-only routing can route the same lexical token to different stable semantic states. | rejected | `spr_s1_falsification.py`: token-only real acc `0.43`, shuffled acc `0.43`; no context signal. | Reopen only after route(token, context) beats token-only and random-hash baselines. |
| S1-C12 | Cross-lingual alignment benefits from SPR path geometry beyond nearest-neighbor token identity. | open | S2 claims suggest cross-lingual AUC, but S1-specific path evidence is not isolated. | SPR path features fail to beat frequency, nearest-neighbor, and bag-of-words MLP baselines. |
| S1-C13 | The minimum viable S1b interface is route(token, context), not route(token). | supported | `spr_context_proof.py`: token-only path buckets are mixed, context-conditioned buckets are pure in the controlled setup. | If adding context does not improve over token-only under controlled polysemy and shuffle tests. |

## Collapse And Handoff

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-C20 | Dynamic routing can be collapsed into a static matrix or lookup artifact for downstream S2. | supported | Historical ARA notes mention frozen L1 encoder and massive checkpoint. | Recomputed dynamic routing and frozen artifact disagree on downstream metrics beyond noise. |
| S1-C21 | S1 output is a valid input contract for S2 fold-stack translation. | open | S2 consumes L1-style vectors conceptually. | S2 performance is unchanged when S1 vectors are replaced by random vectors with matched norm/frequency. |

## Shallow Real-Sentence TreeHeap

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-C30 | A learnable shallow TreeHeap write can encode real short sentences into queryable `root/subject/object` slots, including OOD lexical copy-by-address. | supported pilot | `shallow_treeheap_s1_probe.py` on ni, 2026-06-25: soft TreeHeap train/test/OOD exact `1.0/1.0/1.0`; learned position writes `subject/root/object`; BoW and seq linear OOD exact `0.0`. | If a matched copy-capable flat baseline or sequence model matches OOD copy and structural query accuracy under the same parameter and data budget, this pilot does not show a TreeHeap-specific S1 advantage. |

## World-Model Coordinates

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-WM-C01 | Frozen external embeddings can be used as a world-coordinate ruler for compound words; a TreeHeap prob vector plus encoder should map compound inputs toward target compound coordinates at least comparably to simple baselines. | rejected pilot | `s1_world_model_compound_probe.py` on io, 2026-06-25: local cached `all-MiniLM-L6-v2` projected to 128D; `vector_add` OOD cosine/top1 `0.7198/0.833`; `concat_mlp` OOD `0.5766/0.0`; `treeheap_prob_vector_plus` OOD `0.3919/0.0`; pilot_pass=false. | Reopen only if a constrained TreeHeap encoder beats `vector_add` and matches/exceeds copy/concat baselines on held-out/OOD compound families without treating frozen embeddings as a trainable teacher. |
| S1-WM-C02 | When the coordinate system is trained from local corpus co-occurrence rather than a pretrained embedding, a structured TreeHeap kernel can use write/compose constraints to beat `vector_add` on OOD cosine and approach concat MLP. | supported pilot | `s1_corpus_embedding_kernel_probe.py` on io, 2026-06-25: local SGNS corpus embedding, no pretrained model; OOD cosine `vector_add=0.5785`, `concat_mlp=0.7321`, `structured_treeheap_kernel=0.7126`; OOD top1 remains `0.0`. | If vector_add or matched simple baselines beat the structured TreeHeap kernel across seeds/corpus variants, or if TreeHeap only matches train while OOD cosine/top1 collapses, this claim is rejected. |

## Architecture Position

Current S1 conclusion:

SPR Echo has proved capacity, order sensitivity, and near-lossless self-mapping. Current token-only S1 routing has failed the first polysemy falsification, so it should be treated as a high-capacity identity/path hash. A controlled context proof now supports the S1b interface, but only as a mechanism proof. The 2026-06-25 shallow sentence pilot adds the first post-M0 S1 bridge: real-word short sentences can be written into shallow TreeHeap slots and queried with OOD lexical copy. The first frozen-embedding world-coordinate probe is a negative result: simple vector addition is already a strong compound baseline, while the current unconstrained TreeHeap prob vector plus overfits train and fails OOD. The local-corpus SGNS probe partially repairs this: once the coordinate system is trained from local co-occurrence and the TreeHeap encoder is structurally constrained, TreeHeap beats `vector_add` on OOD cosine and approaches concat MLP, but top1 retrieval is not solved. The next gate is multi-seed/corpus robustness and stronger kernel constraints for top1.
