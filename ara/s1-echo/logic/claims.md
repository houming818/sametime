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

## Architecture Position

Current S1 conclusion:

SPR Echo has proved capacity, order sensitivity, and near-lossless self-mapping. Current token-only S1 routing has failed the first polysemy falsification, so it should be treated as a high-capacity identity/path hash. A controlled context proof now supports the S1b interface, but only as a mechanism proof. The 2026-06-25 shallow sentence pilot adds the first post-M0 S1 bridge: real-word short sentences can be written into shallow TreeHeap slots and queried with OOD lexical copy. The next architecture gate is to make this less position-template-like by adding variable length, modifiers, passive/OSV order, and matched copy-capable baselines.
