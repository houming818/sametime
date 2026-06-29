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

## WMT Echo Kernel

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-WMT-ECHO-C01 | A structured TreeHeap kernel can write and read real WMT SentencePiece short sequences in an echo setting, using tree addresses and shared compose/read kernels rather than only a flat memorization map. | supported pilot | `s1_wmt_echo_kernel_probe.py` on io, 2026-06-25: WMT17 English side, 3000 samples, length 3-8, vocab limit 2048; `treeheap_kernel_echo` OOD token/exact `0.9818/0.9000` with `423104` params; `seq_mlp` OOD `0.5986/0.0533` with `16794112` params; `bow_linear` OOD `0.1659/0.0033`. | If matched copy-capable baselines or larger flat/sequence models match TreeHeap OOD exact under similar parameter and sample budgets, or if longer/noisy/variable-depth WMT echo collapses, this remains only a short-sequence structure pilot. |

## Multi-Kernel Specialization

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-MK-C01 | Structural perturbation tasks can push a TreeHeap kernel bank toward task-dependent specialization, analogous to the opportunity for Transformer multi-head attention to differentiate. | open / mixed pilot | `s1_wmt_multikernel_specialization_probe.py` on io, 2026-06-26. Full vocab run: WMT17, 4000 samples, vocab 2049, length 4-8; gate argmax used all 4 kernels (`echo->2`, `mask_restore->1`, `left/right->3`, `mirror->0`), max OOD ablation exact drop `0.1100`, but OOD mean exact only `0.0600` vs single `0.0495`. Common-token run: vocab 513; gate argmax used all 4 kernels (`echo->0`, `mask_restore->1`, `left->0`, `right->3`, `mirror->2`), max OOD ablation exact drop `0.3050`, but OOD mean exact only `0.1420` vs single `0.1275`. | Reject or downgrade if multi-kernel gates specialize only because of explicit task labels while task accuracy stays low, or if matched single-kernel / flat / small Transformer baselines match specialization and OOD task performance. Support requires both reliable task accuracy and task-specific ablation drops. |

## Probabilistic Read Collapse

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-READ-C01 | TreeHeap read should be modeled as query-conditioned probabilistic collapse over `stop/left/right` from `arr[1]`; `stop` at an internal node is meaningful and should return a subheap state, not only a leaf token. | open / mixed pilot | `s1_probabilistic_read_kernel_probe.py` on io, 2026-06-29. WMT17 short BPE, 3000 samples, length 4-8, vocab 513. Main run with 128 checksum buckets: root OOD acc `0.0638`, read-kernel OOD hard acc `0.6124`, route acc `1.0000`, leaf acc `0.9989`, internal acc `0.2214`. Diagnostic run with 32 checksum buckets: root OOD acc `0.1184`, read-kernel OOD hard acc `0.7177`, route acc `1.0000`, leaf acc `0.9989`, internal acc `0.4332`. | Support requires route collapse, leaf read, and internal subheap read to all pass. Current evidence supports address/path collapse and near-perfect leaf read, but internal subheap summary is not solved. Reject if matched root/flat baselines reach the same OOD route/read behavior, or if route accuracy collapses without teacher-forced routes. |

## Algebraic Internal Readout

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-READ-C02 | After a read kernel reaches an internal node, the first readout targets should be algebraically natural subheap attributes, not arbitrary checksum labels. | supported pilot | `s1_algebraic_readout_probe.py` on io, 2026-06-29. WMT17 short BPE, 5000 samples, length 4-8, vocab 513. On internal OOD queries, routed node state strongly beats the root bottleneck on natural targets: `length=0.9886` vs `0.8388`, `first=0.9277` vs `0.5543`, `last=0.8725` vs `0.2387`, `prefix0=0.9267` vs `0.5543`, `prefix1=0.8725` vs `0.3756`. The deterministic algebraic oracle is `1.0000`. A residue/checksum diagnostic was also recorded, but it is not part of this claim. | If root-only or matched flat/pointer baselines match routed internal readout under the same data/parameter budget, the TreeHeap route advantage is not established. If routed state does not preserve length/first/last/prefix under larger WMT slices or longer variable-depth trees, the natural internal-readout claim should be downgraded. This does not prove translation, semantic phrase meaning, unsupervised routing, or Transformer superiority. |

## Architecture Position

Current S1 conclusion:

SPR Echo has proved capacity, order sensitivity, and near-lossless self-mapping. Current token-only S1 routing has failed the first polysemy falsification, so it should be treated as a high-capacity identity/path hash. A controlled context proof now supports the S1b interface, but only as a mechanism proof. The 2026-06-25 shallow sentence pilot adds the first post-M0 S1 bridge: real-word short sentences can be written into shallow TreeHeap slots and queried with OOD lexical copy. The first frozen-embedding world-coordinate probe is a negative result: simple vector addition is already a strong compound baseline, while the current unconstrained TreeHeap prob vector plus overfits train and fails OOD. The local-corpus SGNS probe partially repairs this: once the coordinate system is trained from local co-occurrence and the TreeHeap encoder is structurally constrained, TreeHeap beats `vector_add` on OOD cosine and approaches concat MLP, but top1 retrieval is not solved. The WMT echo probe moves from curated/local corpus to real WMT SentencePiece data: a structured TreeHeap kernel strongly beats BoW and flat seq MLP on short-sequence OOD echo with far fewer parameters. The multi-kernel perturbation probe adds a more cautious result: gates and ablations show that structural tasks can induce kernel differentiation, but current root-bottleneck reconstruction accuracy is too low to claim a solved multi-kernel learner. SPR-032 removes the root bottleneck for read by using a probabilistic `stop/left/right` kernel from `arr[1]`: route collapse and leaf read are solved in the pilot, but arbitrary internal checksum summaries remain weak. SPR-034 repairs the framing by replacing arbitrary checksum with algebraically natural internal targets. Routed internal-node readout is strongly better than the root bottleneck on length, first/last token, and prefix targets. Residue/mod was only a side diagnostic motivated by possible modular folding from linear order into a tree; it is not the current S1 gate. The next gate is to connect probabilistic route collapse to these natural internal readouts, then test longer WMT spans and matched flat/pointer/Transformer read baselines.
