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

## Explicit Echo Encoder / Decoder

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-ECHO-ED-C01 | An explicit TreeHeap echo encoder/decoder can close the hard echo interface on real WMT SentencePiece short sequences: ordered leaf write, internal summary compose, path-addressed leaf read, subheap decode, and full sequence decode. | supported pilot | `s1_echo_encoder_decoder_probe.py` on io, 2026-06-30. Main run: WMT17 SentencePiece, 2000 samples, train/test/OOD `1600/200/200`, length 3-8, vocab limit 1024, learned parameters `0`, target heap not used in decoder; all sequence/leaf/subheap/summary metrics `1.0000`. Expanded run: `s1_echo_encoder_decoder_probe_expanded`, 20000 samples, train/test/OOD `16000/2000/2000`, length 3-16, vocab limit 4096, train/test/OOD sequence exact, leaf acc, subheap exact, and summary exact all `1.0000`; expanded run includes padded/empty subheap queries and should not be read as semantic evidence. | Reject or downgrade if the decoder secretly uses target heap labels, if ordered leaf writes cannot reconstruct full sequences and subheaps, if internal summaries disagree with decoded subheaps, or if the interface fails when connected to learned write/read kernels. |

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

## Ordered Fold Kernel

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-FOLD-C01 | Natural internal readout requires an order-preserving TreeHeap fold: leaf address/path structure must be preserved before any bag or modulo/cyclic folding is used. | supported pilot | `s1_ordered_fold_kernel_probe.py` on io, 2026-06-29. Pure toy proof, 5000 random length-8..16 sequences, max_len 16, vocab 257. `ordered_tree_fold` natural readout mean/exact `1.0000/1.0000`; `bag_root_fold` mean/exact `0.2766/0.0888`; `modulo_fold_base4` mean/exact `0.3405/0.0888`. The ordered fold beats bag by `+0.7234` mean and early modulo fold by `+0.6595` mean. | Reject or downgrade if a non-addressed bag/global summary or early modulo/cyclic fold preserves the same natural subheap readout under comparable conditions. This does not reject modulo as a later folding operator; it only says modulo should not replace the first order-preserving TreeHeap fold. |

## Latent Plane Fold

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-PLANE-C01 | Language fold should be modeled first as latent placement and attraction over tokens and phrases; TreeHeap is a coordinate and partition system for computing over that placement, not the language law itself. | open theory | `latent_plane_fold.md`, 2026-06-30. This reframes Transformer attention as a dynamic relation field and TreeHeap as a binary/spatial partition over latent placement. No experiment has been run yet. | Reject or downgrade if real-language relation-layout probes show no stable neighborhoods for simple relations such as det-head, aux-verb, quant-head, modifier-head, or predicate-argument, or if TreeHeap partitions do not improve over random/linear partitions after a latent relation layout is established. |

## Controllable Fold Manifold

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-MANIFOLD-C01 | If TreeHeap fold is driven by kernel controls over a latent relation field, product-like structure should emerge as a measurable control surface: increasing relevant control variables should move output from noisy blocks toward stable target blocks on a toy relation task. | supported pilot | `s1_controllable_manifold_probe.py` on io, 2026-06-30. Four short sentence cases, 64 seeds, relation/order weight sweep. Low-control mean F1 `0.0828`; best mean F1 `0.8148`; diagonal gain `0.7319`; product cells `22`; pilot_pass=true. | Reject or downgrade if the sweep is non-controllable across seeds, if quality does not improve from low-control to high-control cells, if random/order-only controls match relation-conditioned controls, or if the same control surface disappears on real relation-layout probes. |

## Heap-State Relaxation

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-RELAX-C01 | A differentiable energy over the current TreeHeap state can generate gradients that relax `arr[i]` toward a lower-energy equilibrium while kernel parameters and address rules remain fixed. | supported pilot | `s1_heap_state_relaxation_probe.py` on io, 2026-06-30. `theta_updated=false`, `heap_state_updated=true`, `target_heap_used_in_loss=false`. Scalar `[2,1,3]` energy ratio `2.47e-31`; vector 7-node TreeHeap mean energy ratio `1.24e-13`, max ratio `3.69e-13`, mean centroid error drop `3.0393`, pass rate `1.0`. | Reject or downgrade if state-only updates cannot lower heap energy across random initial states, if equivalent improvement requires updating kernel parameters, if target heap labels are secretly used in the loss, or if the mechanism fails when connected to real relation fields and probabilistic collapse. |

## Kernel Parameter Learning

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-KERNEL-LEARN-C01 | A parameter TreeHeap `Theta` can learn a local subheap convolution rule from scalar loss and gradient, distinct from SPR-038's heap-state relaxation. | supported pilot | `s1_kernel_parameter_learning_probe.py` on io, 2026-07-01. Hidden scalar kernel `[1,1,1]`; learned theta `[0.9999999999999998,0.9999999999999998,1.0000000000000004]`; theta L2 error `5.44e-16`; TreeHeap test/OOD MSE `8.78e-31/8.93e-30`; wrong-address test MSE `5.9285`; matched-size flat-global test MSE `3.5601`; pilot_pass=true. | Reject or downgrade if `Theta` cannot recover hidden kernels beyond the clean scalar toy, if wrong-address baselines match it, if future proofs only move `H` while `Theta` stays fixed, or if matched flat baselines win without using address/path/subheap structure. |

## Mirror / Chiral Kernel Flip

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-KERNEL-MIRROR-C01 | TreeHeap local convolution is equivariant under mirror / chiral flip: `P_m K_theta(H) = K_{P_lr theta}(P_m H)`, where `P_m` mirrors heap addresses and `P_lr([root,left,right])=[root,right,left]`; scalar loss can learn this mirrored structural slot assignment. | supported pilot / medium reliability | `s1_mirror_kernel_symmetry_probe.py` on io, 2026-07-01. Asymmetric theta `[0.5,1.25,-0.75]`; flipped-kernel test max error `8.88e-16`; OOD max error `3.55e-15`; unflipped-kernel mean error `6.4372`; learned mirrored theta `[0.5000000000000002,-0.7499999999999998,1.2499999999999996]`; theta-mirror L2 error `5.44e-16`; left->original-right error `2.22e-16`; right->original-left error `4.44e-16`; learned test/OOD MSE `1.01e-30/9.76e-30`; pilot_pass=true; Runner audit: structure-assignment level, not rotation/3D-fold, not learned mirror trigger. | Reject or downgrade if mirror equivariance fails for deeper/vector kernels, if the unflipped kernel performs equally well on mirrored heaps, if learned mirrored kernels do not recover `[root,right,left]`, if the proof only works for symmetric left/right kernels, or if later writing promotes this result into a rotation/full-3D-fold/learned-trigger claim. |

## S1 Echo Entry Gate

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-ECHO-GATE-C01 | A controlled S1 echo loop can choose between separate identity/mirror read kernels and decode tokens. | downgraded / misdirected pilot | `s1_echo_entry_gate_probe.py` on io, 2026-07-01. It reached OOD exact `1.0000`, but the design selected output-specific read kernels instead of first canonicalizing mirror input back to echo state. Houming818 rejected this direction. | This claim is not the accepted S1 entry gate. It remains only as a negative design lesson: choosing separate output read kernels is weaker than inverse-gate canonicalization. |
| S1-ECHO-CANON-C01 | A controlled S1 echo loop can learn inverse structural canonicalization before token collapse: observed identity/mirror token leaves are mapped into one canonical echo state and decoded by one shared echo decoder. | supported pilot | `s1_echo_inverse_gate_probe.py` on io, 2026-07-01. Corrected task: target is canonical `[t0,t1,t2,t3]` for both identity and mirrored observations. OOD exact `1.0000`; inverse route argmax `1.0000`; identity inverse gate `0.999794`; mirror inverse gate `0.999784`; canonical state MSE `0.000988962`; no-inverse baseline OOD exact `0.218750`; pilot_pass=true. | Downgrade if canonical-state loss is removed and route/canonical state collapses, if a no-inverse or matched flat baseline solves the same canonicalization task, if routes do not align with identity/mirror addresses, or if later writing promotes this supervised transform-flag proof into natural-language trigger discovery or translation. |

## Sentence-Level Flip Echo

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-ECHO-SENT-C01 | On real short English sentences, TreeHeap S1 echo can be measured as same-algebra flip perturbation/recovery: `WriteLeaves(sentence) -> Flip(root, full_depth) -> learned inverse route -> canonical state -> shared decoder`, with exact/token/edit metrics by sentence length and readable examples. | supported pilot | `s1_sentence_flip_echo_probe.py` on io, 2026-07-02. WMT17 English side, whitespace tokens, 20,000 samples, length 3-32, vocab 8192, train/test/OOD `16000/2000/2000`; perturbation is TreeHeap `Flip(root, full_depth)` on balanced TreeHeap leaves. Hard closure exact `1.0000`; learned inverse OOD exact/token/edit `0.9645/0.9978/0.9979`; no-inverse OOD exact `0.0000`; readable examples included in `README.md`; pilot_pass=true. | Downgrade if hard TreeHeap flip closure is not exactly `1.0`, if perturbation is implemented as external array reverse, if readable examples are absent, if no-inverse or matched flat baselines reach similar exact recovery, or if longer/variable local `Flip(node, depth)` tasks collapse. |
| S1-ECHO-LOCAL-FLIP-C01 | The same-algebra flip echo proof also works at realistic local subheap scale: only a contiguous sentence span is converted into a local TreeHeap and flipped by `Flip(span_root, full_depth)`, while the rest of the sentence remains unchanged; a learned inverse route can restore the canonical sentence through a fixed codebook decoder. | supported pilot | `s1_local_flip_echo_probe.py` on io, 2026-07-03. WMT17 English side, whitespace tokens, 20,000 samples, length 8-32, vocab 8192, span length 2-8, train/test/OOD `16000/2000/2000`. Hard local TreeHeap closure exact `1.0000`; learned local inverse OOD exact/token/edit `1.0000/1.0000/1.0000`; no-inverse baseline OOD exact/token/edit `0.0010/0.7698/0.7376`; all span lengths 2..8 and front/middle/back buckets exact `1.0000`; pilot_pass=true. | Downgrade if the perturbation is array slicing rather than TreeHeap `Flip(span_root, full_depth)`, if fixed codebook decoding is replaced by a target-leaking decoder, if no-inverse or matched flat baselines catch up, or if the next version cannot learn/discover `node, depth` instead of receiving span start/length as metadata. |

## WMT Canonical Echo

| ID | Claim | Status | Evidence | Falsification |
|----|-------|--------|----------|---------------|
| S1-CANON-WMT-C01 | S1 echo should learn a low-entropy canonical TreeHeap state for WMT parallel surface forms: true English/Chinese pairs should map closer than mismatched pairs while each side remains echo-readable. | weak positive / small TreeHeap advantage | `s1_wmt_canonical_echo_probe.py` on io, 2026-07-03. WMT17 `train.zh-en`, 50,000 pairs, train/test/OOD `40000/5000/5000`, max eval 2,000, max_len 48, dim 128, epochs 5. TreeHeap OOD positive/negative distance `0.3458/0.9929`, margin `0.6472`, retrieval@1/@5 `0.6300/0.8195`, entropy `4.0443`, EN/ZH echo token `1.0000/0.9981`. BoW OOD margin `0.5939`, retrieval@1/@5 `0.6285/0.8085`, entropy `4.3442`. Untrained TreeHeap margin `0.0030`, retrieval@1 `0.0010`. | Downgrade if BoW or matched sequence/Transformer baselines match or beat TreeHeap across margin, entropy, and retrieval; if root canonical state cannot be separated from leaf echo memory; if retrieval collapses on larger candidate pools or longer/noisier WMT; or if the result is promoted into BLEU/semantic grounding without S2 decoding evidence. |

## Architecture Position

Current S1 conclusion:

SPR Echo has proved capacity, order sensitivity, and near-lossless self-mapping. Current token-only S1 routing has failed the first polysemy falsification, so it should be treated as a high-capacity identity/path hash. A controlled context proof now supports the S1b interface, but only as a mechanism proof. The 2026-06-25 shallow sentence pilot adds the first post-M0 S1 bridge: real-word short sentences can be written into shallow TreeHeap slots and queried with OOD lexical copy. The first frozen-embedding world-coordinate probe is a negative result: simple vector addition is already a strong compound baseline, while the current unconstrained TreeHeap prob vector plus overfits train and fails OOD. The local-corpus SGNS probe partially repairs this: once the coordinate system is trained from local co-occurrence and the TreeHeap encoder is structurally constrained, TreeHeap beats `vector_add` on OOD cosine and approaches concat MLP, but top1 retrieval is not solved. The WMT echo probe moves from curated/local corpus to real WMT SentencePiece data: a structured TreeHeap kernel strongly beats BoW and flat seq MLP on short-sequence OOD echo with far fewer parameters. The explicit echo encoder/decoder proof now pins down the hard contract: ordered leaf write, internal summary compose, path-addressed leaf/subheap read, and full sequence decode close exactly on WMT short BPE without learned parameters. The multi-kernel perturbation probe adds a more cautious result: gates and ablations show that structural tasks can induce kernel differentiation, but current root-bottleneck reconstruction accuracy is too low to claim a solved multi-kernel learner. SPR-032 removes the root bottleneck for read by using a probabilistic `stop/left/right` kernel from `arr[1]`: route collapse and leaf read are solved in the pilot, but arbitrary internal checksum summaries remain weak. SPR-034 repairs the framing by replacing arbitrary checksum with algebraically natural internal targets. Routed internal-node readout is strongly better than the root bottleneck on length, first/last token, and prefix targets. SPR-035 adds a pure ordered-fold control: preserving linear leaf order through TreeHeap path/address is necessary for natural subheap readout, while bag/global collapse and early modulo/cyclic folding lose locality. SPR-036 reframes the larger theory: language fold should be discovered as latent placement/attraction in a relation field first; TreeHeap is then a coordinate and partition system for computing over that field. SPR-037 adds a product bridge: kernel controls over a toy relation field form a measurable fold-quality surface, moving from noisy blocks toward stable blocks. SPR-038 supports Houming818's state-gradient hypothesis: fixed TreeHeap rules can generate gradients over `arr[i]` and relax heap state without updating kernel parameters. Residue/mod remains a separate folding hypothesis, not the S1 readout gate. SPR-041 corrects the entry gate: mirror should be inverted into canonical echo state before shared decoding. S1-ECHO-SENT-C01 adds the missing readable sentence proof: on 20k real WMT English short sentences, same-algebra TreeHeap full-root flip echo reaches OOD exact/token/edit `0.9645/0.9978/0.9979`, with examples showing observed flipped sentences restored to target sentences. S1-ECHO-LOCAL-FLIP-C01 repairs the main realism objection: local span subheaps of length 2..8 can be flipped by the same TreeHeap algebra and restored exactly on OOD when span start/length are given. S1-CANON-WMT-C01 then reframes S1 correctly: the goal is not random repair, but a WMT-scale canonical state where English and Chinese surface forms for the same pair move closer while both sides remain echo-readable. The first 50k-pair probe is a weak positive, with only a small TreeHeap advantage over BoW; the next gate is stronger sequence baselines and separating root canonical meaning from leaf echo memory.
