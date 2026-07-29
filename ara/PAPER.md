# ARA Paper: Semantic Prefix Routing and TreeHeap

Status: living research artifact
Created: 2026-06-22
Updated: 2026-07-01
Owner: nio / Houming818
Review Engineer: Codex
Repository: SameTime
ARA reference: https://arxiv.org/abs/2604.24658v3

This file is the root ARA manifest for the public SameTime research record.
It is not a conventional paper. It is a compact research state that lets a
human or an AI agent reconstruct:

```text
claim -> predict -> experiment -> evidence -> decision -> next step
```

The human-readable narrative is published as SPR blog posts. This file is the
machine-readable and reviewer-readable map.

## ARA Layout

SameTime uses four ARA layers:

| Layer | Path | Purpose |
|---|---|---|
| Logic | `ara/*/logic/` | Claims, predicts, problems, experiment designs |
| Source | `ara/*/src/` | Reproducible scripts and environment notes |
| Trace | `ara/*/trace/` | Pivots, rejected paths, decision DAG |
| Evidence | `ara/*/evidence/` | Summaries, metrics, pointers to raw logs |

Main topics:

| Topic | Path | Role |
|---|---|---|
| S1 Echo | `ara/s1-echo/` | Capacity, path hashing, context routing |
| S2 Translation | `ara/s2-translation/` | Fold stack, graph builder, probability container |
| M0 TreeHeap Math | `ara/m0-treeheap-math/` | Algebra, plus, kernel search, trainability |
| S3 Generation | `ara/s3-generation/` | Placeholder for later generation / decoder work |

## Status Vocabulary

| Status | Meaning |
|---|---|
| `verified` | Evidence exists and includes a baseline or falsification check |
| `supported` | Positive evidence exists, but baselines or scale are incomplete |
| `open` | Claim is plausible but not yet tested |
| `design` | Architecture or mathematical proposal, not yet evidence-backed |
| `rejected` | Tested and failed |
| `downgraded` | Earlier stronger claim was narrowed by later evidence |

## Global Research Claim

The current global claim is deliberately narrow:

```text
SPR / TreeHeap is not a replacement for Transformer by assertion.
It is a family of addressable, path-aware, tree-structured operators that may
share the same ML capabilities as MLP / CNN / Transformer while providing
stronger inductive bias for address, substructure migration, prefix reuse,
probability containers, and delayed collapse.
```

The research must prove this through small tasks before WMT-scale claims:

```text
M0 math toolbox
-> S1 capacity and context routing
-> TreeHeap trainability
-> structural existence proofs
-> S2 translation / fold-stack tasks
```

## Claim Tree

### C0: ARA Process Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C0-001 | SPR-001, SPR-008 | Every strong claim must have evidence and falsification criteria. | verified | `ara/README.md`, `ara/*/logic/claims.md`, `ara/*/evidence/README.md` | A claim is promoted without evidence pointer or failure condition. |
| C0-002 | SPR-008 | The correct work loop is `predict -> claim -> experiment -> evidence -> trace`. | supported | Blog SPR-008 and current ARA layout | Research decisions continue to depend on chat memory only. |
| C0-003 | SPR-019 | Blog narrative is not enough; claims must be preserved in registry artifacts. | verified process debt / repaired 2026-06-30 | DSPro audit found `claims.md` current through SPR-038 while this root manifest stopped at SPR-021. This update reconciles root map and traces. | Future SPR claims diverge from registry and are not reconciled. |
| C0-004 | DSPro audit 2026-06-30 | ARA has a "storytelling tax": evidence and local claim registries can be healthy while `PAPER.md` and trace DAGs go stale. | verified | `ara/s1-echo/logic/claims.md` was current; `ara/PAPER.md` and `ara/*/trace/research_dag.yaml` were stale before this repair. | Root manifest and trace DAG are updated in the same commit as future claims. |

### C1: S1 Capacity and Order Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C1-001 | SPR-001, SPR-002 | S1 token path hash has enough capacity for WMT14 word echo. | verified | `ara/s1-echo/logic/claims.md` S1-C01, solo rate 99.7% | Solo rate drops below 95% on same slice / seed. |
| C1-002 | SPR-002 | Pure cyclic shift is order-colliding; sign alternation breaks the symmetry. | verified | `ara/s1-echo/logic/claims.md` S1-C02 | Sign alternation still gives same representation for non-degenerate A,B vs B,A. |
| C1-003 | SPR-002 | Echo reconstruction can reach near-perfect BLEU without learned Transformer attention. | supported | `ara/s1-echo/logic/claims.md` S1-C03, BLEU-4 99.99 | Shuffled leaf labels preserve BLEU, proving only lookup. |
| C1-004 | SPR-003 | Token-only routing does not encode contextual semantics. | rejected old claim / verified rejection | `ara/s1-echo/logic/claims.md` S1-C11 | Reopen only if token-only route beats context and random baselines on polysemy. |
| C1-005 | SPR-007 | Context-conditioned routing can encode semantic distinctions in a controlled proof. | supported | `ara/s1-echo/logic/claims.md` S1-C10, S1-C13 | Real corpus or BoW/random baselines match or beat context route. |
| C1-006 | SPR-006, SPR-007 | S1b minimum interface is `route(token, context)`, not `route(token)`. | supported | `ara/s1-echo/logic/claims.md` S1-C13 | Context does not improve over token-only under controlled and real tests. |
| C1-007 | SPR-004 | S1 output can be a contract for downstream S2. | open | `ara/s1-echo/logic/claims.md` S1-C21 | S2 unchanged when S1 vectors are replaced by random matched vectors. |

### C2: S2 Fold Stack and Graph Builder Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C2-001 | SPR-005 | Semantic vectors contain fold-action information. | verified | `ara/s2-translation/logic/claims.md` C-001 | Fold action prediction falls to chance under proper splits. |
| C2-002 | SPR-005 | 32D semantic space is enough for the measured fold-action prediction task. | verified | `ara/s2-translation/logic/claims.md` C-002 | 32D materially underperforms 128D under recomputation. |
| C2-003 | SPR-005 | Cross-lingual fold structure prediction is feasible in both EN->ZH and ZH->EN directions. | verified | `ara/s2-translation/logic/claims.md` C-003, C-004 | Cross-lingual AUC drops to chance against frequency / bag baselines. |
| C2-004 | SPR-005 | Fold action types can be predicted by small MLPs. | supported | `ara/s2-translation/logic/claims.md` C-005 | Baselines match after leakage controls. |
| C2-005 | SPR-005, SPR-008 | PP/VP/NP patterns show different collapse regimes. | verified | `ara/s2-translation/logic/claims.md` C-006..C-010 | Recomputed grammar atlas removes collapse pattern. |
| C2-006 | SPR-005 | Head, span, and action detection are feasible with supervised probes. | verified | `ara/s2-translation/logic/claims.md` C-011..C-013 | Probe fails under clean train/test split. |
| C2-007 | SPR-005 | Fold representation can reconstruct sentence with oracle or template edges. | verified / supported | `ara/s2-translation/logic/claims.md` C-014, C-015 | Reconstruction collapses under oracle-free setting. |
| C2-008 | SPR-008 | Current graph assembly bottleneck is child/parent allocation, not fold action representation. | verified | `ara/s2-translation/logic/claims.md` C-016..C-020 | Oracle child/parent ablations do not close UAS gap. |
| C2-009 | SPR-008 | Parent candidates should be kept as probability containers instead of early argmax. | verified for graph-builder stage | `ara/s2-translation/logic/claims.md` C-028 | Top-k parent coverage is low or later modules cannot use distributions. |
| C2-010 | SPR-008 | Current 3-epoch TreeHeap vectors are over-collapsed and cannot support a syntax-energy claim. | verified | `ara/s2-translation/logic/claims.md` C-027, strategy audit | Longer/retrained vectors show separable syntax energy under baselines. |
| C2-011 | SPR-008, SPR-009, SPR-010 | Historical checkpoints cannot carry the stronger "TreeHeap syntax energy is solved" claim. | downgraded | `ara/s2-translation/evidence/strategy_audit/`, `tmerge_diagnostic/` | New independently trained checkpoint passes energy and syntax baselines. |
| C2-012 | SPR-010 | Current world-model / background-field training evidence is diagnostic, not a positive translation claim. | downgraded | `ara/s2-translation/evidence/world_model_long_20260617_180554/` | New loss shows downstream translation or structure gains with controls. |

### C3: M0 TreeHeap Algebra Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C3-001 | SPR-011 | TreeHeap must first become a mathematical toolbox before language claims. | design | `ara/m0-treeheap-math/logic/problem.md`, `logic/solution/algebra.md` | Language-level work proceeds without closure / inverse / projection tests. |
| C3-002 | SPR-011, SPR-013 | Minimal TreeHeap algebra should support closure, non-commutativity, inverse-like operations, projection, subheap matching, and probability normalization. | supported pilot | `ara/m0-treeheap-math/logic/predicts.md` P-MATH01; `evidence/treeheap_math_probe/` | Synthetic exact mode fails closure or order distinction. |
| C3-003 | SPR-012 | Subheap kernel search is the TreeHeap analogue of convolution over local structure. | design / partly supported by toy | `ara/m0-treeheap-math/logic/experiments.md` existence suite B; blog SPR-012 says no full claim yet | Kernel relocation fails beyond seen positions or degenerates to root matching. |
| C3-004 | SPR-013 | A pure-math probe can establish a non-language mathematical pilot. | supported | `evidence/treeheap_math_probe/README.md` | Probe cannot reproduce closure / inverse / subheap matching. |
| C3-005 | SPR-014 | TreeHeap order should be generated by primitive + plus, not only assigned by an external index. | design | `ara/m0-treeheap-math/logic/predicts.md` P-MATH02 | No plus candidate produces successor, cycle, or order margin. |
| C3-006 | SPR-015 | An addressable TreeHeap can use plus as successor, information gain, mod-base fold, and cyclic-kernel addressing in a toy. | supported toy | `evidence/primitive_plus_probe/` | Toy trace cannot reproduce address successor or mod fold. |
| C3-007 | SPR-016 | Before TreeHeap encoder/decoder, local learning stack must pass linear, nonlinear, and modular toy tasks. | verified for local stack | `evidence/trainability_quiz/` | Linear regression, XOR, or modular addition fails under deterministic rerun. |
| C3-008 | SPR-016 | Passing trainability quiz proves ML capacity of the stack, not TreeHeap language intelligence. | verified boundary | `ara/m0-treeheap-math/logic/experiments.md` trainability interpretation | Blog or registry promotes toy tasks as language evidence. |

### C4: TreeHeap Existence Bias Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C4-001 | SPR-017 | TreeHeap's existence claim is not "it can also learn"; it is a structural inductive-bias claim. | design | Blog SPR-017 | TreeHeap only matches generic MLP/Transformer without structural advantage. |
| C4-002 | SPR-017 | Addressable closure / mod-fold tasks should test whether TreeHeap preserves address and overwrite semantics under length extrapolation. | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST01 | Flatten baselines equal or beat TreeHeap on extrapolated address tasks. |
| C4-003 | SPR-017 | Subheap kernel relocation should show local pattern transfer across unseen tree addresses. | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST02 | Kernel only works at trained positions or has high false positives. |
| C4-004 | SPR-017 | Prefix compression and delayed collapse should show reuse of shared prefixes and calibrated candidates. | open | `ara/m0-treeheap-math/logic/predicts.md` P-EXIST03 | Prefix tree gives no compression or probability container calibration. |
| C4-005 | SPR-018 | Pattern matching alone is the wrong B experiment; real TreeHeap proof requires learned encoder plus paired query/decoder kernel. | design | Blog SPR-018 | Hand-designed pattern matching remains the only working mechanism. |
| C4-006 | SPR-018 | Learned ordered TreeHeap search should encode key/value data into a searchable tree and use a query kernel to decode. | open | Blog SPR-018 planned experiment | Learned encoder cannot form searchable structure beyond trivial memorization. |
| C4-007 | SPR-018 | Learned weighted prefix TreeHeap should resemble Huffman-like compression for skewed symbol distributions. | open | Blog SPR-018 planned experiment | Expected path length fails to beat fixed-length baseline. |

### C5: Soft TreeHeap / Gradient Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C5-001 | SPR-019 | TreeHeap needs differentiable operator lifting before it can be trained like MLP / Transformer. | design | Blog SPR-019 | Useful TreeHeap training succeeds with only hard non-differentiable operators. |
| C5-002 | SPR-019 | Soft TreeHeap should be a probability lifting of Hard TreeHeap operators: `SoftO(H)=sum_a p(a) O_a(H)`. | mathematical design | Blog SPR-019 | One-hot soft operation fails to recover hard operation. |
| C5-003 | SPR-019 | Naive soft memory write is insufficient as a TreeHeap claim because it updates array slots, not the plus algebra. | design / critique | Blog SPR-019 | Slot interpolation is shown equivalent to TreeHeap plus under a formal mapping. |
| C5-004 | SPR-019 | Soft Plus should lift TreeHeap plus: `H_next=sum_a p(a|H,x) (H ⊕_a x)`. | supported pilot | `ara/m0-treeheap-math/evidence/soft_plus_probe/` | Gradient cannot flow through soft plus or collapse gives illegal TreeHeap. |
| C5-005 | SPR-019, SPR-020 | Kernel-guided Soft Plus should use a TreeHeap convolution kernel to decide write/merge route. | open / scoped pilot evidence | `ara/m0-treeheap-math/evidence/soft_plus_probe/`: `pilot_pass=true`; GLM audit shows current collapse depends on engineered alignment features | Kernel-guided plus is worse than naive memory write or encoder soft plus in clean-feature ablation. |
| C5-006 | SPR-019 | Multi-kernel / staged training is preferable to a single "big pot" loss. | open | Blog SPR-019 Experiment 3 loss ablation | Single total loss is more stable and generalizes better under matched budget. |
| C5-007 | SPR-019 | Soft collapse should recover legal Hard TreeHeap structure. | supported pilot | `ara/m0-treeheap-math/evidence/soft_plus_probe/`: `collapse_accuracy_tau_0.05=1.0` | Collapse legality, route interpretability, or hard-soft gap fails in larger/noisy tests. |
| C5-008 | SPR-021 | C05 must expose TreeHeap structure: path, subheap, and recursive route/plus; otherwise it degenerates into flat soft memory. | supported pilot | `ara/m0-treeheap-math/evidence/structural_c05_probe/`: flat/path-only test acc 0.0; subheap/path+subheap test acc 1.0 | Flat address or path-only baselines match subheap/path-subheap kernels on unseen-depth relocation. |

### C6: Post-M0 S1 Encoder, Kernel, and Fold Claims

These claims are registered in `ara/s1-echo/logic/claims.md`. They are listed
here because SPR-026..SPR-038 moved S1 beyond the original hash/capacity stage.

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C6-001 | SPR-026 | A learnable shallow TreeHeap write can encode real short sentences into queryable `root/subject/object` slots, including OOD lexical copy-by-address. | supported pilot | `ara/s1-echo/evidence/shallow_treeheap_s1_probe/`; TreeHeap train/test/OOD exact 1.0/1.0/1.0, flat baselines OOD exact 0.0 | Matched copy-capable flat or sequence baseline matches OOD copy and slot query accuracy. |
| C6-002 | SPR-028 | Frozen external embeddings are not yet a valid TreeHeap world-coordinate proof. | rejected pilot | `ara/s1-echo/evidence/s1_world_model_compound_probe/`; vector_add OOD cosine/top1 0.7198/0.833 vs TreeHeap 0.3919/0.0 | Reopen only if constrained TreeHeap encoder beats vector_add and copy/concat baselines on held-out compounds. |
| C6-003 | SPR-029 | When coordinates are trained from local co-occurrence, a structured TreeHeap kernel can beat vector_add on OOD cosine and approach concat MLP. | supported pilot | `ara/s1-echo/evidence/s1_corpus_embedding_kernel_probe/`; OOD cosine vector_add 0.5785, TreeHeap 0.7126, concat MLP 0.7321 | Vector_add or matched simple baselines beat TreeHeap across seeds/corpus variants, or top1/cosine collapses. |
| C6-004 | SPR-030 | A structured TreeHeap kernel can write/read real WMT SentencePiece short sequences in echo mode with far fewer parameters than flat sequence MLP. | supported pilot | `ara/s1-echo/evidence/s1_wmt_echo_kernel_probe/`; TreeHeap OOD token/exact 0.9818/0.9000 with 423104 params vs seq MLP 0.5986/0.0533 with 16794112 params | Copy-capable or larger matched baselines match OOD exact under similar parameter/sample budgets. |
| C6-005 | SPR-031 | Structural perturbation can induce multi-kernel differentiation, but current accuracy is too low to claim solved specialization. | open / mixed pilot | `ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe*/`; all kernels used, max ablation drops up to 0.3050, mean exact remains low | Gate specialization depends only on task labels, accuracy stays low, or matched single/flat/Transformer baselines match. |
| C6-006 | SPR-032 | Read should be query-conditioned probabilistic collapse over `stop/left/right` from `arr[1]`; internal `stop` is meaningful. | open / mixed pilot | `ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe*/`; route acc 1.0, leaf acc 0.9989, internal hard acc up to 0.4332 | Support requires route, leaf read, and internal subheap read all pass under matched baselines. |
| C6-007 | SPR-034 | Internal-node readout should target algebraically natural attributes before arbitrary checksum labels. | supported pilot | `ara/s1-echo/evidence/s1_algebraic_readout_probe*/`; routed internal readout beats root bottleneck on length/first/last/prefix targets | Root-only or matched flat/pointer baseline matches routed internal readout under same budget. |
| C6-008 | SPR-035 | Natural internal readout requires an order-preserving TreeHeap fold before bag/mod/cyclic folding. | supported pilot | `ara/s1-echo/evidence/s1_ordered_fold_kernel_probe/`; ordered fold exact 1.0 vs bag/modulo exact 0.0888 | Non-addressed bag/global summary or early modulo fold preserves the same subheap readout. |
| C6-009 | SPR-036 | Language fold should first be modeled as latent token/phrase placement and attraction; TreeHeap is a coordinate and partition system over that field. | open theory | `ara/s1-echo/logic/latent_plane_fold.md`; blog SPR-036 | Relation-layout probes show no stable neighborhoods or TreeHeap partitions do not improve over random/linear partitions. |
| C6-010 | SPR-037 | Kernel controls over a latent relation field should form a measurable fold-quality control surface. | supported pilot | `ara/s1-echo/evidence/s1_controllable_manifold_probe/`; low-control F1 0.0828, best F1 0.8148, diagonal gain 0.7319 | Sweep is non-controllable, order/random controls match relation controls, or the surface disappears on real relation fields. |
| C6-011 | SPR-038 | A differentiable energy over current TreeHeap state can relax `arr[i]` while kernel parameters and address rules remain fixed. | supported pilot | `ara/s1-echo/evidence/s1_heap_state_relaxation_probe/`; scalar energy ratio 2.47e-31, vector mean energy ratio 1.24e-13 | State-only updates cannot lower energy, require hidden theta updates, or use target heap labels in the loss. |
| C6-012 | post SPR-038 | Explicit hard echo encoder/decoder closes the WMT short-BPE interface: ordered leaf write, internal summary compose, path leaf/subheap read, and full sequence decode. | supported pilot | `ara/s1-echo/evidence/s1_echo_encoder_decoder_probe*/`; main and expanded runs reach 1.0 sequence/leaf/subheap/summary metrics | Decoder uses target heap labels, hard interface fails under learned kernels, or non-empty/noisy subheap metrics collapse. |
| C6-013 | SPR-039 | A parameter TreeHeap `Theta` can learn local subheap convolution kernels by gradient, distinct from SPR-038 state relaxation. | supported pilot | `ara/s1-echo/evidence/s1_kernel_parameter_learning_probe/`; hidden `[1,1,1]`, learned theta `[1.0,1.0,1.0]`, theta L2 error `5.44e-16`, test/OOD MSE `8.78e-31/8.93e-30`, wrong-address test MSE `5.9285` | `Theta` fails to recover hidden kernels beyond clean scalar toy, wrong-address baselines match it, or only heap state `H` moves while parameters stay fixed. |
| C6-014 | SPR-040 | TreeHeap local convolution is equivariant under mirror / chiral flip; geometric left/right mirror is implemented as algebraic permutation of heap addresses and kernel slots, and scalar loss can learn the mirrored structural slot assignment. | supported pilot / medium reliability | `ara/s1-echo/evidence/s1_mirror_kernel_symmetry_probe/`; flipped-kernel test max error `8.88e-16`, unflipped mean error `6.4372`, learned mirrored theta `[0.5,-0.75,1.25]`, left/right assignment errors `2.22e-16/4.44e-16`; Runner audit: structure-assignment level only | Mirror equivariance fails for deeper/vector kernels, unflipped kernels work equally well, learned mirrored kernels do not recover `[root,right,left]`, or later claims promote this into rotation/full-3D-fold/learned-trigger without new evidence. |
| C6-015 | SPR-041 | Corrected S1 entry requires inverse-gate canonicalization before shared decoding, but the recorded learned inverse route is a flat matrix, not recursive TreeHeap routing. | supported numeric / mechanism-limited | `ara/s1-echo/evidence/s1_echo_inverse_gate_probe/`; OOD exact `1.0000`, inverse route argmax `1.0000`, identity inverse gate `0.999794`, mirror inverse gate `0.999784`, canonical state MSE `0.000988962`, no-inverse baseline OOD exact `0.218750`. 2026-07-05 audit: route is `L x L` matrix. | No-inverse or matched flat baselines solve the same canonicalization task, canonical-state loss is necessary but unjustified, or this proof is promoted into TreeHeap routing / natural-language trigger discovery / translation. |
| C6-016 | SPR-043 | Real-sentence S1 echo with same-algebra TreeHeap perturbation has valid numeric recovery evidence, but its learned inverse route is a flat matrix. | supported numeric / mechanism-limited | Full-root evidence: `ara/s1-echo/evidence/s1_sentence_flip_echo_probe/`; TreeHeap `Flip(root, full_depth)` hard closure exact `1.0000`; learned inverse OOD exact/token/edit `0.9645/0.9978/0.9979`. Local evidence: `ara/s1-echo/evidence/s1_local_flip_echo_probe/`; learned OOD exact/token/edit `1.0000/1.0000/1.0000`. 2026-07-05 audit: learned inverse is length/span-conditioned `L x L` route matrix. | Hard closure fails, perturbation is external array reverse/slicing rather than TreeHeap flip, examples are absent, no-inverse/matched baselines catch up, or future writing claims this already proves recursive path routing. |
| C6-017 | SPR-044 | S1 WMT canonical echo has weak positive TreeHeap-style compose evidence, but it is not recursive path-route evidence. | weak positive / small TreeHeap-style advantage | `ara/s1-echo/evidence/s1_wmt_canonical_echo_probe/`; TreeHeap-style compose OOD margin `0.6472`, retrieval@1/@5 `0.6300/0.8195`, entropy `4.0443`; BoW margin `0.5939`, retrieval@1/@5 `0.6285/0.8085`, entropy `4.3442`. Full streaming queue is stability evidence only; TreeHeap-style last loss `0.6209` trails LSTM `0.5158`. | BoW or matched sequence/Transformer baselines match or beat TreeHeap-style compose; result is promoted into BLEU, semantic grounding, path-route learning, or S2 decoding evidence. |
| C6-018 | SPR-045 | S1 mirror recovery can be implemented as recursive TreeHeap routing with actual `stop/left/right` action traces, distinct from a flat `L x L` route matrix. | supported pilot | `ara/s1-echo/evidence/s1_recursive_treeheap_route_probe/`; WMT-massive English side, 20,000 samples, heap max_len 32, train lengths 3..24, OOD lengths 25..32. Hard TreeHeap oracle OOD exact `1.0000`; learned recursive route OOD exact/token `1.0000/1.0000`; old length-indexed flat matrix OOD exact/token `0.0000/0.0097`; pilot_pass=true. | Flat shared route or matched pointer baseline solves the same unseen-length route task, action traces are absent, route kernel stops using heap addresses, or target-position supervision is promoted into natural-language trigger discovery / translation. |

### C7: S3 Real-Text Generation Claims

| ID | Source | Claim | Status | Evidence / Pointer | Falsification |
|---|---|---|---|---|---|
| C7-001 | SPR-052 | Under a matched `K=4` decoder-memory budget, a learned TreeHeap frontier should beat fixed/random tree frontiers and flat four-vector compression on WMT; route replacement should causally damage translation. | main claim not supported / weak route signal | `ara/s3-generation/evidence/s3_wmt_frontier_smoke/`; learned/fixed/random/flat NLL `6.5988/6.6012/6.7020/6.5080`; same-checkpoint fixed/random route replacement delta `+0.0097/+0.0285`, below registered `0.05` gate | Reopen only after value-preserving compose redesign beats flat `K=4`, fixed, and random controls across preregistered seeds. |
| C7-002 | SPR-054 | A frozen TreeHeap root compressor plus one bounded addressed detail code per internal node forms a progressive multiresolution codec without changing the root predictor. | supported mechanism / three-seed 1M-block proof | `ara/s3-generation/evidence/s3_multiresolution_treeheap_pyramid/main_v2/`; MSE `1.9334 -> 0.5526`, k=64 token top-1 `0.9964`, sequence exact `0.8268`, address-shift MSE `3.3147`, root NLL delta `0`, flat/Haar k=64 MSE `1.6042/1.2881` | Keep bounded until random-pair training, per-level ablation, quantization, and stronger equal-rate baselines are tested. |
| C7-003 | SPR-055 pending | Raw-token echo loss can jointly train shared recursive WRITE/FOLD/DETAIL/UNFOLD/READ kernels into an address-sensitive continuous protocol, but does not by itself force a useful global root or finite compression. | partial support / one-seed mechanism result | `ara/s3-generation/evidence/s3_treeheap_native_codec/`; continuous v1/v2 top-1 `0.9955/0.9915`, detail-shift `0.0024/0.0014`, root-zero `0.9956/0.9889`; 632-bit v3 top-1 `0.2073` | Reject root-plus interpretation when root ablation is neutral. Reopen finite-code compression only with explicit quantized rate-distortion; test root utility on prediction of missing information. |
| C7-004 | post SPR-057 | A learned WMT TreeHeap frontier should contain functionally equivalent subheap groups whose same-group exchange is less damaging than a distance-matched different-group exchange. | not supported at current checkpoint | `ara/s3-generation/evidence/s2_treeheap_functional_equivalence/`; learned same/different NLL delta `0.01865/0.02053`, gap `0.00187`, bootstrap 95% `[-0.00395, 0.00745]`; cosine-distance match error `0.000052`, coverage `500/500` | Reopen only with a training objective that directly rewards reusable subheap behavior and preregistered held-out causal exchange. Geometric clusters alone are not semantic evidence. |
| C7-005 | post SPR-057 | Translation-only complete-subheap descendant masking should move useful S2 information upward and outperform plain and equal-count random-mask TreeHeaps. | main claim not supported / weak robustness trend | `ara/s3-generation/evidence/s2_treeheap_emergent_mask_translation/`; plain/structured/random clean NLL `6.5251/6.5062/6.5033`, cut NLL `6.6989/6.6047/6.6201`; structured selected-root contribution gain `-0.00073`; only finite/non-empty gate passed | Reopen only with root-exclusive span readout and a preregistered structured-over-random gain across seeds. Current result supports generic masking regularization, not upward information concentration or S2 superiority. |
| C7-006 | post SPR-057 | A root-exclusive variable-resolution frontier should close decoder bypasses and make recursively composed TreeHeap roots both causal and better for S2 than full memory or equal-frontier flat span means. | partial mechanism support / S2 gain rejected | `ara/s3-generation/evidence/s2_treeheap_root_exclusive_decoder/`; full/tree/flat native NLL `6.5038/6.6072/6.5770`; exclusive root-removal damage `0.10032` vs full-training forced-frontier damage `0.01877`; frontier `0.5126` states/token | Root causality passed, quality did not. Reopen S2 gain only after a value-preserving residual FOLD beats flat span means under the same root-exclusive decoder and bandwidth. |
| C7-007 | post SPR-057 | Complete-input echo from non-leaf TreeHeap states should establish a private multiresolution codec only if parent/root/address interventions show causal use beyond a leaf/string bypass. | partial support / one-level parent codec supported / multiresolution rejected | `ara/s1-echo/evidence/s1_multiresolution_internal_echo/main/`; equal-parameter mean/single/multichannel NLL `0.616107/0.004979/0.002010`; multichannel top-1/exact `0.999428/0.991211`; closest-parent/address damage `+74.4254/+54.3687` NLL; root and all higher levels neutral | The experiment proves shared FOLD can encode adjacent pairs into parent-only states without leaf access. It does not prove multiresolution storage: READ placed `99.9786%` mass on level 1. Reopen with a task that disjoint pair codes cannot solve and an independent higher-level causal gate. |
| C7-008 | post SPR-057 | A lifting FOLD/UNFOLD protocol should act as an information pump: only parents recurse upward, addressed details remain local, root-first UNFOLD closes exactly, and natural next-token loss trains the shared predictor. | registered full claim not supported / algebraic pump and inductive root learning supported | `ara/s1-echo/evidence/s1_lifting_information_pump/main/`; depth-6 closure max `3.70e-6`, state MSE `3.14e-14`, native token/block echo `1.0/1.0`; learned/frozen NLL `8.03468/8.06377`; root-shuffle next NLL `+0.21359`; all four pairing depths causal; root-zero block exact `0`, but token drop only `0.06958` | P3 required root echo token drop `>=0.10` and failed. Retain the narrower trainable-pump result; do not infer semantic hierarchy from a private invertible code. Reopen language-level claims only with downstream scale-specific readout and matched sequence baselines. |
| C7-009 | post SPR-075 | A frozen source encoder plus one-shot target-H synthesis and algebraic UNFOLD should retain source/address/detail causality and reduce cyclic generation. | strong claim rejected / UNFOLD mechanism feasible | `ara/s3-generation/evidence/s3_stone1_c12_hstate_unfold/`; UNFOLD/GRU NLL `7.8480/7.4115`, source shuffle `+0.0010`, address `+0.0002`, distinct-2/4 `0.0079/0.0080`, repeat run 128 | Do not blame the already-closed inverse algebra or scale the one-shot MLP. Reopen only with causal target-state migration. |
| C7-010 | post SPR-075 | Joint training and repeated multi-head source-subheap communication should allow a task-private TreeHeap protocol to emerge without internal labels. | partial content/round protocol / address, head and generation gates failed | `ara/s3-generation/evidence/s3_stone1_c13_emergent_protocol/`; NLL `10.3917 -> 7.9772`, source shuffle/empty/last-round `+0.0627/+0.2204/+0.0115`, three detail depths causal; address `+0.0018`, only one head causal, output 128 commas | Retain source-content, iteration and detail evidence only. Reopen full private-protocol support with collective-head/root-only controls, address causality and noncollapsed generation. |
| C7-011 | post SPR-075 | A target TreeHeap grown token by token with incremental path FOLD can carry autoregressive history without a GRU and drive recursive source reads. | partial target-state mechanism support / generation rejected | `ara/s3-generation/evidence/s3_stone1_c14_target_tree_autoregressive/`; NLL `9.6855 -> 6.9601`, zero-history `+4.3857`, target-root-only `+0.7718`, source-root-only `+0.5325`, closure MSE `1.08e-14`; source shuffle only `+0.0979`, route collapsed to deepest depth, BLEU-4 `0.2150` | Retain causal target TreeHeap state. Reject usable translation and variable-depth source protocol; require best-checkpoint selection and stronger source/diversity evidence before scaling. |

## SPR Blog Source Map

| SPR | File | Main role in ARA |
|---|---|---|
| 001 | `blogs/.../spr/001-problem.md` | Problem definition and ARA rule: every claim needs evidence / falsification |
| 002 | `blogs/.../spr/002-s1-evidence.md` | S1 echo capacity and order evidence |
| 003 | `blogs/.../spr/003-s1-falsification.md` | Token-only semantic route rejected |
| 004 | `blogs/.../spr/004-architecture-decision.md` | SPR layer split and S1/S2 architecture framing |
| 005 | `blogs/.../spr/005-s2-fold-stack.md` | Fold stack and S2 direction |
| 006 | `blogs/.../spr/006-next-experiments.md` | Baseline battle and claim decision workflow |
| 007 | `blogs/.../spr/007-context-proof.md` | Controlled context routing proof |
| 008 | `blogs/.../spr/008-s2-strategy-audit.md` | Strategy audit, probability container, downgrade of historical syntax claims |
| 009 | `blogs/.../spr/009-world-model-frames.md` | Terminology: world model / background field |
| 010 | `blogs/.../spr/010-world-model-night-run.md` | Night-run diagnostic; no positive WMT claim |
| 011 | `blogs/.../spr/011-treeheap-algebra.md` | TreeHeap algebra design before language claims |
| 012 | `blogs/.../spr/012-subheap-kernel-search.md` | Subheap kernel / convolution-like reasoning proposal |
| 013 | `blogs/.../spr/013-treeheap-math-probe.md` | M0 pure math probe |
| 014 | `blogs/.../spr/014-primitive-plus-order.md` | Primitive + plus as order source |
| 015 | `blogs/.../spr/015-primitive-plus-probe.md` | Toy proof for addressable plus / mod fold |
| 016 | `blogs/.../spr/016-trainability-quiz.md` | ML entrance quiz: linear, XOR, modular addition |
| 017 | `blogs/.../spr/017-treeheap-existence-proofs.md` | Existence proof claims A/B/C |
| 018 | `blogs/.../spr/018-treeheap-structural-inductive-bias.md` | Learned encoding and paired query/decoder kernel refinement |
| 019 | `blogs/.../spr/019-soft-treeheap-gradient.md` | Soft algebra, kernel-guided soft plus, multi-kernel training |
| 020 | `blogs/.../spr/020-soft-treeheap-audit.md` | GLM audit, ARA scope repair, clean-kernel next proof |
| 021 | `blogs/.../spr/021-c05-structural-proof.md` | C05 structural proof: path, subheap, recursive plus |
| 022 | `blogs/.../spr/022-treeheap-math-foundation.md` | Math background: rooted-tree algebra, operad/BCK positioning |
| 023 | `blogs/.../spr/023-treeheap-kernel-convolution-ops.md` | TreeHeap operations as convolution-kernel combinations |
| 024 | `blogs/.../spr/024-probabilistic-kernel-kl-world-model.md` | Probabilistic kernels, KL-style world-model fitting |
| 025 | `blogs/.../spr/025-deductive-inductive-kernel-proof.md` | Split deductive operator proof from inductive probability learning |
| 026 | `blogs/.../spr/026-s1-shallow-treeheap-write.md` | Shallow real-sentence TreeHeap slot write pilot |
| 027 | `blogs/.../spr/027-treeheap-diff-algebra.md` | TreeHeap diff/distance and finite-difference learning signal |
| 028 | `blogs/.../spr/028-s1-world-coordinate-negative.md` | Frozen embedding world-coordinate negative result |
| 029 | `blogs/.../spr/029-local-corpus-coordinate-kernel.md` | Local corpus coordinate system and structured kernel pilot |
| 030 | `blogs/.../spr/030-wmt-echo-kernel.md` | Real WMT SentencePiece echo kernel pilot |
| 031 | `blogs/.../spr/031-multikernel-specialization.md` | Multi-kernel specialization mixed pilot |
| 032 | `blogs/.../spr/032-probabilistic-read-kernel.md` | `stop/left/right` probabilistic read collapse |
| 033 | `blogs/.../spr/033-algebraic-decoders.md` | Algebraic decoders for finite-field TreeHeap state |
| 034 | `blogs/.../spr/034-algebraic-internal-readout.md` | Natural internal-node readout targets |
| 035 | `blogs/.../spr/035-ordered-fold-kernel.md` | Ordered fold before bag/mod/cyclic fold |
| 036 | `blogs/.../spr/036-latent-plane-fold.md` | Latent placement / relation-field fold theory |
| 037 | `blogs/.../spr/037-controllable-fold-manifold.md` | Controllable fold-quality surface pilot |
| 038 | `blogs/.../spr/038-heap-state-relaxation.md` | Heap-state relaxation and state-gradient pilot |
| 039 | `blogs/.../spr/039-parameter-treeheap-kernel-learning.md` | Planned parameter TreeHeap / local convolution kernel learning proof |
| 040 | `blogs/.../spr/040-mirror-kernel-symmetry.md` | Mirror / chiral TreeHeap kernel flip proof |
| 041 | `blogs/.../spr/041-s1-echo-entry-gate.md` | Controlled S1 echo entry gate: token write, structural route, token collapse |
| 043 | `blogs/.../spr/043-s1-sentence-flip-echo.md` | Sentence-level same-algebra TreeHeap flip echo with readable examples |
| 044 | `blogs/.../spr/044-s1-wmt-canonical-echo.md` | WMT canonical echo: bilingual canonical state plus same-language echo |
| 052 | `blogs/.../spr/052-treeheap-frontier-bottleneck.md` | Fixed-bandwidth WMT frontier result; main advantage claim not supported |
| 053 | `blogs/.../spr/053-treeheap-algebraic-operator-codec.md` | Learned short operator program plus fixed TreeHeap executor; address OOD positive and depth OOD failure |
| 054 | `blogs/.../spr/054-treeheap-multiresolution-pyramid.md` | Frozen-root addressed-detail codec and three-seed rate-distortion evidence |

The canonical local blog source is:

```text
../../blogs/www.grepcode.cn/src/spr/
../../blogs/www.lostmap.cn/src/spr/
```

The public blog URLs are:

```text
https://www.grepcode.cn/spr/
https://www.lostmap.cn/spr/
```

## Evidence Registry Pointers

| Evidence | Supports |
|---|---|
| `ara/s1-echo/evidence/README.md` | S1 capacity, token-only falsification, controlled context routing |
| `ara/s2-translation/evidence/strategy_audit/` | S2 graph-builder bottleneck, vector collapse, probability containers |
| `ara/s2-translation/evidence/tmerge_diagnostic/` | t_merge / background-field diagnostics |
| `ara/s2-translation/evidence/world_model_long_20260617_180554/` | World-model night-run diagnostic, not positive WMT claim |
| `ara/m0-treeheap-math/evidence/treeheap_math_probe/` | Minimal algebra probe |
| `ara/m0-treeheap-math/evidence/primitive_plus_probe/` | Addressable plus and mod-fold toy |
| `ara/m0-treeheap-math/evidence/trainability_quiz/` | Linear regression, XOR, modular addition learning checks |
| `ara/m0-treeheap-math/evidence/soft_plus_probe/` | Soft Plus gradient path, toy collapse, GLM feature-ablation audit |
| `ara/m0-treeheap-math/evidence/structural_c05_probe/` | C05 structural ablation: flat/path-only vs subheap/path-subheap relocation |
| `ara/m0-treeheap-math/evidence/kernel_convolution_ops_probe/` | Deterministic TreeHeap convolution operations: search, plus/write, mirror flip |
| `ara/m0-treeheap-math/evidence/deductive_inductive_kernel_probe/` | Deductive hard/soft operator equivalence plus inductive probability learning with KL metrics |
| `ara/m0-treeheap-math/evidence/treeheap_diff_algebra_probe/` | TreeHeap diff, norm, distance, finite-difference and theta-gradient checks |
| `ara/m0-treeheap-math/evidence/algebraic_decoder_probe/` | Finite-field algebraic decoders: path, subheap, mirror, residue, ordered leaves |
| `ara/s1-echo/evidence/shallow_treeheap_s1_probe/` | Shallow real-sentence slot write and OOD copy-by-address |
| `ara/s1-echo/evidence/s1_world_model_compound_probe/` | Frozen external embedding world-coordinate negative result |
| `ara/s1-echo/evidence/s1_corpus_embedding_kernel_probe/` | Local corpus co-occurrence coordinate and structured TreeHeap kernel |
| `ara/s1-echo/evidence/s1_wmt_echo_kernel_probe/` | Real WMT short-BPE echo kernel versus BoW/seq MLP baselines |
| `ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe*/` | Multi-kernel specialization and ablation mixed pilots |
| `ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe*/` | Probabilistic `stop/left/right` read collapse diagnostics |
| `ara/s1-echo/evidence/s1_algebraic_readout_probe*/` | Algebraic internal readout from routed node state |
| `ara/s1-echo/evidence/s1_ordered_fold_kernel_probe/` | Ordered fold versus bag/modulo fold controls |
| `ara/s1-echo/evidence/s1_controllable_manifold_probe/` | Controllable relation/order fold surface |
| `ara/s1-echo/evidence/s1_heap_state_relaxation_probe/` | Heap-state-only relaxation under fixed kernel/address rules |
| `ara/s1-echo/evidence/s1_echo_encoder_decoder_probe*/` | Hard WMT short-BPE echo encoder/decoder closure |
| `ara/s1-echo/evidence/s1_echo_entry_gate_probe/` | Misdirected first SPR-041 pilot: separate output read kernels, retained as a negative design lesson |
| `ara/s1-echo/evidence/s1_echo_inverse_gate_probe/` | Corrected SPR-041 pilot: inverse-gate canonicalization before shared echo decoding |
| `ara/s1-echo/evidence/s1_sentence_flip_echo_probe/` | Real-sentence same-algebra TreeHeap flip echo, length metrics, readable examples |
| `ara/s1-echo/evidence/s1_local_flip_echo_probe/` | Real-sentence local subheap same-algebra TreeHeap flip echo, span-position metrics, readable examples |
| `ara/s1-echo/evidence/s1_wmt_canonical_echo_probe/` | WMT canonical echo: parallel-pair contrast, entropy, retrieval, and echo metrics |
| `ara/s1-echo/evidence/s1_recursive_treeheap_route_probe/` | Recursive `stop/left/right` route proof versus flat length matrix |
| `ara/s1-echo/evidence/s1_content_treeheap_route_probe_20k_e5/` | Content-aware recursive TreeHeap route using dense subheap vocab-count summaries |
| `ara/s1-echo/evidence/s1_compact_content_treeheap_route_probe_20k*/` | Compact subheap embedding route attempts; memory reduction positive, OOD route exact still below gate |
| `ara/s1-echo/evidence/s1_semantic_prefix_compression_probe/` | Supervised toy semantic-prefix compression proof for deductive transfer to unseen leaves |
| `ara/s1-echo/logic/s1_encoder_world_observer.md` | Design note for TreeHeap encoder-as-world-observer and compressibility-driven prefix induction |
| `ara/s3-generation/evidence/s2_treeheap_functional_equivalence/` | Held-out distance-matched TreeHeap exchange audit; current WMT functional-group claim not supported |
| `ara/s3-generation/evidence/s2_treeheap_emergent_mask_translation/` | WMT translation-only structured subheap mask versus plain and equal-count random-mask TreeHeaps |
| `ara/s3-generation/evidence/s2_treeheap_root_exclusive_decoder/` | Root-exclusive WMT frontier: decoder bypass closed, root causality positive, S2 quality gain negative |

## Current Open Proof Queue

Highest priority open experiments:

1. `Learned Echo Encoder/Decoder`
   - Claim: C6-012.
   - Status: hard interface is closed exactly in `ara/s1-echo/evidence/s1_echo_encoder_decoder_probe*/`.
   - Update: C6-015 corrects the entry gate: mirror input must be inverted into canonical echo state before a shared decoder reads it. The older separate-read-kernel pilot is downgraded as misdirected.
   - Update: C6-016 adds readable sentence-level same-algebra flip echo over 20k real WMT English short sentences.
   - Update: C6-016 now includes local span subheap flip evidence. Given span start/length, local `Flip(span_root, full_depth)` restores OOD exactly; no-inverse baseline does not.
   - Update: C6-017 reframes S1 as WMT canonical echo. The first 50k-pair run is a weak positive for bilingual canonical alignment plus echo, with only a small TreeHeap advantage over BoW.
   - Update: C6-018/C6-019 repair the route mechanism boundary after audit: recursive route and dense content-aware route pass on WMT-massive OOD lengths, but compact subheap embeddings are still mixed because route-level exactness falls below the dense gate.
   - Update: C6-020 adds a supervised toy semantic-prefix proof: prefix class structure can support deductive transfer (`amoxicillin -> medicine -> consumable`, `eat` accepts `consumable`) where pair memorization fails. This is not natural-language semantic learning.
   - Update: C6-021 reframes the next blocker as encoder induction: `Encode_Theta(raw observations) -> H_tree` must learn placement/fold/prefix structure from compressibility signals, not receive semantic labels.
   - Next: run structured-vs-shuffled unlabeled corpus proof with echo, context prediction, InfoNCE, replacement consistency, and description-length pressure.
   - Pass gate: learned prefixes appear in structured data, disappear under shuffled controls, preserve echo, and beat flat/bag/pair baselines on held-out transfer.

1.5. `Parameter TreeHeap Kernel Learning`
   - Claim: C6-013.
   - Status: executed; supported pilot in `ara/s1-echo/evidence/s1_kernel_parameter_learning_probe/`.
   - Result: `Theta` moved and recovered hidden local convolution kernel `[1,1,1]`; wrong-address and flat-global matched-size baselines failed.
   - Next: extend from scalar shared kernel to vector/matrix kernels, noisy restore, and multi-kernel tasks.

2. `Real Relation Layout Probe`
   - Claim: C6-009, C6-010.
   - Status: latent-plane theory and toy controllable manifold exist.
   - Next: build real short-sentence relation fields for det-head, aux-verb, quant-head, modifier-head, and predicate-argument neighborhoods.
   - Pass gate: TreeHeap partition/kernel improves relation recovery over random/linear partitions.

3. `Probabilistic Internal Read`
   - Claim: C6-006, C6-007.
   - Status: route and leaf read pass; arbitrary internal checksum failed; algebraic readout is positive.
   - Next: evaluate internal read on natural attributes, non-empty subheaps, and longer variable-depth WMT slices.
   - Pass gate: routed node state continues to beat root/flat/pointer baselines.

4. `Multi-Kernel Specialization Retry`
   - Claim: C6-005.
   - Status: gates specialize but accuracy is weak.
   - Next: retry after path-conditioned read and better task heads.
   - Pass gate: task accuracy rises materially and ablation drops remain task-specific.

5. `Real Context S1b Baseline Battle`
   - Claim: C1-005, C1-006.
   - Status: controlled proof exists; real-corpus baseline battle remains open.
   - Pass gate: context-conditioned path features beat BoW, keyword, random-hash, nearest-neighbor, and token-frequency baselines.

## Current Downgraded or Rejected Claims

| Claim | Decision |
|---|---|
| Token-only routing encodes contextual semantics | rejected |
| Current 3-epoch TreeHeap vectors solve syntax energy | downgraded |
| Historical checkpoint proves TreeHeap syntax | downgraded |
| Naive soft memory write proves TreeHeap algebra is trainable | rejected as too weak |
| WMT / Transformer replacement claim | not allowed yet |
| Frozen pretrained embeddings prove TreeHeap world coordinates | rejected pilot |
| Multi-kernel specialization is solved | open / mixed only |
| Probabilistic internal subheap read is solved | open / mixed only |
| Hard echo encoder/decoder proves semantic learning | not allowed; interface proof only |

## Reviewer Notes

The research is currently not allowed to claim:

```text
TreeHeap beats Transformer on WMT.
TreeHeap has learned syntax from the current historical checkpoint.
Soft TreeHeap training is already proved.
Kernel-guided soft plus has learned clean routing from raw TreeHeap geometry.
Hard exact echo proves language semantics.
Multi-kernel gates already equal Transformer multi-head capability.
Heap-state relaxation already solves S1/S2 learning.
```

The research is allowed to claim:

```text
S1 has strong path-hash capacity.
Token-only semantic routing failed.
Context-conditioned routing works in a controlled proof.
S2 fold/action signals are probe-predictable.
Current graph builder needs probability containers and better allocation.
TreeHeap math/toy probes support a narrow algebraic toolbox direction.
Soft Plus has a working gradient-path toy proof.
Kernel-guided clean route learning remains open.
Subheap structure carries unseen-depth relocation in the C05 structural toy.
TreeHeap kernels can beat flat/BoW baselines on short WMT echo under current pilot constraints.
Probabilistic read can route and read leaves, while internal subheap read remains open.
Algebraic internal readout is a better target than arbitrary checksum.
Ordered fold is necessary before bag/mod/cyclic fold for natural subheap readout.
Heap-state relaxation is a valid state-gradient pilot, not a translation claim.
Supervised semantic-prefix structure can support toy deductive transfer where pair memorization fails.
The next S1 gate is encoder induction: TreeHeap prefixes must be learned from observation statistics, not supplied as labels.
```

## 2026-07 S2 Lifting-Pump Result

`S2-LIFT-WMT-C01` is supported as a mechanism claim on a `27K/2K/2K`
English-to-Chinese WMT run. Translation-only loss learned a root-first
probabilistic READ that used several TreeHeap resolutions. Recursive READ
reached test NLL `5.0903`, versus root-only `5.4337` and full UNFOLD `5.1342`.
Source, root, every detail depth, and every recursive pair depth were causal
under registered interventions; FOLD/UNFOLD closure MSE was `1.73e-14`.

The quality claim is not supported. Flat sequence attention remained better
at NLL `4.8103` and token BLEU-4 `3.169`, versus TreeHeap `2.528`. No
compression or compute advantage follows because the current implementation
materializes all resolutions before probabilistic READ. See
`s3-generation/logic/s2_lifting_pump_wmt.md` and
`s3-generation/evidence/s2_lifting_pump_wmt_full/`.

## 2026-07 Adaptive Lifting Result

`S2-ADAPTIVE-LIFT-WMT-C01` is partial. A 30K attribution experiment showed
that learned update improved old-pump NLL by `0.0892`, while alternating
orientation degraded it by `0.1251`; the combined kernel improved only
`0.0394` and failed its non-antagonism gate. The preregistered winner rule
therefore advanced learned update without alternation.

On a `200K/5K/5K` WMT-massive run, learned update improved old-pump NLL from
`4.6743` to `4.6335` and token BLEU-4 from `9.609` to `9.909`. It missed the
registered `0.05` gain gate by `0.0092`, but closed `30.8%` of the flat gap.
Closure MSE was `2.35e-14`; source, root, every detail depth, and five of six
pair depths were strongly causal. Flat sequence still led at NLL `4.5419` and
BLEU-4 `10.572`, so quality superiority remains rejected.

## 2026-07 Lifting Subheap Pretraining Result

`S3-LIFT-SUBHEAP-PRETRAIN-C01` is partial. On real unlabeled Chinese
news/wiki/web text, three identical 34.18M-parameter learned-update lifting
models received 5,000 updates of token-only, matched arbitrary-span, or
address-aligned subheap masking. The aligned model beat the matched span model
by `0.0364/0.0540` NLL on common width-4/8 held-out recovery. Width-8 free
generation was non-empty, had `0.8262` unique outputs, and used sample-specific
source state: source shuffle cost `0.1345` NLL. Every detail depth was causal.

The complete TreeHeap protocol claim failed. Root zero cost only `0.0042` NLL
and a pre-FOLD adjacent left/right swap cost only `0.0022`; exact width-8
recovery was zero. The evidence therefore supports a small aligned-curriculum
gain and a source/detail-dependent seq2seq pretraining mechanism, not causal
root/address organization or WMT transfer. See
`s3-generation/logic/lifting_subheap_pretraining.md` and
`s3-generation/evidence/s3_lifting_subheap_pretrain_5k/`.

## 2026-07 Annealed Contraction Proposal

`S3-ANNEAL-TOY-C01` and `S3-ANNEAL-REAL-C01` are preregistered. Houming818
proposed that TreeHeap should not learn by masking information before FOLD;
complete input should enter the tree, then progressively narrower frontiers
should retain the information that removes the most future-generation loss.
The first proof separates a data-defined predictive event core from independent
nuisance details without core labels in training. The second predicts the next
16 real-text tokens from a complete 64-token history while the readable
frontier anneals from 64 leaves toward one root. See
`s3-generation/logic/annealed_contraction_protocol.md`.

## 2026-07 Annealed Contraction Result

The overnight proof completed on io. The controlled toy supports the narrow
claim that future-only loss can select a predictive core into a root: two of
three seeds exceeded `0.99` exact, core shuffle cost more than `16` NLL, and
nuisance shuffle cost approximately zero. It did not establish a
TreeHeap-specific advantage because the registered mean/flat margin failed in
all seeds.

On real Chinese text, the annealed frontier curriculum improved root NLL over
uniform depth training by `0.0918`, kept root within `0.0603` NLL of leaves,
and produced an ordered seven-depth profile (Spearman `-0.75`). However,
source shuffle cost only `0.0852` and a pre-FOLD sibling swap had no cost.
The retained result is therefore multi-resolution predictive contraction, not
causal sample/address organization. Full records are in
`s3-generation/evidence/s3_annealed_contraction_toy/` and
`s3-generation/evidence/s3_annealed_frontier_pretrain/`; checkpoints are on
NAS with committed SHA-256 pointers.

## 2026-07 Post-FOLD Damage Repair Proposal

`S3-MULTIRES-REPAIR-C01` is preregistered. It changes the evaluation target
from one-shot ablation sensitivity to damage followed by repair. Complete real
text is encoded first; only then are addressed residuals erased. A frozen
annealed encoder/decoder evaluates zero fill, parent-only latent regression,
and a shared cross-scale repair kernel using parent, root, surviving sibling
details, depth, and path. The proof separates generic latent repair from
TreeHeap-specific address/context use and records latent, token-retrieval, and
future-generation recovery curves. See
`s3-generation/logic/multiresolution_damage_repair.md`.

## 2026-07 Post-FOLD Damage Repair Result

The frozen real-text proof found a stable but narrower repair mechanism.
Erasing addressed details reduced affected token retrieval from `1.0000` to
`0.5573`. A depth-shared parent-only kernel restored `0.9095`, recovered
`72.38%` of leaf-state error, and reduced future NLL from `15.5926` to `6.2226`
against clean `5.6895`. This supports learnable redundancy between lifting
parent `U` and residual `D` after complete-input FOLD. Adding root, neighbors,
and path underperformed parent-only (`69.94%` repair), and wrong addresses were
cheap, so address-conditioned repair is rejected. Evidence is under
`s3-generation/evidence/s3_multiresolution_damage_repair/`; the next gate must
erase both detail and its direct-parent information to test genuinely recursive
grandparent/sibling/path repair.

## 2026-07 Full-Corpus Repair-Aware Seq2Seq Proposal

`S3-FULL-REPAIR-SEQ2SEQ-C01` is preregistered as the first product-pressure
run built on the stable parent-detail repair result. It streams the complete
verified local io files for Chinese continuation, Web/百科 QA, BELLE instruction
following, and English-Chinese translation. The 34M annealed TreeHeap and
repair kernel are jointly trained on clean leaf, random-resolution, and
post-FOLD repair generation tasks. Training uses `/home/nio/datasets` only;
NAS is excluded from the hot path. The run records per-source coverage,
conditional-generation audits, repair tolerance, free examples, and exact
resume. See `s3-generation/logic/full_corpus_repair_seq2seq.md`.

## 2026-07 Private Protocol Battle Result

`S3-PRIVATE-PROTOCOL-BATTLE-C01` is partially supported.  On the registered
three-seed `30K/2K/2K` WMT run, all recursive heads trained, every four-head
ablation caused damage, source/root shuffle increased NLL by
`1.9532/2.1113`, and crossing independently trained encoders and decoders
increased NLL by `2.2507`--`4.2958`.  This is evidence for a structurally
causal, seed-private TreeHeap encoder/decoder protocol.

The competitive predictions failed.  Mean flat/h1/h2/h4 NLL was
`6.0401/6.1231/6.1341/6.1934`; four heads were worse than one head and the
matched flat baseline.  The result therefore does not support multi-head or
TreeHeap quality superiority.  Evidence is under
`s3-generation/evidence/s3_private_protocol_battle_full/`; see
`s3-generation/logic/private_protocol_battle.md`.

## 2026-07 Small Transformer Reality Check

`S3-PRIVATE-PROTOCOL-TF-C02` passed its narrow registered comparison.  On the
same frozen `30K/2K/2K` WMT split, a `27.28M` 2+2-layer Transformer reached
mean NLL `6.4423` with the identical old recipe and `6.5330` with a warmup,
dropout, and label-smoothing recipe.  TreeHeap h1 at `6.1231` beat both, while
the flat GRU remained best at `6.0401`.

This shows that TreeHeap is not merely worse than the tested small Transformer
at this data scale.  It does not establish a top-model result: TreeHeap was
substantially slower, the nominal standard Transformer recipe underperformed,
and its validation loss was still improving at the final epoch.  See
`s3-generation/logic/private_protocol_transformer_benchmark.md` and evidence
under `s3-generation/evidence/s3_private_protocol_transformer_benchmark_full/`.

## 2026-07 Controlled Data-Dose Result

`S3-PRIVATE-PROTOCOL-DATA-DOSE-C03` is supported as a single-seed pilot.  With
validation/test identity, initialization, optimizer, batch size, and 15,625
updates frozen, increasing unique training pairs from 30K through 100K and
300K to 1M reduced TreeHeap h1 test NLL monotonically from `6.2671` to
`5.1454`, `4.2558`, and `4.0198`.  The 30K-to-1M gain was `2.2473` and
Spearman dose/NLL correlation was `-1.0`, passing every preregistered gate.

This resolves the narrow question: SPR-064's 30K result was strongly limited
by training-data diversity.  It does not establish TreeHeap superiority.  Flat
GRU and small Transformer also improved monotonically and reached `3.9365`
and `3.9201` at 1M versus h1 `4.0198`.  The 30K arms peaked at step 1,000 and
then severely overfit, while the 1M h1 and Transformer arms were still best at
the final step.  The endpoint is therefore not a convergence or dataset-optimum
claim.  Noisy and sometimes mojibake/misaligned web pairs further limit product
interpretation.  See `s3-generation/logic/private_protocol_data_dose.md` and
`s3-generation/evidence/s3_private_protocol_data_dose_full/`.

## 2026-07 STONE-1 Translation Engineering Milestone

`S3-STONE1-PRIVATE-PROTOCOL-C01` is preregistered as the engineering extension
of the controlled data-dose result. It fixes the one-million-pair evaluation
platform and requires a non-teacher-forced English-to-Chinese CLI, mean NLL at
most `3.90`, BLEU-4 at least `13.5`, and three-seed stability. Product quality
alone is insufficient: a learned local TreeHeap direction kernel must beat
identity and frozen-random controls, and gate/address interventions must each
cause registered held-out damage while FOLD/UNFOLD remains closed.

The decoder cannot read the finest leaf level. Rotation is not supervised or
required; local mirror-like choices are audited only after final seq2seq
training. See `s3-generation/logic/stone1_private_protocol_translation.md`,
code `s3-generation/src/s3_stone1_private_protocol.py`, and CLI
`s3-generation/src/treeheap_cli.py`.

The formal 1M-pair, three-seed run completed in 7.57 hours and did not support
the registered STONE-1 recipe. Learned/identity/frozen-random mean NLL was
`4.2269/4.0719/4.1210`, and BLEU-4 was `9.7875/11.1735/10.7198`. Forcing the
learned checkpoint to identity slightly improved NLL (`-0.0118`), and forcing a
random orientation changed it by only `+0.0034`; the learned local direction
therefore did not become a useful private protocol. In contrast, swapping
left/right addresses caused `+1.4948` NLL damage, giving strong positive
evidence that the decoder uses TreeHeap address/order structure. All engineering
gates passed, while only 2/5 product-quality and 1/6 structural gates passed.
This rejects the hard straight-through direction-gate recipe, not TreeHeap as a
whole. Formal evidence is in
`s3-generation/evidence/s3_stone1_private_protocol/`.

STONE-1 remains incomplete. Its preregistered C02 iteration fixes handedness as
algebra rather than learning it. The canonical transform is
`detail=R-L`, `parent=0.4L+0.6R`, with exact inverse
`L=parent-0.6detail`, `R=parent+0.4detail`. A learned arm starts exactly at this
codec and may learn only continuous residuals in `P/U`; fixed algebraic and
frozen-random residual codecs are controls. The original product gates remain
unchanged, and new causal gates require learned `P/U` to beat both controls,
suffer damage when forced back to the algebraic codec, retain address
causality, and show positive held-out improvement as finer TreeHeap depths are
exposed. See `s3-generation/logic/stone1_canonical_codec.md`.

C02 completed in 7.26 hours. Learned/algebraic/frozen mean NLL was
`4.0538/4.1138/4.0910`; learned BLEU-4 was `11.2865` and NLL standard
deviation was `0.0914`. Product quality, stability, and frozen-control gates
failed, so C02 did not complete STONE-1. Structural evidence was positive:
forcing the learned checkpoint back to exact algebra damaged NLL by `1.0984`,
swapping addresses damaged it by `1.3545`, and opening all six levels improved
NLL monotonically from `4.6366` to `3.9876`. Formal evidence is under
`s3-generation/evidence/s3_stone1_canonical_codec/`.

`S3-STONE1-CAPACITY-RATE-DISTORTION-C03` rejected capacity scaling under the
registered contract. Across three seeds, 28M-long reached mean test NLL/BLEU
`3.7495/12.7444`, while 50M at the original update count reached
`4.1469/10.1225`; the larger model was also worse than frozen C02. The 50M
route stopped entirely at level zero: post-fold child-address swap and positive
depth growth had zero effect, although forcing the algebraic codec damaged NLL
by `1.1021`. The swap left the already-formed root unchanged, so this proves
that the decoder ignored unfolded children but does not determine whether
encoder left/right paths were compressed into root. Thus more updates improved
the seq2seq product signal, while more parameters did not improve the registered
quality/depth-read contract. STONE-1 remains incomplete and the conditional
91.93M stage is not authorized. A pre-fold subtree mirror is required for the
next encoder-path causality test.
See `s3-generation/logic/stone1_capacity_rate_distortion.md` and formal evidence
under `s3-generation/evidence/s3_stone1_capacity_rate_distortion/`.

`S3-STONE1-PROTOCOL-GROWTH-C04` supports path-sensitive root compression and
rejects the simple "recursive decoding only needs more updates" explanation
under one frozen 50.27M run. From 15,625 to 62,500 updates, validation NLL
improved `4.1879 -> 3.6613`, but non-root route mass, root-to-full gain, and
post-fold child-address damage all remained approximately zero. The decoder
therefore continued to stop at root. This was not a bag-like collapse:
mirroring left/right subheaps before root formation damaged final NLL at every
fold depth, with maximum damage `0.6747`, and forcing the algebraic codec
damaged NLL by `1.3456`. The observed mechanism is an ordered recursive encoder
compressed into root and a root-only surface decoder. Final test NLL/BLEU-4 was
`3.5818/12.5107`; exact recovery was `0.004`, so generation remains weak and
STONE-1 is incomplete. This single seed does not establish stable emergence or
the intended coarse-to-detail recursive decoder. See
`s3-generation/logic/stone1_protocol_growth_trajectory.md` and evidence under
`s3-generation/evidence/s3_stone1_protocol_growth_trajectory/`.

`S3-STONE1-FROZEN-PRESSURE-C05` supports a trainable forced recursive decoder
channel over the frozen C04 `H_state`. Two equally initialized decoder arms
trained for 15,625 updates while every encoder parameter remained frozen. The
leaf-pressure arm improved validation NLL `4.6301 -> 3.5496` and reached test
NLL/BLEU-4 `3.4636/13.9564`, better than the root control's
`3.5149/13.5999`. Its shared branch kernel received nonzero gradient in every
observed update, encoder checksums were unchanged, and shuffling frozen
intermediate details damaged NLL by as much as `0.6909`. Thus the previous
root-only behavior was not evidence that deeper `H_state` lacked usable
information; its decoder route was effectively closed. Because depth was
forced, the result does not establish spontaneous route selection or complete
STONE-1. See `s3-generation/logic/stone1_frozen_encoder_pressure_decoder.md`
and formal evidence under
`s3-generation/evidence/s3_stone1_frozen_encoder_pressure_decoder/`.

`S3-STONE1-DECODER-DEPTH-FLOOR-C06` supports bounded-pressure learnable depth
on one seed. It kept the
C04 encoder frozen and compares native sequential stopping, mandatory deepest
reading, and a learnable depth distribution with a fixed two-percent mass floor
at each of six visible levels. The floor is an architectural pressure supply:
it prevents gradient starvation while leaving 88 percent of route probability
learnable. Equal-update arms test whether this bounded pressure can remain
within `0.10` NLL of both controls and retain frozen-detail causality. Formal
native/forced-leaf/depth-floor test NLL was `3.5156/3.4636/3.4117`, and BLEU-4
was `13.4823/13.9564/14.4886`. All six gates passed. The result supports
learnable depth under a nonzero pressure floor, not spontaneous routing after
the floor is removed. See `s3-generation/logic/stone1_decoder_depth_floor.md`.

`S3-STONE1-DECODER-RANDOM-PULSE-C07` completed two 4K-data smoke runs and was
deferred before the one-million-pair formal run. Shared decoder-cell gradients
from all depth pairs were aligned rather than antagonistic (mean cosine about
`0.70-0.75`), but every temporary-depth schedule overfit the small stream. The
registered primary `random_32` arm had mean probe gain `-0.2858` and maximum
forgetting `0.4736`. This rejects accumulation at smoke scale but does not
decide the formal C07 claim.

`S3-STONE1-FIXED-ROOT-NOISE-REPAIR-C08` tested the fixed-coordinate issue
directly. One 64-leaf physical root was retained; the C04 encoder was frozen;
decoder arms trained on clean masked tails, visible repeated-EOS tails, or
visible deterministic-random tails over one million WMT pairs. Masked random
values were exactly invariant. EOS matched valid NLL improved
`5.8130 -> 3.5370` and beat random-tail training at `3.7162`; matched test NLL
was `3.4517` versus the clean arm's `3.4117`. However, the EOS-trained decoder
scored `3.8258` on clean input, `0.3256` behind the clean-trained decoder and
above the registered `0.15` retention limit. Five of six gates passed.
Repeated EOS is therefore supported as a learnable fixed-frame convention,
while protocol-independent clean-compatible repair is rejected. See
`s3-generation/logic/stone1_fixed_root_noise_repair.md` and evidence under
`s3-generation/evidence/s3_stone1_fixed_root_noise_repair/`.

`S3-STONE1-FROZEN-PLATFORM-REPLICATION-C09` completes STONE-1 against the
machine-readable `S3-STONE1-C09-PLATFORM-V1` contract. The contract freezes the
1M/2K/2K split and hashes, tokenizer, C04 checkpoint, 64-leaf TreeHeap,
320/512 dimensions, EOS-tail convention, two-percent depth floor, AdamW
settings, batch 64, and 15,625 updates. Seeds `71901/71902/71903` reached test
NLL `3.4546/3.4517/3.4510` and BLEU-4
`13.7066/13.8713/13.5945`. Mean NLL/BLEU-4 was `3.4524/13.7241`; NLL standard
deviation was `0.00157`. Generation was nonempty in every seed, maximum severe
repetition was `0.0215`, every branch observation received nonzero gradient,
encoder checksums stayed unchanged, and shuffling frozen details caused at
least `+0.5634` NLL damage. All registered product, TreeHeap/integrity, and
engineering gates passed.

This signs a reproducible TreeHeap translation PoC on one exact platform. It
does not establish state-of-the-art translation, full-corpus scaling, a
removable pressure floor, conversation, world knowledge, or general
intelligence. See `s3-generation/logic/stone1_c09_replication.md` and evidence
under `s3-generation/evidence/s3_stone1_c09_replication/`.

## 2026-07 Bounded Rotation Search Result

`M0-ROT-C01` is supported as a synthetic pilot. The registered construction
recursively applied `H_(n+1) = CAT(H_n, R_n(H_n))`, where every supplied
rotation was invertible and order preserving. Across 52,898 sampled or
exhaustive queries, deterministic retrieval, inverse recovery, and a shared
two-parameter route kernel all reached exact `1.0`; the learned kernel trained
at depths 1--4 and transferred through depth 16.

At depth 16 the lazy TreeHeap represented 2,031,616 logical values. Retrieval
used 20.1682 comparisons on average, versus 1,014,613.8684 for the unordered
equality-scan control. Explicit payload storage was 32,247.873 times the lazy
base-plus-descriptor storage. Permuting payload positions inside rotated slices
reduced the unchanged one-path kernel to `0.036` exact, showing that searchable
order, rather than copying alone, carried the result. Over-budget growth and
query requests returned `BUDGET_EXHAUSTED`.

The boundary is decisive: the rotation law was supplied, not learned; a fully
materialized sorted array has the same logarithmic comparison complexity; and
the proof says nothing about semantic, arbitrary-space, or cryptographic
search. See `m0-treeheap-math/logic/bounded_rotation_search.md` and
`m0-treeheap-math/evidence/bounded_rotation_search_probe/`.

The recursive orbit is explicitly rejected as the runtime architecture. The
active follow-up, `M0-ROT-C02`, fixes one node pool `H_C` and allows rotation
only as a reversible transformation of an existing subheap. Shared kernels may
grow their receptive field through a bounded number of full-tree passes, but
capacity and node count cannot change; exhaustion returns `UNRESOLVED`.

### Fixed-Capacity Private Protocol Carrier Pilot

The first `M0-ROT-C02` component probe used two hard six-operation encoder
programs over overlapping subheap mirrors in a fixed 127-node state. A decoder
with only six trainable logits learned each inverse program from echo MSE:
paired hard reconstruction was exact, cross-protocol MSE was `2.012999`, and
identity/one-bit interventions were damaged. State shape remained
`[batch,127,4]` throughout.

The preregistered universal order gate failed. Protocol A was genuinely
order-sensitive (`0.440443` wrong-order MSE), while protocol B decoded exactly
in forward or reverse order because its selected operator composition was
order-equivalent. The result supports rotation as a fixed-capacity private
protocol carrier, but corrects the recursion claim: order itself carries extra
information only for a noncommutative selected program. The encoder programs
were fixed, so natural emergence remains untested. See
`m0-treeheap-math/logic/fixed_capacity_rotation_protocol_probe.md`.

### Rotation Selection Without an Order Label

The next `M0-ROT-C02` probe removed order from the objective. A fixed
24-member population contained six exact tree automorphisms, six corrupted
automorphisms, and twelve random within-depth permutations. A shared local
decoder and global gate received only masked-state prediction MSE. Tree-edge
preservation was measured after training.

The first seed strongly selected exact operators but its IID control also
locked onto one exact candidate, exposing neutral gate/decoder co-adaptation.
An eight-seed preregistered audit resolved the confound. Structured branching
worlds selected exact operators in `8/8` runs; exact-group mass mean/min was
`0.999657/0.999215`, and edge-preservation/loss Pearson averaged `-0.997404`.
IID winners dispersed as `2 exact / 1 mild / 5 random`, exact mass averaged
`0.283272` near the population share `0.25`, and edge/loss Pearson was
`-0.018631`. Giving every candidate its exact inverse tied all echo losses at
zero.

Thus, arbitrary invertible private codes survive paired echo, while structured
local prediction selects tree-edge/prefix-preserving rotations in this
controlled world. Order was an observed survivor property, not a loss label.
The operators still came from a fixed bank, so learning new rotation formulas
and language-scale selection remain open. See
`m0-treeheap-math/logic/rotation_selection_multiseed.md`.

## Maintenance Rule

When a new SPR blog changes a claim, update this file in the same commit or add
a follow-up commit before running long experiments.

Every new claim should have:

```text
ID
source
claim sentence
status
evidence pointer
falsification condition
next experiment
```
