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

## Current Open Proof Queue

Highest priority open experiments:

1. `Learned Echo Encoder/Decoder`
   - Claim: C6-012.
   - Status: hard interface is closed exactly in `ara/s1-echo/evidence/s1_echo_encoder_decoder_probe*/`.
   - Update: C6-015 corrects the entry gate: mirror input must be inverted into canonical echo state before a shared decoder reads it. The older separate-read-kernel pilot is downgraded as misdirected.
   - Update: C6-016 adds readable sentence-level same-algebra flip echo over 20k real WMT English short sentences.
   - Update: C6-016 now includes local span subheap flip evidence. Given span start/length, local `Flip(span_root, full_depth)` restores OOD exactly; no-inverse baseline does not.
   - Update: C6-017 reframes S1 as WMT canonical echo. The first 50k-pair run is a weak positive for bilingual canonical alignment plus echo, with only a small TreeHeap advantage over BoW.
   - Next: stronger sequence/Transformer baselines, larger candidate retrieval windows, and separating root canonical meaning from leaf echo memory.
   - Pass gate: non-empty subheap metrics, noisy/masked restore, and matched copy-capable baselines.

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
```

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
