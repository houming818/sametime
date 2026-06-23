# ARA Paper: Semantic Prefix Routing and TreeHeap

Status: living research artifact
Created: 2026-06-22
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
| C0-003 | SPR-019 | Blog narrative is not enough; claims must be preserved in registry artifacts. | open | This `PAPER.md` is the first root manifest | Future SPR claims diverge from registry and are not reconciled. |

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
| C4-005 | SPR-018 | Pattern matching alone is the wrong B experiment; real TreeHeap proof requires learned encoder plus conjugate query/decoder kernel. | design | Blog SPR-018 | Hand-designed pattern matching remains the only working mechanism. |
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
| 018 | `blogs/.../spr/018-treeheap-structural-inductive-bias.md` | Learned encoding and conjugate kernel refinement |
| 019 | `blogs/.../spr/019-soft-treeheap-gradient.md` | Soft algebra, kernel-guided soft plus, multi-kernel training |
| 020 | `blogs/.../spr/020-soft-treeheap-audit.md` | GLM audit, ARA scope repair, clean-kernel next proof |

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

## Current Open Proof Queue

Highest priority open experiments:

1. `Kernel-guided Soft Plus Autograd`
   - Claim: C5-004, C5-005
   - Status: executed pilot in `ara/m0-treeheap-math/evidence/soft_plus_probe/`.
   - Result: gradients reach `K_write` and plus parameters; low-temperature collapse is correct in the toy.
   - Boundary: current success depends on engineered alignment features.
   - Next: compare against naive memory write and encoder soft plus using clean features.

2. `Write Mechanism Ablation`
   - Claim: C5-003..C5-005
   - Compare:

```text
A: naive soft memory write
B: encoder soft plus
C: kernel-guided soft plus
```

3. `Hard/Soft Consistency`
   - Claim: C5-002, C5-007
   - Measure hard-soft output gap, collapse legality, route interpretability.

4. `Existence Proof Suite`
   - Claim: C4-002..C4-004
   - Measure address extrapolation, subheap relocation, prefix compression.

5. `Real Context S1b`
   - Claim: C1-005, C1-006
   - Controlled proof must face BoW, keyword, random-hash, and real-corpus baselines.

## Current Downgraded or Rejected Claims

| Claim | Decision |
|---|---|
| Token-only routing encodes contextual semantics | rejected |
| Current 3-epoch TreeHeap vectors solve syntax energy | downgraded |
| Historical checkpoint proves TreeHeap syntax | downgraded |
| Naive soft memory write proves TreeHeap algebra is trainable | rejected as too weak |
| WMT / Transformer replacement claim | not allowed yet |

## Reviewer Notes

The research is currently not allowed to claim:

```text
TreeHeap beats Transformer on WMT.
TreeHeap has learned syntax from the current historical checkpoint.
Soft TreeHeap training is already proved.
Kernel-guided soft plus has learned clean routing from raw TreeHeap geometry.
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
