# S2 Experiment Registry

> Maps blog claims to executable experiments with evidence pointers.

## Phase A: Semantic→Fold Action

**Question**: Does the semantic vector encode structural decision information?

**Design**: EN sentence → sentence-transformer (384D) → PCA(128D) → MLP(128→256) → predict 47 fold action types (multi-label)

**Variants tested**:
- Multiple dimensions: 32, 64, 128, 256
- Cross-source validation (different WMT sources)
- Shuffle test (null hypothesis)

**Evidence**: `phase_a_results.json`, `falsification.py` output

**Verdict**: ✅ AUC=0.64-0.70, 32D≈128D

---

## Phase D1: Cross-lingual Structure Prediction

**Design 1** (ZH→EN): ZH semantic (384D) → MLP → EN fold actions
**Design 2** (EN→ZH): EN semantic (384D) → MLP → ZH fold actions + Chinese-specific constructions

**Evidence**: `phase_d1_results.json` (ZH→EN AUC=0.701), `e2a_results.json` (EN→ZH AUC=0.671)

**Verdict**: ✅ Cross-lingual structure prediction is possible

---

## Phase D2a: Structure Binding

**Design**: Token-level: for each token, predict (is_head, span_membership, action_type)

**Evidence**: `phase_d2a.py` output (Head F1=87.6%, Span F1=88.2%, Action Top-5=96.7%)

**Verdict**: ✅ Fold node discovery is reliable

---

## Phase D3: Oracle Ablation

**Design**: Oracle replace each module (Head/Span/Child) with gold → measure UAS gain

**Evidence**: `phase_d3_results.json` (Gold Child → +41% UAS, 100% of gap)

**Verdict**: ✅ Child assignment = sole bottleneck

---

## Graph Builder Methods (Benchmark)

**Design**: Compare 5 methods on same data: Oracle, Template, Template+Dist, Distance only, Left parent

**Evidence**: `graph_bench.py` output

**Verdict**: ✅ Distance only (56.5%) beats template (50.1%) and learned (48%)

---

## Error Atlas

**Design**: For all edges where nearest ≠ oracle, classify error type by dep label

**Evidence**: `diagnose_gap.py` output (PP_ATTACHMENT 35%, OBJECT 20%, SUBJECT 14%)

**Verdict**: ✅ PP attachment ambiguity dominates

---

## Probability Container (P3)

**Design**: Top-3 candidate graphs → evaluate whether scorer identifies best graph

**Evidence**: `p3_container.py` output (scorer correct 80%, but top-3 UAS = nearest)

**Verdict**: ❌ Container can't improve over nearest with current features

---

## Tensor PoC

**Design**: Non-commutative tensor → energy minimization over permutations

**Variants**: Sum outer product, position-weighted concat, centrality-weighted concat, pure path

**Evidence**: `tensor_poc.py`, `path_tensor.py` output

**Verdict**: △ Non-commutative works, but energy doesn't align with syntax with current vectors

---

## Strategy Audit: Tensor / Slot / Container Gates

**Design**: Run one queue over 12K WMT massive English lines, extracted SVO/SVOA cases, vector modes (`random`, `L0`, `TreeHeap`), role bases (`onehot`, `random`, `orthogonal`), tensor operators, fold-node degree patterns, and parent top-k container coverage.

**Evidence**: `ara/s2-translation/evidence/strategy_audit/strategy_audit_summary.json`

**Verdict**:

- Current TreeHeap 3-epoch vectors are highly collapsed (`cos_offdiag_mean=0.9849`) and do not beat L0/random in role-slot template ranking.
- Non-commutative tensor operators are permutation-sensitive, but raw energy does not align with gold syntax.
- Role-slot FoldNode design is supported: degree <= 4 covers 99.0% of fold nodes.
- Probability container is supported at the parent-candidate level: gold parent is in top-3 for 99.9% of evaluated child sets.
