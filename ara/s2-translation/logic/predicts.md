# S2 Predict Registry

This file is the ARA logic entry for predictions that must be tested before
they can become claims.

## P-FRAME01: World-model reference-frame probe

### Predict

If TreeHeap contains a useful world model, then a composite concept should not
only be close to its surface words. Its difference from a base concept should
point toward interpretable relation anchors in the right local frame.

Examples:

```text
football - ball -> foot / kick / field / goal
basketball - ball -> hand / throw / court / basket
baseball - ball -> bat / glove / pitch / base
tennis - ball -> racket / court / net / serve
```

### Why this is the right next test

The previous `t_merge` diagnostic showed that raw final TreeHeap vectors can
look collapsed because of a strong common direction, while centered geometry
and CMul/pre-merge geometry may still carry information. Therefore the next
test should compare multiple readout points:

```text
random
L0
path
CMul pre-merge
merge_no_bias
tree
centered_tree
```

The experiment should not treat a legacy checkpoint as final positive evidence.
For now it is a diagnostic artifact. A positive pilot only means the predict is
worth retesting on a newly trained checkpoint.

### Evidence gate

For a set of `(composite, base, positive anchors, negative anchors)` probes:

1. Compute `delta = vector(composite) - vector(base)`.
2. Compute each anchor direction as `vector(anchor) - vector(base)`.
3. Rank anchors by cosine similarity to `delta`.
4. Compare positive-anchor ranking against hard negative anchors.

Pass criteria for a pilot:

```text
best TreeHeap internal mode MRR > L0 MRR
best TreeHeap internal mode AUC > L0 AUC
best TreeHeap internal mode hit@3 > L0 hit@3
random baseline near chance
```

Fail criteria:

```text
L0 >= all TreeHeap modes
or random/path performs similarly to semantic modes
or centered_tree loses the signal that CMul had
```

### Planned implementation

```text
src/frame_probe.py
evidence/frame_probe_2h/
```

### Status

Executed as diagnostic evidence:

```text
evidence/world_model_long_20260617_180554
verdict: inconclusive
```

The local-context objective did not make P-FRAME01 stronger over long training.
This suggests the next step should test the algebraic basis before claiming a
world-model topology.

## P-ALG01: TreeHeap algebra closure

### Predict

If TreeHeap is a usable mathematical substrate rather than just a neural
feature extractor, then a small set of algebraic operations should be
approximately closed over TreeHeap objects.

In plain terms:

```text
TreeHeap object op TreeHeap object -> TreeHeap object
```

The first algebraic operations to test are:

```text
compose      children -> parent
decompose    parent -> candidate children container
transpose    edge(parent, child, role) -> edge(child, parent, inverse_role)
project      heap object -> frame coordinates
unproject    frame coordinates -> heap object
normalize    heap object -> canonical heap object
energy       heap object -> scalar consistency score
```

### Why this comes before language reasoning

Language reasoning is an upper-layer task. Before asking whether TreeHeap can
translate or reason, we need to know whether its objects support stable
mathematical manipulation.

The previous world-model night run showed:

```text
t_merge did not necessarily collapse the vector space.
local context training did not create a strong reference-frame topology.
```

Therefore the immediate question is not:

```text
Can TreeHeap translate WMT yet?
```

It is:

```text
Can TreeHeap define operations whose outputs remain interpretable TreeHeap objects?
```

### Minimum object definition

A pilot TreeHeap object can be represented as:

```text
H = (v, p, s, q)

v: semantic/world vector
p: heap path or structural coordinate
s: latent slot distribution
q: probability mass / confidence
```

This definition is intentionally small. If the algebra cannot be made stable
for this object, it is too early to claim larger language-level reasoning.

### Evidence gates

E-ALG01 compose/decompose consistency:

```text
children -> compose -> parent -> decompose -> children'
children' should recover the original children in top-k.
```

Pass criteria:

```text
top1 recovery > nearest/random baseline
top3 recovery >= 0.80 on controlled synthetic heaps
roundtrip energy lower for gold children than shuffled children
```

E-ALG02 transpose consistency:

```text
edge -> transpose -> transpose -> edge'
edge' should return to the original edge up to tolerance.
```

Pass criteria:

```text
double_transpose_error < single_transpose_distance
inverse role mapping improves graph reconstruction over no-transpose baseline
```

E-ALG03 projection roundtrip:

```text
H -> project(frame) -> unproject(frame) -> H'
```

Pass criteria:

```text
roundtrip cosine(H, H') > random projection baseline
frame-specific projection improves relation-anchor ranking
```

E-ALG04 closure stress test:

```text
compose(project(transpose(H))) should still be a valid TreeHeap object.
```

Pass criteria:

```text
no norm explosion
no global cosine collapse
energy remains bounded under repeated operations
```

### Fail criteria

P-ALG01 fails if:

```text
operations only preserve information but do not change relation geometry
or decompose cannot recover children above random
or transpose twice does not approximately return
or project/unproject is just identity/echo with no frame selectivity
```

### Planned implementation

```text
logic/solution/treeheap_algebra.md
src/treeheap_algebra_probe.py
evidence/treeheap_algebra_probe/
```

### Status

Design phase.

## P-ALG02: SubHeap kernel search

### Predict

If TreeHeap supports topology-level reasoning, it should support a convolution-
like operation over heap structure:

```text
match_subheap(H, K) -> ProbabilityContainer[SubHeap]
```

Where:

```text
H: full TreeHeap object or graph
K: local SubHeap kernel
```

This is analogous to image convolution, but the kernel is not a rectangular
pixel mask. It is a local heap pattern:

```text
K = event {
  center: action-like
  slot_1: agent-like
  slot_2: object-like
  slot_3: location-like
}
```

The operation should find matching local topology even when surface order
changes.

### Motivation

In a matrix, a kernel such as:

```text
1 0 1
0 1 0
1 0 1
```

can be slid over the whole matrix to find regions with the same local pattern.

The lower-level interpretation is:

```text
linear order + local window + modular shift + repeated kernel action
```

This means P-ALG02 also depends on a TreeHeap address algebra:

```text
path
parent(path)
left(path)
right(path)
sibling(path)
next_dfs(path) mod N
next_bfs(path) mod N
```

The first implementation should not jump directly to rich linguistic kernels.
It should first test whether cyclic/local neighborhood search over TreeHeap
addresses can reproduce simple convolution-like matching.

For TreeHeap, the analogous question is:

```text
Can a learned local heap kernel find equivalent reasoning structure inside a
larger heap?
```

Examples:

```text
cat eats fish
fish is eaten by the cat
the cat quickly eats fish
```

All three should activate an event kernel close to:

```text
eat_event(agent=cat, object=fish)
```

### Evidence gates

E-ALG05 kernel invariance:

```text
same kernel, different surface order -> same subheap match
```

Pass criteria:

```text
active/passive/paraphrase variants rank the same event subheap in top-k
kernel match beats token-order and nearest-neighbor baselines
```

E-ALG06 kernel selectivity:

```text
same words, wrong role assignment -> lower score
```

Pass criteria:

```text
score(agent=cat, object=fish) > score(agent=fish, object=cat)
```

E-ALG07 cross-lingual kernel transfer:

```text
source-language kernel match -> target-language equivalent subheap
```

Pass criteria:

```text
ZH/EN aligned event structures activate compatible kernels above shuffled
alignment baselines
```

### Fail criteria

P-ALG02 fails if:

```text
kernel matching only tracks token order
or only tracks lexical overlap
or active/passive variants do not share a high-scoring subheap
or role-swapped structures score the same as gold structures
```

### Planned implementation

```text
src/subheap_kernel_probe.py
evidence/subheap_kernel_probe/
```

### Status

Design phase.
