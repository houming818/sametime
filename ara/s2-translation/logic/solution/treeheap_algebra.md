# TreeHeap Algebra Draft

This is a mathematical design note, not an evidence claim.

The goal is to define TreeHeap as an algebraic substrate before using it for
language-level reasoning.

## Object

The pilot TreeHeap object is:

```text
H = (v, p, s, q)
```

Where:

```text
v: semantic/world vector
p: heap path or structural coordinate
s: latent slot distribution
q: probability mass / confidence
```

The object is not just an embedding. It is a structured state that can enter
TreeHeap operations.

## Closure Principle

For TreeHeap to be useful mathematically, operations should be approximately
closed:

```text
TreeHeap object op TreeHeap object -> TreeHeap object
```

For scalar readouts:

```text
energy(TreeHeap object) -> scalar
distance(TreeHeap object, TreeHeap object) -> scalar
```

Approximate closure is acceptable because learned operators and probabilistic
containers are expected. But the output must remain interpretable as a TreeHeap
object.

## Primitive Operations

### Compose

```text
compose(H1, H2, ..., Hn) -> H_parent
```

Meaning:

```text
combine child heap objects into a parent heap object
```

The operation must update:

```text
v: semantic/world state
p: structural coordinate
s: latent slot distribution
q: confidence/probability mass
```

### Decompose

```text
decompose(H_parent) -> Distribution[child-set]
```

This is the probabilistic inverse of compose.

It should not return one hard answer too early. It should return a probability
container:

```text
{
  child_set_1: 0.52
  child_set_2: 0.31
  child_set_3: 0.17
}
```

### Transpose

```text
transpose(edge(parent, child, role)) -> edge(child, parent, inverse_role)
```

This is a relation-direction operator.

Expected property:

```text
transpose(transpose(edge)) ~= edge
```

Transpose is important for translation because source and target languages may
realize the same relation in different directions.

### Project

```text
project(H, frame) -> H_frame
```

Meaning:

```text
interpret a heap object inside a local reference frame
```

Examples:

```text
sport frame
body-part frame
motion frame
container frame
cause-effect frame
```

### Unproject

```text
unproject(H_frame, frame) -> H
```

This is an approximate inverse of project.

Expected property:

```text
unproject(project(H, frame), frame) ~= H
```

### Normalize

```text
normalize(H) -> H
```

Normalize keeps objects inside a stable region:

```text
bounded norm
valid probability mass
valid slot distribution
canonical path coordinate
```

### Energy

```text
energy(H) -> scalar
```

Energy measures internal consistency.

Lower energy should mean:

```text
the object is a more valid TreeHeap state
```

Energy should be used for ranking, not as a direct claim of truth.

### SubHeap Kernel Match

```text
match_subheap(H, K) -> ProbabilityContainer[SubHeap]
```

This is the TreeHeap analogue of convolution or correlation.

In a matrix, a convolution kernel scans local neighborhoods. In TreeHeap, a
SubHeap kernel scans local heap neighborhoods.

The convolution analogy should first be understood as an address-space
operation, not only as a 2D image operation:

```text
linear order
+ local window
+ modular shift
+ repeated kernel action
```

For a sequence, the neighborhood can be:

```text
i - 1 mod n
i
i + 1 mod n
```

For a flattened matrix, the neighborhood can be generated from fixed offsets:

```text
index = row * width + col
left  = index - 1
right = index + 1
up    = index - width
down  = index + width
```

Therefore TreeHeap kernel matching also needs an explicit address algebra:

```text
path
parent(path)
left(path)
right(path)
sibling(path)
next_dfs(path) mod N
next_bfs(path) mod N
```

Without this address space, a kernel has no well-defined way to move.

The kernel is not a fixed 2D mask. It is a small structured pattern:

```text
K = {
  center: event/action
  slot_1: agent-like
  slot_2: object-like
  slot_3: location-like
}
```

The output is not a single hard position. It is a probability container:

```text
{
  subheap_12: 0.71
  subheap_4:  0.18
  subheap_29: 0.07
}
```

This operation is important because many reasoning tasks are local topology
search:

```text
find the event
find the agent/object relation
find the modifier scope
find the attachment site
```

Unlike ordinary sequence convolution, SubHeap kernel matching must be able to
handle:

```text
non-grid structure
latent slots
optional children
role swaps
active/passive direction changes
approximate matches
```

## Minimal Algebra Tests

### E-ALG01: Compose/Decompose Roundtrip

```text
children -> compose -> parent -> decompose -> children'
```

Measure:

```text
top-k recovery
roundtrip energy
gold-vs-shuffled margin
```

### E-ALG02: Double Transpose

```text
edge -> transpose -> transpose -> edge'
```

Measure:

```text
double transpose error
role inverse consistency
graph reconstruction change
```

### E-ALG03: Projection Roundtrip

```text
H -> project(frame) -> unproject(frame) -> H'
```

Measure:

```text
roundtrip similarity
frame selectivity
relation-anchor ranking
```

### E-ALG04: Closure Stress

```text
repeat compose/project/transpose/decompose operations
```

Measure:

```text
norm stability
cosine collapse
energy drift
probability entropy
```

### E-ALG05: Kernel Invariance

```text
same reasoning pattern, different surface order -> same kernel match
```

Measure:

```text
top-k match consistency
active/passive invariance
paraphrase invariance
```

### E-ALG06: Kernel Selectivity

```text
same tokens, wrong roles -> lower kernel score
```

Measure:

```text
gold-vs-role-swapped margin
kernel score calibration
```

### E-ALG07: Cross-Lingual Kernel Transfer

```text
source subheap kernel -> target equivalent subheap
```

Measure:

```text
aligned event recovery
target-side slot compatibility
shuffled alignment baseline
```

## Engineering Decision

Do not ask whether TreeHeap can solve WMT before the algebra passes a small
closure test.

Language tasks should be introduced after:

```text
compose/decompose works
transpose is stable
project/unproject is not just identity echo
energy ranks gold structures above shuffled structures
SubHeap kernels find topology rather than surface order
```
