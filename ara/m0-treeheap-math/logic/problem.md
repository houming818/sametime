# M0 TreeHeap Math Problem

## Question

Can TreeHeap be defined first as a mathematical object with a small but useful
operator toolbox, before using it for echo, structure induction, or translation?

## Motivation

Direct WMT training is too large as an early signal. A bad BLEU score cannot
tell whether the failure came from:

```text
operator design
world model learning
loss design
decoder weakness
data scale
training instability
evaluation noise
```

M0 isolates the lowest layer:

```text
TreeHeap as algebraic object
```

The goal is not to prove language understanding. The goal is to test whether
TreeHeap supports closed, composable, approximately invertible, and locally
searchable operations on synthetic structures.

## Minimal Object

The pilot object is:

```text
H = (name, v, head_v, slot, q, children)
```

Where:

```text
name: symbolic identifier for synthetic tests
v: vector representation
head_v: root/head vector used for exact synthetic recomposition
slot: role or structural coordinate
q: probability mass
children: ordered child heaps
```

## Scope

Included:

```text
compose
decompose
transpose
inverse_transpose
project
unproject
energy
match_subheap
probability container
```

Excluded:

```text
tokens
syntax labels
WMT
BLEU
real checkpoints
```
