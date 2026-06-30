# Latent Plane Fold Theory

Date: 2026-06-30
Author: Codex Review
Status: theory draft

## Motivation

SPR-035 proved only a narrow engineering fact:

```text
If we want to read natural internal subheap attributes, an ordered TreeHeap
fold must preserve leaf address, path, and subheap locality.
```

That is not yet a theory of language fold.

Houming818's correction is stronger:

```text
We should not invent a fold and force language into it.
We should observe language phenomena and discover the folding laws already
present in the world.
```

This document reframes TreeHeap fold as a latent placement problem.

## Core Idea

Language understanding may not begin as a symbolic table:

```text
subject = ...
verb = ...
object = ...
```

It may begin as a placement process:

```text
tokens are placed in a latent plane / field;
related tokens attract each other;
local clusters stabilize;
clusters are recursively composed or collapsed.
```

For:

```text
a cat is eating some food
```

the first mental object may be closer to:

```text
a -> cat
is -> eating
some -> food
cat -> eating
eating -> food
```

than to a hand-written grammar table.

The grammar table is an explanation after the geometry stabilizes.

## Transformer Analogy

A Transformer does not explicitly store a human-readable plane. It computes:

```text
score(i, j) = Q_i dot K_j
attention(i, j) = softmax(score(i, j))
```

But operationally this resembles a dynamic relation field:

```text
each token asks which other tokens are near / relevant / useful;
attention weights form a soft graph over the sequence;
different layers and heads refine that graph.
```

So a useful engineering analogy is:

```text
Transformer attention = learned dynamic co-occurrence / relation field.
```

This is not subjective feeling. It is an architectural reading of the math.

## TreeHeap Reframe

TreeHeap should not be treated as a pre-invented binary grammar.

Instead:

```text
latent plane placement
-> local attraction / graph formation
-> binary or k-ary partition for computation
-> TreeHeap address/path representation
-> read kernel / collapse
```

In this view, heap structure is not the source of language law.
It is a computational coordinate system for a latent placement problem.

## Plane Partition View

Any heap can be interpreted as repeated binary partition of a plane or ordered
space.

For an array:

```text
[0, 1, 2, 3, 4, 5, 6, 7]
```

a binary heap induces:

```text
[0..7]
  [0..3]
    [0..1]
    [2..3]
  [4..7]
    [4..5]
    [6..7]
```

For a latent plane, the analogous operation is:

```text
split a region into two subregions;
preserve address/path of each region;
allow kernels to operate on local neighborhoods.
```

This links:

```text
array fold
tree fold
quadtree / kd-tree style partition
attention relation field
spatial clustering
```

under one idea:

```text
fold is placement plus partition plus local collapse.
```

## Discovery Principle

The research order should be:

```text
1. Observe language phenomena.
2. Infer the latent relation / placement pattern.
3. Choose a partition or heap coordinate that preserves the pattern.
4. Define kernels over that coordinate.
5. Test whether the kernel learns or preserves the observed relation.
```

Not:

```text
1. Invent a fold.
2. Force language into it.
3. Declare victory because a toy task works.
```

## Candidate Linguistic Phenomena

Initial phenomena to observe:

```text
determiner-head attraction: a -> cat
quantifier-head attraction: some -> food
auxiliary-verb attraction: is -> eating
predicate-argument attraction: cat -> eating, eating -> food
compound attraction: foot -> ball, basket -> ball
modifier-head attraction: red -> apple
attachment ambiguity: with a telescope -> saw / man
```

These should become measured relation patterns before they become TreeHeap
kernels.

## Claim Draft

```text
S1-PLANE-C01:
Language fold should be modeled first as latent placement / attraction over
tokens and phrases. TreeHeap is a coordinate and partition system for computing
over that placement, not the language law itself.
```

Status:

```text
open theory
```

## Predict Draft

```text
P-S1-PLANE01:
If the latent placement view is useful, then weakly supervised relation
signals from real text should produce stable neighborhoods:

  det-head, aux-verb, quant-head, verb-object, subject-predicate

and TreeHeap kernels should perform better when their partition preserves
those neighborhoods than when the same tokens are folded by a purely positional
or random partition.
```

## Future Experiment Sketch

Use short real sentences:

```text
a cat is eating some food
the dog is chasing a ball
a child drinks some water
```

Build candidate relations without committing to full grammar:

```text
nearby function word -> content word
verb-like center -> noun-like neighbors
stable pair under substitution
```

Compare three layouts:

```text
1. linear order layout
2. learned co-occurrence / attraction layout
3. random layout
```

Then test whether TreeHeap partition over each layout preserves readout and
composition quality.

## Boundary

This theory does not claim:

```text
TreeHeap already understands language.
Binary heap is the true grammar.
Transformer has subjective feelings.
Latent plane is literally two-dimensional.
```

It claims only:

```text
Language fold should be discovered as relation geometry first;
TreeHeap should then be evaluated as a computational partition of that geometry.
```
