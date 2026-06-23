# Soft TreeHeap Extension

Status: supported pilot
Created: 2026-06-22
Updated: 2026-06-23
Source: SPR-019

## Question

Can Soft TreeHeap be integrated into M0 as a differentiable extension of the
Hard TreeHeap operator algebra?

This is not a language claim. It is an M0 algebra / learning claim.

## Hard Operator

Hard TreeHeap uses discrete operators:

```text
H_next = H ⊕_a x
```

Where:

```text
H: current TreeHeap
x: input node / token / primitive
a: address or path
⊕_a: TreeHeap plus at address a
```

## Probabilistic Lifting

Soft TreeHeap lifts hard operators into a probability container:

```text
SoftO(H) = sum_a p(a) * O_a(H)
```

For plus:

```text
H_next = sum_a p(a | H, x) * (H ⊕_a x)
```

If `p(a*) = 1` and all other probabilities are zero, Soft TreeHeap collapses
back to Hard TreeHeap:

```text
H_next = H ⊕_{a*} x
```

## Kernel-guided Soft Plus

The address distribution should not be produced by a generic memory write.
It should be produced by a TreeHeap kernel:

```text
score(a) = K_write(subheap(H, a), x)
p(a) = softmax(score(a))
H_next = sum_a p(a) * (H ⊕_a x)
```

This keeps the operation inside the TreeHeap algebra:

```text
subheap kernel -> route probability -> plus candidates -> collapse
```

## What This Replaces

Naive soft memory write:

```text
arr_new[i] = (1 - p[i]) * arr_old[i] + p[i] * write_vector
```

is differentiable, but it does not prove TreeHeap algebra is trainable. It
updates array slots. Kernel-guided Soft Plus updates over TreeHeap plus
candidates.

## Evidence

Pilot:

```text
src/soft_plus_probe.py
evidence/soft_plus_probe/
```

Result:

```text
pilot_pass = true
dL/dK_write = 0.09024252282755797
dL/dPlus    = 0.14386881319232503
initial_loss = 0.6774554273075396
final_loss = 0.0007743342108376414
collapse_accuracy_tau_0.05 = 1.0
```

## Boundary

This evidence supports only the M0 pilot claim:

```text
kernel-guided Soft Plus can be differentiable, trainable in a toy,
and collapsible to the hard plus address.
```

It does not prove:

```text
language understanding
syntax induction
WMT translation
Transformer replacement
general superiority over neural memory
```

It also does not prove that the write kernel has learned a route from raw
TreeHeap geometry. The current probe uses engineered alignment features:

```text
root_alignment = root_diff * root_side
child_alignment = diff * child_side
```

The audit in `.squad/outbox/006-runner-reviewer-01-soft-treeheap-ara-audit.md`
found:

```text
current features: collapse_acc = 1.000
remove alignment/sum features: collapse_acc = 0.625
raw/basic features: collapse_acc = 0.250
```

So the current proof is a gradient-path and collapse smoke proof. It is not yet
a proof that a TreeHeap convolution kernel can independently learn routing.

## Next Experiments

1. Compare write mechanisms:

```text
A: naive soft memory write
B: encoder soft plus
C: kernel-guided soft plus
```

2. Test unseen address relocation.

3. Test multi-kernel staged loss against big-pot loss.

4. Add stochastic / noisy subheap features and check collapse legality.

5. Replace hand-coded alignment features with a learned kernel over clean
   subheap observations:

```text
key
parent key
left child key
right child key
leaf flag
depth/address metadata
```

The clean-kernel experiment is the bridge from "Soft Plus can receive gradient"
to "TreeHeap can learn a searchable structure."
