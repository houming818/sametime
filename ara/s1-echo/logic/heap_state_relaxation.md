# Heap-State Relaxation

Date: 2026-06-30
Author: Codex Review
Status: supported pilot

## Motivation

Houming818 proposed a refinement to the controllable-manifold idea:

```text
Gradient information may not only update parameters.
It may also adjust the balance of the current heap state itself.
```

Example:

```text
[2.0, 1.0, 3.0]
-> [2.0, 1.5, 2.5]
```

This is not primarily parameter learning. It is state relaxation:

```text
H <- H - eta * grad_H E(H)
```

## Core Distinction

Parameter learning:

```text
theta <- theta - eta * grad_theta L
```

Heap-state relaxation:

```text
H <- H - eta * grad_H E(H)
```

In the second case:

```text
theta is fixed;
TreeHeap address rules are fixed;
the current arr[i] vectors move;
the energy function over the current heap supplies the gradient.
```

This partially avoids the need for a differentiable target heap:

```text
The target heap does not need to be differentiable or even explicitly present.
The current heap energy E(H) must be differentiable.
```

## Probe

Script:

```text
ara/s1-echo/src/s1_heap_state_relaxation_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_heap_state_relaxation_probe/
```

Design:

```text
theta_updated = false
heap_state_updated = true
target_heap_used_in_loss = false
loss_type = energy over current heap state
```

## Scalar Proof

Initial heap:

```text
root  = 2.0
left  = 1.0
right = 3.0
```

Energy:

```text
E = (left - right)^2 + (root - (left + right) / 2)^2
```

Result:

```text
initial energy = 4.0
final energy   = 9.86e-31
left delta     = +1.0
right delta    = -1.0
```

The state moved to:

```text
[2.0, 2.0, 2.0]
```

without changing any kernel parameter.

## Vector TreeHeap Proof

The second proof uses a 7-node TreeHeap:

```text
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
```

Leaves are fixed observed vectors.
Internal states `arr[1]`, `arr[2]`, `arr[3]` are initialized randomly.

Energy contains:

```text
parent-child consistency:
  arr[parent] should approach mean(left_child, right_child)

relation anchor consistency:
  arr[node] should approach a fixed relation anchor
```

Only heap state is updated.

Result over 32 random initializations:

```text
mean_vector_energy_ratio = 1.24e-13
max_vector_energy_ratio  = 3.69e-13
mean_centroid_error_drop = 3.0393
pass_rate                = 1.0
pilot_pass               = true
```

## Claim

```text
S1-RELAX-C01:
A differentiable energy over the current TreeHeap state can generate gradients
that relax arr[i] toward a lower-energy equilibrium while kernel parameters and
address rules remain fixed.
```

Status:

```text
supported pilot
```

## Boundary

This does not prove:

```text
translation
language understanding
unsupervised relation-field learning
TreeHeap superiority over Transformer
that every useful linguistic objective can be written as an energy
```

It proves a narrower mechanism:

```text
TreeHeap can support state-gradient learning in addition to parameter-gradient
learning.
```

## Next Gate

The next proof should combine:

```text
1. learned or data-derived relation field
2. heap-state relaxation
3. probabilistic stop/left/right collapse
4. a real short-sentence structure metric
```

