# Controllable Fold Manifold

Date: 2026-06-30
Author: Codex Review
Status: pilot supported on toy relation field

## Motivation

SPR-036 reframed language fold as:

```text
latent placement / attraction
-> partition
-> TreeHeap address/path
-> kernel read and collapse
```

Houming818's next correction is that the project needs a product-facing
control story:

```text
At first, a generated structure may be chaotic.
Can we adjust kernel details as control variables until normal product
behavior appears?
If the process is controllable, the control path itself may teach us the
emergence law.
```

This document turns that into an ARA claim.

## Core Idea

A TreeHeap kernel is not only a fixed algebraic operator. It can also expose
knobs:

```text
relation_weight: how much to trust the latent relation field
order_weight:    how much to trust linear neighborhood/order
temperature:     how early to collapse a probability container
noise_scale:     how much perturbation remains in routing
```

The claim is not that these exact knobs are final. The claim is that fold
quality should be measurable as a function over such controls.

That function is the controllable fold manifold:

```text
controls -> generated fold state -> product quality metric
```

In a mature system, the controls may be learned parameters. In the current
probe, they are explicit variables so that we can observe the surface directly.

## Why This Matters

Without this layer, the project jumps between two disconnected worlds:

```text
algebra layer: plus, diff, compose, decompose, kernel
product layer: can it generate / parse / translate?
```

The controllable manifold is the bridge:

```text
algebraic controls
-> visible structure changes
-> product-like behavior
```

If this bridge works, then a failed product result is not just "bad model".
We can ask:

```text
Was relation attraction too weak?
Was order preservation too strong?
Did collapse happen too early?
Was the subheap partition wrong?
Was the kernel missing a mirror, path, or decomposition operation?
```

## Pilot Experiment

Script:

```text
ara/s1-echo/src/s1_controllable_manifold_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_controllable_manifold_probe/
```

The probe uses four short sentence cases:

```text
the cat is running for a car
a dog is chasing the ball
the book that i bought yesterday is expensive
some kids are playing in the park
```

For each sentence, we write weak target blocks:

```text
the cat
is running
a car
for a car
book + relative clause
```

Then we build a transparent relation field:

```text
tokens in the same weak block attract each other;
smaller blocks create stronger attraction.
```

The fold kernel greedily merges blocks using:

```text
score(A, B)
  = relation_weight * relation(A, B)
  + order_weight    * order(A, B)
  - balance_penalty
  + noise
```

The experiment sweeps:

```text
relation_weight in {0, 0.25, 0.5, 1, 2, 4}
order_weight    in {0, 0.25, 0.5, 1, 2, 4}
seeds = 64
```

Quality is measured by constituent/block F1 against the weak target blocks.

## Result

From `summary.json`:

```text
low_control_mean_f1      = 0.0828
max_control_mean_f1      = 0.8148
high_sum_control_mean_f1 = 0.8148
diagonal_gain            = 0.7319
product_cell_count       = 22
pilot_pass               = true
```

The diagonal path is especially important:

| relation_weight | order_weight | mean_f1 |
|---:|---:|---:|
| 0.00 | 0.00 | 0.0828 |
| 0.25 | 0.25 | 0.4036 |
| 0.50 | 0.50 | 0.6398 |
| 1.00 | 1.00 | 0.7603 |
| 2.00 | 2.00 | 0.8119 |
| 4.00 | 4.00 | 0.8148 |

This is the desired pilot shape:

```text
weak/no controls -> noisy fold
stronger controls -> better block structure
high controls -> plateau
```

## Interpretation

This supports a narrow claim:

```text
On a transparent toy relation field, fold quality can be controlled by kernel
variables and moves along a measurable surface.
```

It does not prove:

```text
translation
language understanding
unsupervised discovery of the relation field
TreeHeap superiority over Transformer
correct grammar
```

The strongest lesson is product-facing:

```text
TreeHeap development should expose control variables and read structure
quality as a surface, not as a single pass/fail generation result.
```

## Claim

```text
S1-MANIFOLD-C01:
If TreeHeap fold is driven by kernel controls over a latent relation field,
then product-like structure should emerge as a measurable control surface:
increasing relevant control variables should move output from noisy blocks
toward stable target blocks on a toy relation task.
```

Status:

```text
supported pilot
```

## Falsification

Downgrade or reject if:

```text
1. the sweep is non-controllable across seeds;
2. quality does not improve from low-control to high-control cells;
3. random or order-only controls match relation-conditioned controls;
4. the same control surface disappears on real relation-layout probes.
```

## Next Gate

The next experiment should replace the toy relation field with a relation field
estimated from data:

```text
co-occurrence window
dependency-like weak relation
masked-token restoration signal
contrastive positive/negative phrase pairs
```

Then rerun the same control-surface analysis.

If the surface remains controllable, S1 can move toward a product debugger:

```text
sentence -> relation field -> TreeHeap fold -> visible blocks / energies /
uncertainty containers
```
