# S1 heap-state relaxation probe

Claim: `S1-RELAX-C01`

This proof tests Houming818's heap-state gradient hypothesis:

```text
The gradient does not have to update only kernel parameters theta. It can update
the current TreeHeap state H so that the heap relaxes to a lower-energy
equilibrium.
```

## Design

- theta updated: `False`
- heap state updated: `True`
- target heap used in loss: `False`
- loss type: `energy over current heap state`

## Metrics

- scalar energy ratio: `0.00000000`
- scalar left delta: `1.0000`
- scalar right delta: `-1.0000`
- mean vector energy ratio: `0.00000000`
- max vector energy ratio: `0.00000000`
- mean centroid error drop: `3.0393`
- pass rate: `1.0000`
- pilot pass: `True`

## Boundary

This does not prove translation, language understanding, or unsupervised world
model learning. It proves only that a differentiable energy over the current
heap state can generate gradients that move `arr[i]` toward a lower-energy
state while fixed kernel/rule parameters stay unchanged.
