# Soft Plus Probe Evidence

This is an M0 synthetic proof for kernel-guided Soft Plus.

## Verdict

`pilot_pass = True`

## What Was Tested

```text
score(a) = K_write(subheap(H, a), x)
p(a) = softmax(score(a))
H_next = sum_a p(a) * Plus_a(H, x)
loss = MSE(H_next, target)
```

The proof checks whether gradients reach both:

```text
K_write parameters
Plus_a parameters
```

and whether low-temperature collapse recovers the correct hard plus address.

## First Gradient Norms

```text
dL/dK_write = 0.45795759649795653
dL/dPlus    = 0.27255371621653673
```

## Training Trace

| Epoch | Tau | Loss | MSE | Route CE | Accuracy | Gold Prob | Hard/Soft Gap | Grad K | Grad Plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.0000 | 1.6642 | 1.3212 | 1.37199 | 0.250 | 0.254 | 4.38233 | 4.580e-01 | 2.726e-01 |
| 1 | 1.9940 | 1.46953 | 1.14469 | 1.29938 | 0.500 | 0.276 | 4.19006 | 3.108e-01 | 1.847e-01 |
| 5 | 1.9702 | 1.17549 | 0.893586 | 1.1276 | 0.500 | 0.334 | 3.75135 | 1.971e-01 | 1.466e-01 |
| 25 | 1.8553 | 0.746452 | 0.544057 | 0.809579 | 0.750 | 0.461 | 2.8786 | 7.826e-02 | 6.676e-02 |
| 100 | 1.4810 | 0.290571 | 0.181858 | 0.434852 | 1.000 | 0.656 | 1.61643 | 5.409e-02 | 6.254e-02 |
| 250 | 0.9437 | 0.0636836 | 0.0213312 | 0.16941 | 1.000 | 0.849 | 0.557263 | 7.314e-03 | 2.230e-02 |
| 500 | 0.4453 | 0.00638603 | 0.00108564 | 0.0212016 | 1.000 | 0.979 | 0.120234 | 2.318e-03 | 6.265e-03 |
| 1000 | 0.2500 | 0.00034799 | 5.15754e-05 | 0.00118566 | 1.000 | 0.999 | 0.0286711 | 2.420e-04 | 1.038e-04 |
| 1800 | 0.2500 | 0.000310608 | 4.68849e-05 | 0.00105489 | 1.000 | 0.999 | 0.0273601 | 1.793e-04 | 9.719e-05 |

## Collapse Examples

| Key | Gold Address | Argmax tau=0.25 | Argmax tau=0.05 | Gold Prob tau=0.25 | Gold Prob tau=0.05 |
|---:|---|---|---|---:|---:|
| 2 | LL | LL | LL | 1.000 | 1.000 |
| 3 | LL | LL | LL | 0.999 | 1.000 |
| 5 | LR | LR | LR | 0.997 | 1.000 |
| 6 | LR | LR | LR | 1.000 | 1.000 |
| 10 | RL | RL | RL | 1.000 | 1.000 |
| 11 | RL | RL | RL | 0.997 | 1.000 |
| 13 | RR | RR | RR | 0.999 | 1.000 |
| 14 | RR | RR | RR | 1.000 | 1.000 |

## Interpretation

This evidence supports only a narrow M0 claim:

```text
kernel-guided Soft Plus can be made differentiable, can receive gradient
through K_write and Plus_a, and can collapse to the hard address in this
synthetic key/address toy.
```

It does not prove language understanding, syntax induction, WMT translation,
or superiority over Transformer.
