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
dL/dK_write = 0.31991300065246825
dL/dPlus    = 0.291900292038898
```

## Training Trace

| Epoch | Tau | Loss | MSE | Route CE | Accuracy | Gold Prob | Hard/Soft Gap | Grad K | Grad Plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.0000 | 1.14607 | 0.802156 | 1.37565 | 0.250 | 0.253 | 3.48909 | 3.199e-01 | 2.919e-01 |
| 1 | 1.9940 | 1.02194 | 0.694932 | 1.30803 | 0.500 | 0.271 | 3.26334 | 2.606e-01 | 2.541e-01 |
| 5 | 1.9702 | 0.770424 | 0.481863 | 1.15424 | 0.500 | 0.321 | 2.7535 | 1.609e-01 | 1.732e-01 |
| 25 | 1.8553 | 0.490201 | 0.271115 | 0.876345 | 0.875 | 0.425 | 2.03447 | 5.248e-02 | 4.855e-02 |
| 100 | 1.4810 | 0.328613 | 0.182106 | 0.586027 | 1.000 | 0.567 | 1.60092 | 3.608e-02 | 3.663e-02 |
| 250 | 0.9437 | 0.0747444 | 0.0159366 | 0.235231 | 1.000 | 0.796 | 0.498041 | 7.910e-03 | 1.514e-02 |
| 500 | 0.4453 | 0.0106362 | 0.00145884 | 0.0367093 | 1.000 | 0.964 | 0.142346 | 3.324e-03 | 7.068e-03 |
| 1000 | 0.2500 | 0.000716328 | 5.62154e-05 | 0.00264045 | 1.000 | 0.997 | 0.0298769 | 3.985e-04 | 1.089e-04 |
| 1800 | 0.2500 | 0.000613639 | 4.87752e-05 | 0.00225945 | 1.000 | 0.998 | 0.0278801 | 3.181e-04 | 9.810e-05 |

## Collapse Examples

| Key | Gold Address | Argmax tau=0.25 | Argmax tau=0.05 | Gold Prob tau=0.25 | Gold Prob tau=0.05 |
|---:|---|---|---|---:|---:|
| 2 | LL | LL | LL | 1.000 | 1.000 |
| 3 | LL | LL | LL | 0.997 | 1.000 |
| 5 | LR | LR | LR | 0.996 | 1.000 |
| 6 | LR | LR | LR | 0.998 | 1.000 |
| 10 | RL | RL | RL | 0.998 | 1.000 |
| 11 | RL | RL | RL | 0.996 | 1.000 |
| 13 | RR | RR | RR | 0.997 | 1.000 |
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
