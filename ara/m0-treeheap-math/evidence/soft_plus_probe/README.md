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
dL/dK_write = 0.09024252282755797
dL/dPlus    = 0.14386881319232503
```

## Training Trace

| Epoch | Tau | Loss | MSE | Route CE | Accuracy | Gold Prob | Hard/Soft Gap | Grad K | Grad Plus |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.0000 | 0.677455 | 0.33324 | 1.37686 | 0.250 | 0.253 | 2.30401 | 9.024e-02 | 1.439e-01 |
| 1 | 1.9940 | 0.661998 | 0.321501 | 1.36199 | 0.250 | 0.257 | 2.26512 | 8.073e-02 | 1.256e-01 |
| 5 | 1.9702 | 0.623132 | 0.296998 | 1.30453 | 0.250 | 0.274 | 2.17559 | 6.585e-02 | 8.245e-02 |
| 25 | 1.8553 | 0.529263 | 0.26093 | 1.07333 | 1.000 | 0.344 | 2.01087 | 5.000e-02 | 5.126e-02 |
| 100 | 1.4810 | 0.308301 | 0.148255 | 0.640183 | 1.000 | 0.536 | 1.49047 | 3.980e-02 | 3.078e-02 |
| 250 | 0.9437 | 0.0783981 | 0.0121967 | 0.264806 | 1.000 | 0.775 | 0.440505 | 8.699e-03 | 1.456e-02 |
| 500 | 0.4453 | 0.0130076 | 0.00139482 | 0.046451 | 1.000 | 0.955 | 0.130012 | 3.984e-03 | 8.348e-03 |
| 1000 | 0.2500 | 0.000955443 | 1.99984e-05 | 0.00374178 | 1.000 | 0.996 | 0.0177602 | 5.509e-04 | 7.776e-05 |
| 1800 | 0.2500 | 0.000774334 | 1.69195e-05 | 0.00302966 | 1.000 | 0.997 | 0.0163691 | 4.168e-04 | 5.541e-05 |

## Collapse Examples

| Key | Gold Address | Argmax tau=0.25 | Argmax tau=0.05 | Gold Prob tau=0.25 | Gold Prob tau=0.05 |
|---:|---|---|---|---:|---:|
| 2 | LL | LL | LL | 1.000 | 1.000 |
| 3 | LL | LL | LL | 0.996 | 1.000 |
| 5 | LR | LR | LR | 0.994 | 1.000 |
| 6 | LR | LR | LR | 0.998 | 1.000 |
| 10 | RL | RL | RL | 0.998 | 1.000 |
| 11 | RL | RL | RL | 0.994 | 1.000 |
| 13 | RR | RR | RR | 0.996 | 1.000 |
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
