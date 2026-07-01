# Deductive vs Inductive Kernel Probe Evidence

Verdict: `pilot_pass = True`

## Deductive Proof

| Check | Result |
|---|---:|
| mirror involution | True |
| mirror patch involution error | 0.000000e+00 |
| conjugate equivalence error | 0.000000e+00 |
| one-hot soft plus error | 0.000000e+00 |

These are algebraic checks. They are expected to hold by definition when the
operators are correctly specified.

## Inductive Proof

The world model defines an operation distribution:

```text
P_W(a|H,q) = softmax(-||patch(a)-query||^2 / temperature)
```

Models are trained to imitate `P_W` and evaluated with KL divergence.

| Model | Train KL | Test KL | OOD KL | OOD top1 |
|---|---:|---:|---:|---:|
| address_prior | 1.436873 | 1.423885 | 2.485674 | 0.027 |
| linear_raw | 1.438356 | 1.419793 | 2.486090 | 0.047 |
| mlp_raw | 0.016925 | 0.017466 | 0.051720 | 0.930 |
| treeheap_prob_kernel | 0.012546 | 0.014143 | 0.075806 | 0.890 |
| oracle_fixed_kernel | 0.000000 | 0.000000 | 0.000000 | 1.000 |

## Interpretation

This pilot separates two proof types:

```text
deductive proof: algebraic operations hold by definition
inductive proof: trainable parameters reduce KL to imitate a world distribution
```

It does not prove language ability or WMT translation. It is a controlled proof
that probability kernels can be evaluated as inductive learners with KL/OOD KL.
