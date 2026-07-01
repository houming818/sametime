# Deductive vs Inductive Kernel Probe Evidence

Verdict: `pilot_pass = False`

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
| address_prior | 1.454618 | 1.431725 | 2.393506 | 0.037 |
| linear_raw | 1.454791 | 1.438613 | 2.385915 | 0.050 |
| mlp_raw | 0.014008 | 0.012530 | 0.041845 | 0.973 |
| treeheap_prob_kernel | 0.013969 | 0.017539 | 0.119512 | 0.763 |
| oracle_fixed_kernel | 0.000000 | 0.000000 | 0.000000 | 1.000 |

## Interpretation

This pilot separates two proof types:

```text
deductive proof: algebraic operations hold by definition
inductive proof: trainable parameters reduce KL to imitate a world distribution
```

It does not prove language ability or WMT translation. It is a controlled proof
that probability kernels can be evaluated as inductive learners with KL/OOD KL.
