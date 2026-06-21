# Trainability Quiz Evidence

This is synthetic M0 evidence for a minimal learning check before TreeHeap-object echo.

The quiz uses NumPy manual gradients, not PyTorch.

## Verdict

`pilot_pass = True`

## Tasks

| Task | Final Loss | Accuracy/R2 | Pass |
|---|---:|---:|---:|
| `linear_regression` | `1.6516923158620557e-30` | `1.0` | `True` |
| `xor` | `0.0007663465151399971` | `1.0` | `True` |
| `modular_addition` | `0.0021191076117503013` | `1.0` | `True` |

## Interpretation

The toy verifies that small trainable modules can learn linear mapping,
nonlinear XOR, and a full base-8 modular addition table. This does not
prove TreeHeap language learning; it is only the ML entrance exam before
building trainable TreeHeap encoder/plus/decoder modules.
