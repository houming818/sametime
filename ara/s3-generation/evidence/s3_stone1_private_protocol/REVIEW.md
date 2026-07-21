# STONE-1 Formal Evidence Review

Date: 2026-07-22
Reviewer decision: `not_supported_under_recipe`

## What Ran

- Host/device: `io`, NVIDIA GeForce RTX 3090
- Runtime: 27,268.44 seconds (7.57 hours)
- Data: frozen 1M train / 2K validation / 2K test WMT-massive split
- Arms: identity, learned structural, frozen random
- Seeds: 71901, 71902, 71903
- Updates: 15,625 per arm
- Parameters: 27,769,097 per arm
- Exit status: 0; `stderr.log` is empty

## Primary Result

| Arm | Mean NLL | NLL std | Mean BLEU-4 |
|---|---:|---:|---:|
| identity | **4.071877** | **0.005801** | **11.173550** |
| learned structural | 4.226905 | 0.112471 | 9.787452 |
| frozen random | 4.120966 | 0.010646 | 10.719838 |

The learned arm loses to identity on every seed. Its NLL penalties are
`+0.305850`, `+0.084650`, and `+0.074583`. It also loses to frozen random on
every seed. The registered quality and comparative-structure predictions fail.

## Causal Result

On learned seed 71903:

| Intervention | NLL | Damage from native |
|---|---:|---:|
| native | 4.141139 | 0.000000 |
| force identity | 4.129340 | -0.011799 |
| force random | 4.144581 | +0.003442 |
| swap left/right addresses | 5.635948 | **+1.494809** |

This is a clean separation. Left/right addresses are causally important, but
the learned direction policy is not. Replacing it with identity helps slightly;
replacing it with the frozen random policy barely changes NLL.

## Mechanism Diagnosis

The learned gate is a hard straight-through bit computed from
`[left, right, left-right, left*right]`. The translation loss is its only
training signal. Deep gates mostly saturate toward identity, while several
shallower levels retain substantial entropy or seed-dependent hard choices.
Identity gives the decoder one stable canonical coordinate system. The learned
arm changes that coordinate system while encoder and decoder are being trained,
and the final choices do not compensate for the optimization cost.

This diagnosis is consistent with, but not proved solely by, the gate
statistics. A follow-up should directly test stability over training rather
than merely adding more data.

## Numerical Closure

Closure MSE remains around `1e-13`, but learned seeds 71901 and 71903 have
maximum absolute errors `2.1458e-5` and `2.8640e-5`, above the preregistered
`1e-5` gate. This looks like accumulated floating-point error, not a categorical
failure to invert, but the registered gate is correctly recorded as failed.

## Product Boundary

The learned checkpoint generates non-empty text, stays below the repetition
limit, loads in the CLI, uses about 1.70 GiB peak VRAM, and has 27.42 ms batch-1
P50 latency. These are engineering successes. Quality is below the registered
milestone and below the identity control, so the checkpoint is not a usable
translation product.

The result does not falsify TreeHeap, recursive lifting, fixed-capacity heaps,
or all learnable private protocols. It falsifies this specific hard learned
local-direction recipe under the frozen one-pass training contract.
