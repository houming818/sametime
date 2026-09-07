# FOLD Observation-Matrix Decode Probe

## Question

With every trained parameter frozen, what text does the decoder produce when
only the numerical FOLD observation operator is changed?

This probe does not use NLL as its decision target.  It inspects decoded text
directly to distinguish a usable sentence regime from empty, repetitive,
language-confused, or unformed output.

## Intervention

For two valid child states, replace the current merge with

```text
parent = s * (left + right)
```

Equivalently, the local observation matrix is `[s I, s I]`.  The same scalar is
shared by every TreeHeap depth and is applied to both the base and residual
TreeHeap channels.  One-child nodes remain unchanged, matching the production
mask contract.

The preregistered scale grid is:

```text
0, 0.25, 0.5, sqrt(0.5), 1.0
```

`sqrt(0.5)` is the current implementation and therefore the native numerical
control.  `0.5` is arithmetic averaging.  `0` removes all direct parents made
by FOLD while retaining leaves.  The decoder's learned `K_up` convolution is
still active and may reconstruct a residual parent from those leaves, so this
is not a strict leaf-only ablation.  `1.0` is unnormalized summation.

## Fixed Contract

- checkpoint and all learned parameters are frozen;
- tokenizer, input, direction, recursion depth, greedy decoder, and maximum
  generation length are identical across scales;
- no result may be selected or rejected using reference NLL;
- the probe records raw decoded text, output length, EOS occurrence, adjacent
  repetition, and unique-piece ratio;
- results describe runtime behavior only and do not establish which scale would
  be optimal after retraining.

## Reading the Result

- If `s=0` still produces similar sentences, the leaves plus learned `K_up`
  path are sufficient for much of the observed generation; a separate
  `K_up` bypass is required before attributing the result to leaves alone.
- If a bounded middle interval produces sentences while either endpoint fails,
  the frozen decoder has a numerical compatibility band for TreeHeap context.
- If output changes gradually with `s`, the observation amplitude acts as a
  controllable semantic direction.
- If output changes abruptly, the intervention crosses a decoder decision
  boundary; this is evidence of sensitivity, not by itself evidence of a better
  representation.

## First 106M Observation

The frozen pass-2 106M checkpoint was decoded at depths 5, 6, and 7 on four
sentences.  The intervention produced a visible compatibility band:

- low scales sometimes collapsed immediately to EOS (empty output) or retained
  only a short, vague lexical fragment;
- `s=0.5` and the native `s=sqrt(0.5)` most consistently produced terminated
  sentence-like output;
- `s=1.0` often expanded to the 48-piece limit, failed to emit EOS, and repeated
  clauses or local motifs;
- increasing `s` sometimes restored source concepts before destabilizing
  generation, so the response is causal but not globally monotone in quality.

This supports the narrow claim that the numerical amplitude of parent
observations changes the frozen model's decoded regime.  It does not identify
the best train-time operator or establish that the model has learned correct
translation.
