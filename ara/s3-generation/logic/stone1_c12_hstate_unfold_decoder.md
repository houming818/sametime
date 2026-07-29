# STONE-1 C12: H_state algebraic UNFOLD decoder

Status: smoke complete; mechanism feasible, strong claim rejected

## Audit premise

C11 established causal source dependence but retained severe free-generation
loops. Code audit found that the so-called recursive decoder resets its active
route to the root at every output token. It recursively reads within one token
step, but does not recursively consume output addresses across steps.

## Claim

`S3-STONE1-HSTATE-UNFOLD-C12`:

Given the same frozen C11 TreeHeap encoder and adjacent-block continuation
stream, a decoder that predicts a target TreeHeap H_state and applies the
existing algebraic UNFOLD to 128 addressed output leaves will preserve more
position-specific information and produce less cyclic free output than the
current teacher-forced GRU decoder.

The target state is

```math
H_y=(r_y,\{d^y_k\},\{g^y_k\}).
```

The learned bridge predicts `r_y`, `d_k`, and `g_k` from source H_state,
target-parent state, source subheap state, depth, and address. Child states are
not produced by an unrelated split MLP. They use the existing inverse FOLD:

```math
a=p-U(d),\qquad \hat b=d+P(a),
```

```math
l=g a+(1-g)\hat b,\qquad r=g\hat b+(1-g)a.
```

Seven shared recursive expansions produce 128 target leaves. All token logits
are emitted in parallel; no predicted token is fed back into the decoder.

## Controlled experiment

- frozen encoder: exact C11 checkpoint;
- data: same adjacent raw blocks and variable source lengths;
- target: fixed 128 pieces;
- arm A: newly initialized historical depth-floor GRU decoder;
- arm B: newly initialized algebraic UNFOLD decoder;
- same examples, optimizer family, update count, seed, and target CE;
- report parameters, elapsed time, tokens, NLL and generation metrics.

The smoke may establish mechanism feasibility without claiming parameter- or
compute-matched superiority. A later formal comparison must match effective
capacity or present a rate-quality frontier.

## Predictions

1. UNFOLD optimization remains finite and lowers held-out NLL.
2. Source shuffle and empty source each add at least `0.05` NLL.
3. Swapping source sibling addresses adds at least `0.01` NLL.
4. Zeroing at least one predicted target-detail depth adds `0.01` NLL.
5. Mean target-leaf variance exceeds `1e-4`.
6. Token distinct-2 and distinct-4 exceed the GRU arm by at least `0.05`.
7. The maximum repeated n-gram run is lower than the GRU arm.

## Falsification

Reject the strong decoder claim if UNFOLD leaves collapse, address/detail
interventions are neutral, source conditioning disappears, or reduced
repetition is explained only by a parameter/compute mismatch. Passing a smoke
does not establish translation, dialogue, semantic consciousness, or
superiority over Transformer.

## Smoke result

The preregistered 500-update smoke ran on `io` as task 56. Both arms used the
frozen C11 encoder and the same adjacent-block corpus protocol. The result is
evidence against the strong claim.

| Metric | Algebraic UNFOLD | Token-stream GRU |
|---|---:|---:|
| Trainable parameters | 18,761,474 | 39,205,698 |
| Initial validation NLL | 10.4029 | 10.4436 |
| Final validation NLL | 7.8480 | 7.4115 |
| Training/evaluation elapsed | 17.51 s | 619.11 s |
| Distinct-2 | 0.0079 | 0.0507 |
| Distinct-4 | 0.0080 | 0.0700 |
| Maximum repeated run | 128 | 62 |
| Unique-output fraction | 0.0625 | 0.9375 |

The UNFOLD computation was finite and fast, and its 128 leaf states were not
numerically identical (`leaf_variance=0.1407`). All seven predicted detail
levels also had non-zero norms. These observations support only the narrow
mechanism statement: the existing inverse FOLD can be placed in a differentiable
seven-level decoder and optimized end to end.

They do not show that the decoder used a TreeHeap private protocol. Causal
interventions were almost neutral:

| Intervention | Validation NLL increase |
|---|---:|
| Shuffle source examples | +0.0010 |
| Empty the source | +0.0019 |
| Swap source sibling addresses | +0.0002 |
| Zero target detail depth 0 | +0.0006 |
| Zero target detail depth 6 | +0.0017 |

Zeroing detail depths 1 through 5 changed NLL by `-0.0002` to `-0.0023`;
removing several details slightly improved the score. The free output collapsed
to a comma at all 128 positions for most samples. Therefore non-zero internal
states were activity, not evidence that the output depended on those states.

## Decision

Predictions 1 and 5 passed. Predictions 2, 3, 4, 6, and 7 failed. Reject the
strong C12 decoder claim.

The useful finding is architectural: a shared local inverse-FOLD kernel is
about 35 times faster than the serial GRU arm in this smoke and uses about 48%
of its trainable parameters, but it learned a corpus-prior shortcut instead of
a source-conditioned target tree. More updates alone are not the next test;
the next design must make target H_state construction conditional on source
subheaps and must prevent a position-wise vocabulary prior from satisfying the
loss without using address/detail state.

Evidence: `../evidence/s3_stone1_c12_hstate_unfold/summary.json`.
