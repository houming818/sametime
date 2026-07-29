# STONE-1 C12: H_state algebraic UNFOLD decoder

Status: registered, implementation pending

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

