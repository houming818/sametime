# A16: Zero-training multiresolution Bayesian READ

## Status

Preregistered before smoke and formal execution.

## Motivation

A15 mixed token identity, frozen corpus geometry, and a trainable downstream
GRU. A high-dimensional random code was therefore a strong token identifier:
the GRU could relearn context associations from the downstream task. That
experiment cannot attribute masked-token performance to installed semantics.

A16 removes downstream learning completely. The only information allowed to
score a masked token is the frozen conditional probability field installed
before the evaluation region.

## Representation

For target token `t` and context token `c`, the installed field is

```text
p_t(c) = P(c | t)
```

For TreeHeap node `v` with token masses `m_t`:

```text
p_v(c) = sum_{t in v} m_t p_t(c) / sum_{t in v} m_t
delta_v(c) = p_v(c) - p_parent(v)(c)
```

At depth `d`, the READ state for token `t` is the prototype of the node reached
by `t`:

```text
p_hat_d(c | t) = p_root(c) + sum_{k=1..d} delta_path_k(c) = p_node(t,d)(c)
```

The final token residual is

```text
delta_token(c) = p_t(c) - p_leaf(t)(c)
```

and therefore `p_leaf + delta_token = p_t` exactly. Node and token probability
states have the same dimension and unit. Every registered node state and final
token state lies on the probability simplex; residuals have zero total mass.

## Frozen Bayesian READ

Held-out WMT English sentences begin at valid line 1,100,000, after A14's
1,000,000 fit lines and 100,000 sealed-test lines. For each occurrence of a
target token, context IDs within a radius-four window are collected. No target
coordinate is included in its own context bag.

For candidate token `t` and observed context bag `C`:

```text
score_d(t | C) = log prior(t) + sum_{c in C} log(p_hat_d(c | t) + epsilon)
posterior_d(t | C) = softmax_t(score_d)
```

There is no trainable embedding, adapter, GRU, decoder, or task loss.

## Arms

- `prior_only`: empirical target frequency, no context field.
- `native_depth_0..D`: root through successive TreeHeap resolutions.
- `native_full_token`: the original per-token probability row; this is the
  uncompressed reference, not a TreeHeap victory condition.
- `row_shuffle_full`: preserve every probability row but randomly change which
  token owns each row.
- `path_shuffle_depth_0..D`: preserve the learned tree and its occupancy while
  assigning complete token paths to different token identities.

Three fixed shuffle seeds are used. A random Gaussian embedding is deliberately
excluded because it is not a conditional probability field and cannot be
evaluated by the frozen Bayesian READ equation.

## Candidates

1. `A14-1M-D3` with its 1,000,000-line count artifact.
2. `A13-200K-D5` with its 200,000-line count artifact.

Both must have exactly the same target and context ID order as their count
artifacts and the evaluation tokenizer.

### Paired-depth control amendment

The first formal execution made the preregistered two-candidate comparison, but
those candidates confound corpus scale and depth (`1M/D3` versus `200K/D5`).
Before interpreting any cross-candidate depth effect, A16-R1 adds the two
already-sealed complementary checkpoints: `1M/D5` and `200K/D3`. The original
formal artifact remains unchanged. The paired run uses the identical evaluation
tensor hash, Bayesian equation, controls, and mechanical gates; it introduces
no tuned parameter or new metric.

## Recorded metrics

For every native depth and control:

- posterior NLL;
- top-1 and top-5 accuracy;
- mean reciprocal rank;
- number of examples and usable context tokens.

The experiment reports the complete resolution curve. It does not stop on a
non-monotonic quality observation.

## Mechanical gates

1. no trainable parameter exists in the evaluation path;
2. all probabilities, scores, and metrics are finite;
3. every probability row sums to one within `1e-6`;
4. stored node residual closure is at most `1e-6`;
5. token residual closure is at most `1e-6`;
6. native depth zero equals `prior_only` within `1e-8` for all metrics;
7. all candidates evaluate the identical held-out examples and context bags.

## Evidence interpretation

The installed field has causal predictive information only if
`native_full_token` beats the median `row_shuffle_full` control. TreeHeap
compression is useful only to the degree that intermediate native depths retain
the gain of `native_full_token` over `prior_only` and outperform their matched
path-shuffle controls.

No fixed percentage-retention threshold is imposed in this first diagnostic.
The measured curve will determine whether a later application gate is
reasonable.

## Claim

`S1-BAYES-READ-A16-C01` remains open until formal evidence is complete.
