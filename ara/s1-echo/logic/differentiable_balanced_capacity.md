# Differentiable Balanced-Capacity Readout

## Claim

`S1-BALANCED-CAPACITY-TRAIN-C01`

An NCE-free TreeHeap placement can remain trainable without losing mutual
exclusion: gradients update shared leaf context distributions, while a
Sinkhorn projection keeps token-to-leaf assignment at equal capacity. The
same leaf distributions act as routing prototypes and context decoders, so no
random token embedding is introduced.

## Model

Real WMT counts define fixed token observations

```text
x_v = sqrt(P_train(context | token=v)).
```

Each leaf owns one trainable context distribution `d_l`. Routing scores are
negative Hellinger distances:

```text
s(v,l) = -||x_v - sqrt(d_l)||^2 / temperature.
```

For the intervention arm, iterative log-domain Sinkhorn normalization turns
these scores into `Q(v,l)` with row sum `1` and column sum `V/L`. Prediction is

```text
P_hat(context | v) = sum_l Q(v,l) d_l.
```

The only learning objective is held-in context cross entropy. There is no
negative sampling, NCE term, trainable token lookup, or random parameter
initialization. Leaf distributions start from the already supported
deterministic balanced semantic tree.

The matched control has identical initialization, optimizer, steps, and loss,
but replaces Sinkhorn with a row softmax; it may choose arbitrary leaf mass.

## Preregistered Smoke

- WMT massive English side, first 100,000 valid lines;
- 256 readable targets, 512 contexts, depth 6, 64 leaves;
- every tenth line held out and never used for gradients;
- 200 full-batch Adam steps, learning rate `0.01`;
- temperature `0.20`, 30 Sinkhorn iterations;
- deterministic construction; seed `19411` is used only by balanced-random
  controls and random-pair audits.

## Gates

- finite loss and gradients;
- balanced train NLL decreases by at least `0.005`;
- trainable leaf parameters move by a nonzero amount;
- Sinkhorn maximum row and column mass error are each below `5e-4`;
- exact hard projection leaves exactly four token types in every leaf;
- balanced held-out hard NLL beats balanced random and is no more than `0.03`
  worse than the frozen semantic initializer;
- balanced held-out neighbor LCP beats random and frequency-balanced controls
  and is no more than `0.10` below the frozen initializer.

The unconstrained arm is diagnostic and cannot make the claim pass. Passing
does not prove sentence generation, bilingual alignment, new-token insertion,
or superiority over Transformer embeddings.

## Numerical Amendment Before Formal Smoke

Task 479 was a 20K/20-step implementation contract at temperature `0.08` and
20 Sinkhorn iterations. Gradients, exact hard capacity, held-out NLL, and
topology passed, but the soft row-mass error was `0.0696`; column error was
already `9.54e-7`. A no-gradient audit with 120 iterations (task 480) still
left row error `0.00324`, showing that simply unrolling many sharp projection
steps is inefficient. A second no-gradient audit changed only temperature to
`0.20` with 30 iterations (task 481) and reached row/column errors
`9.78e-6/9.54e-7` without changing the exact hard route result.

Therefore temperature `0.20` replaces `0.08` before the registered formal
smoke. This is a numerical conditioning amendment derived without gradient
training or held-out model selection; all quality gates remain unchanged.

## Formal Smoke Result

Task 483 passed optimization and capacity mechanics but failed the discrete
readout gates. Balanced train NLL fell `4.89265 -> 4.77961`; row/column mass
errors were `3.06e-5/1.19e-6`; the hard projection remained exactly four
tokens per leaf; and held-out neighbor LCP improved from the frozen `2.2500`
to `2.3503`. However, held-out soft NLL was `4.78887` while hard single-leaf
NLL was `5.61469`, worse than frozen `4.77779` and random `4.85159`.

The model therefore learned a useful multi-leaf mixture, not a valid discrete
TreeHeap readout. `S1-BALANCED-CAPACITY-TRAIN-C01` is rejected for the exact
zero-sharpening objective. Capacity solved marginal crowding but did not make
each token assignment decisive.

## Registered Sharpening Successor

`S1-BALANCED-CAPACITY-SHARP-C01` adds only

```text
lambda * mean_v H(Q(v, :))
```

to the same held-in context objective. Sinkhorn already fixes equal column
mass, so conditional entropy can be reduced without rewarding global leaf
collapse. A bounded `lambda = 0, 0.05, 0.20, 1.00` ladder repeats the exact
100K/256/512/depth-6/200-step contract. All arms are reported; held-out data
does not select the weight.

A weight supports the successor only if row/column mass gates and exact hard
capacity pass, normalized route entropy is at most `0.25`, mean top-1 route
mass is at least `0.75`, hard-minus-soft held-out NLL is at most `0.10`, hard
held-out NLL beats balanced random, and held-out neighbor LCP beats both
balanced controls. The matched unconstrained arm receives the same entropy
weight and remains diagnostic.

## Sharpening Ladder Result

Tasks 484--487 completed the registered `lambda=0/0.05/0.20/1.00` ladder.
The balanced arm kept exact `4..4` leaf capacity and mass errors below the
registered thresholds at every weight. However, normalized route entropy only
moved `0.8730 -> 0.8367`, top-1 mass only moved `0.1581 -> 0.1846`, and the
hard-soft NLL gap stayed `0.8258..0.8729`. No weight passed the sharpening
gates, so `S1-BALANCED-CAPACITY-SHARP-C01` is rejected for this bounded range.

The matched unconstrained arm exposed the role of capacity: at `lambda=0.05`
its hard leaf occupancy became `0..189`, whereas every balanced arm remained
`4..4`. Entropy pressure supplies concentration but not mutual exclusion;
capacity projection supplies mutual exclusion but does not by itself supply
concentration.

Before any larger weight ladder, the next audit measures the step-0 gradient
norms of context NLL and route entropy on the unchanged 100K contract. The
ratio `||grad L_context|| / ||grad H_route||` defines the scale unit; held-out
metrics are not involved in choosing the next weights.

Task 488 measured balanced step-0 gradient norms
`||grad L_context||=0.00305738` and `||grad H_route||=0.01181207`, giving the
equal-gradient scale `lambda*=0.258835`. The calibrated successor therefore
uses `lambda=4/16/64`, approximately `15.5/61.8/247.3` times `lambda*`.
These three values are registered as one bounded ladder before training.

`S1-BALANCED-CAPACITY-CALSHARP-C01` uses the same success gates as the first
sharpening claim. All three runs must be reported; held-out metrics cannot
select a weight or change the ladder.

## Calibrated Sharpening Result

Tasks 489--491 (`lambda=4/16/64`) also failed to create a discrete regime.
Balanced normalized entropy was `0.8285/0.8259/0.8253`, top-1 mass was
`0.1869/0.1863/0.1856`, and hard-soft NLL gaps were
`0.9266/0.9344/0.8881`. Capacity stayed exactly `4..4`, but increasing
entropy pressure degraded context NLL without materially sharpening routes.
The calibrated constant-weight hypothesis is rejected.

This identifies an objective mismatch: when prediction is computed from a
soft leaf mixture, the model is rewarded for retaining that mixture. A larger
penalty cannot reliably turn the downstream protocol into a discrete one.

## Hard-Protocol Successor

`S1-BALANCED-CAPACITY-HARD-C01` changes the forward contract, not the corpus or
TreeHeap geometry. Each step computes soft Sinkhorn mass, projects it to an
exact equal-capacity one-hot assignment, and uses that one-hot assignment for
context prediction. Backpropagation uses the soft Sinkhorn derivative through
a straight-through estimator:

```text
Q_forward = Q_soft + stop_gradient(Q_hard - Q_soft).
```

The intervention and soft reference both begin from a deterministic blend of
`50%` semantic leaf distribution and `50%` global context background, leaving
room for measurable learning without random initialization. Both use the
unchanged 100K/256/512/depth-6/200-step contract, `lambda=0`, and seed `19411`
only for controls.

The hard-protocol arm passes only if finite gradients and exact `4..4`
capacity hold, train NLL decreases by at least `0.02`, final hard held-out NLL
beats balanced random and is no more than `0.03` worse than the frozen semantic
tree, held-out neighbor LCP beats both controls, and final hard NLL beats the
matched soft-forward reference by at least `0.10`.

## Hard-Protocol Result

Tasks 492--493 completed the paired test. With identical blended
initialization, the soft-forward reference reduced train NLL but ended at hard
held-out NLL `5.51873`. The straight-through hard protocol reduced train NLL
`4.81023 -> 4.75460` and ended at hard held-out NLL `4.77457`, an advantage of
`0.74416` over the soft reference. It also slightly beat the frozen semantic
tree (`4.77779`) and clearly beat balanced random (`4.85159`). Exact capacity
remained `4..4`, row mass error was `9.42e-6`, gradients were finite, and
held-out neighbor LCP `2.30339` exceeded random/frequency controls
`0.99870/1.15234`. Every registered gate passed.

This supports the narrow mechanism: the discrete protocol must be present in
the forward computation if deployment will use a discrete route. A soft
mixture is not an adequate surrogate merely because its NLL is low.

## Scale Successor

`S1-BALANCED-CAPACITY-HARD-SCALE-C01` repeats only the supported hard protocol
at two deterministic scale points: `(200K lines, 512 targets, 1024 contexts,
depth 7)` and `(500K, 1024, 2048, depth 8)`. Both retain four token types per
leaf, 200 steps, blend `0.50`, learning rate `0.01`, temperature `0.20`, and
30 Sinkhorn iterations. Each point must independently retain finite gradients,
exact capacity, train-NLL decrease `>=0.02`, hard NLL below balanced random and
within `0.03` of the frozen tree, and neighbor LCP above both balanced controls.
No automatic expansion beyond 1024 targets is allowed.

## Scale Result

Tasks 494--495 passed every registered scale gate:

| Targets / contexts | Lines | Train NLL decrease | Hard held-out NLL | Frozen / random NLL | Neighbor LCP | Capacity |
|---:|---:|---:|---:|---:|---:|---:|
| 512 / 1024 | 200K | 0.07070 | 5.25649 | 5.26116 / 5.32978 | 2.48307 | 4..4 |
| 1024 / 2048 | 500K | 0.07902 | 5.70199 | 5.70540 / 5.77995 | 2.82096 | 4..4 |

Both runs had finite gradients, full utilization and occupancy entropy `1.0`,
and held-out neighbor LCP above balanced frequency/random controls. The raw
summaries retain the script's base `S1-BALANCED-CAPACITY-TRAIN-C01` claim
label, while their experiment field, task commands, preregistration, and this
result map them to `S1-BALANCED-CAPACITY-HARD-SCALE-C01`; raw evidence was not
rewritten.

The supported result remains a token-context compression/readout mechanism.
It does not yet demonstrate sentence composition, translation, generation,
polysemy conditioned on a current sentence, or online insertion of new token
types.
