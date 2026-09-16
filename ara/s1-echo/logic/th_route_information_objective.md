# TreeHeap Hierarchical Route Information Objective

## Claim

`S1-TH-ROUTE-INFO-C01`

The pair-NCE fixed-point experiment leaves every soft TreeHeap split almost
maximally uncertain. A hierarchical information objective should make routes
token-dependent without sending every token to one branch.

For token state `X`, current node `N`, and binary branch `B`, each level uses:

```text
L_info = H(B | X, N) - H(B | N) = -I(X; B | N)
```

Minimizing `L_info` lowers the conditional entropy of each token's branch while
maximizing marginal branch entropy within occupied nodes. This differs from a
hard occupancy penalty: it asks for confident token-specific decisions and
balanced aggregate use at the same time.

## Fixed-Seed Smoke

- immutable smoke pair cache;
- seed `19301` for every arm;
- rounds 2, depth 6, dimension 32, 300 steps;
- pair-NCE control is the existing task 458;
- interventions use route-information weights `0.05` and `0.20`;
- no seed search and no early stop.

The Smoke is useful only if route entropy falls measurably without destroying
held-out pair ordering. The selected formal weight, if any, must be declared
from Smoke before a 100K-line run.

## Smoke Gates

- all losses and gradients finite;
- held-out pair accuracy no more than `0.01` below the fixed-seed control;
- minimum soft branch entropy decreases by at least `0.01` bit;
- hard leaf utilization remains at least `0.10`;
- saturated route mass does not exceed `0.95`.

Passing these gates proves only that the intervention changes the intended
router variable without immediately destroying the pair task. It does not
prove semantic quality or generation.

## First Smoke Result and Calibrated Retry

Fixed-seed tasks 469--470 completed, but weights `0.05/0.20` failed the route
gate. Minimum soft entropy changed only from control `0.9999493` to
`0.9999479/0.9999423` bits. Pair accuracy stayed unchanged and hard occupancy
spread slightly, but the target soft decision remained effectively uniform.

At step 1, increasing weight from `0` to `0.20` changed the route-weight
gradient norm by only about `1.9e-7`, versus the pair objective's route-weight
gradient norm `2.13e-4`. The information term is second-order near the uniform
router. A bounded gradient-scale retry therefore uses weights `256` and
`1024`, still with seed `19301` and the same batches. This is a scale
calibration, not a new objective or seed search. The original gates remain
unchanged.
