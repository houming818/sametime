# S1 Probability-Residual Tree Replication and Depth Ladder A12

Status: preregistered

## Why This Experiment Exists

A11 showed that one depth-5 Monte Carlo search improved both development and
sealed-test context reconstruction while preserving probability FOLD and path
residual closure. One successful seed is not enough to call the embedding
formation method stable. It may be a favorable search trajectory, and depth 5
may be accidentally well matched to the 512-token field.

A12 separates two variables that A11's original CLI coupled:

- `split_seed` fixes the fit/dev count split and therefore fixes the problem;
- `search_seed` changes only the Monte Carlo proposal trajectory.

This is required for a valid search-replication claim.

## Frozen Inputs

```text
counts: A11 frozen real-WMT 512 x 1024 token-context field
split seed: 20260924 for every arm
fit ratio: 0.8
alpha: 0.05
temperature: 0.02
iterations: 512 per arm
depths: 3, 4, 5, 6
search seeds: 20260925, 20260926, 20260927
```

The matrix therefore contains 12 independent search arms over one fixed data
split. The sealed WMT test field is never used for proposals, acceptance, best
state selection, depth selection, or early stopping.

## Claim

`S1-F-MC-A12-C01`: Probability-residual TreeHeap F search is reproducible
across Monte Carlo trajectories on a fixed corpus field, and at least one depth
in the preregistered 8/16/32/64-leaf ladder provides stable held-out gains
without violating probability conservation or collapsing leaf use.

## Per-Arm Mechanical Gates

```text
all scores finite
512 trace rows present
at least one proposal accepted
FOLD conservation max abs <= 1e-10
path residual closure max abs <= 1e-10
leaf utilization >= 0.50
```

## Replication Gates

A depth is called replicated only when all conditions hold:

```text
at least 2 of 3 seeds have dev gain >= 0.005
median dev gain > 0
median sealed-test best-minus-initial < 0
all 3 seeds beat their random-route sealed-test control
all 3 seeds pass the mechanical gates
```

The A12 claim is supported when at least one of the four depths is replicated.
Results from every depth and seed remain evidence; failing arms are not removed.

## Interpretation Rules

NLL is not required to improve monotonically with depth. Greater depth raises
capacity but also reduces observations per leaf and enlarges the search space.
The ladder is descriptive: it identifies a stable operating region, not a
stopping rule and not a proof that deeper or shallower trees are universally
better.

Even a fully supported A12 claim establishes only reproducible corpus-field
embedding formation. It does not establish semantic categories, READ,
translation, generation, or Decoder compatibility. Those require a later
frozen-embedding downstream probe.

## Result

Tasks `563` through `574` completed successfully on `io`. Every arm contains
512 trace rows, finite scores, at least one accepted proposal, valid FOLD and
path-residual audits, and leaf utilization above the preregistered threshold.

```text
depth  leaves  median dev gain  median sealed-test delta  min utilization
3      8       0.053624         -0.053989                 1.00000
4      16      0.044896         -0.045092                 1.00000
5      32      0.048298         -0.050505                 1.00000
6      64      0.029383         -0.029595                 0.96875
```

For every depth, all three search seeds exceeded the `0.005` dev-gain gate,
all three improved over the deterministic initial tree on sealed test, and all
three beat their random-route control. All four depths therefore meet the
replication definition, so `S1-F-MC-A12-C01` is supported.

The ladder is not monotonic. Depth 3 has the largest median reconstruction
gain, while depth 6 has the smallest. This does not prove that eight leaves are
the best semantic embedding capacity: the objective measures context-field
reconstruction, not downstream semantic resolution. The next experiment
should freeze representative shallow and middle-depth embeddings and compare
their downstream information, rather than selecting a depth from NLL alone.
