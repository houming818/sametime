# C10 Observer-Resolution STOP Smoke

Date: 2026-08-12

Diagnostic: `S3-C10-OBSERVER-RESOLUTION-STOP-D02`

Parent: `S3-TREEHEAP-PRETRAIN-POSTERIOR-C10`

Status: smoke completed; numerical-tail mechanism supported only as a bounded
compute cleanup, not as the explanation of C10 leaf collapse

## Question

The current recursive READ has a soft STOP probability, not a computational
termination condition. With finite logits, sigmoid STOP is never exactly zero
or one. A path with arbitrarily small remaining mass therefore continues until
the leaf boundary.

This smoke asks whether an observer resolution can turn negligible remaining
mass into a real stop without materially changing the model prediction.

## Resolution rule

Let `a_(t,d,i)` be the probability mass arriving at node `i`, depth `d`, while
generating target step `t`. For observer resolution `epsilon`:

```text
if a_(t,d,i) <= epsilon:
    deposit all a_(t,d,i) at this node
    do not visit its children
else:
    apply the learned STOP and branch kernels normally
```

The forced stop conserves mass. It does not delete the low-flux path; it reads
the current coarser node instead of expanding that path further.

`epsilon=0` exactly reproduces the original soft recursion.

## Frozen sweep

The C10-PT pilot checkpoint is never updated. The smoke uses the same held-out
WMT rows for every arm:

```text
epsilon = 0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1
```

For each arm record:

- token NLL and delta from `epsilon=0`;
- probability mass deposited at every depth;
- mass forced to stop by observer resolution;
- mean stop depth;
- mean visited nodes per generated token;
- compute reduction relative to `epsilon=0`.

## Predictions

### P1: numerical-tail explanation

If leaf arrival is mainly caused by infinitesimal nonzero tails, some small
`epsilon <= 1e-2` should:

- reduce visited nodes by at least 20%;
- move visible stop mass to internal nodes;
- add no more than 0.02 NLL.

### P2: learned downward-flow explanation

If C10 sends substantial probability mass downward, small epsilon values will
barely change the computation. Only large epsilon values will force internal
stops, and they will either damage NLL or expose that intermediate nodes are
already usable despite the failed learned STOP.

This smoke distinguishes the two mechanisms. It does not claim that any fixed
epsilon is the final STOP algorithm.

## Smoke result

Taskd job `167` completed on `io` in 14 seconds after job `166` exposed and
invalidated an aliased-statistics accumulator. Job `167` is the valid run. It
froze C10-PT state SHA-256
`811fe2c00de5aaa27a90f22660ffceca0444d3c53f2ad8a04ed764c480b55f71`
and scored 2,532 target pieces from 128 held-out WMT rows.

| Epsilon | Delta NLL | Node visits reduced | Forced internal-stop mass | Leaf stop mass |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0% | 0% | 100.0000% |
| 1e-6 | +0.0000001 | 4.07% | 0.0000036% | 100.0000% |
| 1e-4 | +0.0000566 | 6.23% | 0.000738% | 99.9993% |
| 1e-3 | +0.001791 | 8.24% | 0.01259% | 99.9874% |
| 3e-3 | +0.01018 | 10.11% | 0.06786% | 99.9321% |
| 1e-2 | +0.06985 | 14.06% | 0.5044% | 99.4956% |
| 2e-2 | +0.29555 | 20.70% | 1.8676% | 98.1324% |
| 5e-2 | +2.45097 | 42.30% | 9.1395% | 90.8605% |
| 1e-1 | +8.83539 | 55.44% | 20.9869% | 79.0131% |

The registered P1 gate failed: no `epsilon <= 1e-2` reduced node visits by at
least 20% while keeping delta NLL at or below 0.02.

The experiment nevertheless establishes a useful bounded result. Observer
resolution around `1e-3..3e-3` removes 8-10% of node visits at almost no NLL
cost. Those paths carry less than 0.07% of probability mass, so this is genuine
finite-resolution tail pruning.

It does not explain the main leaf behavior. Even at `epsilon=0.02`, where
compute falls by about 21%, 98.13% of total mass still reaches leaves and NLL
already worsens by 0.296. C10 therefore has both:

1. a small mathematical nonzero tail that an observer threshold can safely
   terminate; and
2. a dominant learned downward flow that deliberately deposits almost all
   probability mass at leaf resolution.

Evidence:

```text
ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/
  pilot_seed10101/observer_resolution_stop_smoke.json
```
