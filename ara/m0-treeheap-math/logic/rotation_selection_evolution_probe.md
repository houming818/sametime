# Rotation Selection Without an Order Label

Status: partially supported; single-seed control failure
Parent claim: `M0-ROT-C02`
Predict: `P-ROT02-B`
Date: 2026-07-21

## Question

If candidate fixed-capacity rotations are not told which ones preserve tree
relations, can task fitness eliminate disordering transforms and concentrate
on structure-preserving transforms?

This experiment does not put edge preservation, path preservation, inversion
count, or an `is_ordered` label in the loss. Those quantities are computed only
after training.

## Fixed Population

Each individual uses a 127-node TreeHeap. The population has 24 registered
bijections, all preserving capacity and depth cardinality:

```text
6 exact tree automorphisms
6 mildly corrupted automorphisms
12 random within-depth permutations
```

The training population size is fixed. No individual creates nodes, and the
inference result is one collapsed operator.

## Worlds

### Structured world

Generate a Gaussian branching process:

```text
child = rho * parent + sqrt(1-rho^2) * noise
rho = 0.92
```

Random nodes are masked. A small shared local decoder predicts a masked node
from its transformed parent, sibling, and children. Its only optimization
target is masked-state MSE.

### IID control

Use the identical training procedure with `rho=0`. Nodes have no parent-child
relation, so structural automorphisms should have no systematic fitness value.

### Exact-echo control

Give every candidate its mathematical inverse. Every bijection should echo
exactly, showing that unrestricted paired echo alone cannot select order.

## Selection

One global gate has 24 logits:

```text
p(R_i) = softmax(logit_i / temperature)
loss = sum_i p(R_i) * masked_prediction_loss_i
```

There is no order regularizer. After a decoder warmup, task gradients update
both decoder parameters and gate logits.

## Post-Hoc Audit

After training, compute for each candidate:

- tree-edge preservation ratio;
- validation masked-prediction MSE;
- final gate probability;
- correlation between edge preservation and validation loss.

## Predictions

```text
P1 structured-world exact-automorphism mass >= 0.75
P2 structured exact mean MSE is at least 0.10 below random mean MSE
P3 edge-preservation/loss Pearson correlation <= -0.70
P4 IID exact-automorphism mass <= 0.50
P5 IID exact-vs-random mean MSE gap <= 0.02
P6 every exact-echo inverse max error < 1e-12
P7 exact-echo candidate-loss variance < 1e-12
P8 every individual remains 127 nodes
```

## Falsification

Reject the selection predict if structured data does not concentrate fitness
on relation-preserving transforms, if the same concentration appears in IID
data, or if exact paired echo itself distinguishes arbitrary bijections.

## Boundary

Passing would show selection of tree-relational order in a controlled world,
not language semantics or universal evolution. Failing would show that the
current task pressure is insufficient; it would not disprove fixed-capacity
rotation as a protocol carrier.

## Single-Seed Result

```text
structured exact gate mass       = 0.999713
structured exact/random MSE      = 0.264948 / 0.678812
structured edge/loss Pearson     = -0.997420

IID exact gate mass              = 0.905888
IID exact/random MSE             = 0.999190 / 1.000591
IID edge/loss Pearson            = -0.268907

exact echo inverse error         = 0
exact echo loss variance         = 0
```

Seven of eight gates passed. `P4` failed because the IID gate eventually put
`90.59%` mass on the exact group even though group validation losses were tied
within `0.0014` and structure/loss correlation was weak. The IID training trace
first moved exact mass down to `0.0616`, then up to `0.9059`, consistent with
neutral drift amplified by gate/decoder co-adaptation.

Decision: structured selection is a strong positive in this seed, exact echo
correctly shows no selection pressure, but one IID trajectory is insufficient
to distinguish structural selection from stochastic winner lock-in. The
registered follow-up is `P-ROT02-C`, an eight-seed drift audit.
