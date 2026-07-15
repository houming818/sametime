# TreeHeap Structural Differential Atlas

Date: 2026-07-15
Status: partial operator signatures / 32K scale-stable
Claim: `S3-TREEHEAP-DIFF-ATLAS-C01`

## Question

Does the current learned TreeHeap state expose repeatable, operator-specific
multiscale responses, or does every perturbation merely create undifferentiated
vector noise?

This claim does not assume semantic differential locality already exists. It
maps several possible structural differences before selecting a later semantic
claim.

## Frozen Subject

Use the single-seed checkpoint from `S3-TREEHEAP-GEOMETRY-C01` without further
training. The checkpoint is weak and not endorsed; the experiment describes
its actual response only.

## Perturbation Basis

Apply six operations to held-out real Chinese 64-token blocks:

1. `WRITE_ONE`: replace one leaf and preserve topology.
2. `WRITE_TWO`: replace two leaves in different subheaps.
3. `MIRROR_8`: recursively mirror one aligned eight-leaf subheap.
4. `SWAP_CHILDREN_8`: exchange only the two four-leaf children of that subheap.
5. `SWAP_SUBHEAPS_4`: exchange two distant aligned four-leaf subheaps.
6. `SHUFFLE_8`: random order inside the same eight-leaf region; non-algebraic control.

Each operation is evaluated at two address settings. Address A constructs the
operator centroids; address B tests whether the multiscale signature transfers
instead of memorizing one position.

## Observation Signature

For every depth, compare original and perturbed node states. Record:

- mean squared state difference;
- maximum node difference;
- mean state cosine change;
- Bag READ output difference;
- adjacency READ output difference.

The concatenated six-depth vector is an operator response signature. A nearest
centroid classifier trained only on address A predicts operation type at
address B. This is an observation instrument, not a language decoder.

## Sample-Size Gate

Report all metrics at validation prefixes `512`, `2,048`, `8,192`, and
`32,768`. A metric is called statistically stable only if its change from 8K
to 32K is at most `0.02` absolute and bootstrap 95% intervals are reported.

The checkpoint itself was trained on 200,000 blocks. Stability over validation
size does not prove that training-data scale is sufficient.

## Predict

```text
P1  Exact inverse controls recover tokens and every TreeHeap state with max
    absolute error <= 1e-6.

P2  Address-A to address-B operator classification exceeds random 1/6 by at
    least 0.20 at 32K blocks.

P3  MIRROR_8 and SWAP_CHILDREN_8 remain distinguishable above 0.60 pairwise
    accuracy at address B; otherwise the state does not expose recursive
    mirror versus one-level child exchange.

P4  WRITE_ONE and WRITE_TWO remain distinguishable above 0.70 pairwise
    accuracy at address B, showing sensitivity to changed structural extent.

P5  The 8K-to-32K change is <= 0.02 for multiclass accuracy and every pairwise
    accuracy in P3/P4.

P6  Order-preserving Bag output change for MIRROR_8, SWAP_CHILDREN_8 and
    SHUFFLE_8 is smaller than their adjacency output change on average.
```

## Decision

- Full support requires P1-P6 and later replication across model seeds.
- Partial support means operators have distinguishable response signatures but
  the Bag/adjacency interpretation or sample stability fails.
- Rejection means address-transfer classification remains near chance or
  inverse controls fail.

Even full support would establish operator observability, not semantic
relations, a private encoder-decoder protocol, multilingual alignment, or an
advantage over Transformer.

## Result

The frozen checkpoint was evaluated on 32,768 held-out real Chinese blocks.
Four of six gates passed:

```text
P1 exact inverse recovery          PASS   max state error 0
P2 six-way address transfer        PASS   0.4776 vs chance 0.1667
P3 mirror vs one-level child swap  PASS   0.7769
P4 one-write vs two-write          FAIL   0.6070 < 0.70
P5 evaluation-scale stability      PASS
P6 Bag/order readout separation    FAIL
```

The six-way result is not uniformly strong. Per-operation accuracy was:

```text
WRITE_ONE          0.8281
WRITE_TWO          0.1687
MIRROR_8           0.5789
SWAP_CHILDREN_8    0.1652
SWAP_SUBHEAPS_4    0.6951
SHUFFLE_8          0.4296
```

`WRITE_TWO` was predicted as `WRITE_ONE` 66.8% of the time, while
`SWAP_CHILDREN_8` was predicted as `MIRROR_8` 50.4% of the time. The state
therefore exposes broad operation families and propagation extent, but not all
operator details.

The recursive distinction is visible in the depth-energy curves. MIRROR and
one-level child swap have nearly identical energy at depths 1 and 2. At depth
3, full MIRROR retains MSE `0.01744`, while child swap falls to `0.00728`; by
the root they are `0.00481` and `0.00071`. Distant four-leaf subheap exchange
keeps a larger root response of `0.01697` because two remote ancestor paths are
changed.

Evaluation sample size was not driving the main result:

```text
blocks             512     2K      8K      32K
six-way accuracy  .4818   .4807   .4796   .4776
mirror/child      .7725   .7849   .7791   .7769
one/two write     .6025   .6096   .6083   .6070
```

The 32K bootstrap 95% interval for six-way accuracy is `[0.4756, 0.4796]`.
This establishes evaluation stability for this frozen checkpoint. It does not
show that its 200,000 training blocks, one model seed, or current objective are
sufficient for semantic learning.

P6 failed in the same direction as the earlier geometry proof: order-changing
operations moved the weak Bag reader more than the adjacency reader. The
checkpoint has structural response signatures but still lacks a trustworthy
directional readout.

## Decision

Keep `S3-TREEHEAP-DIFF-ATLAS-C01` as **partial support**. Multiple independent
TreeHeap-native perturbations produce stable, cross-address multiscale
signatures, so differential observation is viable as an instrument. The result
does not yet support semantic differences, decoder causality, multilingual
alignment, or superiority to Transformer. The next experiment should vary
training-data scale and model seed before using this atlas for semantic claims.

Evidence: `evidence/s3_treeheap_structural_differential_atlas/summary.json`.
