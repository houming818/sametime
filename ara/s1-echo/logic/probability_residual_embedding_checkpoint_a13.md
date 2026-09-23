# S1 Probability-Residual Embedding Checkpoint A13

Status: preregistered

## Objective

A11 and A12 established that probability-residual F search improves held-out
context reconstruction reproducibly. They did not save the learned F as a
loadable embedding artifact. A13 closes that engineering gap.

Two checkpoints will be trained from the frozen 200K-line WMT context field:

```text
target tokens: 512
context tokens: 1024
fixed split seed: 20260924
new search seed: 20260928
search budget: 2048 proposals per arm
depth arms: 3 and 5
```

Depth 3 represents the strongest A12 reconstruction region. Depth 5 preserves
more routing resolution and is retained as the middle-depth comparison. A13
does not select one as universally superior.

## Checkpoint Contract

Each `embedding_checkpoint.pt` must contain:

```text
SentencePiece hash and target/context token IDs and pieces
learned node axes and thresholds
token leaf IDs, path bits, and route margins
token square-root context-probability coordinates
node probability prototypes and child-parent residuals
smoothed leaf probabilities used for held-out reconstruction
```

After saving, the script must reload the checkpoint, reroute every token, and
recompute sealed-test NLL. The token assignments must match exactly and NLL
absolute error must be at most `1e-12`.

## Claim

`S1-F-CKPT-A13-C01`: The reproducible probability-residual F search can produce
a self-describing, frozen, reload-stable TreeHeap embedding checkpoint at both
the shallow and middle-depth operating points.

Both arms must pass all A11 mechanical gates, improve sealed-test NLL over
their initial tree, beat random routing, and pass the exact reload contract.

## Boundary

This creates embedding artifacts. It does not yet show that READ, a Decoder,
translation, or semantic probes can use them. The checkpoints are frozen inputs
for that next experiment; downstream labels must not retroactively alter A13.

## Result

The two formal arms completed on `io` as tasks `576` and `577`.

```text
depth                         3                 5
leaves                        8                32
initial dev NLL               5.468152         5.373436
best dev NLL                  5.418536         5.318546
dev gain                      0.049616         0.054890
initial sealed-test NLL       5.486780         5.396372
best sealed-test NLL          5.436473         5.341358
sealed-test improvement       0.050307         0.055014
leaf utilization              1.000000         1.000000
checkpoint reload exact       true             true
reload NLL absolute delta     0                0
```

Both arms pass every preregistered gate. `S1-F-CKPT-A13-C01` is supported for
the frozen 512-target/1024-context WMT field. The resulting artifacts are valid
TreeHeap probability-residual embedding checkpoints and may be used as frozen
inputs to downstream probes.

Depth 5 has a modestly larger reconstruction gain in this longer search, but
that does not select it as the better semantic representation. Both checkpoints
must remain frozen and be compared under the same downstream data, model,
initialization, and optimization budget.
