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
