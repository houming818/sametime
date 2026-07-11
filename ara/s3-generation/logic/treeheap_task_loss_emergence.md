# Task-Loss TreeHeap Structural Emergence

## Claim

`S3-TREEHEAP-EMERGENCE-C01` asks a narrow causal question:

> Can output loss alone make a TreeHeap's internal child states and recursive
> route functionally necessary for held-out generation?

The word "emergence" is used operationally.  It does not mean that a specific
numeric loss value is universal.  It means that, during training, a held-out
task improvement appears together with measurable dependency on TreeHeap
address, child state, and compose history.

## What Is And Is Not Supervised

Optimized objective:

\[
L(\theta) = \operatorname{CE}(D_\theta(\operatorname{Read}_\theta(H), q), y)
\]

where `y` is the two-token surface output.

Not optimized or supplied as labels:

```text
- route direction
- stop position
- tree depth
- merge pair
- category identity
- code length
- semantic class
```

The known bracket structure is part of the input data structure.  It is not
converted into a direct shape-bit feature for the TreeHeap model.  It only
changes which recursive `Compose(left,right)` calls construct the root.

## TreeHeap State

For a three-leaf item, leaf states are `h_a`, `h_b`, and `h_c`.  The local,
ordered convolution is:

\[
\operatorname{Plus}_\theta(h_l,h_r) =
\operatorname{MLP}_\theta([h_l + e_L; h_r + e_R])
\]

with distinct `e_L` and `e_R`, so that swapping children changes the state.

The two possible heaps are:

\[
H_0 = \operatorname{Plus}(\operatorname{Plus}(h_a,h_b),h_c)
\]

\[
H_1 = \operatorname{Plus}(h_a,\operatorname{Plus}(h_b,h_c))
\]

At the root the read kernel computes:

\[
p = \operatorname{softmax}(K_\theta([h_{root};h_L;h_R;q]))
\]

where `p = [p_stop, p_left, p_right]`.  It reads the soft state:

\[
h_{read}=p_{stop}h_{root}+p_{left}h_L+p_{right}h_R
\]

and a shared decoder maps `h_read` to two output token distributions.

The route is not trained against a gold left/right target.  It receives gradient
only because choosing a useful child lowers surface cross-entropy.

## Why The Dataset Is Contradictory Without Structure

Both examples contain the same ordered leaves `[a,b,c]`:

```text
((a b) c) -> [a,b]
(a (b c)) -> [b,c]
```

Consequently a structure-blind model observes the same input paired with two
different outputs.  It can minimize average loss but cannot make both exact.
This is a constructive lower-bound control, not a claim that all sequence
models lack structural capacity.  A model receiving an explicit tree bit can
solve the task; `shape_oracle` records that upper control.

## Causal Tests

Training accuracy is insufficient because the root could retain the answer.
The completed model is evaluated without retraining under interventions:

```text
full          ordinary recursive read
root_only     replace the read result by h_root
zero_internal zero the actual internal child before readout
mirror        reverse the tree and test the correspondingly reversed output
```

The claim is supported only if the full model generalizes and the first two
ablations cause a material loss.  `route_internal_acc` is a post-hoc observer:
it compares `argmax(p)` with the child that is structurally internal.  It must
be high but is never part of the gradient objective.

## Limits

This is not a proof that language structures emerge from raw text.  The input
tree bracketing is supplied, and the task is symbolic.  It only tests whether
the TreeHeap's local compose/read mechanism can be selected by task loss rather
than by manually supervising its path.
