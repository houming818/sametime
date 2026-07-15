# S2 TreeHeap Emergent-Mask Translation

## Question

Earlier real-text evidence established two bounded facts:

1. a recursively composed TreeHeap root can carry task-relevant information;
2. removing lower-level subheap details during training can move some usable
   information upward, although the previous depth trend was not monotonic.

This experiment asks whether that mechanism helps S2 translation.  It does not
provide grammar, semantic categories, routes, merge labels, or desired internal
states.  The only learned objective is English-to-Chinese translation
cross-entropy.

## Claim

`S2-TREEHEAP-EMERGENT-MASK-C01`:

> When a multi-head TreeHeap encoder is trained for WMT translation, hiding
> complete lower portions of several already-composed subheaps while retaining
> their roots should make translation gradients place more usable information
> in upper TreeHeap states.  This should improve clean or detail-cut translation
> relative to an identical unmasked TreeHeap and an equal-count random-memory
> dropout control.

"Emergent" is operational here.  It means that no target internal structure is
specified; only output translation loss selects what the TreeHeap stores.

## Encoder

Tokens are written as leaves.  At every level, a shared multi-head
`[state,left,right]` kernel composes adjacent children:

\[
z_h = U_h\,\mathrm{GELU}(D_h[state,left,right]+b_h)
\]

\[
parent = \mathrm{LayerNorm}\left(\sum_h
\mathrm{softmax}(g(state,left,right))_h z_h\right).
\]

All valid leaves and internal states are exposed to the same autoregressive
decoder.  The three TreeHeap variants have identical parameters and start from
the same initialization.

## Structured Multi-Scale Mask

The complete TreeHeap is built before masking.  For each selected subheap at
depth `d`, its leaves and internal descendants below depth `d` are hidden from
decoder attention, while the selected subheap root and ancestors remain
visible.  Multiple depths and positions may be selected in one sentence.

Therefore information is not deleted before composition.  The decoder must use
the already-composed upper states if they contain what translation needs.

Controls:

- `plain`: no training-time memory removal;
- `structured_mask`: aligned complete subheap descendants are removed;
- `random_mask`: exactly the same number of states is removed at every level,
  but at random addresses.

All variants use only target-token cross-entropy.  The mask is an information
path perturbation, not a supervised answer.

## Evaluation

Each checkpoint is evaluated in three modes:

1. `normal`: complete TreeHeap, measuring ordinary S2 quality;
2. `cut`: deterministic held-out structured cuts, keeping subheap roots;
3. `cut_root_zero`: the same cuts plus removal of the selected subheap roots.

The root contribution is:

\[
\Delta_{root}=NLL(cut\_root\_zero)-NLL(cut).
\]

## Registered Predictions

For the first WMT smoke:

1. structured-mask normal NLL beats plain by at least `0.03`;
2. structured-mask cut NLL beats plain cut NLL by at least `0.10`;
3. structured-mask root contribution exceeds plain by at least `0.05`;
4. structured-mask cut NLL beats equal-count random mask by at least `0.05`;
5. all models complete with finite gradients and non-empty generation.

Predictions 2-4 test the mechanism.  Prediction 1 tests whether it transfers
to the actual clean S2 task.

## Falsification

The S2 improvement claim is not supported if clean NLL does not improve.  The
upward-information mechanism is not supported if structured cuts are no more
robust than plain/random masks or selected-root removal is neutral.  A result
that improves only cut robustness is a useful regularization result, not an S2
quality improvement.  No smoke result may be promoted to semantic emergence,
world knowledge, or architecture superiority.

## Result

The registered FP32 smoke completed on io in 203 seconds using 5,000 training,
500 validation, and 500 test WMT pairs.  Each model had 12,775,685 parameters
and the three models started from the same state.

| Model | Clean NLL | Clean token-BLEU4 | Detail-cut NLL | Cut plus selected-root removal NLL | Selected-root contribution |
|---|---:|---:|---:|---:|---:|
| plain TreeHeap | 6.5251 | 0.4071 | 6.6989 | 6.7030 | 0.00413 |
| structured mask | 6.5062 | 0.4799 | 6.6047 | 6.6081 | 0.00340 |
| equal-count random mask | 6.5033 | 0.7722 | 6.6201 | 6.6248 | 0.00472 |

Derived effects:

- structured clean gain over plain: `0.01896` NLL;
- structured detail-cut gain over plain: `0.09419` NLL;
- structured detail-cut gain over random mask: `0.01541` NLL;
- structured selected-root contribution gain over plain: `-0.00073` NLL.

The gates were `false/false/false/false/true`.  The main claim is therefore
**not supported**.  Structured masking produced a small clean regularization
effect and a larger detail-cut robustness trend, but random masking matched or
nearly matched it.  Removing the specifically retained subheap root caused
almost no additional loss and no gain over plain training.  The current
decoder can continue through many visible states and is not forced to read the
selected root, so this experiment did not create evidence of upward TreeHeap
information concentration.

The next valid redesign is architectural rather than a larger mask sweep:
after selecting a subheap, expose its root as the only representation of that
span to the decoder.  Generic random dropout must remain as a control.  Only a
reproducible structured-over-random gain would justify a stronger emergence
claim.
