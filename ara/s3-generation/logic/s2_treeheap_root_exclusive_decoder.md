# S2 TreeHeap Root-Exclusive Decoder

## Motivation

The previous structured-mask experiment improved detail-cut WMT NLL by
`0.0942`, but did not increase the causal contribution of the retained subheap
root.  Its decoder could still attend to overlapping ancestors and many other
states.  The mask changed availability without enforcing a TreeHeap decoding
protocol.

This experiment removes that bypass.

## Claim

`S2-TREEHEAP-ROOT-EXCLUSIVE-C01`:

> If a source span is represented to the S2 decoder only by its recursively
> composed subheap root, translation loss can train that root into a useful
> compressed state.  A variable-resolution TreeHeap frontier should match or
> improve full-tree translation and outperform an equal-frontier flat span-mean
> encoder, while removal of compressed roots should causally damage output.

Only English-to-Chinese target-token cross-entropy is optimized.  There are no
route labels, syntax labels, semantic classes, mask-recovery targets, or
internal-state targets.

## Root-Exclusive Frontier

For every source sentence, choose aligned, non-overlapping spans whose union is
the complete token sequence.  Each span is represented by exactly one state:

\[
F(T)=\{v_1,\ldots,v_m\},
\]

\[
\bigcup_i Leaves(v_i)=Leaves(T),
\qquad
Leaves(v_i)\cap Leaves(v_j)=\varnothing.
\]

If `v_i` is an internal subheap root, none of its descendants or ancestors are
visible as alternative copies of the same span.  Uncollapsed one-token spans
remain leaves.  Training samples different valid frontiers; held-out evaluation
uses a deterministic frontier derived from the source tokens.

## Models

All variants have the same declared parameters and identical initialization:

1. `full_tree`: decoder sees every valid leaf and internal TreeHeap state;
2. `exclusive_tree`: decoder sees only the variable-resolution TreeHeap
   frontier;
3. `exclusive_flat`: decoder sees the same frontier addresses and span widths,
   but every internal root is replaced by a learned projection of the mean of
   that span's token embeddings.

The flat-span control distinguishes recursive FOLD information from bandwidth
reduction and generic span pooling.

## Interventions

- `native`: the representation used by that model during evaluation;
- `frontier`: force the deterministic recursive TreeHeap frontier;
- `frontier_root_zero`: use the same frontier but hide all internal roots;
- `leaf_only`: expose ordered token leaves only.

For an exclusive model:

\[
\Delta_{compressed\ root}
=NLL(frontier\_root\_zero)-NLL(native).
\]

## Registered Predictions

1. exclusive TreeHeap native NLL beats full-tree native NLL by `0.03`;
2. exclusive TreeHeap native NLL beats exclusive flat-span NLL by `0.05`;
3. removing compressed roots raises exclusive TreeHeap NLL by `0.10`;
4. exclusive training increases compressed-root damage by `0.05` over forcing
   the same frontier on the full-tree model;
5. training and generation remain finite and non-empty.

## Falsification

The S2 gain is not supported if prediction 1 fails.  A specifically recursive
TreeHeap gain is not supported if flat span means match the recursive roots.
Root causality is not supported if root removal is neutral.  A positive result
would remain a WMT smoke mechanism result, not evidence of human semantics,
world knowledge, consciousness, or superiority to production Transformers.

## Result

The registered FP32 smoke completed on io in 183 seconds with 5,000 training,
500 validation, and 500 test WMT pairs.  The deterministic frontier used
`0.5126` states per source token; `42.79%` of its states were internal roots.

| Model | Native NLL | Native token-BLEU4 | Forced TreeHeap frontier NLL | Compressed-root removal NLL | Leaf-only NLL |
|---|---:|---:|---:|---:|---:|
| full tree | 6.5038 | 0.3460 | 6.8831 | 6.9019 | 6.5207 |
| exclusive TreeHeap | 6.6072 | 0.1841 | 6.6072 | 6.7075 | 6.5549 |
| exclusive flat span | 6.5770 | 0.2503 | 6.7227 | 6.7981 | 6.6394 |

Derived effects:

- exclusive TreeHeap clean gain over full tree: `-0.10345` NLL;
- recursive TreeHeap gain over equal-frontier flat span mean: `-0.03020` NLL;
- compressed-root removal damage after exclusive training: `+0.10032` NLL;
- root-removal damage when the same frontier is forced on full-tree training:
  `+0.01877` NLL;
- exclusive-training root-causality gain: `+0.08155` NLL.

The gates were `false/false/true/true/true`, so the result is **partial
mechanism support**.  Root exclusivity solved the decoder-bypass problem: when
a span was represented only by its root, removing that root materially damaged
translation, and exclusive training made this dependence much stronger than
full-tree training.  However, it did not improve S2 quality.  The recursive
root lost to both full overlapping memory and the much simpler flat span mean.

This localizes the next blocker.  The decoder protocol can force TreeHeap roots
to matter; the current FOLD kernel does not preserve enough translation value
inside those roots.  The next experiment should keep the root-exclusive
decoder fixed and improve only the encoder/FOLD path, using a residual
value-preserving root initialized from the flat span mean plus a learned
non-commutative correction.  That comparison can ask whether TreeHeap adds
order/structure without first discarding the strong mean-pooling baseline.
