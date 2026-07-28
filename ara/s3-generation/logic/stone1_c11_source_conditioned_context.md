# STONE-1 C11: source-conditioned variable context

Status: registered, running

## Problem

C10 optimized teacher-forced next-piece cross entropy, but its training frame
always exposed 128 source pieces followed by 128 visible EOS pieces.  Free
generation then fell into a few frequent Chinese templates.  Low validation NLL
did not establish that the decoder used the source.

## Claim

`S3-STONE1-SOURCE-CONDITIONED-C11`:

A fixed-size TreeHeap can learn Chinese continuation that causally depends on
its source when (1) the source contains two ordered 128-piece subheaps, (2)
short contexts use an invisible mask instead of a visible EOS tail, and (3)
training explicitly penalizes a wrong-source example scoring as well as the
matched source.

This claim is about conditional continuation.  It is not a translation, chat,
world-model, or general-intelligence claim.

## Data contract

The existing deterministic C10 raw stream is reused without reparsing the
corpus.  Adjacent packed rows produce one example:

```text
row[i]   (256 pieces) -> source TreeHeap
row[i+1][:128]        -> future target
```

The source therefore consists of two ordered 128-piece regions.  During
training, the most recent `L` pieces are retained for
`L in {16, 32, 64, 128, 256}`.  They are copied to the visible prefix of the
fixed 256-leaf heap; the remaining leaves are PAD and invisible.  Text is not
duplicated or mirrored.

## Objective

Let `NLL(x,y)` be teacher-forced target cross entropy for matched source `x`,
and let `x_shuffle` be another source in the same batch.

```math
L_{dep} = max(0, m + NLL(x,y) - NLL(x_{shuffle},y))
```

```math
L = L_{native} + lambda * L_{dep}
```

The second term does not tell the model what sentence to write.  It only says
that the correct history must explain its own future better than an unrelated
history.

## Predictions and stop rules

1. `source_shuffle_damage = NLL(shuffle) - NLL(native) >= 0.05` after the
   smoke stage.  Otherwise stop the long run.
2. An empty/EOS-only source must be worse than the native source.
3. First-step distributions must change across sources; identical logits reject
   source conditioning.
4. Free outputs from different held-out sources must not all collapse to one
   normalized string.
5. All metrics are reported separately by context length.  A good average may
   not hide failure on 16/32-piece input.

## Falsification

The claim is rejected if matched and shuffled sources remain interchangeable,
if short contexts still require a visible EOS tail, or if free generation is a
single dominant template despite lower teacher-forced NLL.  Passing this test
does not prove that TreeHeap is better than a Transformer; matched baselines
belong to the next experiment.

