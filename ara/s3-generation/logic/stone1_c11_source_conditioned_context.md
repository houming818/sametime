# STONE-1 C11: source-conditioned variable context

Status: completed; source conditioning supported, nondegenerate generation rejected

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

## Result

The io run completed 10,000 updates in 27,038 seconds with 50,679,947
parameters. Every registered numerical source-dependence gate passed:

| context pieces | initial NLL | final NLL | gain | final shuffle damage | final empty damage |
|---:|---:|---:|---:|---:|---:|
| 16  | 5.8125 | 4.9922 | 0.8203 | +0.2188 | +0.3281 |
| 32  | 5.3359 | 4.9297 | 0.4062 | +0.2708 | +0.3958 |
| 64  | 5.1380 | 4.9453 | 0.1927 | +0.3281 | +0.4661 |
| 128 | 5.1354 | 4.8776 | 0.2578 | +0.3099 | +0.4635 |
| 256 | 5.1354 | 4.9870 | 0.1484 | +0.2474 | +0.4349 |

This is positive evidence that the corrected TreeHeap frame reads its source:
replacing or emptying the source makes the same target less probable at every
tested length. Variable-length training particularly repaired the short-input
distribution mismatch.

The original free-generation gate was inadequate. Eight outputs were eight
different strings, so the automatic `unique_output_fraction` was `1.0`, but
manual inspection found severe within-output loops such as repeated variants
of “我当时就医了”, “不忘初心”, and empty book-title templates. Mean character
distinct-1/2/4 was only `0.2774/0.3568/0.4514`; one output had distinct-4
`0.1111`. String identity measures inter-sample diversity, not intra-sample
degeneration.

Therefore the composite claim is only partially supported. C11 repairs source
visibility and establishes conditional dependence, but it does not yet produce
usable free continuation. The next experiment must register token-level
distinct-n, repeated-span/run length, EOS rate, and source-output relevance
before training rather than accepting unique strings as a generation gate.
