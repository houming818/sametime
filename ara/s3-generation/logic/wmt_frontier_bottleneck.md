# WMT Fixed-Bandwidth TreeHeap Frontier

## Claim

`S3-WMT-FRONTIER-C01`:

> Under the same decoder and a fixed `K`-state source-memory budget, a
> translation-loss learned TreeHeap frontier should outperform fixed-tree,
> random-tree, and flat `K`-vector frontiers.  Perturbing the learned frontier
> should cause a measurable held-out translation loss increase.

This claim follows earlier ARA evidence that subheap-aware kernels can transfer,
ordered folds preserve natural internal readout, content-aware recursive routes
can work, and frozen internal prefixes can drive limited surface decoding.  It
does not assume those controlled results already solve WMT.

## Frontier Definition

For a source sentence tree `T(x)`, a `K`-node frontier is:

\[
F_K(T)=\{v_1,\ldots,v_K\}
\]

such that the nodes do not overlap and their leaves cover the source exactly:

\[
\bigcup_{v\in F_K}Leaves(v)=Leaves(T)
\]

\[
Leaves(v_i)\cap Leaves(v_j)=\varnothing\quad(i\ne j)
\]

The implementation starts with ordered token leaves and performs adjacent
merges until exactly `K` live subheaps remain.  Those live subheaps are the only
memory visible to the decoder.  There is no all-leaf attention bypass.

## Compared Encoders

All compressed encoders expose exactly `K=4` states to the same autoregressive
Chinese decoder:

1. `learned_frontier`: straight-through Gumbel selection over learned adjacent
   merge scores;
2. `fixed_frontier`: deterministic adjacent balanced/Huffman-like merge order;
3. `random_frontier`: deterministic untrained input-hash merge schedule;
4. `flat_k`: four ordered contiguous pooled vectors without a tree;
5. `leaf_oracle`: an unrestricted flat GRU over all source tokens, reported as
   a capacity upper bound rather than a matched competitor.

The source length is restricted to more than `K`, so every matched model must
compress.

## Predictions

Primary smoke prediction:

```text
learned_frontier < fixed_frontier NLL
learned_frontier < random_frontier NLL
learned_frontier < flat_k NLL
```

Upgrade gate after smoke:

- five seeds;
- learned frontier improves held-out NLL by at least `0.05` over the strongest
  matched `K=4` baseline;
- 95% confidence interval of the paired seed difference excludes zero;
- resetting or permuting learned merge decisions raises NLL by at least `0.05`;
- learned routes beat an untrained route control, not merely exhibit diversity.

## Falsification

Reject or redesign if fixed/random/flat frontiers match learned frontier, if
learned routes are non-causal under intervention, if performance only returns
when `K` approaches source length, or if gains disappear on held-out lengths.

A pass proves only that TreeHeap substructure is a useful, unavoidable WMT
information carrier under a fixed bandwidth.  It does not prove consciousness,
a persistent world model, semantic Huffman optimality, or Transformer-level MT.

## Smoke Result

The first io run does not pass the primary claim gate.

| Encoder | Test NLL | token-BLEU4 |
|---|---:|---:|
| learned frontier | 6.5988 | 0.226 |
| fixed frontier | 6.6012 | 0.159 |
| random frontier | 6.7020 | 0.185 |
| flat `K=4` | **6.5080** | **0.431** |
| unrestricted leaf oracle | 6.3541 | 0.792 |

The same learned checkpoint worsened from `6.5988` to `6.6085` when its route
was replaced by fixed choices and to `6.6272` under random choices.  This is a
small causal route signal (`+0.0097/+0.0285` NLL), but it is below the registered
`0.05` gate and does not overcome flat four-vector compression.

Decision: main claim not supported at smoke; retain the route intervention as
weak mechanism evidence.  Do not run the five-seed upgrade until compose-state
information preservation is redesigned.
