# S1 Probability-Residual Corpus Scale A14

A14 isolates corpus exposure while holding the tokenizer, 512 target tokens,
1024 context tokens, sealed test field, split seed, search seed, search budget,
and TreeHeap depth arms fixed.

## Formal Contract

```text
training scales: 50K, 100K, 200K, 500K, 1M valid English examples
depths: 3, 5
sealed test: 100K examples after the 1M training prefix
sealed test pairs: 3,670,609
split seed: 20260924
search seed: 20260929
search proposals: 1024 per arm
taskd jobs: 587-598
```

The formal comparison is in `formal/aggregate/comparison.json`; the compact
table is `formal/aggregate/scale_curve.csv`. Per-arm summaries and complete
search traces are under `formal/runs/`. The 1M depth-3 and depth-5 runs contain
the two full reload-audited embedding checkpoints.

Large `context_counts.pt` files remain on `io` under the same evidence path and
are intentionally excluded from Git. Their hashes and the data-region hashes
are recorded in `formal/counts/manifest.json` and each scale manifest.

## Result

The mechanical contract passes for all ten arms. Depth 3 reaches its lowest
fixed-test NLL at 1M (`5.409086`). Depth 5 reaches its lowest value at 200K
(`5.328784`) and then fluctuates slightly upward. Both endpoint curves improve
relative to 50K, so A14 does not show general corpus-scale degradation.

Context-neighbor stability rises steadily and reaches `0.815` from 500K to 1M,
while label-invariant tree partition F1 remains low (`0.20-0.45`). The current
evidence points to unstable F-search topology before it points to an embedding
capacity collapse. One seed cannot distinguish search variance from a true
depth-5 saturation point.

The `smoke_r1/` directory verifies the builder, fixed-test identity, routing
index, and aggregate pipeline. The earlier 50-line smoke correctly failed its
all-target test-coverage guard and is not part of the formal evidence.
