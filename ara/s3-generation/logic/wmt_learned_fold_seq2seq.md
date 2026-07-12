# WMT Translation-Loss Learned Fold

## Claim

Given real aligned English-Chinese pairs, translation cross entropy can train
not only token values and a surface decoder, but also an adjacent-fold kernel
that chooses which source subheaps to compose.

## Encoder

For the current node sequence `h_1 ... h_n`, every adjacent pair proposes a
parent:

\[
c_i = Compose_\theta(h_i + e_L, h_{i+1} + e_R)
\]

The merge kernel scores each proposal:

\[
s_i = K^{merge}_\theta(h_i,h_{i+1},c_i)
\]

During training, straight-through Gumbel-Softmax selects one adjacent pair.
The forward pass therefore contains one hard merge, while the backward pass
passes translation gradients into all merge scores.  Repeating `n-1` times
produces a concrete binary TreeHeap merge history.

No parse labels, merge labels, POS tags, or syntax trees are provided.  The
only task loss is Chinese sequence cross entropy.

## Prediction

If WMT alignment contains useful structural pressure, learned-fold TreeHeap
should:

1. train without route labels;
2. retain readable generation;
3. beat or match the fixed balanced TreeHeap;
4. suffer a measurable loss when internal nodes are hidden;
5. learn non-trivial, input-dependent merge histories.

## Controls

- fixed balanced TreeHeap;
- flat GRU;
- BoW;
- full, leaf-only, and root-only evaluation from the same learned checkpoint;
- held-out WMT examples and deterministic argmax merge histories.

## Falsification Boundary

The structural claim remains open or is rejected if learned fold matches the
fixed tree and BoW, if full equals leaf-only, or if merge histories collapse to
one position-independent rule.  A positive smoke would still not establish a
persistent world model, semantic Huffman compression, or Transformer-level
translation quality.

## Smoke Result

The 5,000-pair smoke completed on io.

| Encoder | Test NLL | token-BLEU4 |
|---|---:|---:|
| learned adjacent fold | 6.0448 | 0.290 |
| fixed balanced TreeHeap | 6.0205 | 0.734 |
| flat GRU | 5.8792 | **0.926** |
| BoW | **5.8761** | 0.779 |

Learned-fold `full` and `leaf_only` were tied (`6.0448` versus `6.0434` NLL,
both `0.290` token-BLEU4).  Root-only degraded to `7.8445` NLL.  Therefore the
decoder used leaves and bypassed learned internal nodes.

The route audit showed high input dependence: 490 unique routes among 500
sentences, mean normalized choice entropy `0.8632`, and all routes changed
after token shuffle.  This is not sufficient positive evidence because route
diversity can exist without causal downstream use.  The claim is rejected for
the current architecture.  Before another topology experiment, training must
constrain decoder bandwidth so that useful information has to pass through
selected internal subheaps.
