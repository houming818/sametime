# S1 explicit echo encoder/decoder probe

Claim: `S1-ECHO-ED-C01`

This proof implements an explicit TreeHeap echo encoder and decoder.

## Design

- learned parameters: `0`
- encoder: `token sequence -> ordered TreeHeap leaves -> internal NodeState summaries`
- decoder: `root length + path-addressed leaf/subheap reads -> token sequence`
- uses target heap in decoder: `False`

## Metrics

| Split | Sequence exact | Leaf acc | Subheap exact | Summary exact |
|---|---:|---:|---:|---:|
| train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ood | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Boundary

This proves the hard/algebraic echo interface closes. It does not prove
translation, learned semantics, compression, or noisy correction.
