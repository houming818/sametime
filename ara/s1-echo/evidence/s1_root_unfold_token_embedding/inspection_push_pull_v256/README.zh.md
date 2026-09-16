# Push / Pull Root-Unfold 坐标检查

这是 task 505 的描述性重放，不改变 `S1-ROOT-UNFOLD-EMBED-C01`。
合同与 100K/256-token/depth-4 正式点一致，只把强制检查词改为
`push,pull`。

| Piece | Capacity leaf | 4-bit path | Largest leaf masses |
|---|---:|---:|---|
| `push` | 5 | `0101` | leaf 5 `0.24628`, leaf 4 `0.20853`, leaf 2 `0.16134` |
| `pull` | 2 | `0010` | leaf 2 `0.23852`, leaf 11 `0.15672`, leaf 4 `0.10898` |

The probability-vector L1/L2 distances are `0.74017/0.28110`. In the
square-root probability embedding used downstream, cosine similarity is
`0.88946` and Hellinger distance is `0.33247`. The two words therefore share
substantial distributional context but are not assigned the same route.

At the root, right-branch conditional probability is `0.2120` for `push` and
`0.4500` for `pull`; both remain on the left in their strongest path. Inside
that left half, the next right-branch probability is `0.6909` for `push` and
`0.3368` for `pull`, which creates the first hard-path divergence.

This coordinate records corpus-context differences. It does not independently
encode a symbolic `push = opposite(pull)` relation.
