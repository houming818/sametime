# WMT Learned-Fold Smoke Result

Host: `io` (RTX 3090, existing 270W limit retained)

Data: 5,000 train / 500 valid / 500 test WMT pairs, source BPE length
3--16 before EOS, five epochs, matched 192-dimensional models.

| Encoder | Test NLL | PPL | token-BLEU4 |
|---|---:|---:|---:|
| learned adjacent fold | 6.0448 | 421.9 | 0.290 |
| fixed balanced TreeHeap | 6.0205 | 411.8 | 0.734 |
| flat GRU | 5.8792 | 357.5 | **0.926** |
| BoW | **5.8761** | **356.4** | 0.779 |

Learned-fold causal ablation from the same checkpoint:

| Read mode | NLL | token-BLEU4 |
|---|---:|---:|
| full leaves + learned internal nodes | 6.0448 | 0.290 |
| leaf-only | 6.0434 | 0.290 |
| root-only | 7.8445 | 0.115 |

`full` and `leaf-only` are effectively identical.  The decoder bypasses the
learned merge states and reads source leaves directly.

The route audit found 490 unique routes across 500 examples, normalized merge
choice entropy `0.8632`, and a route change for every token-shuffled source.
This rules out one fixed merge schedule, but it does not prove useful learned
structure: an untrained content-sensitive scorer can also produce diverse
routes.  Since destroying access to all internal nodes has no cost, the
translation task did not use those routes in this architecture.

Claim status: `rejected at smoke / learned route is non-causal`.

The next architectural correction must remove or limit the leaf-attention
bypass before spending more compute on learned topology.
