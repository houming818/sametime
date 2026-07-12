# Large Checkpoint Artifacts

The five checkpoints remain on `io` and are intentionally not stored in Git.

Remote directory:

```text
/home/nio/log/holds/SameTime/ara/s3-generation/evidence/s3_wmt_frontier_smoke/
```

| File | SHA-256 |
|---|---|
| `checkpoint_best_fixed_frontier.pt` | `8c90d4be747d01b6f41d9914db0b515597489b740bbf5eb4a4fda489b98a56e6` |
| `checkpoint_best_flat_k.pt` | `1b0ff53e11f0d8489abbed96a29d4fd96535de748a09cf83b984252933090cb7` |
| `checkpoint_best_leaf_oracle.pt` | `5e60187b7a087bdb4e9c985f71dfe89bbb159387c5e9151d329203dcaf6ad331` |
| `checkpoint_best_learned_frontier.pt` | `d0ce05b71035ccd414dfad4135ddab2a2866f3b5cc4da7de7f0df90c458fc69a` |
| `checkpoint_best_random_frontier.pt` | `6aa6dfc912ddb22a09faa436055a58a599a8dd93d3c7156d9df706c3deeff556` |

`intervention.json` was produced from the learned-frontier checkpoint above.
