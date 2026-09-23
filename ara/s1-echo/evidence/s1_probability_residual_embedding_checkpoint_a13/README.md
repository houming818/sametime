# S1 Probability-Residual Embedding Checkpoints A13

A13 converts the reproducible A11/A12 F search into loadable embedding
artifacts. The formal runs use the frozen WMT 200K context field, split seed
`20260924`, new search seed `20260928`, and 2048 proposals per arm.

## Artifacts

```text
formal/depth_3_seed20260928/embedding_checkpoint.pt  4,615,253 bytes
formal/depth_5_seed20260928/embedding_checkpoint.pt  5,804,693 bytes
```

Both contain the SentencePiece identity, target/context token IDs and pieces,
F axes and thresholds, token leaf/path/margin coordinates, node probability
prototypes, child-parent residuals, and smoothed leaf probabilities.

Depth 3 improves sealed-test NLL by `0.05030662`; depth 5 improves it by
`0.05501387`. Both retain 100% leaf utilization, beat random routing, reload to
identical assignments, and reproduce test NLL with zero measured error.

The checkpoints are frozen embedding inputs. They do not yet establish
semantic readout, generation, translation, or Decoder compatibility.

Task logs are stored under `taskd_logs/`. The two-step smoke verifies the file
contract only and is excluded from the formal Claim.
