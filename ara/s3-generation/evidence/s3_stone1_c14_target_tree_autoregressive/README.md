# C14 controlled target-TreeHeap experiment

- Claim: `S3-TARGET-TREE-AUTOREGRESSIVE-C14`
- Host: `io`
- Taskd task: `61`
- Date: 2026-07-29
- Code: `../../src/s3_stone1_c14_target_tree_autoregressive.py`
- Machine-readable result: `summary.json`

## Command

```bash
python3 ara/s3-generation/src/s3_stone1_c14_target_tree_autoregressive.py \
  --train-samples 5000 \
  --valid-samples 500 \
  --test-samples 500 \
  --max-scan 100000 \
  --epochs 5 \
  --batch-size 16 \
  --dim 192 \
  --hidden 384 \
  --evidence-dir ara/s3-generation/evidence/s3_stone1_c14_target_tree_autoregressive
```

## Verdict

Partial mechanism support, strong generation claim rejected.

The target TreeHeap is causally used: zeroing all target history costs `4.3857`
NLL and retaining only its root costs `0.7718`. Source root-only read costs
`0.5325`. However, source shuffle costs only `0.0979`, source route mass
collapses to the deepest level, BLEU-4 is `0.2150`, and only 18.4% of outputs
are unique. This proves neither usable translation nor a learned
variable-resolution source protocol.
