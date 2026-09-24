# A15 evidence index

Claim: `S1-F-HANDOFF-A15-C01`

## Runs

- `smoke/summary.json`: route-only 30-step pipeline smoke, task 599.
- `smoke_calibration/summary.json`: route-only convergence calibration, task 600.
- `smoke_r1/summary.json`: corrected probability-field plus route smoke, task 601.
- `formal/summary.json`: two candidates, three seeds, five arms, task 602.

The first two smokes are retained because they falsified treating the compact
route summary as the complete embedding. They are not formal evidence.

## Formal command contract

```text
python3 ara/s1-echo/src/s1_frozen_embedding_handoff_gate_a15.py
  --checkpoint A14-1M-D3=ara/s1-echo/evidence/s1_probability_residual_corpus_scale_a14/formal/runs/train_1000000_depth_3/embedding_checkpoint.pt
  --checkpoint A13-200K-D5=ara/s1-echo/evidence/s1_probability_residual_embedding_checkpoint_a13/formal/depth_5_seed20260928/embedding_checkpoint.pt
  --data /home/nio/datasets/wmt_massive/train.massive.zh-en.tsv
  --spm-model /home/nio/datasets/wmt_massive/sp_bpe_massive.model
  --out ara/s1-echo/evidence/s1_frozen_embedding_handoff_gate_a15/formal
  --seeds 20260930,20260931,20260932
  --start-valid-line 1100000
  --scan-valid-lines 250000
  --train-sequences 20000
  --test-sequences 5000
  --steps 600
  --read-steps 500
  --batch 256
  --hidden 64
  --device cuda
```

The full machine-readable command, checkpoint hashes, tokenizer hash, data
region hash, tensor hashes, traces, intervention results, and gates are embedded
in `formal/summary.json`.

