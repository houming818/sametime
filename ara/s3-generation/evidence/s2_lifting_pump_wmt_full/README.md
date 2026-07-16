# S2 Lifting Pump WMT Full

This is the full real-data evidence for `S2-LIFT-WMT-C01`.

- Host: `io`, NVIDIA RTX 3090, FP32
- Data: WMT English-to-Chinese, `27K/2K/2K` train/valid/test
- Model: 256-dimensional lifting TreeHeap, ten epochs
- Selection: lowest validation NLL
- Runtime: 1398.49 seconds
- Decision: `supported_full` as a mechanism claim

The recursive model reached test NLL `5.0903`. It beat root-only (`5.4337`)
and full UNFOLD (`5.1342`), but did not beat the flat sequence baseline
(`4.8103`). All eight preregistered mechanism gates passed.

Important causal deltas:

```text
source shuffle     +1.4450 NLL
root shuffle       +1.7204 NLL
detail shuffles    +0.1171, +0.1422, +0.2402, +0.4960, +0.4857
pair breaks        +0.4807, +0.4414, +0.4107, +0.4354, +0.2105
force root         +1.1773 NLL
force leaves       +0.6236 NLL
closure MSE         1.73e-14
```

Files:

- `summary.json`: configuration, metrics, interventions, gates, and boundary
- `trace.jsonl`: epoch traces for all five models
- `examples.json`: held-out greedy generations
- `command.sh`: exact rerun command

The five checkpoints are not stored in Git. They are archived with this
metadata at:

```text
/mnt/nas/ara/s3-generation/evidence/s2_lifting_pump_wmt_full/
```
