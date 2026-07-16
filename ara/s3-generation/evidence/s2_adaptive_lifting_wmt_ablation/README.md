# S2 Adaptive Lifting WMT: 30K Attribution

The preregistered 30K ablation separated learned update from alternating
orientation before the larger run.

```text
old_recursive       NLL 6.1488
learned_update      NLL 6.0596   gain +0.0892
alternate_fixed     NLL 6.2740   gain -0.1251
adaptive_alternate  NLL 6.1094   gain +0.0394
```

All pumps remained numerically closed and had finite gradients. The learned
update delta RMS was `0.1477`, so its parameterized update did not remain at
the fixed-half initialization.

Decision: `partial`. Learned update passed; alternating orientation was
harmful; the combination failed the registered non-antagonism gate. The 200K
winner-selection rule therefore advances `learned_update`, not the combined
kernel.

Exact checkpoints are archived at:

```text
/mnt/nas/ara/s3-generation/evidence/s2_adaptive_lifting_wmt_ablation/
```

Git stores `summary.json`, `trace.jsonl`, `examples.json`, and `command.sh`.
