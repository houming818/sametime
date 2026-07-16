# S2 Learned-Update Lifting WMT: 200K Scale

This directory contains the scale evidence for the winner selected by the
preregistered 30K attribution experiment.

```text
flat_seq       NLL 4.5419   BLEU-4 10.572
old_recursive  NLL 4.6743   BLEU-4  9.609
learned_update NLL 4.6335   BLEU-4  9.909
```

The learned update improved over the old pump by `0.0408` NLL, missing the
registered `0.05` gate, but closed `30.8%` of the old pump's gap to flat. All
causal structure, multiresolution, closure, finite-gradient, and non-empty
generation gates passed. Decision: `partial`.

Data: deterministic reservoir sample from two million raw WMT-massive rows;
`200K/5K/5K` train/valid/test; English to Chinese; 32K tokenizer; 64-leaf heap;
256 dimensions; five epochs. Runtime was 7,592.61 seconds on `io`.

Git stores the summary, traces, examples, and exact command. Checkpoints are
archived at:

```text
/mnt/nas/ara/s3-generation/evidence/s2_adaptive_lifting_wmt_200k/
```
