# Lifting Subheap Pretraining Evidence

Preregistered token-only versus address-aligned multi-scale subheap masking.

Checkpoints: /mnt/nas/ara/s3-generation/evidence/s3_lifting_subheap_pretrain_5k

Run: real Chinese news/wiki/web text, WMT Massive 32K tokenizer, 256D,
batch 64, 5,000 updates per curriculum on `io`.

Decision: partial support. Aligned subheap masking slightly beat the matched
random-span control at width 4/8, and generation depended on the source plus
all detail depths. Root and left/right interventions did not pass their causal
gates. See `audit_correction.json` for the corrected matched greedy-accuracy
comparison; `summary.json` retains the original generated record.
