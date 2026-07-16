# Annealed Frontier Pretraining

Complete 64-token context predicts the next 16 tokens; no MASK.

Decision: partial support. Annealing improved root NLL over uniform training
and yielded an ordered frontier curve, but source shuffle and pre-FOLD sibling
swap did not meet the causal gates. See `summary.json` for all results.

Checkpoints: /mnt/nas/ara/s3-generation/evidence/s3_annealed_frontier_pretrain
