# S3 World TreeHeap Analogy Smoke

Host: `io`, RTX 3090. Seed: `71503`.

The model trained for 6,000 steps on a finite symbolic world and evaluated
4,096 held-out hash-blocked combinations. TreeHeap and flat Transformer both
reached sequence exact `1.0`; deterministic lexical replacement reached
`0.5466`.

The full World TreeHeap claim failed: zeroing W retained token accuracy
`0.9984`, address shuffling retained `0.9867`, and route entropy collapsed near
zero. Preserve the narrower finite-analogy representability result only.

`summary.json` contains metrics and readable A/B/C/D outputs. `trace.jsonl`
contains optimization and route-collapse history. `checkpoint.pt` is a toy
evidence artifact, not a language-model checkpoint.
