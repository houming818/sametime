# C15 budgeted conditional index evidence

- Claim: `S3-BUDGETED-CONDITIONAL-INDEX-C15`
- Host: `io`
- Taskd task: `62`
- Date: 2026-07-29
- Code: `../../src/s3_stone1_c15_budgeted_index_probe.py`
- Machine-readable result: `summary.json`

## Command

```bash
python3 ara/s3-generation/src/s3_stone1_c15_budgeted_index_probe.py \
  --evidence-dir ara/s3-generation/evidence/s3_stone1_c15_budgeted_index
```

## Verdict

Supported controlled index toy; natural-language and compression claims remain
open. Exact path NLL is layout invariant and the exact count tree is larger
than the flat table. Under a ten-node search budget, discrete leaf placement
optimization improves independent-test Hit@3 from a random-layout mean of
`0.6921` to `0.8125`, matching flat exact Top-3. The gain falls when shared
conditional structure is shuffled.
