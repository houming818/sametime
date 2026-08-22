# Local Qwen judge smoke

Date: 2026-08-22

Status: completed / all registered gates passed

Claim: `NIO-LOCAL-JUDGE-C01`

## Question

Can an official local `Qwen/Qwen3-8B` checkpoint run reproducibly on io's
24 GiB RTX 3090 and produce auditable structured judgments for the mono, QA
and medical calibration samples?

This judge does not rewrite source data and does not certify factual or
medical correctness. It is a third-stage calibration observer behind
deterministic checks and the BGE relation scorer.

## Fixed configuration

```text
checkpoint: Qwen/Qwen3-8B
precision: bfloat16
device: cuda
decoding: greedy (do_sample=false, temperature omitted)
thinking: disabled
maximum input: 4096 tokens
maximum output: 192 tokens
```

The checkpoint repository revision, local file hashes, runtime versions and
GPU state must be recorded.

## Sample

Use 60 records total: 20 each from the completed seed-15101 `mono`, `qa` and
`medical` scorer-smoke review manifests. Within each family, select records
from the low, middle and high BGE score regions. Selection is deterministic
and the source text remains byte-for-byte unchanged.

The required JSON fields are:

```json
{
  "relation": "matched|partial|mismatch|uncertain",
  "text_quality": "usable|noisy|corrupt|uncertain",
  "domain_risk": "ordinary|medical_unverified",
  "reason_code": "short_machine_readable_code",
  "reason_zh": "short Chinese explanation"
}
```

## Predictions and gates

```text
P0 checkpoint downloads completely and its manifest is recorded
P1 BF16 load and inference finish on one RTX 3090 without OOM or non-finite values
P2 at least 57/60 generations parse as exactly one JSON object
P3 repeated inference over six fixed records is byte-identical
P4 every medical record is marked medical_unverified
P5 source text and the BGE evidence remain unchanged
```

Failure of P0 or P1 stops the route. Failure of P2 or P3 means the model is
not yet an auditable judge and must not score a larger sample. Passing this
smoke authorizes only a labeled calibration pilot, not full-corpus deletion or
automatic medical approval.

## Result

The final warning-free run was taskd task `273`.

| Measure | Result |
|---|---:|
| Samples | 60 (20 mono, 20 QA, 20 medical) |
| Exactly parseable JSON | 60/60 |
| Six repeated generations | 6/6 byte-identical |
| Medical risk label | 20/20 `medical_unverified` |
| Source review hashes unchanged | yes |
| Peak allocated GPU memory | 16,637,488,128 bytes (15.50 GiB) |
| End-to-end runtime | 109.48 seconds |

The model did not merely reproduce the BGE ordering. It marked some
high-BGE records as mismatches or noisy, including repetitive instruction
answers and a repeated adjacent-text fragment. It also retained some
low-BGE long mono spans as matched or partial when their continuation was
visible. This is useful disagreement for calibration, not proof that Qwen is
the correct judge.

Observed limitations:

- `relation` boundaries between `partial` and `mismatch` are not yet
  calibrated against an external label set.
- Qwen may call a medically relevant but indirect answer a mismatch. That is
  a policy judgment, not a factual determination.
- The sample was deliberately score-stratified and is not an estimate of the
  corpus-wide acceptance rate.

Decision: `Qwen/Qwen3-8B` is technically suitable for the next small labeled
calibration pilot. It is not authorized to rewrite source rows or approve
medical content, and no full-corpus LLM judgment is scheduled from this result.

## Evidence target

```text
ara/data-quality/evidence/local_qwen_judge/qwen3_8b_smoke_seed15102/
```
