# ARA Experiment Report Template

Use this template for every training or proof run. The human-readable report
and `summary.json` must describe the same experiment contract.

## 1. Experiment Card

| Field | Value |
|---|---|
| Experiment ID | `<stable-id>` |
| Claim | `<claim-id>` |
| Predict | `<predict-id>` |
| Status | `preregistered / running / supported / partial / not_supported / failed` |
| Author / reviewer | `<names>` |
| Code commit | `<git-sha>` |
| Host / accelerator | `<host, GPU, power policy>` |
| Started / completed | `<ISO-8601>` |

## 2. Question And Falsification

**Question:** State one causal question.

**Prediction:** State the expected metric direction and numeric threshold before
running the experiment.

**Falsification:** State the result that would reject or narrow the claim.

**Boundary:** State what a positive result would still not prove.

## 3. Dataset Card

| Field | Value |
|---|---|
| Source artifact | `<absolute path or public dataset ID>` |
| Source size / rows | `<bytes, rows>` |
| Sampling algorithm / seed | `<algorithm, seed>` |
| Eligibility filter | `<tokenizer, min/max length, language direction>` |
| Train split | `<rows and SHA-256>` |
| Validation split | `<rows and SHA-256>` |
| Test split | `<rows and SHA-256>` |
| Leakage / duplicate control | `<method and count>` |
| Tokenizer | `<path, vocabulary, SHA-256>` |

Never compare NLL across runs whose validation/test hashes differ without
labeling the comparison as non-controlled.

## 4. Variable Contract

| Type | Variables |
|---|---|
| Independent | `<the variable deliberately changed>` |
| Controlled | `<model, seed, steps, optimizer, splits, tokenizer, batch...>` |
| Dependent | `<NLL, PPL, BLEU, latency... with better direction>` |
| Nuisance / known confounders | `<remaining uncontrolled variables>` |

## 5. Model And Training Card

| Field | Value |
|---|---|
| Architecture / parameters | `<name, count>` |
| Initialization seeds | `<seeds>` |
| Optimizer / LR / schedule | `<complete recipe>` |
| Batch / update budget | `<batch, steps, token or sample exposures>` |
| Precision / clipping | `<dtype, AMP, gradient clip>` |
| Validation cadence | `<steps>` |
| Checkpoint selection | `<best validation metric>` |

## 6. Results

| Model | Train unique rows | Reuse factor | Best step | Valid NLL | Test NLL ↓ | PPL ↓ | BLEU ↑ | Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `<model>` | | | | | | | | | |

Also preserve per-step records in `trace.jsonl`; a final table alone is not
sufficient evidence.

## 7. Decision Gates

| Gate | Threshold | Observed | Pass |
|---|---|---|---|
| `<gate-id>` | `<preregistered threshold>` | `<value>` | `true/false` |

Conclude with `supported`, `partial`, `not_supported`, or `failed`. Do not
rewrite thresholds after seeing results.

## 8. Evidence Manifest

Every formal evidence directory should contain:

```text
command.sh
config.json
dataset_manifest.json
trace.jsonl
runs.json
summary.json
REPORT.md
stdout.log
stderr.log
```

Large checkpoints may live on NAS, but the report must include their path and
SHA-256.

## 9. Limitations And Next Decision

List the strongest alternative explanation, the missing control, and the one
next experiment that follows from the result.
