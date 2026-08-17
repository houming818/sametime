# SameTime / ARA TODO

Updated: 2026-08-17

## NEXT-001: Local bilingual corpus-cleaning pilot

Priority: **active data-quality calibration**

Status: 100K numeric pilot and model-assisted cross-audit complete; 1M
non-destructive shadow run is now permitted.

### Objective

Determine whether io's RTX 3090 can filter the current
`/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv` corpus locally, so
commercial APIs are needed only for uncertain pairs.

### Pipeline

1. CPU deterministic filters: UTF-8/mojibake, empty fields, language direction,
   exact/near duplicates, length ratio, boilerplate, and split leakage.
2. Primary multilingual pair scorer: `BAAI/bge-reranker-v2-m3` (0.6B).
3. Secondary gray-zone scorer: `Qwen/Qwen3-Reranker-0.6B` only if the primary
   pilot is useful.
4. Commercial API or human review only for disagreements and low-confidence
   rows. Models must judge pairs, not rewrite reference translations.

### First experiment

- Run a 100,000-pair pilot on io after the GPU queue becomes idle.
- Include confirmed positives, random length-matched negatives, number/name/
  negation perturbations, and a manually reviewed sample.
- Record throughput, VRAM, wall time, score distributions, AUROC/precision/
  recall at candidate thresholds, and representative false accepts/rejects.
- Produce `clean`, `gray`, and `reject` manifests without deleting source data.
- Hash the input rows, model revisions, code, and output manifests.

### Gate

Do not run the 14.17M-pair corpus until the pilot demonstrates a calibrated
high-confidence reject precision acceptable for training-data removal. If the
reranker cannot distinguish hard mismatches, keep it only as a triage feature
and send the gray zone to API/human review.

### 100K result

- Task: io taskd `210`
- Runtime: 628.30 seconds
- Shuffled-control ROC AUC: 0.9779678884 (numeric gate passed)
- Throughput: 318.32 scored pairs/second
- Peak CUDA memory: 2,159,980,544 bytes
- Unvalidated triage: 89,991 clean / 8,788 gray / 1,221 reject
- Decision: do not delete data and do not start the 1M pass until manual reject
  precision is measured at the preregistered 95% gate.

### Cross-audit result

- DeepSeek blind review: 200/200 stratified reject candidates were mismatches.
- Approximate 95% Wilson lower bound: 98.12%.
- High-score control: 96 correct / 4 partial / 0 mismatch.
- Codex reviewed all low-confidence and non-default labels; low-score partials
  were still unusable as faithful translations.
- Decision: proceed to a reversible 1M shadow manifest; source deletion remains
  forbidden until downstream matched-training evidence exists.
