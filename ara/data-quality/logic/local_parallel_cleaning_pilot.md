# Local parallel-corpus cleaning pilot

Date: 2026-08-17

Status: 100K numeric pilot and model-assisted cross-audit complete

## Objective

Calibrate a local multilingual scorer before filtering the 14,170,275-row WMT
massive corpus. This pilot never deletes or rewrites source data. It emits
candidate `clean`, `gray`, and `reject` manifests for review.

## Data and controls

- Deterministically sample 100,000 non-empty TSV pairs.
- Score the observed pair with `BAAI/bge-reranker-v2-m3`.
- Construct a length-bucketed shuffled target for every sampled source and
  score it as an obvious mismatch control.
- Record deterministic flags for mojibake, language direction, length ratio,
  exact duplicates, and numeric mismatch.
- Export low-, middle-, and high-score examples for manual review.

Observed corpus pairs are not treated as ground-truth positives. Synthetic
shuffles measure only gross mismatch separation; they cannot establish real
reject precision.

## Predict

The local scorer should assign higher scores to observed pairs than shuffled
controls, with ROC AUC at least 0.90 and useful throughput on the RTX 3090. A
candidate reject threshold may be proposed, but no row may be removed until a
human-reviewed calibration set establishes at least 95% reject precision.

## Gate

Proceed to a one-million-row shadow run only if:

1. the run is finite and reproducible;
2. shuffled-control ROC AUC is at least 0.90;
3. manual review confirms reject precision at least 95%;
4. hard cases involving numbers, names, and negation are separately audited;
5. projected full-corpus runtime is acceptable.

## Result

The formal 100,000-row run completed on `io` in 628.30 seconds. It scored both
the observed pair and one length-bucketed shuffled control for each row.

| Measurement | Result |
|---|---:|
| sampled rows | 100,000 |
| corpus valid rows | 14,170,275 |
| shuffled-control ROC AUC | 0.9779678884 |
| observed median score | 0.9860979319 |
| shuffled median score | 0.0000164425 |
| observed 10th percentile | 0.0011972112 |
| scored pairs per second | 318.32 |
| wall time | 10.47 minutes |
| peak CUDA memory | 2,159,980,544 bytes |

The preregistered numeric AUC gate passed. At the unvalidated percentile
thresholds, the manifest contains 89,991 clean candidates, 8,788 gray rows,
and 1,221 reject candidates. These are triage labels, not deletion labels.

The lowest reviewed examples contain clear semantic mismatches, such as a
Chinese sentence about diabetes paired with an English sentence about atopic
dermatitis. The highest examples are close translations. This is encouraging
qualitative evidence, but the review file deliberately remains unlabeled; no
human precision or recall number has yet been established.

## Interpretation

`AUC = 0.978` answers a narrow question: the reranker usually ranks an observed
corpus pair above an artificially shuffled pair. It does not answer whether
every naturally occurring low-score pair is unusable. Legitimate translations
with names, numbers, partial translation, unusual terminology, or substantial
paraphrase can also receive low scores.

Therefore this pilot supports using the reranker as a local triage instrument.
It does not yet support deleting 1.221% of the corpus or starting the full
14.17M-row destructive filter.

## Next gate

1. Blind-label the 200 reject candidates and a boundary-focused gray sample.
2. Report reject precision with a confidence interval and audit names, numbers,
   and negation separately.
3. Only if reject precision is at least 95%, run a one-million-row shadow pass.
4. Keep the original TSV immutable; any accepted filter remains a manifest.

Evidence:

- `ara/data-quality/evidence/local_parallel_cleaning_pilot/formal_seed14101/summary.json`
- `ara/data-quality/evidence/local_parallel_cleaning_pilot/formal_seed14101/manifest.jsonl`
- `ara/data-quality/evidence/local_parallel_cleaning_pilot/formal_seed14101/manual_review.jsonl`
- `ara/data-quality/evidence/local_parallel_cleaning_pilot/formal_seed14101/human_review_blind.csv`
- `ara/data-quality/evidence/local_parallel_cleaning_pilot/formal_seed14101/human_review_key.csv`

The blind CSV contains 400 rows: 200 samples distributed across the reject
bucket, 100 at the lower gray boundary, 50 at the upper gray boundary, and 50
at the lower clean boundary. Reviewers should fill only `review_label` and
`reviewer_notes` without opening the key CSV. Allowed labels are
`correct`, `partial`, `mismatch`, and `uncertain`. The key is joined by
`review_id` only after labeling.

## Model-assisted cross-audit

Because the project owner does not claim professional Chinese-English review
ability, the 400 rows were blindly reviewed by `deepseek-chat`. Codex then
reviewed every low-confidence result, every non-`mismatch` label in the low
score strata, and every non-`correct` label in a supplemental high-score
control.

| Audit slice | DeepSeek result |
|---|---:|
| stratified reject candidates | 200 / 200 mismatch |
| lower gray boundary | 97 mismatch / 3 partial |
| upper gray boundary | 46 mismatch / 4 partial |
| lower clean boundary | 47 mismatch / 3 partial |
| extreme high-score controls | 96 correct / 4 partial / 0 mismatch |

The strict reject precision estimate is 100% on 200 sampled reject candidates;
the 95% Wilson lower bound is approximately 98.12%. Codex's cross-review found
that the low-score `partial` cases shared only superficial topic words and
should still be treated as translation mismatches. The four high-score
`partial` decisions show that the judge is somewhat conservative; none was
classified as a mismatch.

This passes the precision gate for a **non-destructive one-million-row shadow
run**. It does not authorize deletion. The labels are model-assisted evidence,
not an independent professional bilingual gold set, so all filtering must
remain reversible and must later be tested by matched downstream training.

Additional evidence:

- `deepseek_review.jsonl`
- `deepseek_review_merged.csv`
- `human_review_high_control_blind.csv`
- `deepseek_high_control_review.jsonl`
- `cross_audit_summary.json`
