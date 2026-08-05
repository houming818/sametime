# S3 TreeHeap Canonical View Dose C06

Claim: `S3-TREEHEAP-CANONICAL-DOSE-C06`

Host: `io` RTX 3090

Date completed: 2026-08-05

Code commits:

```text
4fca6e1  preregister claim and initial implementation
f6e9991  combine base/replay loss into one token-weighted optimizer step
```

Taskd:

```text
105       A_base
110       install corrected implementation
111       corrected four-arm smoke
112-114   S, BB and BI formal arms
115       aggregate, gate decision and email notification
116       evidence packaging
```

The earlier task 104 smoke is rejected implementation evidence: it exposed a
second-optimizer-step confound before formal additive arms ran. Tasks 106--109
were cancelled. Corrected task 111 passed before tasks 112--114 started.

## Result

```text
native-dose recovery                  0.390900  FAIL
JS specificity BB-BI                  0.137528  PASS
equal-compute native cost BI-BB       0.002812  PASS
structural/source gate                          PASS
dose/replay match                               PASS
screening status                     not confirmed
```

Canonical aggregate: `summary.json`.

Each arm directory contains `summary.json`, `trace.jsonl`, and fixed Dreams.
The result is one seed and cannot upgrade the claim without a new registration.
