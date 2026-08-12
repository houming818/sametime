# C10: Pretrain-to-Task Posterior-Collapse Pipeline

Date: 2026-08-11

Claim: `S3-TREEHEAP-PRETRAIN-POSTERIOR-C10`

Status: pilot completed; matched task transfer is supported, posterior superiority is weakly supported, and multi-resolution READ is rejected for this checkpoint.

## 1. Motivation

C09 tested whether a frozen upper `H_state` was easy for a fresh Decoder to
read. That is a representation-transfer question, not a Bayesian-collapse
test. Aggregate single-reference NLL cannot tell whether the model assigned
probability to several valid continuations or what it actually generated.

C10 establishes one reproducible pipeline:

```text
data build
-> raw-text pretrain
-> matched task train
-> frozen posterior/collapse proof
-> structural interventions
-> report
```

The pipeline separates three roles. Pretraining estimates a broad language
prior, task training teaches how that prior is used for WMT, and the proof
stage inspects probability movement and actual collapse without updating any
parameter.

## 2. Registered claim

Under one unchanged Butterfly/FOLD/READ architecture, raw next-span
pretraining should produce context-conditioned next-token probability buckets
that predict held-out empirical continuation sets better than a unigram prior.
The same pretrained state should then improve a matched WMT task run over the
same initialization trained only on WMT. Both effects must depend causally on
the source and on native TreeHeap communication or pairing.

This is a joint pipeline claim. A lower pretraining NLL alone is insufficient.

## 3. Frozen architecture

Both pretraining and task training use exactly the same model state:

```text
token WRITE
-> dynamic-width XOR Butterfly communication
-> learned lifting FOLD
-> root plus addressed details
-> recursive READ
-> shared Decoder
-> token probability bucket
```

Pilot contract:

| Item | Value |
|---|---:|
| Vocabulary | existing 32K bilingual SentencePiece plus PAD/direction slots |
| State dimension | 256 |
| Hidden dimension | 256 |
| Maximum leaves | 256 |
| Butterfly scale | 0.25 |
| Numeric mode | FP32 |
| Host | `io`, existing 270 W power limit unchanged |

No active lexical `BLANK` is inserted into the Butterfly. A pretraining source
is a fully observed natural prefix; its target is the following natural span.

## 4. Stage A: raw-text pretraining

The stream uses only training partitions from:

- news2016zh, wiki_zh and webtext2019zh;
- the Chinese and English monolingual sides of WMT Massive, without pair labels.

Validation/test documents and WMT validation/test partitions are excluded.
Context width is sampled from `4/8/16/32/64/128`; the following `32` pieces are
the target. The source is complete at its selected width. The short widths
make the evidence used by the posterior proof an in-distribution model input.
The only optimization loss is token cross-entropy:

```text
L_pre = -sum_t log P(x_t | H_context, x_<t)
```

This loss supplies gradients. It is not itself the proof of a useful prior.

The random initial state is saved before the first update. The best checkpoint
is selected on held-out continuation NLL, while posterior/collapse reports are
descriptive until the frozen proof stage.

## 5. Stage B: matched WMT task training

Two arms consume the same WMT rows, direction assignments, batches, optimizer
updates and validation/test sets:

| Arm | Initialization |
|---|---|
| `PT` | Stage-A best pretrained state |
| `SC` | exact pre-Stage-A random state |

Both train all parameters. No architecture or parameter-count difference is
allowed. This isolates initialization transfer rather than extra task data or
extra model capacity.

## 6. Stage C: posterior and collapse proof

An independent held-out raw-text stream constructs empirical next-token
distributions for repeated exact contexts of lengths `4/8/16`:

```text
Q(token | evidence)
```

For each selected context, the frozen model reports:

- complete top-20 model probabilities;
- probability mass assigned to the empirical candidate set;
- JS divergence between model and empirical buckets, with an OTHER bucket;
- empirical-prior JS as a matched unigram baseline;
- greedy top-1 collapse and whether it belongs to the empirical set;
- top-p samples and a bounded generated continuation;
- entropy and unique-collapse rate.

The proof also records human-readable evidence ladders, but manual examples do
not contribute to registered gates.

## 7. Structural interventions

The frozen pretraining proof is repeated with:

1. source rows rolled across samples;
2. Butterfly communication switched to `identity` at runtime;
3. right children reassigned across samples at one pre-FOLD pairing depth.

The old whole-tree mirror is excluded because synchronously mirroring nodes and
masks is an automorphism of the current permutation-equivariant READ.

## 8. Registered predictions

### P0: pipeline integrity

- identical architecture and parameter count in all stages;
- `PT` parent hash equals the Stage-A best hash;
- `SC` parent hash equals the saved initial hash;
- identical WMT stream hashes and scored-token counts;
- finite gradients, atomic checkpoint reload and no raw target leakage.

Any failure invalidates the run.

### P1: a conditional prior forms

On the frozen Stage-A checkpoint:

- held-out NLL improves by at least `0.20` from initialization;
- mean model-to-empirical JS is at least `0.02` lower than unigram-to-empirical
  JS;
- empirical candidate-set mass exceeds unigram candidate mass by at least
  `0.02`;
- greedy collapse lies in the empirical candidate set for at least 20% more
  contexts than unigram top-1;
- at least 25% of selected contexts have different greedy collapses.

### P2: pretraining transfers to the task

After a matched WMT budget, `PT` must beat `SC` by at least `0.02` validation
NLL and may not reduce chrF or token-BLEU4. Actual translations from both arms
must be saved.

### P3: the posterior depends on TreeHeap computation

For the Stage-A checkpoint, wrong-source, runtime-identity and pre-FOLD
pair-break interventions must each worsen empirical JS by at least `0.01` or
reduce empirical candidate mass by at least `0.01`. Wrong-source must also
change greedy collapse on at least 25% of contexts.

### P4: actual collapse remains usable

The frozen native model must produce non-empty bounded generations for at least
95% of contexts, severe adjacent repetition must remain below 25%, and no one
greedy token may occupy more than 50% of selected contexts.

## 9. Execution ladder

```text
smoke
  20 pretrain updates
  20 PT + 20 SC task updates
  16 posterior contexts
  code-validity only

pilot
  100M pretraining target tokens
  matched limited WMT task budget
  >=256 posterior contexts
  one seed, mechanism screen

formal
  separately registered only if pilot passes P0-P4
```

Smoke success does not support the claim. The pilot may reject any or all
predictions without being rerun under revised thresholds.

## 10. Evidence contract

```text
ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/
  config.json
  environment.json
  data_manifest.json
  initial_state.json
  pretrain/{trace.jsonl,summary.json,checkpoint_best.pt}
  task/{PT,SC}/{trace.jsonl,summary.json,checkpoint_best.pt}
  proof/{candidate_bank.json,posterior_rows.jsonl,summary.json}
  report.json
  README.md
```

Every stage records its parent checkpoint SHA-256, tokenizer/data hashes,
optimizer budget, processed examples/tokens, GPU time and exact command.

## 11. Boundaries

A positive result would support a learned, context-conditioned probability
protocol and matched task transfer in this finite pipeline. It would not prove
consciousness, semantic understanding, human Bayesian optimality, world-model
completeness, architecture superiority or product readiness.

## 12. Smoke result

The registered code-validity smoke ran on `io` as taskd job `156` on
2026-08-11. It used 20 pretraining updates, 20 matched task updates per arm and
16 posterior contexts. All nine P0 integrity checks passed, including exact
parent-state hashes, one shared task-stream hash, finite gradients and strict
checkpoint reloads.

| Observation | Smoke value | Interpretation |
|---|---:|---|
| Pretrain initial -> best valid NLL | `10.8661 -> 9.1168` | optimization path works |
| PT task valid NLL | `9.0647` | too few updates for transfer evidence |
| SC task valid NLL | `9.0430` | PT is `0.0217` worse in this smoke |
| Native model -> empirical JS | `0.6750` | worse than the unigram smoke baseline |
| Unigram -> empirical JS | `0.6136` | stronger than the 20-step model |
| Native empirical-candidate mass | `0.00857` | below unigram `0.06615` |
| Non-empty generation | `1.0` | generation code runs |
| Severe adjacent repetition | `1.0` | every generation collapsed to repetition |
| Maximum greedy-token share | `1.0` | all contexts selected the same token |

These are expected negative quality observations from a 20-update run. They do
not reject C10 and cannot support it. They verify that the later pilot will
measure real failure modes rather than only aggregate NLL.

A separate batch-32 timing run, taskd job `157`, executed 200 pretraining
updates in `45.13` seconds, about `4.5K` target pieces/second. A 100M-target-
piece pretraining pilot therefore costs about `6.1` GPU hours at the measured
rate. Matched PT/SC task training and proof raise the first complete pilot
estimate to `9-12` GPU hours.

The timing run also exposed two pre-pilot repairs:

1. Cycling 960 rows overfit after the best validation point near step 100, so
   pilot pretraining must stream fresh corpus windows instead of repeatedly
   cycling a small in-memory table.
2. Smoke posterior contexts were topically clustered, while empirical buckets
   used only a 2/4/8-piece suffix although the model observed a 128-piece
   prefix. Before the pilot, the contract was therefore amended: pretraining
   includes 4/8/16-piece inputs, proof contexts are stratified across source
   documents, and the model receives exactly the same 4/8/16-piece evidence
   used to construct each empirical bucket. This amendment was made before any
   pilot result exists.

Evidence: `evidence/s3_pretrain_task_posterior_pipeline/smoke/`.

After the amendment, taskd `158` validated the matched 4/8/16-piece posterior
path, and taskd `159` validated the pilot-only fresh-window streaming branch.
Both exited successfully and passed P0. Their 20-step model quality remains
non-evidence. Metadata is stored in `smoke_v2/` and `fresh_stream_smoke/`; the
large reloadable checkpoints remain on `io`.

## 13. Pilot result and post-hoc structural audits

The registered pilot completed on `io` with 100,000,768 pretraining target
pieces and two matched 25,000-step WMT arms. PT and SC consumed the identical
20,198,612-piece task stream.

| Metric | PT | SC | PT - SC |
|---|---:|---:|---:|
| WMT test NLL | `5.403696` | `6.291975` | `-0.888279` |
| token BLEU4 | `5.065443` | `1.171101` | `+3.894342` |

The posterior audit found a native empirical JS of `0.609404`, compared with
`0.641004` for the unigram baseline. Native empirical-candidate mass was
`0.062018`, compared with `0.033277` for unigram. Wrong-source intervention
changed the greedy token on `78.64%` of contexts. These measurements support a
source-conditioned signal, but the absolute posterior quality remains weak.

The pilot also exposed a structural failure not anticipated by the parent
claim. Native READ delivered essentially all mass to the leaf level. On 256
held-out rows, native NLL (`5.299829634`) matched forced-leaf NLL
(`5.299829704`). Runtime Identity and depth-0 pair breaking still damaged NLL
by `0.360457` and `0.391156`, respectively. Therefore Butterfly/FOLD remains
causal, while the learned STOP protocol has collapsed to leaf resolution.

Two post-hoc diagnostics refined that result:

1. `path_shape_audit.json` rejected a single linked-path explanation. Top-1
   leaf mass was only `0.210366`; Top-1 truncation increased NLL by `4.004781`.
   Yet uniform leaf pooling increased NLL by only `0.027111`, so a strong
   semantic tree-index interpretation is also unsupported.
2. `observer_resolution_stop_smoke.json` showed that a finite observer
   threshold can prune numerical tails. `epsilon=0.001` reduced visited nodes
   by `8.24%` for `+0.001791` NLL; `epsilon=0.003` reduced them by `10.11%` for
   `+0.010178` NLL. More than `99.93%` of mass still reached leaves, so finite
   resolution does not repair the dominant learned collapse.

Final claim split:

```text
matched pretrain -> WMT transfer: supported in this pilot
context-conditioned posterior signal: weakly supported
Butterfly/FOLD causal participation: supported
learned multi-resolution STOP/READ: rejected for this checkpoint
product readiness or semantic reasoning: not supported
```

Evidence: `evidence/s3_pretrain_task_posterior_pipeline/pilot_seed10101/`.
