# STONE-1 Butterfly V1 Product Training Contract

Date: 2026-08-05

Status: selector complete / BI replay selected / full-corpus product run active

Milestone: `STONE-1-BUTTERFLY-V1`

Claim family: `S3-TREEHEAP-BUTTERFLY-PRODUCT-C07`

Execution update (2026-08-07): taskd runs 118-126 completed the formal
three-seed selector and run 127 produced the aggregate decision. `BI_replay`
passed all ten registered gates and was selected. After two product-runner
smokes (runs 128-129), full-corpus Phase B started as taskd run 130 on `io`
with release seed 9301, 20% additive Identity replay, atomic recovery every
250,000 raw rows, and the registered wake windows.

## 1. Decision to Make

This contract does not assume that the TreeHeap architecture is final. It asks
whether the supported Native Butterfly path or an additive Identity-replay
variant is the better bounded product recipe for bilingual translation.

The experiment has two phases:

1. a matched, three-seed selector compares Native, extra-Butterfly and
   additive-Identity continuation from one frozen checkpoint;
2. only the selected recipe receives one deterministic full-corpus product
   continuation and artifact audit.

C06 is not silently upgraded. Its preregistered recovery gate failed at
`0.3909`. C06 nevertheless found a strong Pareto candidate: against matched
extra-Butterfly compute, additive Identity reduced cross-view JS by `0.1375`
at only `+0.00281` Native NLL. C07 tests whether this trade-off is stable and
useful to product behavior.

## 2. Claim Boundary

The strongest possible result is:

> On the frozen V1 platform, one selected TreeHeap continuation recipe produces
> a reloadable Chinese-English translation checkpoint with source-dependent,
> low-repetition output, retained TreeHeap causality and registered validation
> quality.

It does not claim state-of-the-art translation, conversation, world knowledge,
general intelligence, compute superiority, or a universal Identity ratio.

## 3. Frozen Starting Artifacts

| Artifact | Path | SHA-256 / value |
|---|---|---|
| Starting checkpoint | `ara/s3-generation/evidence/s3_treeheap_butterfly_bilingual_full/checkpoint_best.pt` | `821ce8123d78817b37ff8f0a68458fd59427a7af555f93c7c87c297f28861c1d` |
| WMT corpus | `/home/nio/datasets/wmt_massive/train.massive.zh-en.tsv` | `3f4a5189a6b2f06a8a928165a69e119d6e0afe71ffece2bbe7c049ecef7a44df` |
| SentencePiece model | `/home/nio/datasets/wmt_massive/sp_bpe_massive.model` | `9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731` |
| Corpus size |  | `2,520,995,022` bytes / `14,170,275` raw pairs |
| Checkpoint state |  | epoch `3`, next line `6,400,000`, `48,811,685` examples, `1,032,196,489` target tokens |
| Best valid mean NLL |  | `3.4223013826` |

The product runner must abort before CUDA initialization if any artifact hash,
size, model key, vocabulary size or split hash differs.

## 4. Platform Contract

| Item | Frozen value |
|---|---|
| Host | `io` |
| GPU | NVIDIA GeForce RTX 3090, 24,576 MiB |
| GPU UUID | `GPU-4c73dc30-0b6f-86fb-f1ae-24f316d8b54c` |
| Power limit | 270 W, enforced by `nvidia-power-limit.service`; never raise it |
| Driver | 580.173.02 |
| PyTorch / CUDA | 2.12.0+cu130 / CUDA 13.0 |
| cuDNN | 92000 |
| SentencePiece | 0.2.1 |
| CPU | AMD Ryzen 7 5700X, 8 cores / 16 threads |
| RAM | 31 GiB, no swap |
| Local free disk at registration | 637 GiB |
| Queue | `nio-taskd`, loopback port 19100, serial GPU execution |

Training reads `/home/nio/datasets`, not `/mnt/nas`. Checkpoints are written to
local disk atomically. Only completed best/final artifacts are copied to NAS.
The GPU watchdog and power limit are infrastructure, not tunable variables.

## 5. Model Contract

| Parameter | Value |
|---|---:|
| Total trainable parameters | 34,448,396 |
| FP32 model state | 131.41 MiB |
| Checkpoint with AdamW state | 394.27 MiB |
| Vocabulary | SentencePiece pieces plus direction/PAD/BOS/EOS controls |
| Maximum TreeHeap leaves | 256 |
| Maximum source/target content | 253 pieces |
| State dimension | 256 |
| Hidden dimension | 256 |
| Butterfly coupling scale | 0.25 |
| Dynamic leaf widths | 32 / 64 / 128 / 256 |
| Batch by width | 64 / 32 / 16 / 8 |
| Numeric mode | FP32; AMP is not introduced in C07 |
| Gradient clip | 1.0 |

The data path remains:

```text
token WRITE
-> dynamic-width XOR Butterfly communication
-> adaptive recursive FOLD
-> H_state
-> recursive READ
-> shared bilingual Decoder
-> token probability bucket
```

No Transformer, flat attention block or external teacher is added.

## 6. Optimizer Integrity

Continuation loads the checkpoint's AdamW moments, then explicitly sets every
optimizer parameter group's learning rate to `2e-4`. A startup assertion writes
the effective learning rate to `environment.json`.

One parent batch produces exactly one `optimizer.step()`. For replay arms, base
and replay losses are combined by non-PAD target-token count before one backward
pass:

$$
L=\frac{n_B L_B+n_R L_R}{n_B+n_R}
$$

This prevents a small replay subset from receiving a second full-strength
update. Update count, Native tokens, replay tokens and mode-specific tokens are
logged independently.

No explicit JS loss is introduced. C07 first tests the already observed
Identity-replay mechanism. Adding `lambda * JS` would be a new claim.

## 7. Phase A: Three-Seed Recipe Selector

### 7.1 Independent data interval

The selector uses raw corpus lines `[6,700,000, 7,700,000)`, which are outside
the C06 interval. Stable hash partitioning still excludes validation and test
rows.

Seeds are `9201`, `9202`, and `9203`. Within a seed, all arms use the same row
order, batch order and deterministic replay subset.

### 7.2 Arms

| Arm | Native Butterfly dose | Added dose | Purpose |
|---|---:|---:|---|
| `N_native` | 100% | none | task-quality baseline |
| `BB_replay` | 100% | 20% Butterfly | equal-compute control |
| `BI_replay` | 100% | 20% Identity | cross-view protocol candidate |

`BB` and `BI` replay exactly the same row IDs and target text. Only the source
coordinate mode differs.

### 7.3 Selector budget

Each arm consumes one million raw lines. Based on C06 throughput, one seed costs
about 9.3 GPU hours: about 1.9 hours for Native and 3.7 hours for each replay
arm. Three serial seeds are budgeted at 28--32 GPU hours.

Wall-clock timeout is only a fault guard: four hours for Native and six hours
for each replay arm. The scientific stop is the registered line/token dose.

### 7.4 Selector gates

`BI_replay` becomes the product recipe only if all gates pass:

```text
mean(JS_BB - JS_BI)                         >= 0.10
minimum per-seed (JS_BB - JS_BI)            >= 0.05
mean Native-NLL cost (NLL_BI - NLL_BB)      <= 0.01
maximum per-seed Native-NLL cost             <= 0.02
mean chrF difference BI - BB                 >= -0.30
mean token BLEU-4 difference BI - BB         >= -0.20
maximum severe-repetition increase           <= 0.02
source-shuffle damage                         >= 1.50 in every seed
adjacent-structure override damage            > 0 in every seed
Butterfly communication delta and gradient   > 0 in every seed
```

There is no hidden weighted score. A candidate either passes every constraint
or it does not. If BI fails, the product recipe is Native. BB never becomes a
product recipe unless it independently improves task quality; its primary role
is to control extra compute.

## 8. Phase B: Full-Corpus Product Continuation

After Phase A is signed, the selected recipe restarts from the frozen starting
checkpoint rather than from whichever selector seed happened to look best.
This prevents selection noise from entering the release artifact.

The run makes one deterministic pass over all `14,170,275` raw rows. Both
directions remain balanced by the existing stable row-hash rule. Scientific
completion is one full pass, not a wall-clock deadline.

Expected cost:

| Selected recipe | Effective dose | Expected wall time | Fault timeout |
|---|---:|---:|---:|
| Native | one full Native pass | 27--32 h | 48 h |
| BI | full Native pass plus 20% Identity replay | 48--55 h | 72 h |

Seed `9301` is the release-training seed. Phase A supplies the three-seed
mechanism check; Phase B is one expensive artifact run and is not relabeled as
three-seed product replication.

## 9. Observation Windows

Evaluation must not wait until the end. The following windows are fixed before
training:

| Window | Raw lines exposed | Purpose | Saved artifacts |
|---|---:|---|---|
| W0 | 0 | exact starting baseline | metrics, core/full Dreams, reload hash |
| W1 | 100,000 | detect immediate collapse or LR error | light metrics, core Dreams |
| W2 | 300,000 | reproduce the C06-scale region | full metrics, full Dreams, checkpoint |
| W3 | 1,000,000 | selector endpoint / early product trend | full metrics, full Dreams, checkpoint |
| W4+ | every 2,000,000 | product growth curve | full metrics, full Dreams, Pareto checkpoint |
| Final | 14,170,275 | locked product decision | test metrics, CLI pack, best/final checkpoints |

Atomic recovery checkpoints are additionally written every 250,000 raw lines.
Only W0/W2/W3/W4/Final checkpoints are retained permanently. At least four
checkpoint roles are tracked independently: best Native NLL, best chrF, best
cross-view JS under quality constraints, and final. Test data is evaluated only
at Final; all wake decisions use validation data.

Every wake records:

```text
Native and Identity NLL by direction and length bucket
cross-view JS by direction and length bucket
token BLEU-4 and chrF
source-shuffle, Identity and adjacent override damage
communication delta RMS and gradient norm
non-empty rate, repetition, distinct outputs and output-length ratio
examples, target tokens, replay tokens, optimizer steps, GPU hours
peak VRAM, temperature/power samples and checkpoint hashes
```

## 10. Dreams Corpus

Dreams are immutable observation probes, never training targets and never used
to choose a batch. C07 uses two files:

```text
dreams_product_v1_core.tsv   32 probes, rendered at every light wake
dreams_product_v1_full.tsv   96 probes, rendered at W0/W2/W3/W4/Final
```

The full set is balanced 48/48 by translation direction and stratified as:

| Category | Count | What it tests |
|---|---:|---|
| simple lexical/compositional | 8 | basic source dependence |
| agent/patient and active/passive | 16 | ordered roles and handedness |
| negation, modality and quantifier scope | 16 | small token, large semantic change |
| temporal and causal direction | 12 | before/after and cause/effect |
| attachment and nested clauses | 12 | long-range structural binding |
| exact entities, numbers, units and dates | 12 | factual retention |
| 33--64 and 65--128 piece composition | 12 | middle/long TreeHeap widths |
| 129--253 piece near-limit inputs | 4 | maximum-depth behavior |
| repetition and generic-answer traps | 4 | conditional collapse |

Each TSV row has:

```text
id<TAB>direction<TAB>category<TAB>source<TAB>reference<TAB>required_facts
```

Preparation rules:

1. write minimal pairs before training, including swapped subject/object,
   positive/negative, before/after and exact-number variants;
2. encode every source and store raw piece count and width bucket in a manifest;
3. scan normalized source and reference strings against the WMT corpus; exact
   matches are replaced or marked contaminated before the hash is frozen;
4. compute and commit SHA-256 for both TSV files and the manifest;
5. never edit the frozen files during a run; new ideas go to
   `dreams_exploratory.tsv` and cannot affect C07;
6. keep outputs from every wake, even when they are bad.

Automatic Dream scoring reports chrF, number/entity retention, polarity and
temporal keywords, repetition, output length and minimal-pair separation.
Human notes are stored separately and cannot overwrite numeric metrics.

## 11. Hard Stops and Scientific Downgrades

Immediate hard stop:

```text
non-finite loss or gradient
checkpoint cannot reload exactly
artifact/data/tokenizer hash mismatch
effective LR differs from 2e-4
more than one optimizer step per parent batch
GPU disappearance or watchdog alert
local free disk below 100 GiB
```

Pause for review after two consecutive full wakes:

```text
Native validation NLL is >0.20 worse than W0
source-shuffle damage falls below 0.50
more than half of core Dreams share one normalized output
severe repetition exceeds 0.30
all non-root structural damage collapses to zero
one translation direction improves while the other persistently degrades
```

These are not opportunities to tune the registered run in place. Resume only
after preserving evidence and registering a repair.

## 12. Product Gates

The full artifact is signed only when:

```text
mean test NLL                              <= 3.45
valid chrF                                 improves over W0 by >= 0.50
valid token BLEU-4                         does not fall by > 0.20
non-empty generation                       = 1.00
severe repetition                          <= 0.10
source-shuffle damage                      >= 1.50
Native Butterfly runtime override damage  >= 1.00
all four length regions                    remain source-dependent
reload reproduces fixed greedy token IDs   exactly
CLI task labels                            match bilingual translation objective
```

For BI, the selected cross-view JS gate must also remain satisfied at Final.
If task quality passes but structural gates fail, the artifact is a translation
demo, not TreeHeap product evidence. If structural gates pass but quality fails,
it is a mechanism checkpoint, not a product release.

## 13. Queue and Reliability

```text
preflight hashes + Dreams contamination scan
-> three-arm 30k-line smoke
-> selector seed 9201: N -> BB -> BI
-> selector seed 9202: N -> BB -> BI
-> selector seed 9203: N -> BB -> BI
-> selector summary and signed decision
-> selected full-corpus product continuation
-> final locked evaluation and reload audit
-> NAS archive, model card and CLI package
-> sendme notification
```

After every task starts, taskd checks the log, process, GPU memory and power
after five minutes. All GPU jobs are serial. A failed arm is preserved and
retried only from its last atomic checkpoint after the failure reason is
recorded.

Evidence root:

```text
ara/s3-generation/evidence/s3_stone1_butterfly_product_v1/
```

Large artifacts:

```text
/mnt/nas/ara/s3_stone1_butterfly_product_v1/
```

Required evidence includes `contract.json`, `environment.json`, hashes,
`command.sh`, stdout/stderr, `trace.jsonl`, per-wake metrics, immutable Dreams,
arm summaries, selected/final checkpoints and a NAS manifest.

## 14. Review Gate

Before formal selector arms start, the smoke reviewer must approve:

1. the three-arm code and one-step optimizer invariant;
2. exact data/checkpoint/tokenizer hashes;
3. extracted validation/test manifests;
4. the frozen Dreams files and contamination report;
5. estimated disk use and taskd commands;
6. the fact that C06 remains not supported as registered.
