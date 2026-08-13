# C12: Strict Multi-Level READ Ablation

Date: 2026-08-13

Claim: `S3-MULTILEVEL-READ-ABLATION-C12`

Status: preregistered; matched-stream smoke passed; formal queue pending.

## Question

C11 combined two changes:

1. removal of the learned STOP gate and mandatory root-to-leaf READ;
2. a second bottom-up parent-left-right convolution `K_up`.

The frozen C11 test audit showed that forcing leaf-only READ increased NLL by
`0.440891`, while bypassing `K_up` increased NLL by only `0.000340`. This
suggests that mandatory multi-level READ, rather than `K_up`, caused the useful
parent-level participation. C11 was not a strict architecture ablation because
its historical C10 comparison used a different training stream and learning
rate.

## Arms

Every arm starts from the exact same C10 natural-text pretrained checkpoint and
receives the exact same ordered WMT mini-batches.

| Arm | Decoder READ | Extra bottom-up pass |
|---|---|---|
| `c10` | learned STOP probability container | no |
| `read` | mandatory multi-level residual READ | no |
| `read_up` | mandatory multi-level residual READ | shared `K_up` |

The `read` and `read_up` arms instantiate the same complete C11 Decoder with
the same random seed. The `read` arm disables `K_up` by a non-parameter
architecture flag, so both arms begin with identical READ weights. Its unused
`K_up` parameters remain present solely to keep initialization and parameter
count matched; they receive no gradient in `read`.

## Controlled variables

The following must be identical across all arms:

```text
parent checkpoint hash
WMT row IDs and deterministic train/valid/test split
ordered mini-batch stream hash
optimizer, learning rate and clipping
steps, batch size and token budget
checkpoint selection rule
test rows and generation examples
```

Formal defaults reproduce the C10 task phase:

```text
seed       = 10101
train rows = 200,000
valid rows = 1,000
test rows  = 1,000
steps      = 25,000
batch      = 16
lr         = 0.002
```

No maximum-length cleaning pass is added after C10 row collection. This is
required to reproduce C10's original task stream hash.

## Predict

### P0: experimental identity

1. all arms load the same parent state hash;
2. all arms report the same row hashes and training stream hash;
3. `read` and `read_up` report the same initial READ parameter hash;
4. every run remains finite and produces a reloadable checkpoint.

If P0 fails, no architecture comparison is admissible.

### P1: mechanism isolation

After matched training:

1. `read` forced leaf-only NLL must be at least `0.05` worse than native;
2. at least two non-leaf depth ablations in `read` must each increase test NLL
   by at least `0.01`;
3. `read_up` bypass-`K_up` delta measures the independent contribution of the
   extra upward pass.

### P2: model selection

The preferred arm is selected by a Pareto report, not a single metric:

1. test NLL;
2. token BLEU4 on identical examples;
3. adjacent repetition rate and maximum repeated-token share;
4. parameter count and wall time;
5. structural intervention deltas.

The narrow prediction is:

```text
READ-only preserves the parent-level causal signal seen in C11,
while matching or improving READ+K_up language metrics and runtime.
```

No preregistered claim requires READ-only to beat C10. If C10 remains better,
the result still isolates the cost of forcing a multi-resolution protocol.

## Falsification

The READ mechanism hypothesis is rejected for this implementation if, under a
matched stream, leaf-only and non-leaf depth ablations become negligible or
helpful. The `K_up` hypothesis is rejected if bypassing it remains below
`0.01` NLL and it does not improve generation metrics. A lower training NLL
alone cannot support either mechanism.

## Evidence contract

```text
evidence/s3_multilevel_read_ablation_c12/
  smoke_seed10101/{c10,read,read_up}/
  formal_seed10101/{c10,read,read_up}/
  comparison.json
```

Each arm stores `summary.json`, `trace.jsonl`, an atomic progress checkpoint
and a final best checkpoint. Large checkpoints remain on `io` and NAS.

## Smoke result

Taskd jobs `178--180` completed normally. All three arms produced exactly the
same train/valid/test row hashes and training stream hash. The two multi-level
arms also produced the same initial READ parameter hash. P0 therefore passes
for the experiment implementation.

At 20 steps, coarse-depth ablations are still harmful in both multi-level
arms: removing a coarse update lowers NLL by about `0.047--0.051`. This is the
same sign seen in the early C11 smoke before long training reversed it. The
result is not mechanism support; it demonstrates that the formal experiment
can observe the preregistered sign transition rather than making parent levels
helpful by construction.
