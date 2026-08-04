# TreeHeap Canonical View Ratio

Status: not supported by the registered one-seed screen

## Claim S3-TREEHEAP-CANONICAL-VIEW-C05

A trained Butterfly TreeHeap may benefit from seeing the unpermuted leaf order
more often during continued training. With a fixed example, token and optimizer
step budget, a moderate probability of sending the original leaf order directly
into the same FOLD and Decoder should improve or stabilize canonical translation
without eliminating the measurable contribution of full Butterfly
communication.

This is an empirical claim about training allocation. It does not assume that
the original order is the unique semantic coordinate system, and it does not
call every alternative Dream a valid viewpoint.

## Why this experiment is needed

The current bilingual model computes one serial path for every training sample:

```text
token order H0
  -> complete Butterfly B(H0)
  -> FOLD
  -> H_state
  -> Recursive Decoder
  -> canonical target string
```

The intermediate Butterfly states are not separate corpus views. In particular,
`H0` never reaches FOLD during native training. The decoder is always asked to
recover the canonical target from the fully transformed state.

Houming818's egg-view hypothesis suggests that the observer may need to see the
desired projection more frequently. The smallest controlled test is therefore
not a new loss and not a mixture of every intermediate depth. It is a binary
view allocation:

```text
with probability p:       H0    -> FOLD -> Decoder
with probability 1 - p:   B(H0) -> FOLD -> Decoder
```

Both routes use the same embedding, FOLD, H_state, Decoder and target. The only
changed variable is whether full Butterfly communication is applied before
FOLD.

## What `p` means

Registered screening ratios:

```text
p = 0.0   current native continuation control
p = 0.2   occasional canonical reminder
p = 0.4   moderate canonical emphasis
p = 0.6   canonical-majority stress arm
```

For every ratio, the same source-target rows appear exactly once. The same
number of target tokens and optimizer updates are used. A deterministic hash of
`line_number + seed` chooses the route, and the selected sets are nested: every
row assigned to `p=0.2` is also assigned to `p=0.4` and `p=0.6`.

This design does **not** duplicate the corpus. It distinguishes a view-ratio
effect from the trivial benefit of performing more updates.

## Training implementation

One mini-batch can contain both routes. The batch is split by its registered
view mask, but both subsets contribute to one token-normalized loss and one
optimizer step:

```text
L = (sum CE_butterfly + sum CE_identity) / number_of_target_tokens
```

The total loss is still ordinary teacher-forced token cross-entropy. No view
label, grammar label, contrastive target or human score enters the gradient.

All arms start from the same immutable `checkpoint_best.pt` produced by taskd
89. They reuse the same optimizer state and use the same reduced continuation
learning rate. This first experiment tests whether an already formed private
protocol can be rebalanced; it does not yet prove how such a protocol forms
from random initialization.

## Evaluation axes

### 1. Native translation NLL

Evaluate every arm with complete Butterfly communication. This is the primary
task metric because native inference remains the intended route.

### 2. Canonical-route NLL

Evaluate the same references while bypassing Butterfly. A lower value shows
that the shared FOLD/Decoder can read the original coordinate more reliably.
It is not sufficient by itself because a flat identity-only solution could
also lower this metric.

### 3. Cross-view Jensen-Shannon divergence

For the same teacher-forced tokens, compare the native and identity output
distributions:

```text
JS(P_B, P_I)
```

A moderate decrease indicates that both internal views lead to a more
consistent output protocol. A value near zero is not automatically good if
Butterfly has become unused.

### 4. Structural causal checks

Record:

```text
identity damage  = NLL_identity - NLL_butterfly
adjacent damage  = NLL_adjacent - NLL_butterfly
source shuffle damage
Butterfly transformed-versus-raw RMS
communication-kernel gradient norm during training
```

The preferred arm must improve the canonical objective while retaining source
dependence and measurable use of Butterfly parameters.

### 5. Grammar Dreams

Render the complete fixed bilingual Dreams suite after every arm. It includes:

```text
agent/patient reversal
active/passive
negation
before/after
every/not every
relative-clause attachment
cause/consequence
exact entities and numbers
nested long composition
```

Dreams are observation probes only. Human semantic annotation is performed
after the numeric screening and never enters training.

## Predictions

### P1: moderate rather than maximal canonical emphasis

At least one of `p=0.2` or `p=0.4` should beat the matched `p=0.0`
continuation arm on native held-out NLL. The expected curve is an inverted U:
some canonical reminders help, while `p=0.6` risks undertraining Butterfly.

Screening signal:

```text
native NLL gain >= 0.02
```

The threshold is a pilot effect size, not a universal constant. It must later
survive multiple view-assignment seeds.

### P2: cross-view consistency

The winning moderate arm should reduce native-versus-identity JS divergence by
at least 20% relative to `p=0.0`.

### P3: Butterfly is not trained away

The winning arm must retain all of the following:

```text
native NLL better than at least one structural override
positive source-shuffle damage
non-zero communication gradient norm
non-trivial transformed-versus-raw RMS
```

The registered report includes raw values; borderline cases remain open rather
than being rounded into success.

### P4: human-visible grammar stability

On the fixed Dreams suite, the winning moderate arm should reduce role,
polarity, temporal, quantity or repetition errors without introducing a larger
number of new semantic errors. This is manually reviewed after the run.

## Controls

| Arm | Canonical probability | Purpose |
|---|---:|---|
| C0 | 0.0 | Same-checkpoint, same-budget continued-training baseline |
| C20 | 0.2 | Small canonical reminder |
| C40 | 0.4 | Main hypothesis arm |
| C60 | 0.6 | Over-canonicalization stress test |

All arms share:

```text
checkpoint initialization
optimizer initialization
source-target rows
target-token count
batch order
optimizer-step count
validation/test rows
greedy decoding settings
```

## Experiment stages

### Stage A: infrastructure smoke

Run `p=0.0` and `p=0.4` on a small line slice. Require finite gradients,
identical starting metrics, deterministic view counts, checkpoint reload and
complete evidence output.

### Stage B: one-seed screening

Run all four ratios on the same continuation slice. Save numeric metrics,
per-arm Dreams and optional arm checkpoints. Select no winner until every arm
has consumed the same registered token/update budget.

### Stage C: confirmation

Only if Stage B produces a registered signal, repeat the baseline and candidate
ratio with at least three view-assignment seeds. The final claim status is based
on Stage C, not on a single favorable run.

## Falsification

Reject or downgrade the claim if any of the following occurs:

```text
p=0.0 is best under the matched budget
moderate ratios improve identity NLL but worsen native NLL
JS falls only because all structural overrides become equivalent
source shuffle no longer damages the model
communication parameters receive no gradient
Dreams gain fluency while role, negation, time or facts deteriorate
the apparent winner changes across view-assignment seeds
```

If canonical mixing improves only identity-route NLL, the result supports
multi-view readability but not better native translation. If it improves native
NLL while preserving causal checks, it becomes evidence that corpus-view
allocation can steer a TreeHeap private protocol.

## Evidence target

```text
ara/s3-generation/evidence/s3_treeheap_canonical_view_ratio/
  README.md
  command.sh
  startup.json
  ratio_*/summary.json
  ratio_*/dreams/latest.txt
  summary.json
  stdout.log
```

## Scope

This experiment tests a continuation-training intervention on one existing
Butterfly TreeHeap checkpoint. It does not establish a unique canonical
semantic space, human perception, consciousness, or superiority over other
architectures.

## Screening result

Taskd 99 completed successfully on `io` on 2026-08-04. Every arm consumed the
same 299,407 examples, 6,308,579 target tokens and 6,056 optimizer updates.
The realized canonical ratios matched the registered ratios within 0.14
percentage points.

| `p` | Native NLL | Change vs `p=0` | Identity NLL | Cross-view JS | Source-shuffle damage |
|---:|---:|---:|---:|---:|---:|
| 0.0 | **3.2710** | 0.0000 | 4.8336 | 0.2378 | +1.8247 |
| 0.2 | 3.2826 | +0.0116 | 3.7668 | 0.1011 | +1.8468 |
| 0.4 | 3.2939 | +0.0229 | 3.6963 | 0.0910 | +1.8575 |
| 0.6 | 3.3090 | +0.0380 | **3.6480** | **0.0838** | +1.8566 |

Lower NLL and lower JS are better. The primary preregistered prediction is not
supported: `p=0` is the best native arm, and native NLL worsens monotonically as
canonical exposure increases. No moderate ratio qualifies for multi-seed
confirmation under the registered gate.

The negative result is not equivalent to "nothing happened." Canonical mixing
reduced cross-view JS by 57.5% at `p=0.2` and by 64.8% at `p=0.6`. Identity
damage also fell from 1.5626 to 0.3390. Meanwhile source shuffling continued to
damage NLL by about 1.85, structural overrides remained harmful, Butterfly
communication stayed non-trivial, and communication parameters received
gradient. The model therefore did not obtain low JS by simply ignoring the
source or deleting the Butterfly path.

The supported observation is narrower than the original claim:

> In this continuation setting, canonical-view mixing acts as a view-invariance
> regularizer. It makes one shared FOLD/Decoder more capable of reading both
> identity and Butterfly coordinates, but this invariance costs performance on
> the specialized native Butterfly route.

The fixed Dreams do not show a consistent monotonic improvement in semantic or
grammatical quality. Some individual outputs change in a favorable direction
and others regress or retain repetition. They therefore do not overturn the
primary numeric result.

This screen does not prove that canonical exposure can never help. It rejects
the registered version: continued training from this checkpoint, with a fixed
budget, one shared cross-entropy objective and sample-level ratios 0.2--0.6.
Future work should not repeat the same ratio sweep without changing the
mechanism or prediction.
