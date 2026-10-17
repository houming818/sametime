# Tree QK path-encoding real-corpus smoke

Status: executed on 2026-10-17; mechanism supported, quality advantage open.

## Question

When TreeHeap relation is defined strictly as QK, does adding an explicit
root-to-node path code make the relation field path-sensitive on real
English-Chinese corpus data?

This experiment does not search a final architecture. It checks whether the
SPR-096 wiring is trainable, causally used, and non-collapsed.

## Fixed contract

- deterministic reservoir sample from the existing WMT massive corpus;
- fixed 6000/1000/1000 train/valid/test rows for the first GPU smoke;
- fixed seed 20261017, tokenizer, row order, optimizer, epochs and batch size;
- balanced 32-slot heap with source lengths 16--31 before EOS;
- identical source embedding, FOLD, Value/READ and Decoder initialization;
- soft recursive routing so QK receives gradients at every depth;
- translation cross-entropy is the only training objective;
- quality metrics are observations, not early-stop gates.

## Arms

1. uniform: local child probabilities are uniform over valid children.
2. qk_no_path: QK sees node content but no path code.
3. qk_fixed_path: QK sees a deterministic code derived from depth and heap
   address.
4. qk_learned_path: the path table is trainable and starts at zero, so its
   step-0 function should match qk_no_path.

The common QK projections exist in every arm. Only the learned path table adds
arm-specific trainable parameters; its size is reported separately.

## Path definition

For left bit 0 and right bit 1:

$$
a_{d+1}=2a_d+b_{d+1},
\qquad
z_d=E_{path}(d,a_d).
$$

At each parent:

$$
q_d=W_Q[h_{decoder}\Vert h_{parent}\Vert z_d],
$$

$$
k_{d,c}=W_K[h_{child}\Vert e_c\Vert z_{d\rightarrow c}],
$$

$$
R_{d,c}=\frac{q_d^\mathsf{T}k_{d,c}}{\sqrt{d_k}}.
$$

The local softmax probabilities are multiplied down the tree. READ is the
probability-weighted sum of leaf Values. R remains QK; Value/READ remains a
separate operation.

## Interventions

- uniform: replace learned relation scores by equal valid-child scores;
- query_shuffle: shuffle Query content across examples;
- key_shuffle: shuffle child Key content across examples;
- path_zero: remove only the path code;
- path_shuffle: permute path codes within each depth;
- left_right_swap: swap local left/right relation scores;
- source_shuffle: shuffle the complete encoded source.

## Measurements

- train, valid and test NLL/PPL;
- short greedy BLEU-4, non-empty rate and repetition;
- route entropy by depth, effective leaf count and maximum leaf mass;
- NLL damage for every intervention;
- route Jensen-Shannon divergence against native routing;
- finite gradients, data/tokenizer hashes, initial shared-state hashes and
  parameter counts.

## Preregistered interpretation

### Wiring gate

All arms must have finite gradients, identical selected rows, identical shared
initial state, complete JSON evidence and valid hashes.

### Path-use evidence

Path encoding is causally used only if a path arm shows both:

- positive route divergence under path_zero or path_shuffle; and
- positive held-out NLL damage under the same intervention.

The exact magnitudes are reported continuously. A tiny nonzero value is not
called practically useful.

### Collapse audit

Leaf effective count near 1 means hard collapse. Effective count near the
number of valid leaves means nearly uniform routing. Either extreme can still
train, but neither alone establishes useful relation search.

### Quality boundary

Lower NLL or higher BLEU in one seed is candidate evidence only. The smoke
cannot establish architecture superiority, power savings, sparse-compute
benefit, or a final TreeHeap relation protocol.

## Stop conditions

Stop only for OOM, CUDA/NVIDIA failure, NaN/Inf, broken data/evidence contract,
or non-advancing execution. Do not stop because NLL or BLEU is non-monotonic.

## Execution record

The first contract task (`685`) stopped before training because the staged
script could not resolve the repository-local WMT loader. The import path was
made explicit and the corrected contract task (`686`) completed. The formal
task (`687`) then completed all four arms on `io` in 216 seconds. This is an
engineering preflight failure, not model evidence, and is retained rather than
silently folded into the successful run. Evidence packaging task `688` also
failed on a staging-directory permission boundary; corrected task `689` wrote
the archive to a task-owned temporary path and completed without altering the
experiment output.

Formal evidence:

```text
ara/s3-generation/evidence/s3_tree_qk_path_encoding_smoke/formal_seed20261017
```

All files listed by both smoke and formal `SHA256SUMS` verified after transfer.
The selected-row hash is
`5271a5fb90633eedff5b8746147c38c768c0f2bd9a0480ca94c3f4ac36c3a3e9` and
the SentencePiece hash is
`9956eff597852f8c684c4ad23243d15889da6a9b138f8fd025570147324cc731`.

## Formal results

| arm | test NLL | PPL | BLEU-4 | repeated-token fraction | effective leaves |
|---|---:|---:|---:|---:|---:|
| uniform | 7.465374 | 1746.51 | 0.9202 | 0.7616 | 21.57 |
| qk_no_path | 7.484448 | 1780.14 | 1.1686 | 0.6903 | 10.74 |
| qk_fixed_path | 7.473946 | 1761.54 | 0.7649 | 0.6724 | 11.91 |
| qk_learned_path | 7.468156 | 1751.37 | 1.1111 | 0.6358 | 11.68 |

The learned-path arm adds 6,048 parameters to the 12,514,977 common
parameters. Its step-zero NLL is exactly equal to `qk_no_path`, as required by
the zero initialization contract.

## Causal audits

The content-only QK arm is not an inert copy of uniform routing:

- forcing uniform routing damages NLL by 0.168691;
- shuffling Query damages NLL by 0.010207 and changes routing by JSD 0.013707;
- shuffling Key damages NLL by 0.213982 and changes routing by JSD 0.366746;
- swapping left/right scores damages NLL by 0.909093.

The learned path is also causally used:

- zeroing it damages NLL by 0.014340 and changes routing by JSD 0.016110;
- shuffling it damages NLL by 0.051478 and changes routing by JSD 0.046671.

Both preregistered path-use conditions are therefore satisfied. The effect is
large enough to distinguish from the formal run's Query-shuffle effect, but it
does not establish a useful quality gain.

## Interpretation

`S3-TREE-QK-PATH-C01` is **supported only as a wiring and mechanism claim**:
recursive QK routing trains on real corpus data, avoids one-leaf collapse, and
learns to use a root-to-node path code. The route is substantially more
selective than uniform routing.

The architecture-quality claim remains **open**. Uniform routing has the best
test NLL by 0.002782 over learned path, while all generated samples remain
poor and highly repetitive. BLEU values around 1 are too low for translation
claims. A lower repetition fraction in the learned-path arm is an observation,
not evidence of fluent generation.

The intervention asymmetry is the most useful optimization signal: Key
content and fixed left/right placement dominate Query content. The current QK
field can select a path, but it does not yet use the decoder-side Query strongly
enough to turn that selection into better prediction. The next experiment
should therefore target relation balance and path parameterization under the
same corpus contract, rather than merely adding epochs or declaring the path
code successful.

## Evidence boundary

This one-seed, 6,000-row run does not prove architecture superiority,
translation quality, sparse-compute benefit, memory reduction, or energy
savings. It also does not show that longer training would reverse the NLL
ordering. Those remain separate claims requiring new preregistration.

## Full-corpus follow-up preregistration

Status: preregistered before execution on 2026-10-17.

The 6K result is a mechanism measurement, not an adequate training-volume
comparison. The next run therefore consumes one complete sequential pass of
the immutable `NioClean-ZHEN-S098-7M-v2` release:

- expected rows: 7,304,358;
- expected bytes: 1,535,169,936;
- expected SHA-256:
  `299134867398720cc6d407eadd6de4fb237812319d113fbe12071758e79d92c8`;
- direction: English to Chinese;
- source: at most 31 content pieces plus EOS in a 32-leaf heap;
- target: at most 63 content pieces plus EOS;
- arms: `uniform`, `qk_no_path`, and `qk_learned_path`;
- one shared sequential corpus pass, fixed seed 20261017, batch size 64;
- AdamW with 1,000-step warm-up and cosine decay from `3e-4` to `3e-5`;
- wake and resumable checkpoint every 5,000 optimizer steps.

The fixed-path arm is omitted because the first formal run already showed less
path intervention response than the learned-path arm. This preserves the two
controls required to separate QK use from learned-path use while spending the
full pass on the most informative comparison.

The prior 8,000-row deterministic WMT sample is regenerated with the original
selection contract. Its final 1,000 valid and 1,000 test rows remain fixed, and
all 8,000 selected token pairs are excluded from the NioClean training stream
when exact matches occur. Its selected-row hash must remain
`5271a5fb90633eedff5b8746147c38c768c0f2bd9a0480ca94c3f4ac36c3a3e9`.

### Full-pass questions

1. Does content QK overtake uniform routing after enough corpus exposure?
2. Does learned path overtake content-only QK rather than merely react to path
   interventions?
3. Do Query, Key and path interventions remain causal after a full pass?
4. Does selectivity remain distributed, or collapse toward one effective leaf?
5. Does lower repetition coincide with improved NLL/BLEU, rather than trading
   one failure mode for another?

No quality metric is an early-stop gate. Stop only for OOM, CUDA/NVIDIA failure,
NaN/Inf, release/evaluation hash mismatch, corrupted checkpoint/evidence, or a
non-advancing cursor. A single full pass remains one-seed evidence; it cannot
establish general superiority or energy savings.
