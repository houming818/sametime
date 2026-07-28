# C10 Condition-Collapse Audit

Date: 2026-07-28

Decision: C10 is invalid as evidence for source-conditioned translation or a
TreeHeap private protocol. The completed training run and checkpoint remain
diagnostic artifacts.

## Manual Free-Running Evidence

Three unrelated English inputs were tested with the C10 CLI at checkpoint step
114796:

```text
The earth is round.
The apple is sweet
why is the window wet? because the sky cried
```

All three generated a repetitive Chinese continuation dominated by
`一带一路`, and none emitted EOS within the requested 48 output pieces. The
route distributions were also close across inputs.

## Code Audit

The training path uses gold target prefixes in `RecursiveDecoder.teacher`:

```text
previous decoder input = target[:, step]
```

The CLI uses its own previous prediction. These are different evaluated
functions. A model can lower teacher-forced NLL by learning Chinese prefix
continuation while its free-running generation collapses from BOS.

The source preparation also calls `fixed_source` with `eos_tail` and a visible
heap width of 256. Short inputs therefore contain only a few real source pieces
and a large visible EOS tail.

## Required Follow-Up

1. Keep gold targets fixed and shuffle source rows; report delta NLL.
2. Replace source with all EOS or a neutral masked source; report delta NLL.
3. Compare first-step logits across unrelated inputs before any target prefix
   is available.
4. Report free-running unique-output rate, repetition rate and EOS rate.
5. Correct source visibility and repeat the audits before retraining at scale.

