# FOLD Observation-Matrix Decode Probe

Checkpoint: TreeHeap-106M, pass 2, step 132873.

The probe freezes every learned parameter and replaces only the two-child FOLD
operator with `parent = s * (left + right)`.  Full outputs and diagnostics are
in `106m_pass2_grid.json`.

## Direct examples

Input: `The child put the red apple on the table.` (en2zh, depth 7)

| s | decoded output |
|---:|---|
| 0 | ` ⁇ 子虫` |
| 0.25 | ` ⁇ 放进食` |
| 0.5 | `放进餐后` |
| 0.707 (native) | `放进餐后,把孩子放在桌子上。` |
| 1.0 | 48 pieces, no EOS, repeated `把孩子放在...` clauses |

Input: `孩子把红苹果放在桌子上。` (zh2en, depth 5)

| s | decoded output |
|---:|---|
| 0 | empty (immediate EOS) |
| 0.25 | empty (immediate EOS) |
| 0.5 | `The table is placed in the table.` |
| 0.707 (native) | `Put a table in the table.` |
| 1.0 | `Put a table in the table.` |

Input: long-road sentence (en2zh, depth 7)

| s | behavior |
|---:|---|
| 0 | short unrelated fragment |
| 0.25 | short unrelated fragment |
| 0.5 | terminated 13-piece sentence with `虽然` and `马路` |
| 0.707 (native) | terminated 25-piece road/driver sentence |
| 1.0 | 48 pieces, no EOS, repetition and numeric drift |

## Result

The frozen decoder is capable of producing text under several transformed
observation matrices.  It is not invariant to the matrix: weak parent amplitude
can collapse to empty or under-specified output, while unnormalized summation
often drives length expansion and failure to stop.  The middle interval is the
observed sentence-like regime for this small prompt set.

At `s=0`, learned decoder-side `K_up` remains enabled; therefore the result is
not evidence that raw leaves alone suffice.
