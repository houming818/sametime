# C11: H_state Multi-Level Tree Convolution

Date: 2026-08-12

Claim: `S3-HSTATE-MULTILEVEL-CONV-C11`

Status: wiring and 40-step optimization smoke passed; formal run pending.

## 1. Problem inherited from C10

C10 proved that natural-text pretraining transfers to matched WMT training and
that Butterfly/FOLD interventions damage held-out NLL. It also showed that the
learned READ sends essentially all probability mass to leaves. Native READ and
forced-leaf READ differed by only `7e-8` NLL.

The representation and the read protocol must therefore be separated:

```text
H_state storage: root + addressed details + topology
READ usage:      learned path from root toward leaves
```

The former remains structure-sensitive. The latter collapsed.

## 2. Houming818 proposal

Treat the unfolded state

```text
H_state = (H_0=root, H_1, ..., H_D=leaves, masks, parent-child addresses)
```

as one TreeHeap, not as a flat concatenation. Before each output token, apply a
shared local kernel over every `(parent, left, right)` subheap, then pass the
Decoder query through every depth from root to leaf. No STOP gate decides
whether an intermediate level is allowed to affect the Decoder.

## 3. Operation

Bottom-up tree convolution uses one shared kernel at all depths:

```text
H'_d[i] = LN(H_d[i] + g_up * K_up(H_d[i], H'_(d+1)[2i], H'_(d+1)[2i+1]))
```

Top-down query convolution preserves the frontier and parent-child address:

```text
c_d     = sum_i(frontier_d[i] * H'_d[i])
q_(d+1) = LN(q_d + g_read * K_read(q_d, c_d, depth_d))
frontier_(d+1) = branch(q_(d+1), children(H'_d))
```

The final `q_(D+1)` is the context used by the existing recurrent token
Decoder. The next generated token updates the recurrent hidden state and starts
another root-to-leaf sweep.

The operation is a TreeHeap convolution because:

1. one local kernel is reused at every subheap;
2. left and right addresses remain distinct;
3. the output remains an addressed tree during convolution;
4. parent-child frontiers, rather than flat all-to-all attention, define the
   top-down traversal.

## 4. Predict

### P0: deductive wiring gate

On a real WMT batch:

1. unfolded widths must be `1, 2, 4, ..., leaf_width`;
2. no STOP parameter may exist in the new Decoder;
3. one token cross-entropy backward pass must produce finite non-zero direct
   gradients on root, every parent level and leaves;
4. encoder embedding, learned FOLD, Butterfly communication, `K_up` and
   `K_read` must all receive finite non-zero gradients;
5. bypassing `K_up` or any one depth update must change logits.

P0 proves connectivity only. It does not prove that the protocol is useful.

### P1: smoke optimization gate

After a short WMT fine-tune from the C10 pretrained checkpoint:

1. held-out NLL must remain finite and improve over its initial value;
2. all H_state level gradients must remain non-zero;
3. at least two non-leaf depth ablations must increase held-out NLL by more
   than `1e-4`;
4. bypassing bottom-up convolution must change held-out NLL by more than
   `1e-4`.

### P2: formal claim gate

Using the same 200K WMT row collection, task-stream seed, 25K optimizer steps
and token budget as C10 PT:

1. test NLL must not regress by more than `0.10` from C10 PT (`5.403696`);
2. at least two non-leaf depth ablations must each increase test NLL by
   `0.01` or more;
3. bypassing `K_up` must increase test NLL by `0.02` or more;
4. source shuffle, runtime Identity and pair-break interventions remain causal;
5. generation, repetition and token BLEU are reported; NLL alone cannot close
   the claim.

## 5. Falsification

The hypothesis is rejected for this implementation if gradients reach every
level by construction but, after matched formal training, non-leaf ablations
have negligible effect or the model is materially worse than C10 PT. That
would mean mandatory traversal creates a computational path without inducing
a useful multi-resolution private protocol.

## 6. Evidence contract

```text
evidence/s3_hstate_multilevel_convolution/
  smoke/
    summary.json
    trace.jsonl
  formal_seed10101/
    config.json
    trace.jsonl
    summary.json
    checkpoint_best.pt       # retained on io / NAS when too large for Git
```

Every result records checkpoint hashes, data-stream hash, gradient norms by
depth, intervention NLL, token count, wall time and exact host.

## 7. Boundary

A positive result would show that a shared TreeHeap convolution can make every
H_state depth trainable and causally useful for this finite WMT pipeline. It
would not prove that higher levels have human-readable semantics, that the
private protocol is unique, or that the model is product ready.

## 8. Smoke result

Taskd job `168` completed on `io` in `35.6` seconds using 512 WMT training rows,
64 validation rows, 40 optimizer steps and 5,865 target tokens.

| Observation | Result |
|---|---:|
| initial -> best valid NLL | `20.8235 -> 10.0273` |
| unfolded widths after training | `1,2,4,8,16,32,64` |
| direct gradient norm, root | `0.001809` |
| direct gradient norm, leaf | `0.003638` |
| bypass bottom-up convolution delta NLL | `+1.7831` |
| leaf-only read delta NLL | `+5.1145` |
| each non-leaf depth ablation delta NLL | `+0.0227 .. +0.0246` |

All required encoder, Butterfly, FOLD, up-kernel and read-kernel parameter
groups received finite non-zero gradients. No STOP parameter exists in the
new Decoder. P0 passed. The first implementation also passed the original
short P1 gate, but it replaced the inherited context with a randomly
initialized readout and started at NLL `20.8235`; it is retained as wiring
evidence, not as the formal initialization.

A residual-preserving v2 smoke (taskd `169`) started at NLL `8.1906` and
reached `7.8100` after 40 steps while retaining non-zero gradients at all
levels. However, ablating each non-leaf depth *improved* NLL by about
`0.0044--0.0055`. The corrected directional P1 gate therefore fails: the
levels participate, but at 40 steps their contribution remains harmful. The
formal run must reverse this sign rather than merely produce non-zero damage.

The result proves wiring and short optimization only. Initial NLL is severely
damaged by the newly initialized convolution/read kernels, and final smoke NLL
remains far above C10 PT (`5.403696`). Formal training must determine whether
the mandatory multi-level route can recover language quality without losing
its depth causality.
