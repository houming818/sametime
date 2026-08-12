# C10 Read Geometry: Tree Search or Linked-Path Shortcut?

Date: 2026-08-12

Parent claim: `S3-TREEHEAP-PRETRAIN-POSTERIOR-C10`

Diagnostic: `S3-C10-PATH-SHAPE-AUDIT-D01`

Status: completed on frozen C10-PT pilot checkpoint; both preregistered
explanations rejected as complete descriptions

## Question

The frozen C10-PT checkpoint stores the complete reversible state

```text
H_state = (root, detail_0, ..., detail_D-1).
```

The previous read-path audit nevertheless found that its learned STOP gate
places essentially all stopping mass on leaves. This does not decide whether
the Decoder uses a distributed tree index or merely follows one narrow path
that behaves like a linked-list cursor.

## Competing explanations

### H1: linked-path shortcut

For each generated token, almost all useful read mass lies on one root-to-leaf
path. Keeping only the most likely leaf should preserve NLL, removing it should
cause large damage, and the selected leaf address should move through a narrow
or approximately sequential trajectory over output time.

### H2: distributed leaf index

STOP occurs at leaf resolution, but the branch kernel still mixes several
root-to-leaf paths. Top-1 truncation should lose measurable information,
Top-K recovery should improve progressively, and more than one branch depth
should remain causal.

Both outcomes are narrower than a multi-resolution reader because neither
uses parent states as final readout values.

## Frozen intervention design

No parameter is trained. The same 256 held-out WMT rows and the same teacher-
forced targets are used in every arm.

The native leaf distribution is reconstructed from the recursive branch
softmax:

```text
a_0(root) = 1
a_(d+1)(child) = a_d(parent) * P(child | parent, hidden_t)
```

At the leaf level, the Decoder context is:

```text
c_t = sum_i a_D(i) * leaf_i.
```

Interventions:

| Arm | Operation |
|---|---|
| `native_model` | Unmodified C10 read, used as the numerical reference |
| `all_leaf` | Reimplemented full leaf mixture; must match `native_model` |
| `top1/top2/top4/top8` | Keep only the K largest leaf masses and renormalize |
| `drop_top1` | Remove the largest leaf mass and renormalize the remainder |
| `uniform_leaf` | Ignore branch scores and average valid leaves |
| `ordered_cursor` | Read leaf `min(output_step, source_length-1)` |
| `uniform_depth_d` | Replace the branch decision at depth d by a valid-child uniform distribution |

The implementation has one shared `decoder.branch.weight`; there are no
independent parameters owned by a particular path. Therefore path-state and
path-mass interventions are the well-defined way to ablate "other paths".

## Recorded measurements

- NLL and delta from native for every arm;
- mean Top-1/2/4/8 cumulative leaf mass;
- leaf entropy and effective leaf count `exp(entropy)`;
- number of unique argmax leaves per sentence;
- stationary, adjacent-forward and monotonic-forward transition rates;
- mean absolute leaf-address jump;
- per-depth uniform-branch damage.

## Interpretation gates

Evidence favors the linked-path shortcut only if all of the following hold:

1. mean Top-1 leaf mass is at least `0.80`;
2. Top-1-only NLL damage is at most `0.02`;
3. removing Top-1 damages NLL by at least `0.10`;
4. stationary plus adjacent-forward address transitions account for at least
   `0.70` of consecutive output steps.

Evidence favors a distributed leaf index if:

1. Top-1-only NLL damage exceeds `0.05`;
2. Top-4 or Top-8 materially recovers that loss;
3. at least two depth-specific uniform interventions each add `0.02` NLL.

If neither set passes, the checkpoint is classified as a mixed or degenerate
leaf-resolution protocol. The audit cannot establish semantic understanding,
multi-resolution reading, or product quality.

## Result

Taskd job `165` completed on `io` in `26.8` seconds. It froze checkpoint state
SHA-256 `811fe2c00de5aaa27a90f22660ffceca0444d3c53f2ad8a04ed764c480b55f71`
and evaluated 256 held-out WMT rows containing 5,226 scored target pieces.

The custom full-leaf reconstruction matched the original model to numerical
precision:

| Read arm | NLL | Delta from native |
|---|---:|---:|
| Native model | 5.29983 | 0.00000 |
| Reconstructed all-leaf mixture | 5.29983 | -0.00000 |
| Top-1 leaf only | 9.30461 | +4.00478 |
| Top-2 leaves | 6.83601 | +1.53618 |
| Top-4 leaves | 5.80485 | +0.50502 |
| Top-8 leaves | 5.43817 | +0.13834 |
| Remove Top-1 leaf | 5.38272 | +0.08289 |
| Uniform valid-leaf average | 5.32694 | +0.02711 |
| Ordered output-position cursor | 9.36251 | +4.06268 |

The native leaf distribution had mean Top-1/2/4/8 mass
`0.2104/0.3369/0.5110/0.7222`, entropy `2.6209`, and effective leaf count
`15.85`. Each sentence used an average of `6.72` distinct argmax leaf
addresses over teacher-forced output time. Stationary plus adjacent-forward
transitions accounted for only `0.3789` of transitions, with mean absolute
address jump `6.30`.

Uniformizing one branch depth at a time changed NLL by only
`-0.0210..+0.0112`. No depth crossed the preregistered `+0.02` causal gate.

## Conclusion

The single linked-path hypothesis is rejected: no dominant path explains the
read, and an ordered cursor fails badly. The strong distributed tree-index
hypothesis is also not supported: several leaves contribute, but learned
hierarchical branch decisions add little over averaging all valid leaves.

The narrow evidence-backed description is:

```text
address-sensitive Butterfly/FOLD transformation
-> forced leaf-resolution reconstruction
-> broad, weakly routed soft pooling over transformed leaves
-> recurrent token Decoder
```

This explains the earlier intervention pattern without calling the read a
semantic tree search. Runtime Identity and pre-FOLD pair breaking alter the
contents delivered to the leaf pool, so they can damage NLL even when the
learned branch hierarchy is nearly dispensable. C10 therefore retains an
address-sensitive TreeHeap encoder but has collapsed its intended
multi-resolution read into a near-unstructured leaf pool.

Evidence:

```text
ara/s3-generation/evidence/s3_pretrain_task_posterior_pipeline/
  pilot_seed10101/path_shape_audit.json
```
