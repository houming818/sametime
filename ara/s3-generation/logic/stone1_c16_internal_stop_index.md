# C16: internal STOP TreeHeap index

Status: implementation smoke partial; Hit@3 gain failed, depth gain supported

## Correction to C15

C15 forced all 16 values into the leaves of a 31-node complete binary tree.
That is a useful leaf-only control but omits the established TreeHeap action
`STOP`. A TreeHeap node should be allowed to hold one terminal payload while
also retaining left and right subheaps.

An internal terminal is unambiguous because paths use the action alphabet
`{STOP, LEFT, RIGHT}`. For example, `LEFT, STOP` and
`LEFT, RIGHT, STOP` are different action sequences.

## Claim

`S3-INTERNAL-STOP-INDEX-C16`:

> Under equal physical-node, payload-slot and node-visit budgets, allowing one
> terminal payload at any TreeHeap node and optimizing payload placement by
> retrieval Hit@K should improve held-out retrieval and shorten successful
> paths relative to a leaf-only TreeHeap when query outcomes are skewed.

This is a controlled index claim only.

## Data structure

Both tree arms use:

```text
31 physical binary nodes
16 unique token payloads
at most one payload per node
10 visited nodes per query
Top-3 output
```

For query `q`, node mass is

\[
C_n(q)=C_{local(n)}(q)+C_{left(n)}(q)+C_{right(n)}(q).
\]

Visiting a node exposes its local STOP payload and places its children on a
best-first frontier. After the visit budget is exhausted, the three exposed
payloads with the largest query counts are returned.

The leaf-only arm restricts payload addresses to nodes 15-30. The internal-STOP
arm may place payloads at any of nodes 0-30. Legal optimization swaps two node
payloads, including a payload and an empty node. Tree size never grows.

## Predictions

1. Exact unlimited path probability remains topology invariant to `1e-12`.
2. Optimized internal-STOP held-out Hit@3 exceeds optimized leaf-only Hit@3 by
   at least `0.03` under the ten-node budget.
3. Mean depth of returned correct payloads falls by at least `0.50`.
4. Internal-STOP train-test Hit@3 gap remains below `0.10`.
5. Internal-STOP uses no more physical nodes or payload slots than leaf-only.

## Falsification

Reject the useful internal-STOP claim if it cannot beat the optimized leaf-only
arm under matched budgets, if gains do not transfer to independent test
events, or if they require a larger bucket, tree or visit budget. Passing does
not prove semantic hierarchy or natural-language generation.

## Local implementation smoke (2026-07-29)

This was a code-path smoke, not formal io evidence. It used eight random
layouts, 200 training and 100 independent test events per query, six swap
rounds, a ten-node budget and Top-3 retrieval.

| Measurement | Prediction | Result | Gate |
|---|---:|---:|---|
| Exact NLL layout range | < `1e-12` | `2.22e-16` | pass |
| Leaf-only held-out Hit@3 | reference | `0.8033` | reference |
| Internal-STOP held-out Hit@3 | leaf + `0.03` | `0.8033` (`+0.0000`) | fail |
| Leaf-only mean correct depth | reference | `4.0000` | reference |
| Internal-STOP mean correct depth | leaf - `0.50` | `2.2842` (`-1.7158`) | pass |
| Internal train-test Hit gap | < `0.10` | `0.0021` | pass |

For `eat`, internal STOP exposed `rice`, `noodles` and `apple` at depths 3, 2
and 0. Both optimized arms matched the flat Top-3 result (`0.8033`), so the
ten-node budget was not tight enough for internal placement to improve
accuracy. Its observed benefit was reaching the same correct set at shallower
addresses.

The preregistered strong claim is therefore not supported by this smoke. The
next valid test is a frozen budget curve, not a post-hoc favorable budget: run
the same optimized layouts at node budgets 1-10 and compare area under the
Hit@3-versus-visits curve. That directly asks whether internal STOP returns
useful answers earlier without changing capacity or selecting one convenient
operating point.
