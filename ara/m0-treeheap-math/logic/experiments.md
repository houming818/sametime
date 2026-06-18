# M0 Experiment Registry

## treeheap_math_probe.py

### Question

Does a minimal synthetic TreeHeap operator toolbox satisfy the P-MATH01 evidence
gates?

### Design

Synthetic atoms:

```text
A, B, C, D, E
```

Synthetic heaps:

```text
H_ab = root(R, left=A, right=B)
H_ba = root(R, left=B, right=A)
H_cd = root(R, left=C, right=D)
H_nested = root(T, left=H_ab, right=E)
```

Operators:

```text
compose
decompose
transpose
inverse_transpose
project
unproject
energy
match_subheap
softmax_probability_container
```

### Outputs

```text
evidence/treeheap_math_probe/summary.json
evidence/treeheap_math_probe/README.md
evidence/treeheap_math_probe/matches.jsonl
```

### Interpretation

This experiment only establishes a mathematical pilot. It does not claim that
TreeHeap has learned language structure.

