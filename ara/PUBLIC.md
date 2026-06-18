# Public ARA Mirror

This directory is the public ARA mirror for SameTime.

It is designed for:

- humans who want to understand why the project changed direction;
- AI agents that need a compact research context before proposing experiments;
- reviewers who want to inspect the claim -> evidence -> decision chain.

## What Is Included

```text
logic/     problem definitions, claims, predicts, experiment registries
trace/     research DAGs, dead ends, pivots, architecture decisions
src/       small scripts and environment notes suitable for publication
evidence/  README files and small summaries
```

## What Is Excluded

```text
checkpoints/
large logs
raw datasets
NAS-only artifacts
private environment files
```

The excluded files are not required to understand the research route. When a
large artifact matters, the public evidence summary names the run and records
where the internal artifact was stored.

## Current Focus

The active line is `s2-translation`, especially:

```text
s2-translation/logic/predicts.md
s2-translation/logic/solution/treeheap_algebra.md
s2-translation/trace/research_dag.yaml
```

The main open question is whether TreeHeap can move from token/path echo toward
algebraic structure operations such as SubHeap Kernel Search.
