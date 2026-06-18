# nio.log — ARA Research Artifacts

Organized per ARA protocol (4-layer: `/logic`, `/src`, `/trace`, `/evidence`).
Supports multiple research topics with cross-referencing.

## Structure

```
ara/
├── index.yaml                    ← multi-topic registry + dependency graph
├── m0-treeheap-math/             ← Active: TreeHeap as algebraic toolbox
│   ├── logic/{problem,predicts,experiments}.md
│   ├── logic/solution/algebra.md
│   ├── src/treeheap_math_probe.py
│   ├── trace/research_dag.yaml
│   └── evidence/treeheap_math_probe/
│
├── s2-fold-stack/                ← Active: Fold Stack replaces Transformer
│   ├── logic/{problem,claims,experiments}.md
│   ├── src/environment.md
│   ├── trace/research_dag.yaml   ← ★ Dead ends + pivots saved
│   └── evidence/README.md        ← Results → claims binding
│
├── s1-echo-session/              ← (planned) Phase 1 Echo
├── bridge-design/                ← (planned) L0/L1/L2 bridge
├── wmt-benchmarks/               ← (planned) WMT baselines
└── sametime-pipeline/            ← (planned) Infrastructure
```

## How to Add a New Topic

```bash
mkdir -p ara/{topic-id}/{logic,src,trace,evidence}
cp template.md ara/{topic-id}/logic/problem.md   # ← fill in
cp template.md ara/{topic-id}/trace/dag.yaml     # ← fill in
# Update ara/index.yaml with new topic entry
```

## Key Rule

**Every claim in `/logic` must have an evidence pointer to `/evidence`.**
**Every dead end in `/trace` must have: hypothesis + failure mode + lesson + pivot_to.**
