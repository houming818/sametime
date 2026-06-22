# SameTime

SameTime is an open experiment notebook for learning and testing language-model
systems. It contains WMT/NMT baselines, Semantic Prefix Routing experiments, and
the public ARA research records used to track claims, failures, pivots, and
evidence.

## Open Research Artifacts

The `ara/` directory is a public, lightweight mirror of the research record:

```text
ara/
├── PAPER.md
├── index.yaml
├── s1-echo/
├── s2-translation/
└── s3-generation/
```

Start from `ara/PAPER.md` for the current claim tree, evidence map, downgraded
claims, and next proof queue.

ARA means Agent-Native Research Artifact. The intended reading order is:

```text
logic/     current claims, predicts, and experiment designs
trace/     why the route changed: dead ends, pivots, decisions
evidence/  small summaries and pointers to reproducible evidence
src/       small scripts and environment notes when they are safe to publish
```

Large checkpoints, raw logs, and local NAS artifacts are intentionally excluded
from this public mirror. Public evidence files keep summaries and pointers, so
humans and AI agents can study the reasoning process without downloading
multi-GB experiment outputs.

## Related Blog

The SPR / TreeHeap notes are published here:

- https://www.grepcode.cn/spr/
- https://www.lostmap.cn/spr/

The blog is the human-readable explanation. The `ara/` directory is the
machine-readable research trail.
