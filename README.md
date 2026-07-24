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

## Downloadable Checkpoint

The first executable fixed-root TreeHeap translation checkpoint is published as
the **STONE-1 Candidate C08** release:

- https://repos.grepcode.cn/houming818/grepcode-sametime/releases/tag/stone1-candidate-c08
- https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.tar.gz
- https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.sha256

GitHub remains a public source mirror:

- https://github.com/houming818/sametime/tree/stone1-candidate-c08

It is a research POC, not `STONE-1: COMPLETE`. The release includes the frozen
encoder, EOS-tail decoder, tokenizer, model card, checksums, and CLI usage.

## Related Blog

The SPR / TreeHeap notes are published here:

- https://www.grepcode.cn/spr/
- https://www.lostmap.cn/spr/

The blog is the human-readable explanation. The `ara/` directory is the
machine-readable research trail.

## License

SameTime is distributed under GPL-3.0. See `LICENSE`. Raw training corpora are
not redistributed.
