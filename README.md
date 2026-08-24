# SameTime

[中文简介](README.zh.md) | English

SameTime is an open research project exploring **TreeHeap**, a recursive,
fixed-capacity neural architecture for language encoding, generation, and
learned private protocols between an encoder and a decoder.

The project was initiated by **Houming818** and has evolved through sustained
human-AI collaboration: mathematical ideas are translated into falsifiable
claims, implemented as experiments, reviewed by multiple agents, and preserved
with both successful and negative evidence. TreeHeap is therefore not presented
as a finished design revealed all at once. Its public development history is
part of the research result.

## The Core Question

Can a language system organize information as a recursively computed TreeHeap,
rather than merely storing a sequence and decorating it with tree-shaped
indices?

The current candidate dataflow is:

```text
tokens
-> WRITE into leaf states
-> fixed-capacity XOR Butterfly communication
-> bounded and reversible recursive FOLD
-> multiresolution H_state (root + parent/detail levels + leaves)
-> recursive READ by the decoder
-> token probability distribution
-> autoregressive generation
```

In this design:

- **FOLD** combines two child states into a coarser parent state while retaining
  recoverable detail;
- **UNFOLD** provides the corresponding algebraic reconstruction path;
- **Butterfly** allows distant addresses to interact through sparse, local
  kernels without allocating an unbounded tree;
- **H_state** is the runtime TreeHeap carrying multiple resolutions;
- **READ** lets the decoder use the whole hierarchy rather than treating the
  root as a sentence hash or the leaves as a flat bypass;
- gradient learning is expected to form a private encoder-decoder protocol
  inside these explicit structural constraints.

The runtime state is a TreeHeap. In the current implementation, learned model
parameters are still tensor collections shared by TreeHeap kernels; parameter
memory itself has not yet been reorganized as a growing TreeHeap.

## Why This Is Not Just a Tree-Shaped Array

Earlier experiments exposed several false shortcuts: flat `L x L` routing,
geometry features that leaked the correct branch, leaf-only decoding, learned
STOP gates collapsing to the finest level, and loss curves improving while
generation collapsed to a fixed phrase. These failures changed the architecture
and remain in the public record.

A result counts as TreeHeap evidence only when the recursive state and its
addresses causally affect output under controlled interventions. Merely placing
tensors in a heap array is not enough.

## Current Research Status

Evidence currently supports the following narrow statements:

- recursive TreeHeap operators and reversible FOLD/UNFOLD paths can be
  implemented and numerically audited;
- TreeHeap states can be trained on real WMT sequence-to-sequence data and can
  produce nonempty language output;
- Butterfly communication, recursive FOLD, and multilevel state interventions
  can produce measurable causal effects;
- pretraining and task training can be connected through one checkpoint and one
  reproducible data pipeline.

The following remain open:

- learning a stable, complementary protocol across coarse, middle, and fine
  resolutions;
- preventing the decoder from collapsing onto a single convenient depth;
- obtaining consistently useful long-form generation and stronger translation
  quality at available compute scale;
- demonstrating general reasoning, dialogue, memory, or a world model.

SameTime does not claim that TreeHeap is complete, conscious, or superior to
other architectures. Its present contribution is a new, testable architecture
family and an unusually complete record of how that family is being discovered.

## How to Read the Project

For a concise technical history, start here:

- [TreeHeap generation evolution map (Chinese)](ara/s3-generation/EVOLUTION.zh.md)
- [ARA paper-style overview (Chinese)](ara/PAPER.zh.md)
- [ARA paper-style overview (English)](ara/PAPER.md)
- [Current claim registry](ara/s3-generation/logic/claims.md)

The ARA directory follows this structure:

```text
logic/     claims, predictions, experiment contracts, and decisions
src/       implementations and evaluation programs
trace/     pivots, dead ends, and the reasons the route changed
evidence/  machine-readable summaries, logs, hashes, and artifact pointers
```

Experiment IDs such as `C03`, `C12`, or `C13` are archive coordinates, not
human-facing capability levels. Readable experiment names and their inheritance
relationships are maintained in the evolution map.

## Public Writing and Reproduction

Human-readable SPR / TreeHeap articles:

- <https://www.grepcode.cn/spr/>
- <https://www.lostmap.cn/spr/>

The first downloadable TreeHeap translation proof-of-concept is the historical
**STONE-1 Candidate C08** release:

- <https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.tar.gz>
- <https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.sha256>
- <https://github.com/houming818/sametime/tree/stone1-candidate-c08>

It is preserved as a research checkpoint, not advertised as the final SameTime
model. Large checkpoints, licensed corpora, and local NAS artifacts are not
stored directly in Git. Evidence files retain hashes, summaries, commands, and
artifact locations needed for audit.

## License

SameTime source and original research materials are distributed under
GPL-3.0. See [LICENSE](LICENSE). Third-party datasets and models retain their
own licenses and are not relicensed or redistributed by this repository.
