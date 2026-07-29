# SameTime / TreeHeap ARA Summary

Status: living research entry point

Updated: 2026-07-29

Branch: `experiment/private-protocol-battle`

Scope: M0 algebra, S1 echo/encoding, S2 translation, and S3 generation/indexing

This document is the shortest complete route into the current TreeHeap research.
It summarizes what the project means, what has actually been measured, what has
failed, and what should be tested next. It is not a substitute for raw evidence.
When this summary and an evidence file disagree, the evidence file and the
registered claim boundary win.

## 1. Executive Verdict

TreeHeap is currently a **research architecture with several verified algebraic
and causal mechanisms, but without a usable general language model**.

The evidence supports these narrow statements:

1. Fixed-capacity ordered heaps support useful local algebra: composition,
   decomposition, path-aware difference, mirror, bounded rotation, recursive
   fold/unfold, and probability-lifted write/search operations.
2. Tree-structured address and subheap information can create measurable
   inductive bias in controlled relocation, routing, and finite-budget retrieval
   tasks.
3. Recursive encoders can make root and intermediate states causally depend on
   ordered children. Forced recursive decoders can read that information.
4. Exact unlimited conditional-count likelihood cannot learn tree placement:
   the path probabilities telescope and become layout invariant. Topology starts
   to matter only when memory, node visits, state width, or compute are bounded.
5. A payload may temporarily stop at an internal node and later move downward as
   pressure increases, while the parent retains a folded summary. A local smoke
   test demonstrates collision splitting, not yet a learned language index.

The evidence does **not** establish syntax emergence, a semantic world model, a
stable encoder-decoder private protocol, superiority over Transformer, usable
chat, or GPT-level generation. Current non-teacher-forced generation is weak and
often repetitive. STONE-1 is incomplete.

## 2. Authoritative Record

| Need | Source |
|---|---|
| Root claim tree and research history | [`PAPER.md`](PAPER.md) |
| Chinese root paper | [`PAPER.zh.md`](PAPER.zh.md) |
| M0 claim registry | [`m0-treeheap-math/logic/claims.md`](m0-treeheap-math/logic/claims.md) |
| S1 claim registry | [`s1-echo/logic/claims.md`](s1-echo/logic/claims.md) |
| S2 claim registry | [`s2-translation/logic/claims.md`](s2-translation/logic/claims.md) |
| S3 claim registry | [`s3-generation/logic/claims.md`](s3-generation/logic/claims.md) |
| Reproducible programs | `ara/*/src/` |
| Metrics and run summaries | `ara/*/evidence/` |
| Failed paths and pivots | `ara/*/trace/` |

ARA uses the chain:

```text
claim -> prediction -> experiment -> evidence -> decision -> next claim
```

Blogs explain the work to readers. Blogs are not evidence. A status such as
`supported` is always narrower than the project vision.

## 3. The Research Question

The project asks whether a fixed-capacity, ordered, recursively addressable state
can learn the same broad mapping tasks as flat neural networks while gaining an
advantage on problems that depend on:

- address and path;
- reusable substructure;
- local composition and decomposition;
- bounded search and prefix reuse;
- multiple resolutions;
- delayed probabilistic collapse.

The desired comparison is not "tree versus intelligence." It is:

```text
same task + matched data + matched memory/compute
-> does TreeHeap's structural bias improve efficiency, causality, or generalization?
```

## 4. Current System Model

```text
events / tokens / vectors
        |
        v
WRITE or placement kernel
        |
        v
ordered leaves and internal STOP payloads
        |
        v
recursive FOLD / information pump
        |
        v
H_state: root + addressed intermediate details + leaves
        |
query + recursive READ(stop, left, right)
        |
        v
probability container
        |
collapse under task context
        v
retrieved value, next token, or generated structure
```

Loss can train kernel parameters, token embeddings, readout parameters, and in a
soft formulation placement probabilities. A geometric or algebraic operator may
remain fixed. Current implementations do not justify saying that every parameter
tensor is itself a TreeHeap.

## 5. Core Vocabulary

| Term | Current precise meaning |
|---|---|
| TreeHeap | A fixed-capacity ordered recursive state with stable parent/child addresses and local operators. A tensor reshaped like a tree is not enough. |
| `H_state` | The runtime contents of all TreeHeap nodes. It may contain exact payloads, summaries, details, and latent vectors. |
| `theta` | Trainable parameters used by write, fold, route, unfold, or read kernels. `theta` and `H_state` must be kept conceptually separate. |
| Kernel | A shared local function applied to a node/subheap. It may emit a state update, score map, or `stop/left/right` distribution. |
| FOLD | A child-to-parent operator. Repeated bottom-up application creates root and multiresolution states. |
| UNFOLD | A parent-plus-detail-to-children operator. Exact algebraic unfold is possible for some codecs; useful language decoding is a separate inductive problem. |
| Information pump | The recursive FOLD process that moves task-relevant information upward. It is a mechanism, not proof that roots contain human-readable summaries. |
| STOP | A valid result at any node. It can represent an exact payload, a coarse category, or a probability bucket, depending on the registered model. |
| Probability container | A distribution kept without immediate argmax, such as `{stop, left, right}` or several candidate graphs. |
| Mirror | The deterministic left/right symmetry operation. This project no longer calls it conjugation. |
| Private protocol | A co-adapted encoder/decoder representation that need not be human-readable. A useful private protocol has not yet been demonstrated for free generation. |
| Resolution | Near-root states have larger receptive fields; near-leaf states retain more local detail. Coarse semantic meaning is a hypothesis unless task evidence supports it. |

## 6. Required Invariants

A result counts as TreeHeap evidence only when the tested mechanism uses the
structure rather than bypassing it.

1. **Fixed capacity:** experiments must state the node and state-width budget.
2. **Ordered address:** left/right and path identity must affect the computation.
3. **Recursive locality:** shared local operators must compose across depth.
4. **No hidden flat bypass:** a decoder must not recover the target from visible
   target tokens, an unrestricted flat table, or leaked route labels.
5. **Conservation or declared loss:** migration/fold must state which quantity is
   preserved and which information is intentionally discarded.
6. **Causal intervention:** shuffle, mirror, root/detail removal, or budget changes
   must alter the output when the corresponding structure is claimed to matter.
7. **Matched baseline:** memory, search visits, updates, data, and evaluation must
   be comparable.

## 7. Layer Status

| Layer | Purpose | Current state |
|---|---|---|
| M0 | Algebraic toolbox | Several deterministic identities and differentiable toy mechanisms are supported. This is the strongest foundation. |
| S1 | Write, echo, context routing | Capacity and controlled routing are supported; token-only semantics is rejected; a natural corpus encoder/private protocol remains open. |
| S2 | Fold stack and translation | Fold/action signal exists, but graph assembly and source-conditioned quality remain bottlenecks. Historical translation is bounded and often behind flat baselines. |
| S3 | Generation, codec, indexing | Many causal mechanisms are supported. Free generation and the strong private-protocol claim are not. C15-C17 moved the frontier toward bounded conditional indexing and dynamic payload placement. |

## 8. Strongest Evidence

### 8.1 M0 algebra and learning access

- Deterministic probes support closure, non-commutativity, inverse-like
  reconstruction, projection, subheap matching, mirror involution, finite-field
  decoders, and bounded rotation.
- The diff algebra has a finite-difference check with absolute error
  `4.21e-10`; one gradient step reduced a toy loss from `29.1384` to `0.00066`.
- Soft Plus received nonzero gradients and collapsed to the correct hard address
  at low temperature in its registered toy. Its hand-crafted alignment features
  prevent promotion to a general learned router.
- Full-tree kernel convolution can express deterministic search/write/mirror
  maps. This supports an operator language, not language intelligence.

Relevant mathematical context includes rooted-tree algebra, Hopf/BCK-style
composition and cuts, operadic many-input/one-output composition, and classical
tree kernels. The project has not proved that its complete learned system is
identical to any one of those frameworks.

### 8.2 Structural causality

- In a controlled relocation toy, path-plus-subheap features generalized where
  flat/path-only variants failed. The scope is synthetic and narrow.
- A four-head root compressor reached valid NLL `6.2365`; breaking address
  pairing added `2.6252` NLL and removing individual heads caused large damage.
  This establishes ordered structural use in that model, not semantic heads.
- A multiresolution codec produced a monotonic rate-distortion curve. At detail
  rate `k=64`, token top-1 was `0.9964`, but this was a bounded reconstruction
  mechanism rather than entropy-coded language compression.
- With a frozen encoder, forced recursive reading improved test NLL from a root
  control `3.5149` to `3.4636`; a 2% depth floor improved it further to `3.4117`.
  This proves accessible intermediate information, not spontaneous routing.

### 8.3 C14-C17: the current frontier

**C14 target TreeHeap history.** Autoregressive WRITE and incremental FOLD made
the target history causal: zeroing history added `4.3857` NLL and root-only target
state added `0.7718`. Generation remained unusable (`BLEU-4 0.2150`, unique
output fraction `0.184`), and source routing collapsed to the deepest level.

**C15 bounded conditional index.** For exact conditional counts,

```text
P(leaf | q) = product P(child | q, node)
            = C(q, leaf) / C(q, root)
```

so unlimited exact NLL is independent of tree placement. Under a 10-node visit
budget, optimized placement improved held-out Hit@3 from `0.6921` to `0.8125`
and matched flat exact Top-3. The tree used more raw count entries, so compression
is not established.

**C16 internal STOP.** Allowing exact payloads at internal nodes did not improve
Hit@3 (`0.8033` versus `0.8033`) but shortened mean correct depth from `4.0000`
to `2.2842`. This is a local smoke result, not formal evidence.

**C17 pressure-split insertion.** A fixed 31-node heap inserted the first payload
at root, displaced it downward on collision, and ended with 16 exact leaf
payloads plus 15 internal summaries. Across eight insertion orders, no existing
payload moved upward; signature routing improved mean budget AUC from `0.5269`
to `0.5574`. Mean payload depth was not monotonic, so that registered gate failed.
The signature was engineered from co-occurrence counts, not learned end to end.

## 9. Important Negative Results

- Token-only path hashing is capacity, not contextual semantics.
- The old 128D checkpoint was highly collapsed (`cosine` near `0.985`) and cannot
  support a solved syntax-energy claim.
- Naive world-field/tensor energy experiments did not establish that the correct
  sentence naturally has minimum energy.
- Exact count-path likelihood cannot train topology because it telescopes.
- Random vector sums lose subheap identity; compose must be non-trivial.
- Root-only causal use is not recursive decoding. More training did not make a
  50M decoder spontaneously read deeper nodes.
- C10 processed 1.410B target tokens and achieved low teacher-forced NLL, but the
  CLI collapsed to repetitive Belt-and-Road text. Full target teacher forcing and
  visible EOS invalidated the translation/private-protocol interpretation.
- C11 proved some source dependence but still repeated short phrases.
- One-shot H-state unfold (C12) and parallel emergent protocol (C13) collapsed.
- Distillation teacher uncertainty did not improve over gold targets.
- Larger parameter count alone did not improve the registered 50M model.
- No result is evidence of consciousness, feelings, a world model, GPT-2/GPT-4
  capability, or commercial readiness.

These failures are part of the result. They prevent repeated research loops.

## 10. What Is Known and Unknown

### Known

- A TreeHeap can carry exact and lossy state at several depths.
- Ordered child structure can be causally important.
- Shared recursive operators can transmit gradients and information.
- A decoder can be forced to use intermediate states.
- Under finite search, placement affects retrieval quality.
- Internal STOP can reduce retrieval depth.
- Collision pressure can move exact payloads downward in fixed memory.

### Unknown

- How raw corpus statistics should learn placement without engineered signatures.
- Which bounded node representation preserves useful conditional distributions.
- Whether a learned encoder and decoder form a stable, useful private protocol.
- Whether parent states become semantic contours rather than arbitrary hashes.
- Whether TreeHeap beats strong Transformer, recurrent, retrieval, or compressed
  flat baselines at matched compute and quality.
- Whether variable-depth routing can emerge without a forced floor or collapse.
- Whether a fixed-capacity TreeHeap can produce fluent, source-relevant free text.

## 11. Next Proof: Selective Bucket Migration

The immediate claim should refine C17 instead of starting another decoder.

Start with a node whose STOP bucket contains:

```text
depth1.node0 STOP -> {word1, word2, word3}
```

After more observations, the model may decide that `word3` needs finer
separation:

```text
depth1.node0 STOP -> coarse summary of {word1, word2, word3}
depth2.node0 STOP -> exact word1/word2 group
depth2.node1 STOP -> exact word3 or its finer bucket
```

`word3` is not forgotten globally. Its **exact payload moves down**, while its
probability mass or compressed influence remains represented in the parent's
subtree summary. The conservation rule should be explicit:

```text
subtree_mass(parent, w)
  = local_mass(parent, w)
  + subtree_mass(left, w)
  + subtree_mass(right, w)
```

Migration must be atomic: write child, verify child, recompute parent, verify
mass/checksum, then remove the parent's exact copy. Otherwise the system can
silently duplicate or forget information.

The next experiment should compare leaf-only, static internal STOP, random
split, engineered pressure split, and learned pressure split under identical:

- fixed node/state-byte capacity;
- streaming event sequence;
- 1..B node-visit budgets;
- update compute;
- no duplicate exact payloads.

Primary metrics: Hit@K-versus-visits AUC, exact full retrieval, parent mass
error, migration/forgetting rate, memory bytes, and held-out stream performance.
Only after the controlled bucket proof passes should it move to real corpus
co-occurrence events and a learned kernel.

## 12. Engineering State

- Main repository under review: `holds/SameTime-depth-growth`.
- GPU host: `io`; keep its configured power/frequency safety limits enabled.
- Preferred execution: serial `taskd` queue; use direct SSH only as fallback.
- Local io corpus mirror: `/home/nio/datasets` (about 25 GB).
- NAS source: `/mnt/nas/datasets` (about 51 GB).
- Corpora include WMT material and Chinese news, web, wiki, BELLE, Baike,
  translation, Zhihu, and medical datasets.
- Formal remote runs must preserve code commit, command, environment, logs,
  checkpoints when required, and `summary.json` under `evidence/`.

## 13. Gates Before Product Claims

STONE-1 is not complete. A release-quality milestone requires at least:

1. source-conditioned free generation, not target teacher-forcing leakage;
2. low repetition and non-trivial conditional diversity;
3. reproducible multi-seed quality;
4. causal TreeHeap address/depth use with no flat bypass;
5. matched strong baselines and compute accounting;
6. a CLI whose declared task matches its training objective;
7. public checkpoint, tokenizer, inference command, model card, and limitations.

The nearest honest product is a bounded TreeHeap research codec/index demo, not
a general conversational AI.

## 14. Reviewer Handoff

A new reviewer should proceed in this order:

1. Read this file.
2. Read the relevant row in the topic `logic/claims.md`.
3. Open its experiment design and source code.
4. Inspect `evidence/*/summary.json` and raw logs.
5. Check that intervention and baselines test the stated mechanism.
6. Update claim status, trace, `PAPER.md`, and this summary together.

Do not infer progress from blog count, training duration, GPU use, low
teacher-forced NLL, or model parameter count alone.
