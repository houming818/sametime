# F-Function Claim Integration (F21)

Date: 2026-09-23

Status: registered integration plan; topology pilot recorded, not a completed
architecture claim.

## Purpose

TreeHeap's local composition rule is denoted by:

```text
h_parent = F_theta(h_child_1, ..., h_child_k; c)
```

`F_theta` is not an independent embedding product that can be declared
complete before TreeHeap READ exists. It is the shared FOLD operator that
creates the parent states later consumed by READ and a decoder. The research
separation below is therefore an *experimental causal separation*, not an
architecture separation.

## Relation To Existing Claims

| Existing claim | Relation to F21 | What F21 may establish | What it cannot establish |
|---|---|---|---|
| `S1-PLANE-C01` / C6-009 | F21 supplies the measurable operator-and-partition part of the latent-placement theory. | A relation/placement field has topology-sensitive compositions. | That language geometry is solved, or that a binary heap is language grammar. |
| `S1-MANIFOLD-C01` / C6-010 | F21 extends the idea of a fold-quality control surface from a toy relation field to real-text topology candidates. | A controlled change to topology or F parameters produces a measurable quality change. | That a broad real-language control surface is already mapped. |
| C6-008 | F21 tests one reason ordered fold may matter: changing composition order changes the composed state. | Order/topology sensitivity under a stated task contract. | Natural internal READ or addressed subheap readout. |
| C6-006 and C6-018 | These consume or traverse `F`-produced states, but are downstream mechanisms. | Nothing directly; they are held out as later bridge gates. | `stop/left/right` route induction, query-conditioned READ, or dynamic topology. |
| C7-001, C7-007, C7-008 | These are full-generation, multiresolution, or codec claims. | A better-defined candidate FOLD family for later controlled tests. | Translation gain, multiresolution use, root causality, information-pump semantics, or decoder-memory advantage. |

## F21 Claim Ladder

The ladder preserves the old claims and avoids treating a single end-to-end
loss as an explanation.

| ID | Claim | Required comparison | Current status |
|---|---|---|---|
| `S1-F-0` | Recursive `F_theta` remains finite, shape-compatible, and has a registered normalization/conservation contract at all tested depths. | Numerical stress and scale/depth controls. | open |
| `S1-F-1` | Under a fixed task and fixed capacity, F topology/parameters change retained predictive information relative to direct and bag-like controls. | Matched topology/F ablation with fixed seed/data/budget. | supported pilot / one seed |
| `S1-F-2` | A minimal, separately trained readout can recover a preregistered attribute from `F` parent states beyond a leaf/bag bypass. | Parent-only versus leaf/bag and ablation controls. | partial support / one-seed next-token bridge |
| `S1-F-3` | Different F depths contain causal, complementary information for a readout. | Per-depth-only and depth-removal interventions. | open; prior multiresolution evidence does not pass this gate |
| `S1-F-4` | Dynamic topology or recursive READ learns useful path decisions beyond fixed-topology and flat controls. | Learned route versus fixed/random/flat controls with route traces. | open; do not infer from F-1 |

## Recorded Pilot: Fixed WMT Topology Search

Evidence: `../evidence/s1_topology_search/real_wmt_mc_20260923/summary.json`

Contract:

```text
real WMT English-side text
4 observed tokens -> compose with a fixed candidate F topology -> predict token 5
vocabulary = 256; train/test windows = 12,000/3,000; 500 steps per candidate
```

The candidate family was `direct`, `balanced`, `left_deep`, `right_deep`, and
`skip`. Each candidate used the same 32-dimensional token table, parent
transform family, decoder, optimizer family, and nominal training budget.

Observed best row:

```text
left_deep = F(F(F(x1, x2), x3), x4)
test NLL = 3.139026; PPL = 23.0814; top-1 = 0.401667
```

This is an exploratory indication for `S1-F-1`: this exact sequential
prediction objective is sensitive to fixed composition topology, and the
left-deep candidate outperformed the listed alternatives in this run. It is
not a confirmatory `S1-F-1` test because `direct`, `skip`, and the three-node
trees instantiate different counts of local transform modules.

It does **not** show that a learned TreeHeap route collapses to a chain. The
topology was enumerated by the experimenter; no differentiable or discrete
router chose it. It also does not test translation, BLEU, full-sentence
generation, semantic grounding, READ, subheap addressing, or multiresolution
storage.

## Next Registered Bridge

Before any dynamic-route claim, run a matched `S1-F-1/F-2` parent-read bridge:

```text
freeze each candidate F after identical pretraining
train the same minimal readout on parent-only state
compare parent-only, leaf-only, bag/mean, and topology-permuted controls
evaluate held-out WMT windows on a preregistered attribute
```

Candidate attributes must be selected before training and must not be a
target-leaking copy task. Suitable first attributes are next-token prediction
from a frozen four-token parent or an explicitly held-out order-sensitive
attribute. The report must separate F quality from READ quality: a parent-only
failure cannot, by itself, identify whether F lost information or the readout
is inadequate.

## Decision Rule

Do not promote `S1-F-1` to a path-collapse or TreeHeap-superiority claim.
Promotion requires replicated seeds, an explicit flat control, and a bridge
result showing that the selected F parent state is usable without leaf bypass.

## F21 Matched Parent-Read Result

Evidence: `../evidence/s1_f_function_parent_read_f21/formal_seed20260923/`

The registered bridge ran on the same WMT short-window contract. Each arm had
the same `11,360` encoder parameters: one 32-dimensional token table and
three same-shaped local F modules. `left_deep`, `balanced`, and `right_deep`
used those three modules at their three binary internal nodes. `bag_3step`
used the same three modules after a leaf mean, without hierarchical child
composition. Every encoder was pretrained for 1,500 steps, frozen, then given
a fresh linear parent-only readout trained for 750 steps. A separate fresh
linear head over the frozen mean of the four leaves was the leaf/bag bypass
control.

| Arm | Frozen parent NLL | Fresh parent-only NLL | Fresh mean-leaf NLL | Parent minus mean-leaf |
|---|---:|---:|---:|---:|
| left_deep | 2.992374 | 2.999877 | 3.278185 | -0.278307 |
| balanced | 3.152645 | 3.149159 | 3.316358 | -0.167199 |
| right_deep | 3.365464 | 3.344524 | 3.333919 | +0.010605 |
| bag_3step | 3.228427 | 3.223870 | 3.358068 | -0.134198 |

Interpretation:

1. The fresh parent-only linear head stayed within `0.021` NLL of the
   pretraining head in every arm. Thus the frozen parent states are linearly
   usable for this specified task; this is the narrow F-2 bridge.
2. Composition topology mattered under matched encoder capacity: left-deep
   was the best parent representation and beat its own mean-leaf control by
   `0.278307` NLL. Balanced also beat its own mean-leaf control; right-deep
   did not.
3. The result supports a one-seed F-1 pilot and partial F-2 bridge only. It
   does not prove that left-deep is a universal language topology, that a
   trained router will choose it, or that recursive TreeHeap READ is solved.

The required successor is a fixed three-seed replication with an independent
held-out order-sensitive attribute. The same F family must be retained, and
all results, including a reversal, must be recorded.
