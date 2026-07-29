# C15: budgeted conditional TreeHeap index

Status: preregistered; implementation prepared; formal evidence pending

## Question

Can a TreeHeap act as an actual index rather than a compressed sentence vector?
The concrete test is whether a query such as `eat` can retrieve likely values
under a fixed node-visit budget, and whether the tree arrangement can be
improved directly by retrieval hit rate.

## Retraction of the naive theory

The earlier conditional-count proposal was incomplete and one of its central
arguments was wrong.

Assume every leaf `v` stores an exact co-occurrence count `C(q,v)`, every parent
stores the sum of its descendants, and routing uses exact child mass:

\[
P(child\mid q,node)=\frac{C(q,child)}{C(q,node)}.
\]

For a leaf whose path is `n_0, n_1, ..., n_d=v`, the path probability is

\[
\prod_{i=1}^{d}\frac{C(q,n_i)}{C(q,n_{i-1})}
=\frac{C(q,v)}{C(q,root)}.
\]

All intermediate terms cancel. Therefore exact full traversal gives the same
leaf distribution for every tree arrangement. Path NLL cannot teach a useful
placement. It is merely a hierarchical factorization of the flat conditional
table.

The naive design also stores one complete query table at every internal node.
With `Q` queries and `V` leaves, the flat table uses `Q*V` entries, while a full
binary count tree uses `Q*(2V-1)` entries plus links. Without compression or a
search budget, the tree is larger and does not improve retrieval.

Other limits:

- co-occurrence is association, not automatically a semantic class;
- one fixed path cannot represent every sense of a polysemous value;
- hard Hit@K is not differentiable;
- maximizing unconstrained hit rate admits trivial solutions such as returning
  every leaf or stopping at one giant bucket;
- an exact inverted index may beat the tree when memory and preprocessing are
  unrestricted.

These are falsifying controls, not implementation details.

## Revised claim

`S3-BUDGETED-CONDITIONAL-INDEX-C15`:

> Given a fixed-capacity binary TreeHeap, a fixed node-visit budget, and
> co-occurrence events `(query, value)`, the leaf placement is a learnable
> structural parameter. Directly minimizing retrieval miss rate by legal leaf
> swaps should improve held-out Hit@K over random placements when several
> queries share candidate structure. Exact unlimited path NLL must remain
> placement invariant.

This is an index claim, not a semantic-understanding, generation, consciousness
or Transformer-superiority claim.

## State and operators

The TreeHeap has a fixed complete binary address space. Its trainable state in
this probe is the permutation of values in leaf addresses:

\[
H_{state}=\pi: value\rightarrow leaf\_address.
\]

Leaves contain sparse query counts learned from training events. Internal
counts are deterministic `plus` folds:

\[
C_n(q)=C_{left(n)}(q)+C_{right(n)}(q).
\]

No external nodes are allocated. A legal structural update swaps two leaf
payloads and recomputes affected ancestors.

READ uses best-first recursive expansion. At each step, the unexpanded node
with the largest training mass `C_n(q)` is visited. Search stops after `B` node
visits or after `K` leaves have been returned.

## Dataset

The controlled corpus contains only `(query,value)` observations. It contains
four overlapping query families around food, medicine, vehicles and places,
including low-probability cross-family events. Category names are retained only
by the data generator for human-readable reporting; the placement optimizer
receives counts and Hit@K only.

Examples include:

```text
eat -> rice/noodles/apple/bread, occasionally medicine
prescribe -> ibuprofen/amoxicillin/aspirin/insulin
drive -> car/bus/bicycle/train
visit -> beach/museum/mountain/hotel
```

Train and held-out events are sampled independently from the same distributions.
A shuffled control destroys shared query-family structure while retaining row
and value marginals as closely as the finite toy permits.

## Experiment arms

1. `exact_random_layouts`: multiple random layouts with unlimited traversal;
   verify identical path NLL and leaf probabilities.
2. `budget_random`: distribution of held-out Hit@K across random layouts.
3. `budget_optimized`: pair-swap hill climbing minimizes training miss rate;
   evaluate the frozen best layout on held-out events.
4. `shuffled_budget`: repeat optimization after destroying shared conditional
   structure.
5. `flat_exact`: full conditional table as an accuracy and memory upper bound.

The discrete optimizer is deliberate. It tests whether an index objective
exists before introducing a differentiable placement relaxation.

## Predictions

1. Unlimited exact path NLL varies by less than `1e-12` across layouts.
2. Exact TreeHeap query-count storage exceeds flat table storage.
3. With a preregistered finite budget, optimized held-out Hit@K exceeds the
   mean random-layout Hit@K by at least `0.05` absolute.
4. The optimized training gain transfers to held-out events; train-test Hit@K
   gap remains below `0.10`.
5. Destroying shared conditional structure reduces the optimization advantage
   by at least `0.02` absolute.
6. The `eat` trace returns several high-count ingestion candidates without
   visiting the entire tree.

## Falsification

Reject the useful-index claim if exact path NLL depends on layout, if the
finite-budget objective cannot beat random placement on held-out events, or if
all gain disappears outside training events. Even a passing toy does not prove
that natural-language co-occurrence has a single tree topology or that a
differentiable placement rule exists.

## Decision after the probe

- If only P1-P2 pass, retain the negative theorem and abandon exact-count path
  NLL as a TreeHeap learning objective.
- If P3-P6 also pass, the next step is a differentiable or online local-swap
  placement kernel on real corpus events, with matched memory and node-visit
  budgets against a flat inverted index.
- Do not reconnect this result to WMT generation until the index can retrieve
  held-out corpus values and accept new observations without a full rebuild.

