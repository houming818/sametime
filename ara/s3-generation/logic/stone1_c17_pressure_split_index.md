# C17: pressure-split TreeHeap insertion

Status: local smoke partial; displacement mechanism supported

## Correction

C16 optimized a static placement and could leave a frequent token permanently
at root. The revised hypothesis is dynamic: a token may temporarily occupy a
shallow node while the tree contains little data, but a collision converts that
node into an aggregate and pushes both the old and new payload downward.

## Claim

`S3-PRESSURE-SPLIT-INDEX-C17`:

> Repeated insertion into a fixed-capacity TreeHeap can create hierarchy by a
> deterministic collision-split rule: payloads migrate downward, internal
> nodes become folded query summaries, and recursive query search uses those
> summaries to recover held-out candidates. No token is assigned a permanent
> root or parent address.

This is a mechanism smoke, not a language or optimal-index claim.

## Insertion

1. The first value occupies root.
2. Inserting a different value into an occupied payload node creates left and
   right children, moves the old and new payloads to them, and clears the
   parent's exact payload.
3. An internal node routes later values to the child with the nearest
   co-occurrence signature; the random arm chooses a child randomly. A child
   whose fixed descendant capacity is full is ineligible, so pressure is
   redirected to its sibling instead of growing an unbounded chain.
4. Every insertion recomputes ancestor query mass by exact `plus`.

The address space is a depth-4 complete heap with 31 fixed nodes and 16 leaf
slots. No node is allocated outside it. Only active occupancy grows.

## Predictions

1. After one insertion root holds the first payload; after two distinct
   insertions root is an aggregate and the first payload depth is at least 1.
2. With `N` distinct values, active nodes equal `2N-1` and exact payloads occur
   only at leaves.
3. Mean payload depth does not decrease as collisions accumulate.
4. Full recursive retrieval reaches the same Top-3 as the flat count table.
5. Under finite visit budgets, signature routing has higher mean held-out
   Hit@3 area than random collision routing.

## Falsification

Reject the mechanism if payloads are copied rather than moved, parent mass does
not equal descendant mass, insertion escapes fixed capacity, or full retrieval
loses candidates. Reject the indexing advantage if signature routing does not
beat random routing across insertion orders. A pass does not establish a
learned differentiable encoder; signatures are an explicit corpus statistic in
this smoke.

## Local smoke result (2026-07-29)

The successful smoke used eight insertion orders, 200 training and 100
independent test events per query, a fixed depth-4/31-node address space and a
Hit@3 budget curve from 1 to 20 node visits.

An initial implementation without capacity repair formed an increasingly deep
single chain and exhausted the fixed address space. The retained implementation
adds one structural rule: a full child subheap is ineligible, so insertion
pressure is redirected to its sibling. This is required for bounded growth.

| Measurement | Result | Gate |
|---|---:|---|
| Root after first insertion | `hotel` payload | pass |
| Root after second insertion | aggregate | pass |
| First payload depth after two | `1` | pass |
| Active/expected nodes after 16 values | `31 / 31` | pass |
| Parent query-mass equality | exact | pass |
| Existing payload ever moved upward | no | diagnostic pass |
| Mean payload depth monotonic | no | preregistered gate fail |
| Full tree/flat Hit@3 | `0.8033 / 0.8033` | pass |
| Signature/random budget-curve AUC | `0.5574 / 0.5269` | pass |
| Signature-routing AUC gain | `+0.0304` | pass |

The failed mean-depth gate does not show an existing payload moving upward.
Every collision moves the old payload down exactly one level, and the
per-payload audit found no upward move. The average can temporarily fall when a
new payload is introduced at a depth shallower than the existing population's
mean. The preregistered aggregate statistic was therefore too strong, and its
failure is retained.

At full capacity all 16 exact payloads end at depth 4, while all 15 internal
nodes contain folded query summaries. The useful result is dynamic formation:
temporary shallow payloads are displaced, parent sums remain exact, and the
resulting signature-routed arrangement retrieves useful candidates earlier
than random routing on average. This is still an explicit co-occurrence
encoder, not a learned private protocol.
