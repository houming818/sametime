# S1 Encoder as World Observer

Date: 2026-07-07
Status: design / open theory
Related claim: `S1-ENCODER-OBS-C01`

## Problem

The current S1 route discussion is downstream of a more basic question:

```text
How does raw data become a TreeHeap state?
```

If the encoder writes a bad `H_tree`, then route, read, collapse, and decode are
all reading a bad state.

So the central object is:

```text
Encode_Theta(raw observations) -> H_tree
```

The encoder is the "world observer": it sees repeated observations and must
place information into a TreeHeap so that later operations can reconstruct,
predict, route, and transfer.

## Claim

TreeHeap semantic structure should not be assumed by philosophical reasoning or
hand-written ontology.

It must be selected by loss:

```text
good placement = lower reconstruction loss
               + lower context prediction loss
               + better contrastive alignment
               + better replacement consistency
               + lower description length
```

In short:

```text
TreeHeap ordering should emerge from compressibility.
```

## Proposed Encoder Interface

```text
tokens / spans / observations
  -> token states
  -> candidate placement scores
  -> soft TreeHeap write
  -> soft compose
  -> H_tree
```

Minimal differentiable form:

```text
score(i, j; Theta_place) = compatibility of two observed units
P_merge(i, j) = softmax(score)
H_parent = Compose_Theta(H_i, H_j)
```

Hard tree placement should be treated as a later collapse of a soft placement
distribution.

## Loss Family

### 1. Echo / Reconstruction

The tree must not destroy the sample:

```text
Decode(H_tree, read_position=k) -> token_k
Decode(H_tree, read_subheap=i)  -> subheap_i
```

This prevents semantic compression from becoming an unrecoverable bag.

### 2. Context Prediction

If a token or subheap appears in a context, the encoded state should help predict
that context:

```text
eat [MASK] -> rice / noodle / apple
take [MASK] -> amoxicillin / ibuprofen
```

This is the unsupervised signal that replaces hand labels.

### 3. InfoNCE / Contrastive Transfer

Positive pairs can be generated from observations, not labels:

```text
same context window
same translation pair
same masked slot
same repeated construction
```

Negative pairs can be sampled from mismatched contexts:

```text
wrong verb-object pair
wrong translation pair
random replacement
shuffled corpus pair
```

Loss:

```text
L_InfoNCE = -log exp(sim(anchor, positive)/tau)
                 / sum_j exp(sim(anchor, candidate_j)/tau)
```

### 4. Replacement Consistency

If two leaves can replace each other in many contexts without hurting prediction,
they should be close in the TreeHeap prefix structure.

Examples:

```text
eat rice
eat noodle
eat apple
```

This should pressure `rice/noodle/apple` toward a shared prefix.

### 5. Description Length / Compression Pressure

A prefix is useful if it reduces the number of rules needed to explain data.

Without a prefix:

```text
eat rice
eat noodle
eat apple
cook rice
cook noodle
cook apple
```

With a prefix:

```text
food = {rice, noodle, apple}
eat food
cook food
```

This is the mathematical reason TreeHeap should discover internal nodes:

```text
shared prefix = shorter explanation
```

## Minimal Gate After Review

DeepSeek and Houming818's review raised a useful constraint:

```text
Do not start with five losses.
First prove that one small learning loop can induce structure.
```

So the first executable proof is deliberately smaller:

```text
L = L_echo + L_context
```

It must still keep the important TreeHeap part:

```text
Theta_place:
  leaf/object -> soft internal prefix slot

Theta_compose:
  leaves assigned to a slot -> prefix/internal-node state

Context prediction:
  observed verb-object pairs are positives
  unobserved held-out pairs are unknown, not direct negatives
  a global density constraint prevents all pairs from becoming positive
  verb/context reads through learned prefix states
```

This is not allowed to be a passive random-vector sum:

```text
bad:  H_parent = sum(child vectors)
good: H_parent = Compose_Theta(assigned child states, slot state)
```

The reason is SPR-047's compact-route failure: random sums can preserve some
mass but lose subheap identity.  The encoder must learn placement and compose
together.

## Three Reasoning Modes

The encoder should eventually support three reasoning modes by how it structures
`H_tree`.

### Deduction

Given a prefix and a rule:

```text
amoxicillin -> medicine -> consumable
eat accepts consumable
therefore eat + amoxicillin
```

### Induction

From repeated observations:

```text
eat rice
eat noodle
cook rice
cook noodle
```

infer a reusable latent prefix:

```text
food-like
```

### Analogy

Use structure-preserving relations:

```text
rice : food :: amoxicillin : medicine
eat : food :: take : medicine
```

The first proof should only target induction and held-out transfer. Full analogy
can wait.

## Minimal Proof Proposal

Use a synthetic unlabeled corpus:

```text
eat rice
eat noodle
eat apple
cook rice
cook noodle
cook apple

take amoxicillin
take ibuprofen
prescribe amoxicillin
prescribe ibuprofen

drink water
drink milk
pour water
pour milk
```

Do not provide `food`, `medicine`, or `beverage` labels during training.

The full future direction may use:

```text
L = L_echo
  + L_context_prediction
  + L_InfoNCE
  + L_replacement_consistency
  + lambda * L_description_length
```

But the immediate gate uses only:

```text
L = L_echo + L_context_prediction
```

Evaluate after training with labels only as an audit tool:

```text
cluster purity
nearest-neighbor class accuracy
held-out context prediction
structured corpus vs shuffled corpus gap
```

Key prediction:

```text
structured corpus should induce stable prefix clusters;
shuffled corpus should not.
```

The current planned script is:

```text
ara/s1-echo/src/s1_encoder_minimal_observer_probe.py
```

It trains on two corpora:

```text
structured:
  eat/cook/order    -> rice/noodle/apple
  take/prescribe/buy -> amoxicillin/ibuprofen/aspirin
  ...

shuffled control:
  same verb/object counts, but category law destroyed
```

Gold labels such as `food` and `medicine` are hidden during training.  They are
used only after training to compute cluster purity, pairwise F1, and held-out
transfer.

## Falsification

This direction weakens if:

```text
1. shuffled corpus produces the same clusters;
2. pair/bag/flat baselines match held-out transfer;
3. echo is preserved only when semantic structure disappears;
4. prefix clusters require hand labels during training;
5. learned prefixes cannot improve route/read over random-sum compact state.
```

## Boundary

This document does not claim TreeHeap has learned natural semantics.

It only defines the next research gate:

```text
Can an encoder learn TreeHeap prefix structure from observation statistics?
```

The current supervised semantic-prefix proof is only a target shape. The next
proof must remove hand-provided prefixes from training.
