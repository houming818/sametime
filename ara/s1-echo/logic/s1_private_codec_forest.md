# S1 Private Codec Forest

Date: 2026-07-10
Status: proof / pilot
Related claim: `S1-PRIVATE-CODEC-C01`

## Problem

The encoder discussion needs one precise split:

```text
Theta = parameter TreeHeap / parameter forest
Q     = query or kernel request
H     = current activation state
Loss  = scalar signal used to move Theta
```

The important point is that the answer should not be hand-written into `Q`.
The answer should be learned into `Theta` by gradient descent.

## Minimal Claim

A TreeHeap can act as a private codec if:

```text
1. a parameter TreeHeap head learns a distribution from scalar loss;
2. another parameter TreeHeap head learns a reusable operator;
3. the two heads can be composed serially on a held-out intermediate state.
```

This corresponds to Houming818's multi-head forest view:

```text
Theta_forest = {Theta_food, Theta_filter, ...}
```

and serial reasoning:

```text
H1 = K_food(H0; Theta_food)
H2 = K_fruit_filter(H1; Theta_filter)
```

The heads do not need to share a human-readable internal code. They only need
to share a working private protocol that the decoder/kernel can use.

## Toy World

Use a small finite vocabulary:

```text
rice, noodle, apple, mango, amoxicillin, ibuprofen, stone, car
```

The source heads learn probability buckets:

```text
food -> rice / noodle / apple / mango
rice_apple -> rice / apple / stone
noodle_mango -> noodle / mango / car
```

The `fruit-filter` head learns an operator:

```text
input distribution -> keep only fruit items in that input
```

The held-out compositions are:

```text
fruit_filter(food()) -> apple / mango
fruit_filter(rice_apple()) -> apple
fruit_filter(noodle_mango()) -> mango
```

Crucially, the filter is not trained directly on these exact source-head
outputs. It is trained on other examples, then applied to intermediate states
from the source heads. This prevents a constant `apple/mango` answer from
passing the proof.

## TreeHeap Read Algebra

Each head is a 7-node complete binary parameter TreeHeap:

```text
        1
      /   \
     2     3
    / \   / \
   4   5 6   7
```

Each node stores a logit vector over the toy vocabulary. A fixed TreeHeap read
kernel gathers a local subheap:

```text
read(i) = arr[i] + alpha * read(left(i)) + alpha * read(right(i))
```

The learned object is the array `arr[i]`, not the address rule.

This keeps the proof aligned with the current TreeHeap story:

```text
address / recursion rule = algebraic scaffold
arr[i]                   = learnable parameter memory
scalar loss              = writes information into arr[i]
```

## Loss

Food head:

```text
L_source = mean_h CE(softmax(read(Theta_source_h)), target_bucket_h)
```

Filter head:

```text
mask = sigmoid(read(Theta_filter))
output = normalize(input * mask)
L_filter = CE(output, target_intersection(input, fruit_set))
```

Total:

```text
L = L_source + L_filter + L_constant_baseline
```

There is no separate direct loss on the held-out serial compositions.

## Predict

If the private-codec forest claim is right:

```text
P1: source head loss should fall and output the expected source buckets.
P2: fruit-filter head should learn a reusable input-dependent operator.
P3: held-out serial compositions should recover apple/mango, apple, and mango.
P4: a constant-output baseline should do worse on held-out composition.
P5: an untrained forest should do worse.
```

## Falsification

Downgrade or reject if:

```text
1. the food head only works by putting answers into Q;
2. the filter is trained directly on food() and no held-out composition exists;
3. the constant baseline matches held-out composition;
4. gradients do not move the parameter TreeHeap arr[i];
5. the result is written as natural-language understanding or WMT evidence.
```

## Boundary

This is a toy proof of a learning mechanism. It does not prove:

```text
natural language semantics
unsupervised ontology discovery
WMT translation
Transformer superiority
```

It only tests whether scalar loss can write a reusable private code into a
TreeHeap parameter forest and whether two heads can compose serially.
