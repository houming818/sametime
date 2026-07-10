# S3 Semantic Huffman Generation

Date: 2026-07-10
Status: system hypothesis / queued proof
Related claims:

```text
S3-SEM-HUFF-GEN-C01
S1-PRIVATE-CODEC-C01
S1-MASK-KERNEL-C01
```

## Reframe

The semantic Huffman idea should not live as an S1 echo claim.

S1 echo is useful for proving that a TreeHeap state can preserve input, but it
also biases the research toward reconstruction. The current problem is larger:

```text
learn a tree code that can be compressed, queried, decoded, and used to
generate surface text.
```

That is an S3 generation-layer claim.

## One-Line Hypothesis

```text
S3 = learn and use a semantic Huffman TreeHeap code as a generation program.
```

More explicitly:

```text
raw observations
  -> encoder writes a TreeHeap code
  -> kernels query internal nodes / probability containers
  -> route stops at useful subheaps
  -> decoder generates surface text
  -> loss pushes useful structures to short, shared, decodable paths
```

## Relation to Previous Probes

```text
SPR-046:
  route must read content, not only geometry.

SPR-047:
  encoder is the world observer; bad H makes route/read useless.

SPR-048:
  scalar loss can write rules into a parameter TreeHeap forest and compose heads.

SPR-049:
  mask kernel can output probability buckets, but BoW matched the toy result.
```

The synthesis:

```text
mask probability buckets are not the destination;
they are intermediate containers used by a generator.
```

## Objects

```text
D
  observed training corpus or controlled generation observations

E_phi
  encoder that writes tokens/spans/events into TreeHeap

H
  TreeHeap state for a sample

Theta
  parameter TreeHeap forest / kernel bank

K_theta
  local convolution kernel over TreeHeap neighborhoods

R_theta
  route / stop / left / right collapse policy

G_psi
  surface generator / decoder
```

The S3 path:

```text
H = E_phi(x)
z = K_theta(q, H)
c = R_theta(z)
y_hat = G_psi(c)
```

## Total Loss

```text
L_total =
    lambda_echo      * L_echo
  + lambda_mask      * L_mask
  + lambda_huffman   * L_huffman
  + lambda_structure * L_structure
  + lambda_route     * L_route
  + lambda_gen       * L_generation
```

### L_echo

Preserve input enough to avoid destructive compression:

```text
L_echo = CE(ReadBack(E_phi(x)), x)
```

### L_mask

Produce probability buckets before final generation:

```text
L_mask = CE(P(token | masked_tree), observed_token)
```

### L_huffman

Reward short paths for useful internal nodes:

```text
L_huffman =
  E[-log P(stop at useful internal node | q, H)]
  + beta * E[route_depth(q, H)]
```

This is not just token frequency compression. A useful internal node is one
that supports future queries or generation.

### L_structure

Reward reusable substructure and held-out transfer:

```text
same relation / same local span role -> closer subheap
incompatible role -> farther subheap
```

### L_route

Prevent shortcut routing:

```text
kernel must read arr[i], arr[left], arr[right], and path/subheap state
```

No target-in-left flags. No flat `L x L` route matrix.

### L_generation

Generate the target surface sequence:

```text
L_generation = CE(G_psi(c), target_text)
```

This is why the claim belongs in S3. The quality of the tree code is judged by
whether it helps generate text, not only by whether it echoes input.

## Why S1 Is Not Enough

If this remains in S1, the optimization target becomes:

```text
make the input recoverable
```

That is necessary but too narrow. A good semantic Huffman tree may deliberately
stop at internal probability buckets, delay collapse, or choose a shorter
generation path instead of preserving every surface detail immediately.

S3 can use echo as one regularizer, but S3 should not be ruled by echo.

## First Proof Requirement

The next proof must specifically address the weakness of S1-MASK-KERNEL-C01:

```text
BoW MLP matched TreeHeap on the simple mask corpus.
```

Therefore the S3 proof must require structure:

```text
same bag of words
different TreeHeap path/subheap structure
different generation target
```

If BoW can solve it, the proof is not testing TreeHeap.

## Candidate Controlled Task

Use paired examples with identical bags but different local tree attachment:

```text
Case A:
  Tree meaning: man-with-telescope saw star
  Generate: "the man has the telescope"

Case B:
  Tree meaning: man saw star-with-telescope
  Generate: "the star has the telescope"
```

The first toy can use artificial tokens if needed:

```text
[A [B C]] -> "B owns C"
[[A B] C] -> "A owns B"
```

The point is not English realism. The point is:

```text
same token bag
different tree
different output
```

## Required Baselines

```text
random TreeHeap
frequency Huffman
BoW MLP
flat sequence MLP
pair memory
shuffled corpus
hard-tree template oracle
```

Frequency Huffman is required because a learned TreeHeap must beat simple
frequency compression, not only random placement.

## Claim

```text
S3-SEM-HUFF-GEN-C01:
S3 should train TreeHeap as a semantic Huffman generation code: encoder, tree
code, kernels, and surface decoder jointly minimize reconstruction, masked
prediction, route depth, reusable-substructure, and generation losses so that
query-useful structures become short, shared, decodable internal subheaps that
can generate surface text.
```

## Predict

If the claim is right:

```text
P1. Learned semantic TreeHeap beats BoW/flat on same-bag different-tree
    generation.
P2. Learned semantic TreeHeap beats random/frequency Huffman on held-out
    transfer.
P3. Echo/readback remains above threshold but does not dominate generation.
P4. Expected useful route depth decreases.
P5. Probability buckets remain meaningful before final collapse.
P6. Shuffled corpus fails.
```

## Falsification

Reject or downgrade if:

```text
1. BoW/flat baselines match on same-bag different-tree generation.
2. Frequency Huffman matches learned semantic Huffman.
3. Echo collapses or dominates so strongly that generation cannot use internal
   probability buckets.
4. Shorter routes do not correspond to reusable generation structures.
5. Shuffled corpus preserves the effect.
6. The proof depends on hand semantic labels during training.
```

## Boundary

This is still not WMT translation.

It is the first S3 system claim:

```text
Can TreeHeap learn a code that is useful for generation when structure matters?
```
