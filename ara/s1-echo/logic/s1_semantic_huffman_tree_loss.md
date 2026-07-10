# S1 Semantic Huffman Tree Loss

Date: 2026-07-10
Status: moved / superseded by S3
Related claims:

```text
S1-SEM-HUFF-C01 (moved)
S3-SEM-HUFF-GEN-C01 (active replacement)
S1-PRIVATE-CODEC-C01
S1-MASK-KERNEL-C01
S1-ENCODER-OBS-C01
```

## Migration Note

This S1 document is retained only as historical design context.

Houming818's review corrected the stage boundary:

```text
semantic Huffman tree + encoder/decoder/generation
```

is not a good S1 echo task. Echo constrains the technical search toward input
reconstruction. The active claim now lives in:

```text
ara/s3-generation/logic/semantic_huffman_generation.md
```

Future proofs should use `S3-SEM-HUFF-GEN-C01`, not `S1-SEM-HUFF-C01`.

## One-Line Hypothesis

TreeHeap's S1 goal is not a collection of kernel tricks. It is to learn a
semantic Huffman-like tree:

```text
raw observations
  -> encoder writes a TreeHeap
  -> shared / multi-head kernels query the TreeHeap
  -> decoder reads token, subheap, or probability bucket
  -> loss pushes the tree toward short, reusable, queryable structure
```

In short:

```text
S1 = learn a compressible, queryable, decodable semantic TreeHeap code.
```

## Why This Document Exists

Recent SPR documents are locally useful but globally fragmented:

```text
SPR-046: content-aware route
SPR-047: encoder as world observer
SPR-048: private codec forest
SPR-049: mask kernel probability bucket
```

Each is a probe. None alone is the architecture.

The unifying claim is:

```text
TreeHeap must optimize encoder + tree code + kernel + decoder together.
```

The object being optimized is not "a token classifier". It is a tree code whose
internal nodes reduce future query cost while preserving reconstruction.

## Objects

```text
D
  observed corpus or controlled observation set

E_phi
  encoder that writes tokens/spans/observations into a TreeHeap

H
  runtime TreeHeap state for one sample or batch

Theta
  parameter TreeHeap forest / kernel parameters

K_theta
  convolution kernel over local TreeHeap neighborhoods

D_psi
  decoder/readout from a node, path, subheap, or probability container
```

The basic path is:

```text
H = E_phi(x)
z = K_theta(q, H)
y_hat = D_psi(z)
L = distance(y_hat, y)
```

The gradient may update:

```text
phi    encoder parameters
theta  kernel / parameter TreeHeap forest
psi    decoder parameters
H      current differentiable heap state in state-relaxation settings
```

## Semantic Huffman Analogy

Classic Huffman coding optimizes:

```text
frequent symbols -> shorter bit paths
prefix-free decoding
minimal expected code length
```

TreeHeap should generalize this:

```text
frequent query-useful structures -> shorter TreeHeap paths
replaceable tokens/spans         -> shared internal prefix/subheap
query kernel can stop early      -> internal node carries enough probability mass
decoder can still reconstruct    -> echo/readout remains valid
```

This is not merely frequency Huffman. A token can be rare but structurally
important if it shares query behavior with many other tokens.

## Total Loss

The proposed training objective is:

```text
L_total =
    lambda_echo      * L_echo
  + lambda_mask      * L_mask
  + lambda_huffman   * L_huffman
  + lambda_structure * L_structure
  + lambda_route     * L_route
```

Do not start by training all terms blindly. The formula is the system map.
Experiments should ablate one term at a time.

## Loss Terms

### L_echo: Do Not Destroy the Sample

```text
L_echo = CE(Decode(E_phi(x)), x)
```

Purpose:

```text
TreeHeap compression must remain decodable.
```

Without this, the model can collapse many samples into one cheap node and lose
the original information.

Evidence already related:

```text
S1-ECHO-ED-C01: hard echo encoder/decoder closes exactly.
S1-WMT-ECHO-C01: learned TreeHeap echo works on short WMT BPE.
```

### L_mask: Query a Probability Bucket

```text
L_mask = CE(P_theta(token | masked_tree), observed_token)
```

Purpose:

```text
masked TreeHeap should output a probability bucket, not only a single echo token.
```

Evidence already related:

```text
S1-MASK-KERNEL-C01:
  TreeHeap beats pair memory and shuffled control,
  but BoW matches on the current toy.
```

Lesson:

```text
Mask alone is not enough. The dataset must require substructure, path, or span.
```

### L_huffman: Prefer Short Paths for Useful Shared Structure

A first differentiable approximation:

```text
L_huffman =
  E_{q,x ~ D} [ - log P_theta(stop at useful internal node | q, H_x) ]
  + beta * E_{q,x ~ D} [ expected_route_depth(q, H_x) ]
```

Interpretation:

```text
If an internal node already contains the information needed by query q,
the route kernel should be rewarded for stopping there.
```

This is where the semantic Huffman idea enters.

Classic Huffman says:

```text
high frequency -> short code
```

Semantic TreeHeap says:

```text
high query reuse / high replacement value -> short readable subheap
```

### L_structure: Reusable Substructure and Held-Out Transfer

```text
L_structure =
  contrastive_or_ranking_loss(
    compatible subheaps close,
    incompatible subheaps far
  )
```

Examples:

```text
ate rice
ate noodles
cooked rice
held out: cooked noodles
```

The model should learn:

```text
rice and noodles share a reusable object structure.
```

But current S1-MASK-KERNEL-C01 showed that a simple verb cue lets BoW tie
TreeHeap. Therefore the next experiment must require structure such as:

```text
same verb, different local span changes target bucket
same bag of words, different TreeHeap arrangement changes target bucket
same token appears in multiple subtrees with different query answers
```

### L_route: Query Should Use TreeHeap, Not Shortcuts

```text
L_route =
  route_supervision_or_energy
  + entropy/collapse regularization
```

Purpose:

```text
Kernel must read arr[i], arr[left], arr[right], and path/subheap state.
```

Avoid:

```text
target-in-left flags
precomputed interval answers
flat L x L route matrices
```

Evidence already related:

```text
S1-CONTENT-ROUTE-C01:
  content-aware route passes but dense representation is expensive.

S1-COMPACT-CONTENT-ROUTE-C01:
  naive compact sums reduce memory but lose path exactness.
```

## What Counts as Success

The semantic Huffman claim should not be upgraded merely because one local
probe passes. It needs a combined signal:

```text
1. Echo remains high.
2. Mask top-k / MRR improves over pair memory and shuffled controls.
3. Average useful route depth decreases versus random tree.
4. Internal-node purity / replacement consistency improves.
5. Held-out composition improves.
6. BoW/flat baselines fail or are materially worse on a task requiring
   substructure/path/local span information.
```

## Required Baselines

```text
random TreeHeap:
  same leaves, random internal placement

frequency Huffman:
  short paths by token frequency only

BoW MLP:
  no path or substructure

flat sequence MLP:
  position-aware, no recursive TreeHeap compose

pair memory:
  seen context-token table

shuffled corpus:
  token frequencies preserved, context relation broken
```

The crucial baseline is frequency Huffman:

```text
If frequency Huffman matches learned semantic Huffman,
then TreeHeap only learned frequency compression, not semantic structure.
```

## Next Proof Design

The next dataset must defeat the failure mode from S1-MASK-KERNEL-C01, where
BoW matched TreeHeap.

Minimal stronger toy:

```text
same bag, different tree:

case A:
  [old [man with telescope]] saw star
  mask asks: who has telescope? -> man

case B:
  old man [saw star with telescope]
  mask asks: instrument of seeing? -> telescope
```

For a simpler controlled version:

```text
left subtree determines bucket;
right subtree contains distractor tokens;
BoW sees same tokens in both examples;
TreeHeap path/subheap tells which structure is active.
```

Pass condition:

```text
learned semantic TreeHeap > BoW on held-out MRR and bucket purity
learned semantic TreeHeap > frequency Huffman on expected useful route depth
learned semantic TreeHeap keeps echo above threshold
shuffled control fails
```

## Claim

```text
S1-SEM-HUFF-C01:
TreeHeap S1 should be trained as a semantic Huffman code: encoder, kernels,
and decoder jointly minimize reconstruction, masked prediction, route depth,
and reusable-substructure losses so that query-useful structures become short,
shared, and decodable internal subheaps.
```

## Predict

If the claim is right:

```text
P1. A learned semantic TreeHeap beats random/frequency Huffman on held-out
    structure-transfer tasks.
P2. It preserves echo/readout above threshold.
P3. It has shorter expected route depth for query-useful buckets.
P4. It beats BoW/flat baselines only when the task actually requires
    substructure or path.
P5. Shuffled controls destroy the effect.
```

If P4 fails, the dataset is not a TreeHeap dataset yet, or the kernel is not
using TreeHeap structure.

## Falsification

Downgrade or reject if:

```text
1. BoW/flat baselines match on substructure-controlled tasks.
2. Frequency Huffman matches learned semantic Huffman.
3. Echo/readout collapses when compression pressure is added.
4. Shorter routes do not correspond to reusable query structure.
5. Shuffled corpus produces the same tree quality.
6. The proof depends on hand labels such as food/place/medicine during training.
```

## Boundary

This is not yet WMT translation and not a claim of language understanding.

It is the system hypothesis that turns scattered SPR probes into one training
objective:

```text
learn the tree code first;
then let language tasks stand on that code.
```
