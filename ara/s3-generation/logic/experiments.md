# S3 Experiment Plan: Generation From Semantic Huffman TreeHeap

Owner: Review Engineer
Writer: Codex
Created: 2026-07-10

## P-S3-ENCODER-GATE00: Minimal S1 Encoder Gate

Status: complete / supported pilot
Claim: upstream `S1-ENCODER-OBS-C01`
Design: `../s1-echo/src/s1_encoder_minimal_observer_probe.py`
Evidence: `../s1-echo/evidence/s1_encoder_minimal_observer_probe_026/`

### Question

Before S3 generation is allowed to claim anything, can the smallest S1 encoder
learn useful internal TreeHeap subheaps from corpus statistics?

The accepted minimal loss is:

```text
L = L_context + lambda_echo * L_echo
```

This intentionally avoids the five-loss S3 system.  If this gate fails, S3
decoder work is blocked because there is no credible encoded TreeHeap for the
decoder to read.

### Pass Gate

```text
structured corpus > shuffled corpus on cluster_purity
structured corpus > shuffled corpus on pairwise_f1
structured corpus > shuffled corpus on heldout_mrr or equivalent transfer
```

### Result

```text
structured - shuffled:
  cluster_purity        +0.2616
  pairwise_f1           +0.3720
  heldout_mrr           +0.1827
  full_context_cell_acc +0.2446
```

### Boundary

This is still S1.  It does not prove generation.  It only decides whether S3
has a frozen encoder output worth consuming.

## P-S3-FROZEN-DECODER01: Decoder From Frozen TreeHeap Encoder

Status: complete / limited positive
Claim: `S3-FROZEN-DECODER-C01`
Design: `logic/frozen_treeheap_decoder_gate.md`
Evidence: `evidence/s3_frozen_decoder_gate_probe/`

### Question

Given a TreeHeap encoder output that is frozen after S1, can a decoder read
internal subheap states and produce surface text or surface labels better than
flat baselines?

### Required Rule

The decoder must not receive hand semantic labels as input.  It may consume:

```text
frozen internal subheap state
path / prefix id created by the encoder
kernel-read probability bucket
```

It must not consume:

```text
gold category label
oracle answer text
precomputed "target is left/right" flags
```

### Minimal Proof Shape

Start from the saved S1 encoder evidence:

```text
ara/s1-echo/evidence/s1_encoder_minimal_observer_probe_026/
```

Freeze the best structured encoder result.  Then train only a small S3 decoder
to map internal subheap states to surface outputs.  Compare against:

```text
BoW decoder
flat object-id decoder
random/frequency TreeHeap
shuffled-corpus encoder output
```

### Pass Gate

```text
frozen TreeHeap decoder > shuffled TreeHeap decoder
frozen TreeHeap decoder > BoW/flat when the task requires subheap structure
generation must be reproducible from saved S1 evidence
```

### Result

```text
structured frozen decoder:
  top1    0.4398
  mrr     0.6736
  entropy 1.9378 bits

shuffled frozen decoder:
  top1    0.0903
  mrr     0.3297
  entropy 2.2394 bits
```

This supports frozen internal bucket readability.  It does not yet satisfy the
stronger BoW/flat same-bag different-tree generation gate.

## P-S3-SEM-HUFF-GEN01: Semantic Huffman Generation Code

Status: roadmap / blocked
Claim: `S3-SEM-HUFF-GEN-C01`
Design: `logic/semantic_huffman_generation.md`
Evidence target: `evidence/semantic_huffman_generation_probe/`

### Question

Can TreeHeap be trained as a semantic Huffman-like generation code, where the
tree is not only echo-readable but also useful for surface generation?

### Why This Is S3, Not S1

S1 echo asks:

```text
Can the system preserve and read back input?
```

That is necessary but too restrictive for this hypothesis. Semantic Huffman
coding is about:

```text
compressing reusable structure
stopping at internal nodes
decoding probability buckets
generating surface forms
```

Those are generation-layer requirements. Therefore this claim is moved to S3.

### Unified Loss

```text
L_total =
    lambda_echo      * L_echo
  + lambda_mask      * L_mask
  + lambda_huffman   * L_huffman
  + lambda_structure * L_structure
  + lambda_route     * L_route
  + lambda_gen       * L_generation
```

### Minimal Proof Shape After The Gate

Use a controlled dataset where BoW cannot solve the task:

```text
same bag of tokens
different TreeHeap local span / path
different generated surface answer
```

Example class:

```text
Tree A:
  [old [man with telescope]] saw star
  generate: "the man has the telescope"

Tree B:
  old man [saw star with telescope]
  generate: "the seeing used the telescope"
```

The exact strings can be simplified for the first proof, but the core property
must remain:

```text
same bag, different tree, different generation target
```

### Required Baselines

```text
random TreeHeap
frequency Huffman
BoW MLP
flat sequence MLP
pair memory
shuffled corpus
template oracle / hard tree upper bound
```

### Pass Gate

```text
learned semantic TreeHeap > BoW/flat on same-bag different-tree generation
learned semantic TreeHeap > random/frequency Huffman on held-out transfer
echo/readback remains above threshold
expected useful route depth decreases
top-k probability buckets remain meaningful before final collapse
shuffled corpus fails
```

### Boundary

This is not yet WMT translation. It is the first S3 gate: can a learned
TreeHeap code drive controllable surface generation when structure matters?

This experiment should not run before `P-S3-ENCODER-GATE00` and
`P-S3-FROZEN-DECODER01` have evidence.  Otherwise the work becomes a large
joint-training guess with no isolated encoder or decoder accountability.
