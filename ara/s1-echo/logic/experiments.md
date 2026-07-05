# S1 Experiment Plan: Falsification First

Owner: Review Engineer
Writer: Codex
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Convert SPR Echo from demo scripts into ARA-verifiable experiments.

## E1: Hash And Echo Smoke

Question: Does the implementation still reproduce the basic capacity and order-sensitivity claims?

Commands on io:

```bash
cd /data/homecicd/sametime/code/wmt
python3 spr_hash_cyclic.py
sudo -n python3 spr_echo_proof.py
```

Acceptance:

- Pure cyclic roll collision is true.
- Sign alternation separates reversed order.
- Effective leaf solo rate is at least 95%.
- BLEU-4 is at least 99 on the same WMT14 slice.

## E2: Shuffle Falsification

Question: Is SPR using semantic structure, or just token identity/capacity?

Design:

- Keep sentence lengths, token frequencies, and train/validation split fixed.
- Shuffle semantic labels or syntax labels across examples.
- Train/evaluate the same path-feature classifier on real vs shuffled labels.

Required metrics:

- real score
- shuffled score
- delta
- bootstrap confidence interval or repeated-seed mean/std

Pass condition:

Real labels must beat shuffled labels by a material margin. If not, claims S1-C10/S1-C11 stay open or become rejected.

## E3: Polysemy Routing

Question: Can one token route to different stable states under different contexts?

Initial target words:

- `light`: illumination vs weight
- `bank`: financial institution vs river side
- `charge`: payment/legal/electric senses

Design:

- Build minimal context pairs or sample from corpus.
- Extract SPR path or collapsed vector features per occurrence.
- Evaluate sense separation against labels.
- Compare to random hash, static token embedding, and bag-of-words context baseline.

Controlled proof result:

`spr_context_proof.py` passes the mechanism test on 2026-06-16:

```text
token_acc=0.429
context_acc=1.000
shuffled_acc=0.482
context_purity=1.000
```

This proves that the S1 path operator can separate senses when the route input includes a context signal. It does not prove real-corpus semantic routing.

Full pass condition:

SPR path/context features must separate senses above random hash and token-only baselines.

## E4: Baseline Battle

Question: Is SPR necessary?

Baselines:

- token frequency template
- nearest neighbor in embedding space
- random high-dimensional hash with same leaf capacity
- bag-of-words MLP

Pass condition:

SPR must beat cheap baselines on at least one semantic task while matching them on echo capacity.

## Reporting Contract

Every experiment output must write:

- command
- git or file timestamp
- dataset slice
- seed
- metrics
- failure mode if failed
- claim IDs affected

Evidence goes in `ara/s1-echo/evidence/README.md`.

---

## E5: Shallow Real-Sentence TreeHeap Write

Question:

```text
Can real short sentences be encoded into a shallow TreeHeap memory and queried
by root/subject/object/subheap probes?
```

Why this is the first post-M0 S1 step:

```text
M0 proved that TreeHeap operations can be defined.
S1 must now test whether data can be written into a TreeHeap-shaped memory.
```

Script:

```text
ara/s1-echo/src/shallow_treeheap_s1_probe.py
```

Remote host:

```text
ni.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/shallow_treeheap_s1_probe/
```

Dataset:

```text
curated real-word short sentences
train = 63
test  = 17
ood   = 10
vocab = 37
slots = root / subject / object
```

The OOD split contains lexical items unseen as train outputs:

```text
erin draws cup
nurse brings water
teacher holds book
...
```

Models:

| Model | Meaning |
|---|---|
| `bow_linear` | unordered bag-of-words linear probe |
| `seq_linear` | position-aware sequence linear probe |
| `soft_treeheap` | learned position-to-slot soft write plus copy-by-address memory |

Result:

| Model | Train exact | Test exact | OOD exact | Test subheap | OOD subheap |
|---|---:|---:|---:|---:|---:|
| `bow_linear` | 0.873 | 0.765 | 0.000 | 0.765 | 0.000 |
| `seq_linear` | 1.000 | 0.765 | 0.000 | 0.765 | 0.000 |
| `soft_treeheap` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Learned TreeHeap write:

```text
position 0 -> subject 0.9968
position 1 -> root    0.9968
position 2 -> object  0.9968
```

Decision:

```text
S1-C30 -> supported pilot
```

Interpretation:

```text
This supports the first S1 bridge: a learned shallow TreeHeap write can encode
real short sentences into queryable slots, and copy OOD lexical items by
address. It does not prove WMT, full syntax, deep TreeHeap, or superiority over
copy-capable sequence models.
```

Next falsification:

```text
Add variable length, modifiers, passive/OSV order, and matched pointer/copy
sequence baselines. If those baselines match TreeHeap, this pilot is only an
existence proof, not a TreeHeap advantage proof.
```

---

## E6: Frozen Embedding Compound Coordinate Probe

Question:

```text
Can a TreeHeap prob vector plus encoder use M0 diff/loss machinery to write two
word vectors into a zero TreeHeap and read out a compound concept vector close
to a frozen external world-coordinate target?
```

Distillation guard:

```text
The external embedding model is frozen and used only as a coordinate ruler.
This experiment must not claim that TreeHeap learned the external model's world
knowledge. It only tests whether TreeHeap can fit and generalize in that
coordinate system.
```

Script:

```text
ara/s1-echo/src/s1_world_model_compound_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Embedding source:

```text
sentence-transformers/all-MiniLM-L6-v2
local cached snapshot:
/home/nio/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41
raw_dim = 384
projected_dim = 128
```

No `proxychains4` was needed for this run because the model snapshot was already
cached on `io`. If a future dataset or embedding is not cached, use
`proxychains4` explicitly and record the download command in the evidence.

Dataset:

```text
compound words
train = 20
test  = 11
ood   = 6
targets = 37
```

Examples:

```text
foot + ball -> football
basket + ball -> basketball
hand + ball -> handball
rain + coat -> raincoat
book + shelf -> bookshelf
flash + light -> flashlight
```

Models:

| Model | Meaning |
|---|---|
| `vector_add` | normalize(left + right), zero trainable parameters |
| `concat_mlp` | MLP over `[left, right, left*right, abs(left-right)]` |
| `treeheap_prob_vector_plus` | write left/right into a zero TreeHeap with probabilistic vector plus, then read out |

Result:

| Model | Train cosine | Test cosine | OOD cosine | Test top1 | OOD top1 |
|---|---:|---:|---:|---:|---:|
| `vector_add` | 0.7117 | 0.7256 | 0.7198 | 0.909 | 0.833 |
| `concat_mlp` | 1.0000 | 0.6269 | 0.5766 | 0.000 | 0.000 |
| `treeheap_prob_vector_plus` | 0.9999 | 0.5051 | 0.3919 | 0.000 | 0.000 |

Decision:

```text
S1-WM-C01 -> rejected pilot
```

Interpretation:

```text
The frozen embedding coordinate experiment did connect TreeHeap to an external
world-coordinate ruler, but the current TreeHeap encoder overfits the small
training set and fails OOD. Simple vector addition is a strong baseline for
compound coordinates and must be treated as the first baseline to beat.
```

Next action:

```text
Do not add more parameters to the current TreeHeap reader. Instead, constrain
the encoder so that TreeHeap structure actually matters: shared family slots,
subheap reuse, copy/read pointer constraints, and route entropy/collapse
controls. Then rerun against vector_add.
```

---

## E7: Local Corpus Embedding + Structured TreeHeap Kernel

Question:

```text
Can we avoid pretrained-vector distillation by training a small coordinate
space from local corpus co-occurrence, then test whether a structured TreeHeap
write/compose kernel improves over vector_add?
```

Script:

```text
ara/s1-echo/src/s1_corpus_embedding_kernel_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Coordinate source:

```text
local SGNS corpus embedding
external_model = false
vocab_size = 85
corpus_sentences = 888
skipgram_pairs = 10448
```

Kernel design:

```text
left token  -> left child
right token -> right child
root        -> compose kernel(left_child, right_child)
```

This is different from E6's unconstrained TreeHeap reader. The kernel design
forces the write operation to use TreeHeap structure.

Models:

| Model | Meaning |
|---|---|
| `vector_add` | normalize(left + right), zero trainable parameters |
| `concat_mlp` | MLP over `[left, right, left*right, abs(left-right)]` |
| `structured_treeheap_kernel` | child writes plus root compose kernel |

Result:

| Model | Train cosine | Test cosine | OOD cosine | OOD top1 |
|---|---:|---:|---:|---:|
| `vector_add` | 0.5705 | 0.5965 | 0.5785 | 0.000 |
| `concat_mlp` | 0.9994 | 0.6957 | 0.7321 | 0.167 |
| `structured_treeheap_kernel` | 0.9999 | 0.6801 | 0.7126 | 0.000 |

Decision:

```text
S1-WM-C02 -> supported pilot, narrow scope
```

Interpretation:

```text
The local-corpus coordinate setup removes the pretrained-vector advantage. In
this setting, structured TreeHeap kernel beats vector_add on OOD cosine and is
close to concat_mlp. However, OOD top1 is still 0, so the probe supports
coordinate closeness but not reliable target retrieval.
```

Next action:

```text
Run multi-seed/corpus variants and improve kernel constraints:
family slots, explicit subheap reuse, route collapse regularization, and
nearest-neighbor margin loss. The next proof should require OOD top1 or MRR
improvement, not only cosine.
```

---

## E8: WMT SentencePiece Echo Kernel Probe

Question:

```text
Can a structured TreeHeap kernel write and read real WMT SentencePiece short
sequences, using address/path structure rather than only flat token
memorization?
```

Why this follows E7:

```text
E7 tested a local co-occurrence coordinate system.
E8 returns to real WMT text, but uses an echo target rather than translation.
This isolates the write/read kernel before asking for semantic generation.
```

Script:

```text
ara/s1-echo/src/s1_wmt_echo_kernel_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_wmt_echo_kernel_probe/
```

Dataset:

```text
WMT17 English side
source file = /mnt/nas/datasets/wmt17/train.zh-en
SentencePiece model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 3000
train/test/ood = 2400/300/300
token length = 3..8
average non-pad length = 5.9533
vocab limit including PAD = 2048
```

Kernel design:

```text
token id -> shared token embedding
position -> fixed heap leaf address
internal nodes -> shared bottom-up compose kernel
leaf readout -> shared decoder
```

This means the model is not free to read from an arbitrary flat vector. It must
write token states into leaf addresses, compose them upward through a TreeHeap,
and decode from leaf states.

Models:

| Model | Meaning |
|---|---|
| `bow_linear` | unordered bag-of-token baseline |
| `seq_mlp` | flat position-aware MLP baseline |
| `treeheap_kernel_echo` | fixed leaf write plus shared TreeHeap compose/read kernels |

Result:

| Model | Params | Test token | Test exact | OOD token | OOD exact |
|---|---:|---:|---:|---:|---:|
| `bow_linear` | 33,570,816 | 0.1679 | 0.0033 | 0.1659 | 0.0033 |
| `seq_mlp` | 16,794,112 | 0.5801 | 0.0567 | 0.5986 | 0.0533 |
| `treeheap_kernel_echo` | 423,104 | 0.9818 | 0.8900 | 0.9818 | 0.9000 |

Training traces:

```text
bow_linear loss:          7.5012 -> 2.6490 -> 1.0412
seq_mlp loss:             6.7628 -> 0.0105 -> 0.0034
treeheap_kernel_echo loss:6.0619 -> 0.0149 -> 0.0049
```

Decision:

```text
S1-WMT-ECHO-C01 -> supported pilot
```

Interpretation:

```text
TreeHeap kernel can write/read real WMT short BPE sequences in this echo
setting. The result supports address/path-sensitive kernel design: the
TreeHeap model is much smaller than the flat sequence MLP but generalizes much
better on held-out and OOD short sequences.
```

What this does not prove:

```text
not translation
not semantic world model
not compression
not long-sequence syntax
not victory over copy-capable sequence baselines
```

Next falsification:

```text
1. length 8 -> 16/32 echo
2. noisy echo: mask/drop/swap one token, ask kernel to restore
3. subheap query: read a phrase/window, not the whole sequence
4. matched copy/pointer sequence baseline
5. multi-seed and larger WMT slices
```

---

## E8b: Explicit Echo Encoder / Decoder Interface

Question:

```text
Can we implement the hard TreeHeap echo encoder and decoder contract directly?
```

Why this follows the recent S1 work:

```text
E8 proved a learned TreeHeap echo kernel can approximate short WMT echo.
SPR-035/034 clarified ordered internal readout.
SPR-038 clarified state gradients.
E8b pins down the algebraic encoder/decoder interface before replacing pieces
with learned kernels.
```

Script:

```text
ara/s1-echo/src/s1_echo_encoder_decoder_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_echo_encoder_decoder_probe/
```

Dataset:

```text
source = wmt_sentencepiece
wmt_path = /mnt/nas/datasets/wmt17/train.zh-en
spm_model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 2000
train/test/ood = 1600/200/200
length = 3..8
vocab_limit = 1024
```

Design:

```text
learned_parameters = 0
encoder = token sequence -> ordered TreeHeap leaves -> internal NodeState summaries
decoder = root length + path-addressed leaf/subheap reads -> token sequence
uses_target_heap_in_decoder = false
```

Result:

| Split | Sequence exact | Leaf acc | Subheap exact | Summary exact |
|---|---:|---:|---:|---:|
| train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ood | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Expanded result:

```text
out = ara/s1-echo/evidence/s1_echo_encoder_decoder_probe_expanded/
samples = 20000
train/test/ood = 16000/2000/2000
length = 3..16
vocab_limit = 4096
sample_nodes = 31
```

| Split | Sequence exact | Leaf acc | Subheap exact | Summary exact |
|---|---:|---:|---:|---:|
| train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ood | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The expanded run includes padded/empty subheap queries because it samples all
internal nodes; future learned/semantic variants should report non-empty
subheap metrics separately.

Decision:

```text
S1-ECHO-ED-C01 -> supported pilot
```

Interpretation:

```text
The hard/algebraic echo encoder-decoder interface is closed. This gives the
next learned experiment a clean target: approximate ordered write, compose,
path read, and subheap decode without using target heap labels at decode time.
```

What this does not prove:

```text
not translation
not learned semantic encoding
not compression
not superiority over neural baselines
not noisy-channel correction
```

---

## E9: WMT Multi-Kernel Specialization Probe

Question:

```text
If TreeHeap has a bank of convolution-style kernels, will structural
perturbation tasks push those kernels to specialize, or will they remain
redundant?
```

Relation to Transformer multi-head:

```text
Transformer multi-head attention gives different heads the opportunity to
learn different relations, but it does not guarantee every head is useful.
This experiment asks the analogous TreeHeap question: do multiple tree kernels
actually differentiate under gradient pressure?
```

Script:

```text
ara/s1-echo/src/s1_wmt_multikernel_specialization_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe/
ara/s1-echo/evidence/s1_wmt_multikernel_specialization_probe_common512/
```

Dataset:

```text
WMT17 English side
SentencePiece model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 4000
train/test/ood = 3200/400/400
token length = 4..8
```

Tasks:

| Task | Meaning |
|---|---|
| `echo` | reconstruct the original sequence |
| `mask_restore` | one token is masked; reconstruct the sequence |
| `left_query` | read the left subheap |
| `right_query` | read the right subheap |
| `mirror` | reconstruct the reversed sequence |

Model design:

```text
token id -> leaf embedding
leaf addresses -> complete binary heap leaves
internal node -> kernel(left_child, right_child)
task query -> soft gate over K kernels
selected node -> task decoder
```

The multi-kernel model has four kernels. The single-kernel model has the same
TreeHeap shape but only one compose kernel.

Predict:

```text
P-S1-MK01:
If structural perturbation creates useful gradient pressure, multi-kernel
TreeHeap should outperform single-kernel TreeHeap, and kernel gates/ablations
should show task-dependent specialization.
```

Pass gate:

```text
multi OOD mean exact - single OOD mean exact >= 0.05
multi OOD mean exact >= 0.65
at least two task argmax kernels are used
max ablation exact drop >= 0.10
```

Run A: full vocab pilot

```text
vocab limit including PAD/MASK = 2049
epochs = 28
```

Result:

| Model | Params | OOD mean exact |
|---|---:|---:|
| `single_kernel_treeheap` | 4,641,545 | 0.0495 |
| `multi_kernel_treeheap` | 4,938,764 | 0.0600 |

Specialization signals:

```text
task_argmax_kernel = {
  echo: 2,
  mask_restore: 1,
  left_query: 3,
  right_query: 3,
  mirror: 0
}
unique_argmax_kernels = 4
max_ood_ablation_exact_drop = 0.1100
```

Run B: common-token pilot

```text
vocab limit including PAD/MASK = 513
epochs = 60
```

Result:

| Model | Params | OOD mean exact |
|---|---:|---:|
| `single_kernel_treeheap` | 1,286,921 | 0.1275 |
| `multi_kernel_treeheap` | 1,584,140 | 0.1420 |

Task exact on OOD:

| Task | Single exact | Multi exact |
|---|---:|---:|
| `echo` | 0.0300 | 0.0375 |
| `mask_restore` | 0.0050 | 0.0025 |
| `left_query` | 0.1475 | 0.1775 |
| `right_query` | 0.4525 | 0.4925 |
| `mirror` | 0.0025 | 0.0000 |

Specialization signals:

```text
task_argmax_kernel = {
  echo: 0,
  mask_restore: 1,
  left_query: 0,
  right_query: 3,
  mirror: 2
}
unique_argmax_kernels = 4
max_ood_ablation_exact_drop = 0.3050
drop_kernel_0 left_query = 0.1775
drop_kernel_3 right_query = 0.3050
```

Decision:

```text
S1-MK-C01 -> open / mixed pilot
```

Interpretation:

```text
The positive signal is real: task gates use different kernels, and kernel
ablation causes task-specific damage, especially for left/right subheap
queries. This supports the idea that structural perturbation can create
kernel differentiation pressure.

The negative signal is also real: OOD mean exact is low, multi-kernel improves
single-kernel by only 0.0105 to 0.0145, and echo/mask/mirror remain weak under
the current root-bottleneck decoder. Therefore this is not yet a supported
multi-kernel capability proof.
```

Next action:

```text
1. Replace single root bottleneck decoding with path-conditioned read kernels.
2. Keep subheap/path/noise tasks, but score token accuracy and exact separately.
3. Add matched flat MLP and small Transformer baselines.
4. Add kernel dropout and task-free query variants to test whether
   specialization survives without explicit task labels.
```

---

## E10: WMT Probabilistic Read Collapse Kernel Probe

Question:

```text
Can TreeHeap read be implemented as a query-conditioned collapse process from
arr[1], where each step chooses stop/left/right, instead of decoding every
answer from the root bottleneck?
```

Why this follows E9:

```text
E9 showed that a root-bottleneck decoder makes multi-kernel tasks hard.
SPR-032 tests the missing read mechanism directly: do not ask the root to
contain the whole answer; let a read kernel walk the TreeHeap address/path.
```

Script:

```text
ara/s1-echo/src/s1_probabilistic_read_kernel_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe/
ara/s1-echo/evidence/s1_probabilistic_read_kernel_probe_b32/
```

Dataset:

```text
WMT17 English side
SentencePiece model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 3000
train/test/ood = 2400/300/300
token length = 4..8
vocab limit including PAD = 513
```

Toy target:

```text
1. Write the short BPE sequence into the leaves of a complete binary heap.
2. Query a target heap node id.
3. If the target is a leaf, return the token id at that leaf.
4. If the target is an internal node, return a checksum bucket of the whole
   subheap span.
```

The internal checksum is intentionally synthetic. It gives `stop` at an
internal node a measurable meaning:

```text
stop at leaf     -> read one token
stop at internal -> read a subheap summary
```

Models:

| Model | Meaning |
|---|---|
| `root_query_decoder` | TreeEncoder root state plus query id directly predicts the answer. This is the root bottleneck baseline. |
| `probabilistic_read_kernel` | TreeEncoder states plus `K_read(q, h_node, path)` choose `stop/left/right`; hard mode is a tail-recursive while loop, soft mode is frontier probability accumulation. |

Predict:

```text
P-S1-READ01:
If TreeHeap read should be a probabilistic path collapse, then a read kernel
should beat root-only decoding on OOD queries, route accurately from arr[1],
and support both leaf stop and internal stop.
```

Pass gate:

```text
tail_recursive_interpreter_ok = true
read OOD hard acc >= 0.80
route acc >= 0.95
read OOD hard acc - root OOD acc >= 0.20
internal stop route acc >= 0.90
leaf stop route acc >= 0.90
```

Run A: 128 checksum buckets

```text
epochs = 40
labels = 641
```

Result:

| Model | Params | OOD acc | OOD internal | OOD leaf | Route acc |
|---|---:|---:|---:|---:|---:|
| `root_query_decoder` | 398,209 | 0.0638 | 0.0205 | 0.1066 | n/a |
| `probabilistic_read_kernel` | 499,588 | 0.6124 | 0.2214 | 0.9989 | 1.0000 |

Run B: 32 checksum buckets diagnostic

```text
epochs = 80
labels = 545
```

Result:

| Model | Params | OOD acc | OOD internal | OOD leaf | Route acc |
|---|---:|---:|---:|---:|---:|
| `root_query_decoder` | 373,537 | 0.1184 | 0.0765 | 0.1598 | n/a |
| `probabilistic_read_kernel` | 474,916 | 0.7177 | 0.4332 | 0.9989 | 1.0000 |

Decision:

```text
S1-READ-C01 -> open / mixed pilot
```

Interpretation:

```text
The positive result is strong: path collapse from arr[1] is learnable, and
the hard tail-recursive read and soft frontier read agree. The read kernel
nearly solves leaf copy, and beats the root bottleneck by 0.55 to 0.60 OOD
accuracy.

The negative result is also important: internal subheap summaries are still
weak. Reducing checksum buckets from 128 to 32 improves internal OOD accuracy
from 0.2214 to 0.4332, but this is not enough to claim solved subheap meaning.
```

What this does not prove:

```text
not translation
not semantic world model
not unsupervised route learning
not long-sequence syntax
not solved internal subheap summaries
```

Next action:

```text
1. Replace arbitrary checksum labels with compositional subheap targets:
   length, first/last token, bag checksum, or learned phrase vector.
2. Add a matched pointer/Transformer read baseline.
3. Remove or weaken route supervision to test whether the path kernel can
   learn collapse from answer loss alone.
4. Feed this read kernel back into the multi-kernel tasks from E9.
```

---

## E11: WMT Algebraic Internal Readout Probe

Question:

```text
After a probabilistic read kernel reaches an internal TreeHeap node, does
internal-node readout become easier if the target is an algebraically natural
subheap attribute instead of an arbitrary checksum bucket?
```

Why this follows E10:

```text
E10 proved that path collapse from arr[1] can route to the target node, but
internal checksum labels stayed weak. E11 tests whether the failure was caused
by asking the internal state to predict an unnatural label.
```

Script:

```text
ara/s1-echo/src/s1_algebraic_readout_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_algebraic_readout_probe/
ara/s1-echo/evidence/s1_algebraic_readout_probe_b16/
```

Dataset:

```text
WMT17 English side
SentencePiece model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 5000
train/test/ood = 4000/500/500
token length = 4..8
vocab limit including PAD = 513
```

Targets:

| Target | Meaning |
|---|---|
| `length` | number of non-pad tokens in the queried subheap span |
| `first` | first non-pad token in the subheap span |
| `last` | last non-pad token in the subheap span |
| `residue` | side diagnostic for possible modular folding; not part of the core claim |
| `prefix0` | first ordered token slot |
| `prefix1` | second ordered token slot |

Models:

| Model | Meaning |
|---|---|
| `root_query_decoder` | TreeEncoder root state plus query id predicts all targets. This is the root bottleneck baseline. |
| `routed_state_decoder` | Target node state plus query id predicts all targets. This assumes E10's route collapse has selected the node. |
| `algebraic_oracle` | Deterministic TreeHeap span decoder. This is the mathematical upper bound, not a trainable model. |

Predict:

```text
P-S1-READ02:
If internal TreeHeap state contains useful algebraic subheap information, then
routed internal-node state should beat root bottleneck on natural subheap
attributes: length, first token, last token, and ordered prefix. The residue
column is recorded only as a side diagnostic for modular-folding speculation.
```

Run A: natural readout plus 64-bucket residue diagnostic

```text
epochs = 80
```

Internal OOD result:

| Model | Length | First | Last | Residue | Prefix0 | Prefix1 | Mean | Exact all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `root_query_decoder` | 0.8388 | 0.5543 | 0.2387 | 0.0847 | 0.5543 | 0.3756 | 0.4411 | 0.0516 |
| `routed_state_decoder` | 0.9886 | 0.9277 | 0.8725 | 0.3675 | 0.9267 | 0.8725 | 0.8259 | 0.3571 |
| `algebraic_oracle` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Run B: natural readout plus 16-bucket residue diagnostic

```text
epochs = 80
```

Internal OOD result:

| Model | Length | First | Last | Residue | Prefix0 | Prefix1 | Mean | Exact all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `root_query_decoder` | 0.8527 | 0.5624 | 0.2303 | 0.1087 | 0.5637 | 0.3675 | 0.4476 | 0.0380 |
| `routed_state_decoder` | 0.9919 | 0.9355 | 0.9011 | 0.5203 | 0.9367 | 0.8849 | 0.8617 | 0.4755 |
| `algebraic_oracle` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Decision:

```text
S1-READ-C02 -> supported pilot
```

Interpretation:

```text
This supports the idea that internal-node readout should first target natural
TreeHeap algebraic attributes. Routed node state is much better than the root
bottleneck on length, first/last token, and ordered prefix. The residue column
is kept as a diagnostic artifact only; it should not decide whether S1-READ-C02
passes or fails.
```

What this does not prove:

```text
not translation
not semantic phrase meaning
not unsupervised route discovery
not long-sequence syntax
not superiority over Transformer/pointer baselines
```

Next action:

```text
1. Add matched pointer/Transformer read baselines.
2. Remove teacher-forced target node state and connect E10 route probabilities
   to E11 algebraic readout.
3. Use the natural algebraic readout as the base for semantic phrase decoders.
```

---

## E12: Ordered Fold Kernel Probe

Question:

```text
When a 1D ordered token array is folded into a TreeHeap, what must be preserved
for natural internal-node readout to work?
```

Why this follows E11:

```text
E11 clarified that residue/mod should not decide SPR-034. E12 separates the
issue: first prove that order-preserving TreeHeap fold is sufficient for
natural subheap readout, then treat modulo/cyclic folding as a later optional
operator.
```

Script:

```text
ara/s1-echo/src/s1_ordered_fold_kernel_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_ordered_fold_kernel_probe/
```

Dataset:

```text
pure toy data
samples = 5000
sequence length = 8..16
max_len = 16
vocab = 257
mod_base = 4
```

Models:

| Model | Meaning |
|---|---|
| `ordered_tree_fold` | Complete binary TreeHeap. Leaf addresses preserve original order; internal nodes compose exact subheap summaries. |
| `bag_root_fold` | Collapses the whole sequence into one global/root summary, losing subheap locality. |
| `modulo_fold_base4` | Folds positions by `position % 4` before answering; this intentionally aliases distant positions. |

Targets:

```text
length, first, last, prefix0, prefix1
```

Predict:

```text
P-S1-FOLD01:
If natural TreeHeap readout depends on ordered path/address preservation, then
ordered_tree_fold should be exact, while bag/global fold and early modulo fold
should lose subheap locality. Modulo may still be useful later, but not as a
replacement for the first ordered fold.
```

Result:

| Model | Length | First | Last | Prefix0 | Prefix1 | Mean natural | Exact all |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ordered_tree_fold` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `bag_root_fold` | 0.0888 | 0.3236 | 0.3236 | 0.3236 | 0.3235 | 0.2766 | 0.0888 |
| `modulo_fold_base4` | 0.0888 | 0.4032 | 0.4034 | 0.4032 | 0.4036 | 0.3405 | 0.0888 |

Derived:

```text
ordered_minus_bag_mean_natural = +0.7234
ordered_minus_mod_mean_natural = +0.6595
pilot_pass = true
```

Decision:

```text
S1-FOLD-C01 -> supported pilot
```

Interpretation:

```text
The first fold from a linear ordered array into a TreeHeap must preserve leaf
address/path locality. A bag/root collapse cannot answer local subheap
queries. An early modulo/cyclic fold aliases positions and also fails natural
subheap readout. This does not reject modulo as a later operator; it only says
modulo should be studied as a separate folding kernel after ordered structure
exists.
```

What this does not prove:

```text
not translation
not learned semantic routing
not a neural baseline battle
not proof that modulo is useless
not long real syntax
```

Next action:

```text
1. Connect ordered fold with probabilistic route collapse from E10.
2. Add learned baselines: flat MLP, pointer network, small Transformer read.
3. Separately design a modulo/cyclic folding experiment where the target is
   truly periodic, not natural subheap readout.
```

---

## E13: Controllable Fold Manifold Probe

Question:

```text
Can TreeHeap fold quality be controlled by kernel variables, so that output
structure moves from noisy blocks toward stable product-like blocks along a
measurable surface?
```

Why this follows E12:

```text
E12 proved that order-preserving fold is necessary for natural readout.
SPR-036 then reframed language fold as latent placement and attraction.
E13 connects the algebra/product gap: expose kernel controls and measure how
the generated fold changes.
```

Script:

```text
ara/s1-echo/src/s1_controllable_manifold_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_controllable_manifold_probe/
```

Dataset:

```text
pure toy relation field
sentence cases = 4
seeds = 64
weights = {0, 0.25, 0.5, 1, 2, 4}
```

Example cases:

```text
the cat is running for a car
the book that i bought yesterday is expensive
```

Kernel score:

```text
score(A, B)
  = relation_weight * relation(A, B)
  + order_weight    * order(A, B)
  - balance_penalty
  + noise
```

Predict:

```text
P-S1-MANIFOLD01:
If kernel controls can form a product-facing fold manifold, then increasing
relation/order control should improve block F1 from low-control noisy output
toward stable target blocks.
```

Result:

| Metric | Value |
|---|---:|
| low-control mean F1 | 0.0828 |
| best mean F1 | 0.8148 |
| high-sum-control mean F1 | 0.8148 |
| diagonal gain | 0.7319 |
| product cells | 22 |
| pilot pass | true |

Decision:

```text
S1-MANIFOLD-C01 -> supported pilot
```

Interpretation:

```text
On a transparent toy relation field, fold quality can be treated as a control
surface over kernel variables. This supports a product-development method:
debug TreeHeap by exposing controls, structure quality, energies, and
uncertainty, not only by asking whether final generation is good.
```

What this does not prove:

```text
not WMT
not unsupervised relation discovery
not natural language understanding
not Transformer superiority
not that these two knobs are final
```

Next action:

```text
Replace the toy relation field with a relation field estimated from real data:
co-occurrence, weak dependency pairs, masked-token restoration, or contrastive
phrase pairs. Rerun the same control-surface analysis.
```

---

## E14: Heap-State Relaxation Probe

Question:

```text
Can gradient information update the current TreeHeap state H itself, rather
than only updating kernel parameters theta?
```

Why this follows E13:

```text
E13 proved a manually controlled fold surface. Houming818 then proposed that
the gradient on this surface may also act as heap balance adjustment:
H <- H - eta * grad_H E(H).
```

Script:

```text
ara/s1-echo/src/s1_heap_state_relaxation_probe.py
```

Remote host:

```text
io.grepcode.cn
```

Evidence:

```text
ara/s1-echo/evidence/s1_heap_state_relaxation_probe/
```

Design:

```text
theta_updated = false
heap_state_updated = true
target_heap_used_in_loss = false
loss_type = energy over current heap state
```

Scalar energy:

```text
E = (left - right)^2 + (root - (left + right) / 2)^2
```

Vector energy:

```text
parent-child consistency + relation-anchor consistency
```

Result:

| Metric | Value |
|---|---:|
| scalar energy ratio | 2.47e-31 |
| scalar left delta | +1.0000 |
| scalar right delta | -1.0000 |
| mean vector energy ratio | 1.24e-13 |
| max vector energy ratio | 3.69e-13 |
| mean centroid error drop | 3.0393 |
| pass rate | 1.0000 |
| pilot pass | true |

Decision:

```text
S1-RELAX-C01 -> supported pilot
```

Interpretation:

```text
TreeHeap has a state-gradient path in addition to parameter-gradient learning.
The current arr[i] values can be moved by gradients from a differentiable
energy function while fixed address rules and fixed kernel coefficients remain
unchanged.
```

What this does not prove:

```text
not WMT
not natural language understanding
not unsupervised relation-field learning
not Transformer superiority
```
## P-S1-KERNEL39: Parameter TreeHeap Learns A Local Convolution Kernel

Status: executed
Claim: S1-KERNEL-LEARN-C01
Design: `logic/kernel_parameter_learning.md`
Script: `src/s1_kernel_parameter_learning_probe.py`
Evidence: `evidence/s1_kernel_parameter_learning_probe/`

### Question

Can a parameter TreeHeap `Theta` learn the hidden local convolution kernel
that maps each internal node and its children to a new state?

### Toy Data

Use complete binary heaps:

```text
        h1
      /    \
    h2      h3
   /  \    /  \
 h4   h5 h6   h7
```

Hidden kernel:

```text
theta* = [1, 1, 1]
```

Target:

```text
y1 = h1 + h2 + h3
y2 = h2 + h4 + h5
y3 = h3 + h6 + h7
```

Example:

```text
H  = [1,2,3,4,5,6,7]
H' = [6,11,16,4,5,6,7]
```

### Pass Gate

- learned `Theta` is close to `[1,1,1]`
- train/test/OOD MSE is low
- wrong-address baseline fails or is materially worse
- result updates `Theta`, not only `H`

### Result

```text
pilot_pass = true
learned_theta = [0.9999999999999998, 0.9999999999999998, 1.0000000000000004]
theta_l2_error = 5.43896e-16
treeheap_test_mse = 8.78177e-31
treeheap_ood_mse = 8.92978e-30
wrong_address_test_mse = 5.92848
flat_global_test_mse = 3.56010
```

Decision:

```text
S1-KERNEL-LEARN-C01 -> supported pilot
```

### Boundary

This is a kernel-learning proof, not a language proof.

## P-S1-KERNEL40: Mirror / Chiral Kernel Flip

Status: executed
Claim: S1-KERNEL-MIRROR-C01
Design: `logic/mirror_kernel_symmetry.md`
Script: `src/s1_mirror_kernel_symmetry_probe.py`
Evidence: `evidence/s1_mirror_kernel_symmetry_probe/`

### Question

Can a geometric left/right mirror on TreeHeap be implemented as algebraic
permutation of heap addresses and local kernel slots?

Can scalar loss learn the mirrored root/left/right slot assignment without
claiming rotation-angle learning or full 3D fold?

### Definition

Physical mirror:

```text
M(1)=1
M(2)=3, M(3)=2
M(4)=7, M(5)=6, M(6)=5, M(7)=4
```

Kernel:

```text
theta = [root,left,right]
P_lr(theta) = [root,right,left]
```

Expected law:

```text
P_m K_theta(H) = K_{P_lr theta}(P_m H)
```

### Pass Gate

- flipped-kernel equivariance error near zero
- unflipped kernel fails on mirrored heaps
- learned mirrored kernel recovers `[root,right,left]`
- learned left slot matches original right coefficient
- learned right slot matches original left coefficient

### Result

```text
pilot_pass = true
test_max_flipped_error = 8.88e-16
ood_max_flipped_error = 3.55e-15
test_mean_unflipped_error = 6.4372
learned_theta = [0.5000000000000002, -0.7499999999999998, 1.2499999999999996]
theta_mirror_l2_error = 5.44e-16
left_slot_learns_original_right_error = 2.22e-16
right_slot_learns_original_left_error = 4.44e-16
learned_test_mse = 1.01e-30
learned_ood_mse = 9.76e-30
```

Decision:

```text
S1-KERNEL-MIRROR-C01 -> supported pilot
```

### Boundary

This is a local algebraic/convolution proof, not a language proof. It is also
not a rotation-angle proof or a full 3D fold proof.

## P-S1-RECURSIVE-ROUTE01: Recursive TreeHeap Route

Status: executed
Claim: S1-RECURSIVE-ROUTE-C01
Design: `logic/s1_recursive_treeheap_route.md`
Script: `src/s1_recursive_treeheap_route_probe.py`
Evidence: `evidence/s1_recursive_treeheap_route_probe/`

### Question

Can S1 mirror recovery be implemented as a real recursive TreeHeap route instead
of a flat `L x L` route matrix?

### Definition

A flat route matrix is:

```text
route_logits[length, out_pos, in_pos]
state = route @ leaf
```

A TreeHeap route is:

```text
i = 1
S_i = [arr[i], arr[2i], arr[2i+1]]
K_theta(q, S_i, address_i) -> stop/left/right
if left:  i = 2i
if right: i = 2i + 1
if stop:  read arr[i]
```

### Pass Gate

- hard TreeHeap mirror oracle reaches OOD exact 1.0
- learned recursive route reaches OOD exact >= 0.99
- evidence includes actual `stop/left/right` action traces
- old length-indexed flat route matrix fails on unseen lengths

### Result

```text
data = /mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv
samples = 20,000
heap max_len = 32
train lengths = 3..24
OOD lengths = 25..32

oracle_ood_exact = 1.0000
recursive_ood_exact = 1.0000
recursive_ood_token_acc = 1.0000
flat_length_matrix_ood_exact = 0.0000
flat_length_matrix_ood_token_acc = 0.0097
pilot_pass = true
```

Example route:

```json
{
  "pos": 0,
  "target_leaf": 31,
  "actions": [2, 2, 2, 2, 2, 0],
  "node": 63
}
```

Here `2` means `right`, and `0` means `stop`.

Decision:

```text
S1-RECURSIVE-ROUTE-C01 -> supported pilot
```

### Boundary

This repairs the mechanism boundary for S1 route proofs. It does not prove
translation, semantic grounding, unsupervised span discovery, or natural
language trigger learning. The target position is still supervised. The next
control should compare against flat shared route and pointer baselines.

## P-S1-CONTENT-ROUTE01: Content-Aware Recursive TreeHeap Route

Status: executed
Claim: S1-CONTENT-ROUTE-C01
Script: `src/s1_content_treeheap_route_probe.py`
Evidence: `evidence/s1_content_treeheap_route_probe_20k_e5/`

### Question

Does recursive routing still work when the kernel reads heap content instead of
precomputed geometry-answer features?

### Definition

Allowed kernel input:

```text
query token summary
arr[i]
arr[2i]
arr[2i+1]
```

Forbidden shortcut:

```text
target-in-left flag
target-in-right flag
precomputed interval answer
```

### Result

```text
data = /mnt/nas/datasets/wmt_massive/train.massive.zh-en.tsv
samples = 20,000
vocab = 1024
heap max_len = 32
train lengths = 3..24
OOD lengths = 25..32

train_route_steps = 352110
ood_route_steps = 44130
dense_feature_memory_mb = 6191.25
content_epoch_sec ~= 12.8

OOD step_acc = 0.9983
OOD route_exact = 0.9902
flat_length_matrix OOD token_acc = 0.0029
flat_length_matrix OOD exact = 0.0000
pilot_pass = true
```

### Boundary

This is still a controlled echo route proof. It uses supervised query tokens,
bag/count summaries, and unique-token query positions. It is not translation,
semantic grounding, or unsupervised span discovery.

The complexity profile says the dense proof form is not scalable:

```text
vocab=1024 => dense feature memory ~= 6.2GB
```

The next implementation should replace vocab-count summaries with compact
64D/128D subheap embeddings and generate route examples on the fly.
