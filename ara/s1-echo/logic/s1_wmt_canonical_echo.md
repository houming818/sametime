# S1 WMT Canonical Echo

Created: 2026-07-03
Owner: Codex Review
Stage: S1 Echo -> S2 Bridge

## Claim

`S1-CANON-WMT-C01`

S1 echo should not be framed as random sentence repair. A better S1 target is a
low-entropy canonical TreeHeap state:

```text
surface sentence
-> canonical TreeHeap state
-> original surface sentence
```

For WMT parallel data, the first measurable version is:

```text
en_sentence -> H_en
zh_sentence -> H_zh
```

where `H_en` and `H_zh` should move closer for true translation pairs than for
random mismatched pairs, while each side can still echo back its own surface
tokens.

## Predict

`P-S1-CANON-WMT01`

If S1 can learn a canonical state useful for the S2 bridge, then on held-out WMT
parallel sentences:

1. Positive pair distance `d(H_en, H_zh)` should be lower than mismatched pair
   distance.
2. Retrieval from `H_en` to `H_zh` should beat random retrieval.
3. Alignment softmax entropy should decrease after training, meaning the model
   becomes less uncertain about which target state matches the source state.
4. English and Chinese echo token accuracy should remain non-trivial, so the
   canonicalization does not collapse into an unreadable global hash.
5. TreeHeap should be compared with a bag-of-words baseline under the same data
   and objective. If BoW matches or beats TreeHeap, the result is a baseline
   challenge, not a TreeHeap win.

## Proof Design

The proof uses WMT17 `train.zh-en` from `/mnt/nas/datasets/wmt17`.

Data:

```text
line format: English<TAB>Chinese
sample target: 50k pairs by default
length filter: 4..48 tokens per side
vocab: separate en/zh vocabularies
```

Models:

```text
TreeHeapCanonical:
  token embedding
  path/address embedding
  balanced binary TreeHeap compose kernel
  root projection -> canonical state
  leaf decoder -> echo tokens

BoWCanonical:
  token embedding
  masked mean pool -> canonical state
  leaf decoder -> echo tokens
```

Loss heads are logged separately:

```text
L_align_en_to_zh
L_align_zh_to_en
L_echo_en
L_echo_zh
```

The alignment loss is contrastive over the batch:

```text
score(i,j) = cosine(H_en_i, H_zh_j) / temperature
target(i) = i
```

This is not translation decoding. It only tests whether S1 can build a
canonical bilingual state that is closer for true parallel pairs while still
supporting echo.

## Falsification

Downgrade or reject if:

```text
positive distance is not lower than negative distance
retrieval@k is near random
alignment entropy does not improve
echo token accuracy collapses
BoW baseline matches or beats TreeHeap across the same metrics
```

This proof does not establish WMT BLEU, semantic understanding, or unsupervised
language law discovery. It is a first WMT-scale canonical-state probe.

## Evidence

Run:

```text
ara/s1-echo/evidence/s1_wmt_canonical_echo_probe/
```

Script:

```text
ara/s1-echo/src/s1_wmt_canonical_echo_probe.py
```

Host:

```text
io.grepcode.cn
```

Config:

```text
samples = 50,000
train/test/OOD = 40,000 / 5,000 / 5,000
max_eval = 2,000 per held-out split
max_len = 48
en_vocab = 8192
zh_vocab = 6805
dim = 128
epochs = 5
```

OOD result:

```text
TreeHeap positive_distance = 0.345768
TreeHeap negative_distance = 0.992941
TreeHeap margin            = 0.647173
TreeHeap retrieval@1       = 0.630000
TreeHeap retrieval@5       = 0.819500
TreeHeap entropy           = 4.044313
TreeHeap positive_prob     = 0.227486
TreeHeap en_echo_token     = 1.000000
TreeHeap zh_echo_token     = 0.998135

BoW margin                 = 0.593911
BoW retrieval@1            = 0.628500
BoW retrieval@5            = 0.808500
BoW entropy                = 4.344241

Untrained TreeHeap margin  = 0.003005
Untrained retrieval@1      = 0.001000
```

Pass checks:

```text
positive_distance_below_negative = true
retrieval_beats_random           = true
positive_probability_beats_random= true
entropy_below_uniform            = true
echo_nontrivial                  = true
treeheap_beats_bow_margin        = true
```

## Interpretation

Supported:

```text
WMT parallel surface forms can be trained into a shared canonical state with
strong contrastive retrieval and near-lossless same-language echo.
```

TreeHeap-specific support is positive but modest:

```text
TreeHeap margin > BoW margin
TreeHeap entropy < BoW entropy
TreeHeap retrieval@1 is only slightly above BoW retrieval@1
```

Therefore the honest status is:

```text
supported pilot, small TreeHeap advantage over BoW on this first canonical
probe.
```

Not proved:

```text
translation BLEU
world-model grounding
semantic consciousness space
unsupervised canonical discovery without parallel supervision
large TreeHeap advantage over strong sequence baselines
```

Next:

```text
add a sequence Transformer/GRU baseline
separate root canonical loss from leaf echo memory
test longer/noisier WMT and sentence retrieval beyond 2k candidate windows
connect canonical state to S2 decoding
```
