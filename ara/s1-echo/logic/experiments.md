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
