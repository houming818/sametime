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
