# S1 Evidence Index

Owner: Review Engineer
Writer: Codex
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Bind SPR Echo evidence to claims.

## E-S1-20260616-hash-cyclic

Command:

```bash
cd /data/homecicd/sametime/code/wmt
python3 spr_hash_cyclic.py
```

Observed on io:

```text
pure roll collision: True
roll + sign alternation separated: True
full tree separated: True
```

Claims:

- S1-C02

Interpretation:

Pure cyclic shift is insufficient because it permits reversed-order collision. Sign alternation adds the required non-commutative break for this toy case.

## E-S1-20260616-echo-proof

Command:

```bash
cd /data/homecicd/sametime/code/wmt
sudo -n python3 spr_echo_proof.py
```

Observed on io:

```text
vocab=41429 d=64 K=4 chunk_dim=16 chunk_leaves=128
total effective leaves = 128^4 = 268,435,456
combined: solo=41311 multi=59 active_leaves=41370
solo% = 99.7%
BLEU-4 = 99.99
```

Claims:

- S1-C01
- S1-C03

Interpretation:

This proves capacity and near-lossless self-mapping. It does not prove semantic routing, because a sufficiently large hash-like space can also echo.

## Open Evidence Slots

- `E-S1-real-corpus-polysemy-routing`: required before S1-C10 can be promoted beyond controlled support.
- `E-S1-random-hash-baseline`: required before saying SPR routing is better than a matched random route.
- `E-S1-baseline-battle`: required before saying SPR is necessary rather than merely possible.

## E-S1-20260616-reproduce

Script:

```text
holds/SameTime/experiments/spr_s1_reproduce.py
```

Remote command:

```bash
cd /data/homecicd/sametime/code/wmt
sudo -n python3 spr_s1_reproduce.py
```

Observed on io:

```text
collision=True
sign_alt_separated=True
solo=41311/41429
solo_percent=99.72
bleu4=99.99
```

Notes:

The script was implemented independently but mirrors prior experiment details. One important reproducibility detail is that the original `spr_echo_proof.py` generates random embeddings on CPU and then moves them to CUDA; generating directly on CUDA changes the random stream and slightly changes solo/BLEU. The reproduce script now mirrors CPU generation exactly.

Claims:

- S1-C01
- S1-C02
- S1-C03

## E-S1-20260616-falsification

Script:

```text
holds/SameTime/experiments/spr_s1_falsification.py
```

Remote command:

```bash
cd /data/homecicd/sametime/code/wmt
sudo -n python3 spr_s1_falsification.py
```

Observed on io:

```text
collision=True
sign_alt=True
solo=41311/41429
bleu4=99.99
token_polysemy=0.43
keyword_polysemy=1.00
```

Detailed metrics:

```text
polysemy examples: 42
targets: bank, charge, light
token-only real accuracy: 0.4286
token-only shuffled accuracy: 0.4286
keyword real accuracy: 1.0000
keyword shuffled accuracy: 0.4524
```

Claims:

- S1-C01: confirmed again.
- S1-C02: confirmed again.
- S1-C03: supported again.
- S1-C10: remains open; current token-only path does not encode contextual semantics.
- S1-C11: falsified for current S1 token-only routing; same-token different-sense contexts receive no contextual route state.

Interpretation:

Current S1 echo routing is a high-capacity identity/path hash. It is not yet a semantic router. The next valid architecture step must add context-conditioned routing or connect S1 to S2 semantic/fold features, then rerun this falsification.

## E-S1-20260616-context-proof

Script:

```text
holds/SameTime/experiments/spr_context_proof.py
```

Remote command:

```bash
cd /data/homecicd/sametime/code/wmt
python3 spr_context_proof.py
```

Observed on io:

```text
examples=56
targets=bank, charge, light
token_acc=0.429
context_acc=1.000
shuffled_acc=0.482
context_purity=1.000
mixed_context_buckets=0
```

Claims:

- S1-C10: supported in a controlled synthetic-context setting.
- S1-C11: still rejected for the old token-only route.
- S1-C13: supported as a new architecture claim for route(token, context).

Interpretation:

This is a proof of mechanism, not a corpus benchmark. It shows that the existing path operator can separate same-token senses when the input vector includes a context signal, and that the gain collapses under label shuffle. It does not prove real-corpus semantic routing, translation quality, or superiority over a matched random hash/BoW baseline.
