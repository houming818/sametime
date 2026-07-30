# OBS-002 Matched-Decoder Resolution Observation

Status: observation only; no Claim registered

Date: 2026-07-30

Host: `io`

Formal run: `taskd` task 69

Runtime: 50.6 seconds on CUDA; script-measured body 49.1 seconds

## Contract

- Reconstruct the frozen C06 held-out WMT split.
- Select 1,000 examples, 250 from each source-length bucket: `8-12`, `13-16`,
  `17-24`, and `25-32` pieces.
- Encode each batch once with the frozen C04 encoder.
- Reuse exactly the same clean `levels/masks` for three compatible C06 decoders.
- Expose D0 through D5 without training or parameter updates.
- Generate 96 examples, 24 from each source-length bucket.
- Calculate target NLL/rank, entropy, Top-10 mass, adjacent JS/context movement,
  route mass, source-shuffle damage, state geometry, and short free generation.

## Checkpoints

```text
dae8e23618bb8ffda7ed00057eb452dad0c3427446ef16033ce84a11de02ccad  growth_step62500.pt
425d8bbbb59d0b32d50e4ddb547b8ef4e1dc67fe21a170b8fde454e6f14f5e2f  decoder_native_control.pt
0ed3b8223a181ab503e7ae7afbd734ad530fce214e61633190ebd449c25b9a5f  decoder_leaf_reference.pt
289ea83f976f8c005ab68ee11588edad962d589f3f56e7365ccede457d9fcbbd  decoder_depth_floor.pt
```

The loaded encoder state dictionary digest was
`f14529d9f5a8...` for every arm.

## Main Aggregate

| Decoder | D0 NLL | D5 NLL | D0-D5 gain | D0 target rank | D5 target rank |
|---|---:|---:|---:|---:|---:|
| Native | 3.5875 | 3.5875 | 0.0000 | 207.1 | 207.1 |
| Forced leaf | 3.8833 | 3.5245 | 0.3588 | 232.1 | 192.2 |
| Depth floor | 3.6495 | 3.4790 | 0.1705 | 213.3 | 195.9 |

Native routing stopped at root at every visible depth. Forced-leaf routing was
non-monotonic before improving at D4-D5. Depth-floor routing improved smoothly
from D1-D5 while keeping about 54.5% mass at root in the complete view.

Raw outputs:

- [`summary.json`](summary.json)
- [`per_example.jsonl`](per_example.jsonl)

Reproduction script:
[`../../src/s3_resolution_observe_matched.py`](../../src/s3_resolution_observe_matched.py)

Chinese report:
[`../../observations/OBS-002-MATCHED-DECODER-RESOLUTION.zh.md`](../../observations/OBS-002-MATCHED-DECODER-RESOLUTION.zh.md)
