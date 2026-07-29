# C08 Resolution Observation (Diagnostic Only)

Status: observation only; no Claim registered

Date: 2026-07-29

Host: `io`

Task: `taskd` task 63

Runtime: 4.5 seconds on CUDA

## Question

What does the already-trained C08 model actually generate when the same encoded
TreeHeap state is exposed to the same decoder at successively greater visible
depths?

This run does not train a model and does not assume that root means summary or
that deeper levels mean lexical detail.

## Fixed Inputs

- Released C08 encoder: `encoder-growth-step62500.pt`
- Released C08 decoder: `decoder-eos-tail.pt`
- Released tokenizer: `sp-bpe-massive.model`
- Five English source scenes
- One encoder pass per scene
- The resulting `H_state` is reused for every depth slice
- Same decoder and greedy decoding rule
- Only `levels[:visible_count]` changes from 1 through 6

## Aggregate First-Token Distribution

| Deepest visible depth | Mean entropy (nats) | Mean effective candidates `exp(H)` | Mean Top-10 mass |
|---:|---:|---:|---:|
| 0 | 5.646 | 311.1 | 0.407 |
| 1 | 5.850 | 365.6 | 0.364 |
| 2 | 5.905 | 384.6 | 0.354 |
| 3 | 5.999 | 420.0 | 0.338 |
| 4 | 6.120 | 473.4 | 0.316 |
| 5 | 6.014 | 439.2 | 0.304 |

For this checkpoint, opening more levels generally broadened the first-token
probability bucket. It did not monotonically sharpen it.

## Visible Generation Changes

| Source | Root only | Depth 4/5 observation |
|---|---|---|
| `The earth is round.` | unrelated/invalid pieces | `圆圆的圆珠轮在地上。` / `圆圆圆珠在地上。` |
| `The apple is sweet.` | unrelated `破碎机` fragment | `是苹果的。` / `甜菜...甜美的甜蜜。` |
| `Why is the window wet?` | unrelated fragments | `窗口, 为什么?` / `窗子为什么会不当?` |
| `A cat is eating some food.` | unrelated fragments | `吃过早餐...` |
| `I arrived home at seven o'clock.` | unrelated fragments | `回家,我回家了。` / `我回家了,我回家了。` |

The deeper slices exposed source-related lexical signal. Root-only output was not
a useful coarse summary; it was dominated by invalid pieces and corpus/web
templates. The deeper output was still poor translation and often repetitive.

## Boundary

This is a diagnostic intervention on one released checkpoint. The decoder was
trained with all visible levels and a depth floor, not as six independently
calibrated resolution decoders. Therefore the run shows what the current model
does under depth restriction; it does not establish a general TreeHeap
coarse-to-fine law.

Raw output: [`result.json`](result.json)

Reproduction script:
[`../../src/s3_resolution_observe_existing.py`](../../src/s3_resolution_observe_existing.py)
