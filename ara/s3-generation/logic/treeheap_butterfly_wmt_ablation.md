# TreeHeap Butterfly WMT Matched Ablation

Status: supported / formal three-seed WMT ablation

Date: 2026-07-30

Owners: Houming818 and Codex Review

## 1. Question

The synthetic Butterfly experiment established sparse address communication, but
it did not establish language value. This experiment asks whether the same
pairing topology improves a real English-to-Chinese TreeHeap model.

The experiment keeps the adaptive lifting TreeHeap and recursive decoder fixed.
Only the communication schedule between token WRITE and TreeHeap FOLD changes.

```text
source pieces
  -> token embedding
  -> identity / adjacent / XOR-Butterfly communication
  -> adaptive lifting FOLD
  -> root + addressed details
  -> exact UNFOLD levels
  -> recursive probabilistic READ
  -> Chinese target cross-entropy
```

## 2. Local reversible kernel

For every paired state `(a, b)`, one coupling step is:

```text
b1 = b + alpha * tanh(F(a))
a1 = a + alpha * tanh(G(b1))
```

Its inverse is computed in reverse order:

```text
a = a1 - alpha * tanh(G(b1))
b = b1 - alpha * tanh(F(a))
```

`F` and `G` are shared learned kernels. Their output layers start at zero, so
all arms begin from exactly the original adaptive lifting model. No `N x N`
attention matrix is constructed. Padding addresses do not participate in a
pair update.

## 3. Matched arms

| Arm | Schedule | Purpose |
|---|---|---|
| `identity` | allocate the same kernels but bypass every pair update | original TreeHeap baseline |
| `adjacent` | repeat address-bit 0 pairs at every stage | controls for extra depth and parameters |
| `butterfly` | stage `s` pairs `i` with `i XOR 2^s` | proposed sparse long-range topology |

All arms use the same parameter allocation, initialization, sampled WMT rows,
batch order, optimizer, update count, decoder, and evaluation split.

## 4. Claim

### S3-TREEHEAP-BUTTERFLY-WMT-C02

A reversible XOR-Butterfly communication layer inserted before adaptive
TreeHeap FOLD should provide useful long-range source interaction on real WMT
translation. Under matched training, it should improve held-out NLL over both
identity and repeated-adjacent schedules, show at least as much benefit on long
sources as on the full test set, and become causally necessary after training.

This claim does not assert Transformer superiority, production translation,
semantic address discovery, or compute savings.

## 5. Predictions and gates

Smoke is a code and mechanism gate, not evidence for the language claim:

1. all three arms train on CUDA with finite loss and gradients;
2. initial communication output is exactly identity within `1e-7`;
3. Butterfly forward/inverse MSE is at most `1e-10`;
4. adaptive FOLD/UNFOLD closure MSE is at most `1e-10`;
5. parameters are equal across arms and no dense address matrix is allocated.

Formal evidence uses three seeds. At least two seeds must satisfy all of:

1. Butterfly test NLL improves over identity by at least `0.02`;
2. Butterfly test NLL improves over adjacent by at least `0.015`;
3. on source lengths 25--32, Butterfly improves over identity by at least
   `0.02`, and the long-source gain is not smaller than the all-source gain by
   more than `0.01`;
4. replacing a trained Butterfly schedule with identity at evaluation worsens
   NLL by at least `0.02`;
5. source shuffling worsens Butterfly NLL by at least `0.50`;
6. route mass uses at least two TreeHeap resolutions with mass at least `0.05`;
7. generation is non-empty and finite.

The aggregate result is supported only if at least two of three seeds pass and
mean Butterfly NLL is better than both controls. Otherwise the real-language
claim remains open or is rejected even though the synthetic mechanism remains
supported.

## 6. Scale

Smoke:

```text
train/valid/test = 5,000 / 500 / 500
epochs = 2
dim/hidden = 128 / 128
one seed
```

Formal:

```text
train/valid/test = 200,000 / 5,000 / 5,000
epochs = 5
dim/hidden = 256 / 256
three seeds
```

The formal queue starts only after smoke passes every mechanism gate and the
GPU process, power, memory, and epoch logging are verified.

## 7. Smoke result

The registered smoke ran on `io` through taskd task 82. It used CUDA, seed
8104, 5,000/500/500 WMT rows, two epochs, and 128-dimensional states. Runtime
was 119.5 seconds. The GPU process used about 5 GB VRAM during the first arm.

| Metric | Identity | Adjacent | Butterfly |
|---|---:|---:|---:|
| held-out NLL | `7.59495` | `7.60190` | `7.55776` |
| long-source NLL | `7.98503` | not a decision metric | `7.93839` |

Derived smoke observations:

```text
Butterfly gain over identity = 0.03719 NLL
Butterfly gain over adjacent = 0.04414 NLL
long-source gain             = 0.04664 NLL
disable-Butterfly damage     = 0.18060 NLL
source-shuffle damage        = 1.00879 NLL
```

All six mechanism gates passed: finite gradients, exact initial identity,
communication inverse, FOLD/UNFOLD closure, equal parameter count, and no dense
attention allocation. Every provisional language gate also passed in this one
small seed. This is permission to run the formal experiment, not support for
the claim.

Formal taskd task 83 used 200,000/5,000/5,000 rows, five epochs, and seeds
8104/8105/8106.

## 8. Formal result

Task 83 completed on `io` in 31,286.5 seconds. All three arms had exactly
34,445,832 allocated parameters. Each seed sampled 210,000 rows from the same
2,000,000-line scan and used matched initialization and batch order across
arms.

### 8.1 Held-out WMT

| Seed | Identity NLL | Adjacent NLL | Butterfly NLL | Gain vs identity | Gain vs adjacent |
|---:|---:|---:|---:|---:|---:|
| 8104 | `4.62285` | `4.62273` | `4.54546` | `0.07738` | `0.07727` |
| 8105 | `4.66175` | `4.67974` | `4.58915` | `0.07260` | `0.09059` |
| 8106 | `4.67127` | `4.65998` | `4.56066` | `0.11061` | `0.09932` |
| Mean | `4.65196` | `4.65415` | **`4.56509`** | **`0.08687`** | **`0.08906`** |

Mean token BLEU-4 was `9.9501/9.9485/10.5462` for
identity/adjacent/Butterfly. Generation remains a small-model WMT result, not a
production-quality translation claim.

### 8.2 Long-source and causal checks

Butterfly NLL gains over identity on 25--32-piece sources were
`0.07451/0.07825/0.10424`, mean `0.08567`. The effect therefore did not vanish
on the preregistered long-source subset.

Disabling a trained Butterfly increased NLL by
`1.15034/1.11919/1.23666`. Replacing its schedule with repeated adjacent pairs
increased NLL by `0.92585/1.12803/1.13160`. Source shuffling increased NLL by
`3.23155/3.18970/3.04751`.

These interventions distinguish the address schedule from merely allocating
extra nonlinear layers: the trained model depended on the native changing-bit
topology.

### 8.3 Algebra and decision

All six mechanism gates and all seven language gates passed in all three
seeds. Butterfly inverse MSE was between `6.73e-16` and `7.15e-16`; lifting
FOLD/UNFOLD closure MSE was between `6.68e-15` and `8.17e-15`. No dense address
attention tensor was allocated.

The preregistered decision is `supported`. The evidence supports a narrow
claim: sparse changing-bit communication improves this matched adaptive
TreeHeap WMT model. It does not establish semantic XOR addresses, Transformer
superiority, production translation quality, or measured compute savings.

## 9. Falsification boundary

If Butterfly merely matches identity, the sparse transport exists but adds no
measurable WMT value under this encoder. If adjacent matches Butterfly, extra
nonlinear depth rather than long-range topology explains the effect. If native
and disabled schedules match after training, the optimizer ignored the layer.
Any of these outcomes blocks promotion of the language claim.
