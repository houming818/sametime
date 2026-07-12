# Conditional Denoising Seq2Seq Gate

## Question

The WMT experiment produced more sentence-like Chinese than open continuation
pretraining.  Is that because a strongly conditioned seq2seq target is easier
to learn than an unconstrained next span?

This experiment uses only raw Chinese text and has two registered variants.
The first reconstructs a complete 64-token block from scattered masks.  The
second, closer to the successful short-translation setting, masks one
contiguous 16-token gap and generates only that missing span:

```text
full: damaged 64-token block -> encoder memory -> clean 64-token block
gap:  64-token block with one gap -> encoder memory -> missing 16-token span
```

The objective is ordinary teacher-forced reconstruction cross entropy:

\[
L_{denoise} = CE(P_\theta(x\mid \widetilde{x}), x)
\]

This is deliberately a conditional seq2seq task.  It does not test open-ended
question answering or world knowledge.

## Models

All models share the same autoregressive decoder:

- `treeheap`: the current fixed adjacent-pair recursive encoder;
- `flat_seq`: a GRU sequence encoder;
- `bow`: an order-free mean embedding encoder.

The current TreeHeap topology is fixed by source order.  Therefore a positive
result is evidence for the conditional generation path, not evidence for
learned semantic placement or persistent TreeHeap memory.

## Predictions

1. Denoising generation should be less repetitive than open next-span P0.
2. Order-aware encoders should beat BoW on masked-token recovery.
3. If recursive internal states add useful information, `treeheap` should beat
   its `leaf_only` read ablation and should be competitive with `flat_seq`.

## Controls

- News, wiki, and web documents are assigned to train/valid/test by stable
  document hash before token blocks are created.
- All models use the same tokenizer, corruption stream, decoder, dimensions,
  optimizer, and number of updates.
- Report teacher-forced NLL, masked-token accuracy, greedy token accuracy,
  exact reconstruction, and generated repetition.

## Falsification

Keep the claim open or reject it if held-out loss does not fall, greedy output
still collapses into repetition, or TreeHeap is matched by BoW after controlling
for parameter count.  Even a pass cannot establish learned TreeHeap topology,
world knowledge, consciousness, or a translation advantage.

## Smoke Results

Two 1,000-update runs were completed on io with matched approximately 12.7M
parameter models.

| Task / Model | Test NLL | Teacher accuracy | Greedy token accuracy |
|---|---:|---:|---:|
| full / TreeHeap | 6.6111 | 0.0913 | 0.0181 |
| full / Flat GRU | **5.7053** | 0.0763 | **0.0280** |
| full / BoW | 6.5707 | **0.0944** | 0.0205 |
| gap / TreeHeap | 7.2434 | 0.0813 | 0.0107 |
| gap / Flat GRU | **7.1996** | **0.0820** | **0.0136** |
| gap / BoW | 7.2371 | 0.0789 | 0.0132 |

All exact reconstruction scores were zero and outputs remained weak.  In both
tasks TreeHeap full-memory and leaf-only evaluations were effectively tied.
The smoke therefore rejects the strong prediction that conditional denoising
alone reproduces the readable WMT behavior.  The remaining interpretation is
that real aligned translation pairs provide a substantially more informative
training signal than either open continuation or raw-text gap recovery at this
scale.
