# Conditional Denoising Seq2Seq: Full Reconstruction Smoke

The input is a 64-BPE Chinese block with 30% span masking.  The target is the
complete clean 64-token block.  All models trained for 1,000 updates.

| Model | Test NLL | Masked teacher accuracy | Greedy token accuracy | Exact |
|---|---:|---:|---:|---:|
| TreeHeap | 6.6111 | 0.0913 | 0.0181 | 0.0 |
| Flat GRU | 5.7053 | 0.0763 | 0.0280 | 0.0 |
| BoW | 6.5707 | 0.0944 | 0.0205 | 0.0 |

TreeHeap full and leaf-only are effectively tied (`6.6111` vs `6.6183` NLL).
The fixed recursive internal nodes did not provide measurable benefit.  Greedy
outputs remain poor and often repetitive.  This smoke does not support the
conditional-denoising claim.
