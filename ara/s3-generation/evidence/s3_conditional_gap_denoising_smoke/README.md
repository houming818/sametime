# Conditional Denoising Seq2Seq: Short Gap Smoke

The input is a 64-BPE Chinese block with one contiguous 16-token gap.  The
decoder generates only the missing 16-token span.  All models trained for
1,000 updates.

| Model | Test NLL | Teacher accuracy | Greedy token accuracy | Exact |
|---|---:|---:|---:|---:|
| TreeHeap | 7.2434 | 0.0813 | 0.0107 | 0.0 |
| Flat GRU | 7.1996 | 0.0820 | 0.0136 | 0.0 |
| BoW | 7.2371 | 0.0789 | 0.0132 | 0.0 |

TreeHeap full and leaf-only are again tied (`7.2434` vs `7.2419` NLL).
Short conditional output alone did not reproduce the readable WMT behavior.
The evidence points to aligned translation pairs, not output length alone, as
the important condition in the earlier WMT result.
