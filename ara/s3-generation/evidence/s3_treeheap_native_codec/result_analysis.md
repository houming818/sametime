# S3 Native TreeHeap Codec Result Analysis

Date: 2026-07-14
Host: `io`
Claim: `S3-TREEHEAP-CODEC-C01`

## Decision

The experiment found a real but narrower result than the original claim.

- **Supported in one-seed smoke:** surface echo loss can train all five shared
  recursive operators into an address-sensitive continuous codec.
- **Rejected:** the learned continuous codec is not root-plus-detail. It ignores
  the root and stores the sequence in addressed detail channels.
- **Not supported:** the current 632-bit straight-through binary codec does not
  achieve useful echo fidelity.

## Evidence

| Run | Blocks | Representation | Valid top-1 | Exact | Root-zero top-1 | OOD-32 top-1 |
|---|---:|---|---:|---:|---:|---:|
| `smoke_v1` | 20,000 | unrestricted continuous | 0.9955 | 0.7715 | 0.9956 | 0.9957 |
| `smoke_v2` | 20,000 | symmetric/difference continuous | 0.9915 | 0.5918 | 0.9889 | 0.9886 |
| `smoke_v3` | 100,000 | 632-bit STE binary | 0.2073 | 0.0000 | 0.1882 | 0.2062 |

For v1/v2, shifting details to the wrong addresses reduced top-1 to
`0.0024/0.0014`, and deleting details reduced it to `0.0000/0.0349`. The
protocol therefore depends causally on TreeHeap addresses. Root deletion did
not reduce accuracy, so the root interpretation is falsified.

For v3, detail shift and deletion reduced top-1 from `0.2073` to
`0.0121/0.0613`. The finite details remain causal, but the representation or
straight-through optimization is inadequate. All five operator gradient norms
were finite and non-zero, ruling out a disconnected training path.

The machine-readable metrics, traces, checkpoints, and queue metadata are in
`smoke_v1/`, `smoke_v2/`, and `smoke_v3/`. Their durable copies are under
`/mnt/nas/sametime/ara/s3-generation/evidence/s3_treeheap_native_codec/`.

## What This Means

The model did learn its own handwriting: WRITE encoded tokens, recursive
analysis produced addressed codes, and recursive synthesis plus READ recovered
tokens. No pretrained embedding or structural label defined that protocol.

It did not yet learn the desired document summary. Echo alone rewards any
reversible copy protocol, including one that places nearly everything in local
details. A root becomes necessary only when the task asks for information that
cannot be copied locally, such as a masked or future span.

## Next Gate

Train a matched masked-span or next-span task with:

1. the same recursive operators;
2. no leaf or cross-attention bypass;
3. normal, root-zero, subheap-zero, and address-shift evaluations;
4. flat and local-detail controls at matched state budget.

The next claim should ask whether hierarchy improves prediction of missing
information. It should not call continuous vector width a bit rate.
