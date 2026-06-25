# S1 WMT echo kernel probe

Real WMT English SentencePiece short-sequence echo.

## Claim

`S1-WMT-ECHO-C01`: a structured TreeHeap kernel can write and read real WMT
SentencePiece short sequences in an echo setting, using tree addresses and
shared compose/read kernels rather than only a flat memorization map.

## Run

```text
host = io.grepcode.cn
device = cuda
script = ara/s1-echo/src/s1_wmt_echo_kernel_probe.py
dataset = /mnt/nas/datasets/wmt17/train.zh-en
spm = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 3000
length = 3..8
train/test/ood = 2400/300/300
```

## Models

| Model | Meaning |
|---|---|
| `bow_linear` | unordered bag-of-token baseline |
| `seq_mlp` | flat position-aware MLP baseline |
| `treeheap_kernel_echo` | fixed leaf write plus shared TreeHeap compose/read kernels |

## Result

| Model | Params | Test token | Test exact | OOD token | OOD exact |
|---|---:|---:|---:|---:|---:|
| `bow_linear` | 33,570,816 | 0.1679 | 0.0033 | 0.1659 | 0.0033 |
| `seq_mlp` | 16,794,112 | 0.5801 | 0.0567 | 0.5986 | 0.0533 |
| `treeheap_kernel_echo` | 423,104 | 0.9818 | 0.8900 | 0.9818 | 0.9000 |

## Decision

```text
S1-WMT-ECHO-C01 -> supported pilot
```

This supports short real-corpus BPE write/read with TreeHeap kernel structure.
It does not prove translation, semantic world modeling, compression, long
syntax, or superiority over copy-capable sequence baselines.
