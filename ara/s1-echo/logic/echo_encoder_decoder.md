# Explicit Echo Encoder / Decoder

Date: 2026-06-30
Author: Codex Review
Status: supported pilot

## Motivation

After SPR-038, Houming818 asked whether we can implement the actual echo
encoder and decoder.

This document defines the hard/algebraic interface before asking a neural
kernel to learn it.

## Interface

Encoder:

```text
token sequence
-> ordered TreeHeap leaves
-> internal NodeState summaries
```

Decoder:

```text
root length + path-addressed leaf reads
-> reconstructed token sequence

internal node path
-> decoded subheap span
```

NodeState:

```text
length
first
last
ordered checksum
```

## Probe

Script:

```text
ara/s1-echo/src/s1_echo_encoder_decoder_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_echo_encoder_decoder_probe/
```

Dataset:

```text
source = wmt_sentencepiece
wmt_path = /mnt/nas/datasets/wmt17/train.zh-en
spm_model = /mnt/nas/datasets/wmt17/sp_bpe.model
samples = 2000
train/test/ood = 1600/200/200
min_len/max_len = 3/8
vocab_limit = 1024
```

Design:

```text
learned_parameters = 0
uses_target_heap_in_decoder = false
```

## Result

| Split | Sequence exact | Leaf acc | Subheap exact | Summary exact |
|---|---:|---:|---:|---:|
| train | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| ood | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Claim

```text
S1-ECHO-ED-C01:
An explicit TreeHeap echo encoder/decoder can close the hard echo interface on
real WMT SentencePiece short sequences: ordered leaf write, internal summary
compose, path-addressed leaf read, subheap decode, and full sequence decode.
```

Status:

```text
supported pilot
```

## Boundary

This does not prove:

```text
translation
learned semantic encoding
compression
superiority over neural baselines
noisy-channel correction
```

It proves:

```text
The echo encoder/decoder contract is well-defined and can be implemented
without using target heaps during decoding.
```

## Next Gate

The next proof should replace hard operations with learned approximations:

```text
learned write kernel
learned compose kernel
learned read/collapse kernel
noise or mask restoration
matched pointer/Transformer baseline
```

