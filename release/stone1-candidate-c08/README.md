# SameTime STONE-1 Candidate C08

This release is the first SameTime TreeHeap checkpoint that simultaneously
crosses the registered single-seed product thresholds and runs through a
non-teacher-forced CLI.

It is a **STONE-1 Candidate**, not `STONE-1: COMPLETE`. Multi-seed stability,
formal latency P50, and the complete same-checkpoint structural-causality audit
remain open.

## Recorded Result

```text
task: English-to-Chinese translation
test NLL: 3.4517
token BLEU-4: 13.8713
non-empty generation: 1.0
severe repetition rate: 0.015
physical TreeHeap leaves: 64
tail convention: repeated EOS, visible to the encoder
decoder route: six visible levels with a 2% floor
```

The model is an early research POC. It often produces grammatical Chinese and
recovers a useful semantic outline, but it can mistranslate relations, numbers,
names, and modifiers.

## Download

Download the model bundle attached to this release:

https://repos.grepcode.cn/houming818/grepcode-sametime/releases/tag/stone1-candidate-c08

Direct CDN downloads:

```text
https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.tar.gz
https://www.grepcode.cn/models/stone1-candidate-c08/sametime-stone1-candidate-c08.sha256
```

```text
sametime-stone1-candidate-c08.tar.gz
sametime-stone1-candidate-c08.sha256
```

The bundle contains:

```text
encoder-growth-step62500.pt
decoder-eos-tail.pt
sp-bpe-massive.model
MODEL_CARD.md
SHA256SUMS
```

The source code and CLI are part of the tagged SameTime repository.

## Run

Create a Python environment with PyTorch and SentencePiece, then run from the
tagged repository:

```bash
python3 ara/s3-generation/src/treeheap_fixed_root_cli.py translate \
  --encoder-checkpoint /path/to/encoder-growth-step62500.pt \
  --decoder-checkpoint /path/to/decoder-eos-tail.pt \
  --tokenizer /path/to/sp-bpe-massive.model \
  --text "Artificial intelligence can help people understand the world."
```

Expected style of output from the recorded checkpoint:

```text
聪明人可以理解世界。
```

Use `--interactive` for a simple prompt, or repeat `--text` and add `--json`
for machine-readable output.

## Evidence

The experiment design, full metrics, falsification gates, and CLI examples are
under:

```text
ara/s3-generation/logic/stone1_fixed_root_noise_repair.md
ara/s3-generation/evidence/s3_stone1_fixed_root_noise_repair/
```

## License

SameTime source code and this release bundle are distributed under GPL-3.0.
There is no warranty. The WMT-derived training corpus is not included.
