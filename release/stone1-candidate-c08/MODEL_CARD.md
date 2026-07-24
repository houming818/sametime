# Model Card: STONE-1 Candidate C08

## Identity

```text
name: SameTime STONE-1 Candidate C08
architecture: fixed-capacity TreeHeap encoder + recursive GRU decoder
task: English-to-Chinese translation
status: research POC / STONE-1 Candidate
date: 2026-07-24
```

## Architecture Contract

The encoder writes the source into one fixed 64-leaf TreeHeap. Short strings do
not move the root. Unused leaves are filled with repeated EOS tokens and marked
visible. A frozen C04 encoder produces a root plus addressed lifting details.
The C08 decoder reads six visible resolutions with a fixed two-percent
probability floor at every depth.

The encoder checkpoint has about 50 million model parameters in the complete
runtime. The release separates the frozen encoder and learned EOS-tail decoder
states so the experimental boundary remains inspectable.

## Training And Evaluation

The C08 decoder was trained for 15,625 updates on one million pairs sampled
from the local WMT-massive English-Chinese collection. The frozen split hashes
are recorded in the ARA evidence directory.

Matched EOS-tail test result:

```text
NLL: 3.4517
token BLEU-4: 13.8713
non-empty generation: 1.0
severe repetition rate: 0.015
peak allocated VRAM during C08: 2.27 GiB
```

Repeated EOS was easier to learn than deterministic random-token tails.
However, the EOS-trained decoder lost 0.3256 validation NLL when switched back
to clean masked input. The result supports a fixed framing convention, not a
universal noise-repair operator.

## Intended Use

- Reproduce SameTime/ARA TreeHeap experiments.
- Inspect fixed-root, multiresolution decoder behavior.
- Run small English-to-Chinese translation demonstrations.
- Build controlled ablations and causality audits.

## Not Intended For

- Production translation.
- Medical, legal, financial, or safety-critical decisions.
- General question answering or claims of world knowledge.
- Claims that STONE-1 is complete.

## Known Limitations

- Only one formal seed has crossed the product thresholds.
- Translation can be fluent but semantically wrong.
- Names, numbers, long relations, and rare terms are unreliable.
- Maximum source width is 64 SentencePiece tokens including EOS.
- The EOS-tail convention is part of the learned protocol.
- Complete same-checkpoint structural-causality closure is pending.

## License

GPL-3.0. No warranty. Raw training data is not redistributed.
