# S1 Mask Kernel Structure Extraction Probe

Claim: `S1-MASK-KERNEL-C01`
Predict: `P-S1-MASK-KERNEL01`
Host: `io.grepcode.cn`
Device: `cuda`

## Result

decision: `weak positive / baseline-contested`
pilot_pass: `True`

```text
treeheap heldout MRR/top5/purity/entropy = 0.2833 / 1.0000 / 0.8667 / 0.6998
pair     heldout MRR/top5/purity/entropy = 0.1200 / 0.0000 / 0.4000 / 2.0564
shuffled heldout MRR/top5/purity/entropy = 0.2532 / 0.6667 / 0.4667 / 1.5737
bow      heldout MRR/top5/purity/entropy = 0.2833 / 1.0000 / 0.8667 / 0.6933
flat     heldout MRR/top5/purity/entropy = 0.1976 / 0.5000 / 0.6000 / 0.6920
```

## Boundary

This is a controlled masked-corpus proof. Gold bucket classes are used only for
audit metrics, not for training. This is not WMT translation or natural-language
understanding.
