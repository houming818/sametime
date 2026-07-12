# P0: World-Observation Pretraining

`Theta_global` is the persistent parameter system.  `H_document` is the
temporary TreeHeap written for one document.  P0 trains both through ordinary
next-span prediction, but only `Theta_global` survives from one document to the
next.

```text
Chinese document
  -> first 64 BPE tokens -> H_document
  -> decoder predicts following 32 BPE tokens
  -> CE gradient updates Theta_global
```

P0 corpus mix is deliberately restricted:

```text
news2016zh  50%  factual events and relations
wiki_zh     30%  encyclopedic definitions and facts
webtext     20%  varied natural wording
```

BELLE, Baike QA, medical dialogues, translation pairs, and Zhihu Q&A are not
used in P0.  They are later adaptation/evaluation data, not evidence that a
general observation model formed its own structure.

The objective is one loss:

\[
L_{P0}=CE(P_\theta(x_{t+1:t+32}\mid x_{t-63:t}),x_{t+1:t+32})
\]

The initial smoke checks only data integrity, tokenizer coverage, CUDA safety,
loss decrease, held-out NLL, and readable continuation samples.  It does not
claim a complete world model, semantic compression, or QA.
