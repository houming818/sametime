# C11 Frozen Independent Test Audit

Taskd job `175` loaded the frozen checkpoint produced by formal task `172` and
evaluated a disjoint deterministic 1,000-row WMT test split. It performed no
optimizer step and changed no model parameter.

Main result:

```text
test NLL                 5.421573
C10 NLL, same rows       5.403696
delta vs C10            +0.017877
leaf-only delta NLL     +0.440891
bypass K_up delta NLL   +0.000340
token BLEU4              3.077426
C10 BLEU4, same 32 rows  4.380892
P2 pass                  false
```

The test supports causal use of coarse parent levels through mandatory
multi-level READ, but does not support the additional bottom-up `K_up`
convolution or a product-quality improvement over C10.

Task `174` was a failed audit attempt caused by reading the nonexistent config
field `leaf_width`; task `175` corrected this by deriving the level count from
the loaded Decoder. The failed attempt did not train or modify the checkpoint.
