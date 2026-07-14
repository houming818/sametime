# Residual TreeHeap Forest Pretraining

Claims:

```text
S3-RESIDUAL-FOREST-C01          rejected as written
S3-TREEHEAP-ROOT-COMPRESS-C01  supported full-corpus / single-seed
```

The full pass processed `38,251,247` blocks from news, Wikipedia, and web
sources. The four-head residual model became non-finite at step `62,400`. The
matched four-head no-residual model remained finite:

```text
valid NLL                         6.236525
top-1 / top-5                     0.119598 / 0.232941
address-destruction delta NLL    +2.625219
head-ablation delta NLL          +1.1853/+2.4668/+1.1574/+3.4853
```

The checkpoint is retained remotely and on NAS because it is approximately
226 MB. Local evidence contains `summary.json`, `trace.jsonl`, and the complete
queue log.

See `summary.json` for the immutable metrics and
`../../../logic/residual_treeheap_forest_pretrain.md` for interpretation and
falsification boundaries.
