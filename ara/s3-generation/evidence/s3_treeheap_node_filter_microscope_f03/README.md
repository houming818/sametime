# S3 TreeHeap Node Filter Microscope F03

Claim: `S3-TREEHEAP-NODE-FILTER-MICROSCOPE-F03`

- preregistration: `ara/s3-generation/logic/treeheap_node_filter_microscope_f03.zh.md`
- specimen: `ara/s3-generation/data/treeheap_filter_specimens_f03.jsonl`
- implementation: `ara/s3-generation/src/s3_treeheap_node_filter_microscope_f03.py`
- remote host: `io.grepcode.cn`, RTX 3090, 270 W limit
- smoke task: `373`
- formal task: `376`

The formal run contains 775 coarse observations and 408 deterministic fine-grid observations. Direct
decoded text is the primary evidence. `summary.json` contains the address atlas and gate decisions;
`coarse_results.jsonl` and `fine_results.jsonl` retain every observed output. P0--P4 passed. The result
supports per-node filter causality, not correct translation or a default filter configuration.
