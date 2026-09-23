# Real WMT Fixed-Topology Search

This evidence records a small real-text FOLD-topology screen, not a translation
benchmark and not a learned routing experiment.

`s1_real_wmt_topology_search.py` reads the English side of WMT parallel text,
forms four-token windows, and predicts token five. For each fixed candidate
topology it trains a 32-dimensional token table, local parent transforms, and
a linear decoder for 500 updates. The only intended experimental variable is
the fixed composition topology.

The best recorded candidate was `left_deep` with test NLL `3.1390259`, PPL
`23.0814`, and top-1 accuracy `0.401667`; see `summary.json`.

Interpretation boundary: this is pilot evidence that the stated sequential
objective is sensitive to composition topology. It does not demonstrate an
emergent chain route, READ, stop/left/right routing, addressability,
multiresolution storage, full sentence generation, translation quality, or
TreeHeap superiority over matched language-model baselines.

The corresponding claim-integration record is:
`../../../logic/f_function_claim_integration_f21.md`.
