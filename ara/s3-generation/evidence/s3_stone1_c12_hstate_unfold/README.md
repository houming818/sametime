# C12 H_state UNFOLD smoke evidence

- Host: `io`
- Task: `56`
- Exit code: `0`
- Updates: `500` per arm
- Seed: `75032`
- Result: mechanism feasible; strong decoder claim rejected

`summary.json` contains configuration, learning traces, causal interventions,
generation metrics, and samples for both the algebraic UNFOLD decoder and the
token-stream GRU comparison arm.

The decisive evidence is not merely the worse UNFOLD NLL (`7.8480` versus
`7.4115`). Source shuffle, empty source, sibling-address swap, and per-depth
detail ablations changed UNFOLD NLL by at most about `0.0023`. Free generation
also collapsed to one repeated punctuation token. Thus the seven-level graph
ran and carried non-zero tensors, but the learned output did not causally use
the intended source/address/detail protocol.
