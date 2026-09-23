# S1 Probability-Residual Tree Monte Carlo A11

## Run

- Claim: `S1-F-MC-A11-C01`
- Host: `io.grepcode.cn`
- Task: `561`
- Seed: `20260924`
- Device: RTX 3090 / CUDA
- Search budget: 256 Monte Carlo proposals
- Tree: fixed complete depth 5, 32 leaves
- Input: frozen real-WMT `512 x 1024` token-context count field

## Outcome

All preregistered gates passed. Dev NLL improved by `0.0160695129`. The sealed
test NLL improved by `0.0183314717` relative to the deterministic initial tree
and by `0.0798474582` relative to random routing. All 32 leaves were occupied.

Probability FOLD conservation and root-to-leaf residual reconstruction closed
to numerical precision (`1.39e-17` and `3.47e-18` maximum absolute error).

## Evidence Boundary

This is evidence for globally scored context-field topology search only. It
does not train or evaluate next-token prediction, translation, generation,
recursive READ, semantic labels, or a decoder. One seed is insufficient for a
general architecture conclusion.

## Files

- `summary.json`: configuration, metrics, gates, and claim status
- `trace.jsonl`: all 256 proposal decisions and scores
- `command.txt`: exact invocation and input hash
- `taskd_561.log`: scheduler execution log
