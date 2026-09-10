# F09 ancestor-chain and upper-jump microscope

- Host: `io.grepcode.cn`
- taskd: `400`
- Seed: `12001`
- Runtime: `194.2 s`
- Interventions: `852`
- Checkpoint/theta: frozen exactly
- Integrity gates: all passed

Single `推。` exposed only one active recursive parent and had no independent
upper node to tune. Repeating `推` eight times expanded the active recursive
coordinates from `1` to `15`, formed four-level ancestor chains, and produced
native hard `push`.

No scalar fine jump, upper jump, ancestor chain, off-path control, or uniform
gain produced hard `push` for either complete Sisyphus input. For the fourfold
repeated short Sisyphus sentence, however, one five-coordinate ancestor chain
raised fixed-history push coverage from `0.002546` to `0.019237`; its matched
off-path control reached only `0.002739`. Across all 96 paired conditions the
chain won 44 times, so this is localized path sensitivity rather than a general
ancestor law.

Claim status: repeated input expands the recursive observation space and local
path-sensitive coarse responses exist. Scalar radial parent gain is insufficient
for hard cross-language recovery in the complete sentence.

See `summary.json` for gates and topology, and `interventions.csv` / JSON for
every decoded condition.
