# S1 algebraic readout probe

SPR-034 tests internal-node algebraic readout targets over WMT short BPE sequences.

Decision: `S1-READ-C02 -> supported / mixed pilot`.

Scope:

- Supported: routed internal-node readout is much stronger than root bottleneck
  on natural algebraic targets: `length`, `first`, `last`, `prefix0`,
  `prefix1`.
- Still open: reducing `residue_buckets` from 64 to 16 improves residue
  accuracy, but does not solve it.
