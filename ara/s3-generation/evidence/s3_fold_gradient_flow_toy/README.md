# STONE-2 FOLD gradient-flow toy

- Task: io `295`
- Seed: `16201`
- Shape: 8 leaves x 4 dimensions
- Dtype: float64
- Result: completed

The diagnostic confirms bounded parents, exact closure and inverse-scale parent
Jacobians. An exact alternating-sign tree produced a root-to-leaf gradient norm
of `70,710,677.94` because two upper scales reached the epsilon floor. The READ
toy did not produce negative cross-depth parameter-gradient cosines, so trained
depth interference is not an algebraic necessity of the shared kernel alone.

See `summary.json` for all Jacobians, scale traces, gradient norms and cosine
matrices. This is controlled calculus evidence, not language-task evidence.
