# STONE-2 recursive energy-carrier smoke

- Final io task: `298`
- Seed: `16211`
- Shape: 8 leaves x 4 dimensions
- Dtype: float64
- Arms: current local normalization, recursive energy carrier, naive geometric product

The energy carrier reduced the exact alternating-sign root-to-leaf gradient
norm from `70,710,677.94` to `0.35355`, while retaining exact numerical closure
and exact root-to-leaf multiplicative energy reconstruction. The naive geometric
product produced parent norms of `70.72` for imbalanced inputs and `7,071.07`
for one-sided-zero inputs.

The carrier does not remove the global inverse-scale Jacobian of normalized
directions. This is a controlled calculus smoke, not language-task evidence.
