# M1 Multiscale Probability-Field Microscope

Owner: Nio / Trinity

Created: 2026-10-11

## Question

T0 produced a 16-leaf probability field and T1 showed that the field can support
held-out lexical posterior updates. M1 asks whether the field contains measurable
structure across depth beyond the identity that every parent equals the sum of its
children.

The microscope has two coupled parts:

- M1a observes the static T0 token-type field over tree leaves.
- M1b observes the dynamic T1 token posterior created for a masked occurrence.

M1 is an observation experiment. It does not modify FOLD, train READ, or claim
that an internal node is a linguistic concept.

## Coordinates

For internal node `n` with left and right mass `m_L`, `m_R`, define

```math
m_n=m_L+m_R,
\qquad
a_n=\frac{m_L}{m_n},
\qquad
d_n=2a_n-1,
\qquad
\theta_n=\log\frac{m_L+\epsilon}{m_R+\epsilon}.
```

The child masses are exactly recoverable:

```math
m_L=m_n a_n,
\qquad
m_R=m_n(1-a_n).
```

For `L` leaves, the `L-1` internal split probabilities are an alternative
coordinate system for the interior of the `L-1` dimensional simplex.

## Measurements

1. Mass conservation at every depth.
2. Entropy-chain decomposition by depth.
3. Held-out cross-entropy decomposition by depth.
4. Prior-weighted Haar detail energy by depth.
5. Reconstruction after retaining the largest `k in {1,2,4,8,15}` split details
   per token and replacing omitted splits by `0.5`.
6. Singular-value spectrum of centered internal-node logits.
7. Held-out-token linear and quadratic prediction of each child logit from its
   parent logit and parent mass.

For T1, let `q_K(t | c)` be the token posterior at scale `K`. Moving from `K`
parent bins to `2K` child bins contributes the local log-evidence residual

```math
\Delta_{K\to2K}(t;c_{1:m})
=\sum_{r=1}^{m}
\log\frac{p_{2K}(b^{2K}_r\mid t)}
{p_K(b^K_r\mid t)}.
```

The fine posterior must satisfy

```math
\log q_{2K}(t\mid c)
=\log q_K(t\mid c)+\Delta_{K\to2K}(t;c)-\log Z.
```

M1b measures numerical closure of this recurrence, KL/JS movement between
successive posterior scales, entropy change, gold-token log-probability change,
and the singular spectrum of scale-residual vectors across held-out events.

## Controls

- learned T0-style balanced context tree;
- three matched balanced random trees;
- three balanced permutations of learned leaf addresses.

All controls use the same corpus counts, vocabulary, smoothing, target rows and
leaf capacity.

## Registered Gates

`S1-MS-MICROSCOPE-M1-I01` (integrity): for every arm, maximum mass-conservation
error, entropy-chain residual and held-out cross-entropy-chain residual are at
most `1e-8`. Full 15-detail reconstruction has maximum absolute error at most
`1e-8`.

`S1-MS-MICROSCOPE-M1-I02` (posterior refinement integrity): for every registered
T1 scale transition and context count, direct fine posterior and recursively
refined coarse posterior agree within `2e-5` maximum absolute log-probability
error.

`S1-MS-MICROSCOPE-M1-C01` (compressibility flag): at `k=4`, the learned tree's
prior-weighted mean JS reconstruction error is at least 5% below both the mean
random-tree error and mean learned-leaf-permutation error.

`S1-MS-MICROSCOPE-M1-C02` (recurrence flag): the learned tree's quadratic
parent-to-child held-out-token R-squared is at least `0.10` and exceeds both
control means by at least `0.02`.

C01 and C02 are discovery flags, not run-stopping gates. A failure is an
informative map of where the proposed regularity is absent.

M1b scale movement and residual-spectrum measurements are observational in this
first run. No direction is promoted to a quality gate after seeing the result.

## Smoke-Informed Formal Gate

Task `653` is retained as exploratory smoke. At four contexts, the learned tree
placed `0.281967 / 0.421811 = 0.668467` of its total `K=1 -> 16` mean gold
log-probability gain before or at `K=4`. The three random controls averaged about
`0.25`; learned-leaf permutations averaged about `0.44`.

Before opening the 200K/50K formal result, register:

`S1-MS-MICROSCOPE-M1-C03` (coarse information front-loading): at four contexts,
the learned tree must have positive total `K=1 -> 16` gold log-probability gain
of at least `0.10`; its fraction achieved through `K=4` must exceed the random
mean by at least `0.15` and the learned-leaf-permutation mean by at least `0.10`.

This gate was designed from smoke and is only confirmatory for a new formal data
budget. It must never be described as preregistered before task `653`.

## Result

Tasks `653` (20K smoke) and `654` (200K formal) completed. All numerical
integrity gates passed, including exact T0 mass/entropy/cross-entropy
decomposition and T1 posterior-refinement closure. Formal maximum absolute
log-probability closure error was `1.144409e-05`, below the registered `2e-5`
limit.

The smoke-informed C03 gate passed on 50,000 formal T1 events. With four context
tokens, the learned topology had total mean gold log-probability gain `0.522812`
from `K=1` to `K=16`. The fraction achieved through `K=4` was `0.607451`, versus
random-control mean `0.263590` and learned-leaf-permutation mean `0.378829`.
The learned margins were therefore `0.343861` and `0.228622`, both above the
registered thresholds.

This supports a narrow topology statement: useful T1 discriminative evidence is
front-loaded into coarse learned scales. It does not support a simple monotone
coefficient-decay law. Learned detail energy by depth was
`0.321586/0.158577/0.312618/0.299835`, and held-out cross-entropy contribution
was `0.518268/0.608847/0.522474/0.531875`.

C01 failed. At four retained details, learned prior-weighted JS reconstruction
error was `0.028009`, random mean was `0.008186`, and permutation mean was
`0.040368`. Random mixing creates a smoother field that is easier to approximate;
therefore low reconstruction error alone is not evidence of a better topology.

C02 also failed. Learned quadratic parent-to-child held-out-token R-squared was
`0.124078`, above random mean `0.038021` but below permutation mean `0.216238`.
A single low-order parent-to-child polynomial is not accepted as the TreeHeap F
law.

The generated formal `posterior_summary.json` retained C03 measurements and the
pass flag under `formal_discovery_gate`, while its top-level `claim` list named
only I02. This metadata omission is preserved in the hashed evidence and fixed in
source for future runs; the preregistration above predates task `654`.

Evidence:
`../evidence/s1_multiscale_probability_field_microscope_m1/`.

## Interpretation Boundary

The exact T0 pyramid, chain rules and T1 posterior-refinement recurrence are
algebraic consequences of the tree and do not count as discovered language
structure. Evidence for additional structure requires learned-vs-control
differences in compressibility, depth spectra, held-out recurrence or posterior
residual spectra. Even a pass does not establish sentence meaning, generation,
scale invariance, or a final TreeHeap F function.


