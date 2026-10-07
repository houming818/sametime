# T0 Multiscale Probability-Tree Embedding Gate

Owner: Nio / Trinity  
Author: GPT-5.6 Sol  
Created: 2026-10-07  
Updated: 2026-10-07

## Question

Can a shared 16-leaf context tree turn fixed corpus co-occurrence counts into a
multiresolution token probability field that preserves held-out lexical evidence?
This is an embedding feasibility gate. It does not train FOLD/READ, generate
sentences, or claim that a root state already contains sentence meaning.

## Data Contract

- Training text: English side of the existing WMT massive corpus.
- Split: source-line modulo 10; residue 0 is sealed held-out data.
- Tokenizer: existing WMT 32K SentencePiece model.
- External ruler: SimLex-999. Its scores are loaded only after construction and
  never enter vocabulary selection, context counts, tree splitting, SGNS, or
  scale selection.
- Seed: `20261007` for the primary contract.
- Background capacity: 16 leaves, obtained by four balanced recursive splits of
  context-token probability vectors.

The smoke uses 20K lines, 2,048 targets and 1,024 contexts. If integrity gates
pass, the night run uses 200K lines, 8,192 targets and 4,096 contexts. Quality
metrics never stop the formal run.

## Construction

For context token `c`, construct the normalized target field

```math
s_c(t)=P(t\mid c).
```

At every tree node, split its context tokens by the leading weighted covariance
axis and a deterministic balanced median. Repeating four times assigns every
context token to one of 16 leaves. For target token `t`, aggregate its context
counts by leaf and normalize:

```math
p_t^{(16)}(i)=
\frac{n(t,L_i)+\alpha}{\sum_j n(t,L_j)+16\alpha}.
```

Coarser fields are sibling sums, not separately trained parameters:

```math
p_t^{(K)}(j)=\sum_{i\in\mathrm{desc}(j)}p_t^{(16)}(i),
\qquad K\in\{1,2,4,8,16\}.
```

The matched random control uses the same counts, smoothing, dimensions and
leaf capacities, changing only the context-to-leaf assignment. A 64D SGNS
model is trained from the same held-in count matrix.

## Measurements

1. SimLex Spearman at every probability-field scale using negative JS distance.
2. Matched random-partition SimLex Spearman.
3. SGNS cosine SimLex Spearman and frequency-only correlation.
4. Held-out target-row retrieval using multinomial Bayesian likelihood:

```math
z_t^{(K)}(q)=\log\pi_t+
\sum_{j=1}^{K}n_q(j)\log p_t^{(K)}(j).
```

5. Leaf utilization, probability conservation, finite-value and source-hash
   audits.
6. Blind local-teacher comparisons over model nearest neighbours. Teacher
   judgments are supplemental and never become training labels.

## Registered Claims

`S1-MS-PROB-EMBED-T0-C01`: at 16 leaves, the learned partition exceeds the
matched random partition by at least `0.03` SimLex Spearman.

`S1-MS-PROB-EMBED-T0-C02`: learned 16-leaf held-out target retrieval NLL is at
least `0.02` below the matched random partition and below the frequency-only
prior.

The full `1/2/4/8/16` curves are observations. No monotonic semantic trend is
required because coarse and fine scales can expose different relations.

## Smoke Integrity Gates

- all 16 leaves are populated;
- all probability fields and metrics are finite;
- simplex error is at most `1e-6`;
- held-in and held-out context counts are non-empty;
- at least 40 SimLex pairs are covered;
- SGNS completes its registered smoke budget;
- summary, pair rows and teacher items are valid JSON/JSONL.

Only these integrity gates control progression to formal. C01/C02 are evaluated
after the complete formal budget and never terminate training early.

## Local Teacher Contract

The existing local `Qwen/Qwen3-8B` checkpoint on io may judge pairwise neighbour
quality after formal construction. It receives hidden, order-randomized A/B
choices, greedy decoding, fixed prompts and an order-swapped duplicate. It must
first be calibrated against held-out SimLex triplets. The teacher is an observer,
not ground truth; standard human labels and deterministic measurements remain
primary.

## Boundaries

A pass establishes a usable static lexical embedding gate under this corpus and
tokenization. It does not establish polysemy resolution, multilingual alignment,
sentence composition, generation, recursive memory, reasoning, or consumer-device
efficiency.

## Result

Io task `649` completed the 20K smoke, the full 200K construction, and the
supplemental Qwen3-8B audit. The smoke passed every integrity gate, so the
formal budget was released without using a quality metric as a stopping rule.

Formal material comprised 180,000 held-in and 20,000 held-out WMT English
lines, `20,660,256/2,317,889` train/test context pairs, 8,192 targets and 4,096
contexts. Every one of the 16 leaves received exactly 256 context tokens and
the largest simplex error was `4.44e-16`.

| bins | Tree SimLex rho | random rho | Tree retrieval NLL | random NLL | Tree MRR |
|---:|---:|---:|---:|---:|---:|
| 1 | degenerate | degenerate | 9.952945 | 9.952945 | 0.001171 |
| 2 | 0.049975 | 0.037087 | 8.898135 | 9.594337 | 0.011370 |
| 4 | 0.084982 | -0.059048 | 7.826886 | 9.021866 | 0.059431 |
| 8 | 0.119851 | 0.008380 | 6.909275 | 8.098312 | 0.146351 |
| 16 | 0.120264 | 0.051110 | 6.274350 | 7.241480 | 0.243256 |

The one-bin similarity score is mathematically undefined because all normalized
one-bin fields are identical; tiny floating-point ordering must not be interpreted
as semantic evidence. At 16 bins, Tree-minus-random SimLex rho was `0.069154`, so
C01 passed. Tree retrieval improved NLL by `0.967129` over random and `3.678595`
over the frequency-only prior, so C02 passed.

These are single-seed point-estimate supports. Tree SimLex rho `0.120264` remained
slightly below the frequency ruler `0.123533`, and the matched 64D SGNS run was
invalid as a positive ruler (`rho=-0.042312`). Therefore the result establishes a
useful corpus-conditional partition and multiscale retrieval signal, not a mature
general lexical-semantic embedding.

The local teacher produced valid JSON for all 120 order-paired questions, but failed
calibration and order robustness: SimLex calibration accuracy was `0.725` and
order consistency only `0.325`. Its neighbour preferences are discarded rather than
used as evidence. This is a teacher-protocol failure, not evidence against or for the
embedding.

Evidence:
`../evidence/s1_multiscale_probability_embedding_t0/formal_200k_seed20261007/`
and `../evidence/s1_multiscale_probability_embedding_t0/teacher_qwen3_8b_seed20261007/`.
