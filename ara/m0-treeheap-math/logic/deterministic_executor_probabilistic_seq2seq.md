# Deterministic Executor and Probabilistic Seq2Seq Boundary

## Claim

`M0-DETSEQ-C01`:

> TreeHeap algebra is deterministic conditional on `(H, program)`.  A
> seq2seq system conditioned on a fixed TreeHeap state is deterministic under
> greedy collapse; response diversity appears only when the outer probability
> distribution is sampled or when the persistent TreeHeap state changes.

This claim locates probability in the architecture.  It does not claim that a
language encoder has already learned a useful TreeHeap code.

## Layers

\[
z^*=\arg\max_z P_\theta(z\mid x,H)
\]

\[
H'=O_{z^*}(H)
\]

\[
y_t=\arg\max_w P_\phi(w\mid H',y_{<t})
\]

`P_theta` and `P_phi` may represent uncertainty.  The native executor `O` is a
function, not a sampler.  Once `z` is fixed, recursive `mirror`, `plus`, and
address traversal must return the same state on every execution.

## Tests

1. Generate random legal TreeHeaps and valid native programs. Execute every
   `(H,z)` twice and require bit-exact equality.
2. Sample programs from a probability bucket. For each sampled program, rerun
   the executor and require the conditional result to remain exact.
3. Hold prompt, parameters, and `H` fixed. Run greedy autoregressive decoding
   repeatedly and require one unique sequence.
4. Hold prompt, parameters, and `H` fixed but enable categorical sampling;
   require more than one sequence and low KL divergence between empirical and
   declared first-token distributions.
5. Change `H` with one deterministic native operator, keep the prompt fixed,
   and require a different greedy sequence; repeated decoding within each
   state must still be exact.

## Gates

```text
executor_repeat_exact             = 1.0
sampled_program_conditional_exact = 1.0
fixed_state_greedy_unique         = 1
sampled_unique_sequences          > 1
sampled_first_token_KL            < 0.02
state_A_greedy_unique             = 1
state_B_greedy_unique             = 1
state_A_output != state_B_output
```

## Falsification

The deterministic boundary is falsified if a fixed `(H,z)` produces different
states, if greedy decoding changes while all inputs and state are fixed, or if
native recursive execution consumes random numbers.  The probability boundary
is falsified if sampling cannot produce diversity from a non-degenerate output
distribution.

Passing this proof establishes an execution contract only.  It does not prove
unsupervised structure induction, TreeHeap semantic compression, useful
conversation, or encoder/decoder superiority.

## Result (2026-07-13)

All registered execution-boundary gates passed on io.

| Test | Result |
|---|---:|
| Native operator repeat exact, 10,000 trials | 1.0000 |
| Native inverse restore exact | 1.0000 |
| Repeat/inverse exact at each recursive depth 1..5 | 1.0000 |
| Conditional exact after sampling one program | 1.0000 |
| Unique greedy outputs, fixed state, 1,000 repeats | 1 |
| Unique sampled sequences, fixed state | 3 |
| Empirical vs declared first-token KL | 0.000275 |

With the same prompt and parameters, state A (`root=1`) greedily decoded
`earth is round`; after one deterministic native plus changed state B to
`root=2`, greedy decoding returned `memory has changed`.  Each state still had
exactly one greedy output across 1,000 repeats.

Interpretation: probability is needed for program/response uncertainty, but it
does not enter native recursive execution.  Changing persistent state can
change a deterministic response without making the decoder random.

Scope remains narrow: the response decoder is a controlled mechanism probe,
not a trained language model, and the state change is supplied rather than
learned by an encoder.
