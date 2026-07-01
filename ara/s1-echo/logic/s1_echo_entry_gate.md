# S1 Echo Entry Gate: Corrected Inverse Canonicalization

Created: 2026-07-01
Corrected: 2026-07-01
Owner: Codex Review
Stage: S1 Echo

## Correction Note

The first SPR-041 pilot (`S1-ECHO-GATE-C01`) was misdirected.

It proved that a model can choose between an identity read kernel and a mirror
read kernel for different output tasks:

```text
task=echo   -> read identity
task=mirror -> read mirror
```

Houming818 pointed out that this is not the desired TreeHeap direction. The
desired operation is:

```text
observed mirror state
-> inverse gate / mirror inverse
-> canonical echo state
-> one shared echo decoder
```

Therefore the old evidence is retained as a negative design lesson, not as the
accepted S1 entry claim.

## Accepted Claim

`S1-ECHO-CANON-C01`

A controlled S1 echo loop can learn inverse structural canonicalization before
token collapse:

```text
observed token ids
-> TreeHeap leaf write
-> inverse gate over structural kernels
-> canonical TreeHeap echo state
-> one shared echo decoder
-> canonical token ids
```

In the controlled task:

```text
canonical = [t0,t1,t2,t3]

observed identity = [t0,t1,t2,t3]
observed mirror   = [t3,t2,t1,t0]

target = canonical
```

The transform flag is still given. This does not prove natural-language trigger
discovery.

## Predict

`P-S1-ECHO041-CORRECTED`

If the corrected S1 entry design is valid:

1. identity input should use an identity inverse kernel;
2. mirror input should use a mirror inverse kernel;
3. both paths should produce the same canonical echo state;
4. one shared echo decoder should reconstruct canonical tokens;
5. a no-inverse single-route baseline should fail.

## Model

The model learns:

```text
E[token]                        # token leaf write
inverse_route_logits[op,out,in] # structural inverse kernels
inverse_gate[transform,op]      # probability container over inverse ops
echo_decoder(vector)            # one shared canonical decoder
```

For observed leaf states:

```text
v_i = E[observed_i]
```

The canonical state is:

```text
h_j =
sum_k p(k | transform)
sum_i p(i | k,j) v_i
```

Then one shared decoder reads:

```text
logits_j = EchoDecoder(h_j)
```

The training loss is:

```text
L =
  CE(logits, canonical_tokens)
  + lambda_state * ||h - E[canonical_tokens]||^2
  + lambda_entropy * entropy(gate/routes)
```

The important addition is the canonical-state loss. Without it, token CE alone
can let the decoder compensate for a soft or non-canonical hidden state.

## Evidence

Script:

```text
ara/s1-echo/src/s1_echo_inverse_gate_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_echo_inverse_gate_probe/
```

Host:

```text
io.grepcode.cn
```

Key metrics:

```text
pilot_pass = true
canonical_echo_ood_exact = 1.000000
inverse_route_argmax_ok = 1.000000
identity_gate_identity_inverse_prob = 0.999794
mirror_gate_mirror_inverse_prob = 0.999784
canonical_state_mse = 0.000988962
no_inverse_baseline_ood_exact = 0.218750
```

## Decision

Status:

```text
S1-ECHO-GATE-C01  -> downgraded / misdirected pilot
S1-ECHO-CANON-C01 -> supported pilot
```

The corrected proof supports S1-echo v0 because it verifies the intended data
flow:

```text
mirror is not read as mirror output;
mirror is first inverted into canonical echo state;
then one decoder reads canonical tokens.
```

## What Is Not Proved

This proof does not prove:

```text
WMT translation
language semantics
unsupervised natural mirror trigger discovery
recursive-depth mirror selection
Transformer superiority
long-sequence syntax
```

The transform flag is still supervised.

## Next Work

The next proof should replace:

```text
given transform flag
```

with:

```text
learned trigger from token/context features
```

Then add:

```text
mask/noise restore
variable-length short WMT BPE
stronger flat/sequence baselines
```
