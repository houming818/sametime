# S1 Echo Entry Gate

Created: 2026-07-01
Owner: Codex Review
Stage: S1 Echo

## Claim

`S1-ECHO-GATE-C01`

A controlled S1 echo loop can be trained end-to-end from scalar cross-entropy
loss:

```text
token ids
-> TreeHeap leaf write
-> path-conditioned read kernel
-> task-conditioned structural route selection
-> token collapse decoder
```

In this proof the structure task is deliberately small:

```text
task = echo:   [t0,t1,t2,t3] -> [t0,t1,t2,t3]
task = mirror: [t0,t1,t2,t3] -> [t3,t2,t1,t0]
```

The task flag is given. Therefore this claim is an S1 entry proof, not a
natural-language trigger proof.

## Predict

`P-S1-ECHO041`

If S1 echo can now begin, a tiny differentiable TreeHeap echo model should
learn all three pieces together:

1. token information is stored in learned leaf embeddings;
2. structural read routes collapse to identity or mirror leaf addresses;
3. decoded tokens reconstruct the requested sequence on held-out samples.

A no-task single-kernel baseline should fail because one fixed read kernel
cannot solve both identity and mirror outputs at the same time.

## Model

The model uses four ordered leaf addresses:

```text
leaf0 leaf1 leaf2 leaf3
```

It learns:

```text
E[token]                  # token write vector
route_logits[op,out,in]   # two structural read kernels
task_gate[task,op]        # probability container over operations
decoder(vector)           # token collapse
```

For each output slot:

```text
read_state[out]
  = sum_op p(op | task)
      sum_in p(in | op,out) E[token_in]
```

Then:

```text
logits[out] = decoder(read_state[out])
loss = cross_entropy(logits, target_token)
```

This is intentionally close to the TreeHeap kernel story:

```text
write -> address/path read -> probability container -> collapse
```

## Evidence

Script:

```text
ara/s1-echo/src/s1_echo_entry_gate_probe.py
```

Evidence:

```text
ara/s1-echo/evidence/s1_echo_entry_gate_probe/
```

Host:

```text
io.grepcode.cn
```

Key metrics:

```text
pilot_pass = true
treeheap_ood_token_acc = 1.000000
treeheap_ood_exact = 1.000000
treeheap_ood_route_argmax_ok = 1.000000
treeheap_ood_gate_echo_identity_prob = 0.968449
treeheap_ood_gate_mirror_mirror_prob = 0.810727
no_task_baseline_ood_exact = 0.090820
```

## Decision

Status:

```text
supported pilot
```

The gate is good enough to start S1-echo v0 because:

1. token information is learned and decoded exactly in the controlled task;
2. leaf-address routes collapse to the correct identity/mirror structure;
3. a baseline without task-conditioned structural routing fails badly.

The mirror gate is a probability container, not a fully hard collapse:

```text
mirror probability ~= 0.81
```

This is acceptable for the entry proof, but it should not be promoted to
perfect operation selection.

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

It only proves that the controlled S1 echo loop is trainable.

## Next Work

1. Replace the artificial task flag with learned triggers from token/context
   features.
2. Move from four leaves to variable-length short WMT BPE sequences.
3. Add mask/noise restoration so echo is not only copying.
4. Split operation parameters into a small forest:

```text
Theta_write
Theta_read
Theta_mirror
Phi_trigger
```

5. Ask Runner/DeepSeek to verify:

```text
Can this claim be accepted as the S1-echo v0 entry gate?
Which baseline should be added before calling it more than controlled pilot?
```
