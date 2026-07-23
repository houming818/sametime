# STONE-1 C08: Fixed-Root Noise Repair

Date: 2026-07-23
Status: preregistered single-seed mechanism experiment
Milestone: `STONE-1` (still incomplete)
Claim: `S3-STONE1-FIXED-ROOT-NOISE-REPAIR-C08`
Predict: `P-S3-STONE1-FIXED-ROOT-NOISE-REPAIR-08`

## Question

A 64-leaf TreeHeap does not have to move its root when the source string is
shorter than 64 tokens. The unused leaves can remain part of the fixed
coordinate system. C08 asks whether a frozen TreeHeap encoder plus a learned
decoder can distinguish three cases:

```text
EMPTY/masked tail       structural zero; must have exactly no effect
repeated EOS tail       regular, content-visible noise
random-token tail       irregular, content-visible noise
```

The experiment does not call padding a semantic fact. It tests whether repeated
EOS is easier for the read protocol to suppress than irregular token noise.

## Fixed Coordinate Contract

All arms use:

```text
heap_width = 64
physical root = unchanged
source payload = original source + tail to leaf 63
encoder = frozen C04 checkpoint
decoder route = C06 two-percent depth floor
```

No logical-root movement, subtree rebuilding, or protocol migration is used.

## Conditions

```text
clean_mask
  tail contains PAD and remains masked

masked_random
  tail bytes contain deterministic random tokens but remain masked
  expected to be numerically identical to clean_mask

eos_tail
  every unused leaf contains EOS and is marked valid

random_tail
  every unused leaf contains a deterministic random token and is marked valid
```

Three independently initialized decoder arms train on `clean_mask`,
`eos_tail`, and `random_tail`. Every final decoder is evaluated against all
four conditions. Training data, batches, optimizer, C04 initialization, fixed
root, and update budget are otherwise identical.

## Registered Prediction

The primary arm is `eos_tail`.

```text
G1 initial clean_mask and masked_random NLL differ by at most 1e-7
G2 eos_tail matched validation NLL improves at least 0.30 from initialization
G3 final eos_tail matched NLL is at least 0.10 below random_tail matched NLL
G4 eos-trained clean NLL is no more than 0.15 above clean-trained clean NLL
G5 unmasked EOS changes the frozen root representation measurably:
   mean root cosine(clean, eos) < 0.995
G6 every encoder checksum is unchanged and gradients remain finite
```

Passing G1 is an implementation control, not evidence for learning. Passing
G2/G4 supports a decoder-side regular-noise suppression protocol. Passing G3
supports the stronger claim that regular EOS noise is easier to absorb than
irregular noise. Failure is also informative: it means a fixed TreeHeap cannot
obtain this repair merely by decoder learning over the frozen C04 state.

## Boundary

C08 does not prove that EOS is the correct universal filler, that private
protocols repair themselves, or that persistent TreeHeap memories survive a
protocol-version change. It tests one fixed-root read-time repair mechanism on
one seed.

Planned evidence:

```text
../evidence/s3_stone1_fixed_root_noise_repair_smoke/
../evidence/s3_stone1_fixed_root_noise_repair/
```
