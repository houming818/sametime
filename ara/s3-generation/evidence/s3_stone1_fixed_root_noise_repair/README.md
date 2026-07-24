# C08 Formal Evidence

Claim: `S3-STONE1-FIXED-ROOT-NOISE-REPAIR-C08`

```text
host: io
taskd task: 17
exit code: 0
elapsed: 10041.4 seconds
code commit: f06b997
data: 1,000,000 WMT pairs
updates: 15,625 per arm
peak allocated VRAM: 2.27 GiB
```

Result: five of six preregistered gates passed. Repeated EOS was substantially
easier to learn than deterministic random-token tails, but the EOS-trained
decoder did not retain the clean-input protocol within the registered limit.
The supported result is a regular fixed-frame convention, not universal noise
repair.

Primary machine-readable evidence is `summary.json`. `stdout.log` contains the
complete training trace, `config.json` records the recipe, and
`dataset_manifest.json` records the frozen split hashes. Decoder checkpoints
remain on io under the same evidence directory and are intentionally excluded
from Git.
