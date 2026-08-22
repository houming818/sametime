# Qwen3-8B local judge smoke evidence

Final taskd run: `273` (`qwen3-8b-local-judge-smoke-v3`)

The official `Qwen/Qwen3-8B` checkpoint was downloaded to
`/home/nio/models/Qwen3-8B` through `hf-mirror` and loaded in BF16 on io's
RTX 3090. `model_manifest.json` records every local model file size and
SHA-256. `judgments.jsonl` contains the immutable source text, original BGE
score, raw Qwen output and parsed JSON for all 60 records.

Execution trace:

- `268`: direct Hugging Face download failed on network connectivity.
- `269`: mirror download completed all 15 files.
- `270`: stopped before inference because host Jinja2 was too old.
- `271`: user-level Jinja upgrade failed because unrelated system package
  metadata was unreadable; the global Python environment was not modified.
- `272`: generated valid JSON but was deliberately stopped after an attention
  mask warning.
- `273`: explicit attention mask, warning-free final run; all gates passed.

This evidence tests local structured-judge mechanics only. It does not certify
the judge's labels, factual correctness, medical safety or downstream training
benefit.
