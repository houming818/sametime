# Butterfly long-range evidence

Claim: `S3-TREEHEAP-BUTTERFLY-LONGRANGE-C01`

Status: `supported_mechanism`

This directory records an algebra contract and a synthetic learned address-routing probe. It is not language evidence.

## Provenance

- Remote host: `io`
- taskd formal task: `78`
- taskd script-hash audit: `79`
- Executed script SHA-256: `35f09e55267245dbfa035186b3330ec410e843ee24ba2ff89e89211471e12aa9`
- Preregistered source commit: `c86d576`
- Formal wall time: `82.2s`

The remote checkout remained on an older experiment branch with unrelated
artifacts, so the preregistered script was copied into its canonical `src/`
path instead of switching or cleaning that worktree. `executed_script.py` is
the exact file audited on `io`. It differs from the source now at `main` only
by two post-copy provenance fields (`dense_attention_allocated` and tensor
rank) and by adding `python3` to the generated replay command; the numerical
implementation that produced the evidence is unchanged.
