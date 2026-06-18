# S1 Runtime Environment

Owner: Review Engineer
Writer: Codex
Created: 2026-06-16
Updated: 2026-06-16
Purpose: Record how S1 SPR experiments are currently run.

## io Access

Use MSYS2 bash locally:

```bash
E:/opt/msys64/usr/bin/bash.exe -lc "ssh nio@io.grepcode.cn hostname"
```

Remote identity:

```text
user: nio
host: io
groups: sudo, docker, libvirt
```

Passwordless sudo works:

```bash
sudo -n true
```

## GPU

Observed 2026-06-16:

```text
NVIDIA GeForce RTX 3090
VRAM total: 24576 MiB
VRAM free: 24124 MiB
driver: 580.159.03
CUDA: 13.0
```

## Paths

Main remote code path:

```text
/data/homecicd/sametime/code/wmt
```

Dataset path:

```text
/data/datasets/wmt14
```

Note: normal `nio` execution may hit `PermissionError` on WMT14 because of parent directory traversal permissions. Use `sudo -n` for current smoke tests or fix ACL/group ownership.

## Known Scripts

- `spr_hash_cyclic.py`: order-collision smoke test.
- `spr_echo_proof.py`: decomposed routing echo proof.
- `spr_s1_eval.py`: larger sentence-level echo evaluation.
- `q.py`: experiment queue; run directly on io with `python3 q.py status`.

## Known Makefile Issue

Running `make status` inside `/data/homecicd/sametime/code/wmt` performs a second SSH to `houming818@io.grepcode.cn`, which fails under the current key/host setup. Prefer:

```bash
cd /data/homecicd/sametime/code/wmt
python3 q.py status
```
