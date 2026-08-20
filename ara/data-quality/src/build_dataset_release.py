#!/usr/bin/env python3
"""Build an immutable, content-addressed dataset release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (1 if last and last != b"\n" else 0)


def parse_pairs(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use NAME=VALUE: {value}")
        key, item = value.split("=", 1)
        if not key or key in result:
            raise ValueError(f"invalid or duplicate {label}: {key}")
        result[key] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--file", action="append", default=[], help="LOGICAL=PATH")
    parser.add_argument("--metadata", action="append", default=[], help="KEY=VALUE")
    args = parser.parse_args()

    files = parse_pairs(args.file, "file")
    metadata = parse_pairs(args.metadata, "metadata")
    if not files:
        raise ValueError("at least one --file is required")

    entries = []
    for logical_name, raw_path in sorted(files.items()):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            {
                "logical_name": logical_name,
                "source_path": str(path),
                "bytes": path.stat().st_size,
                "lines": line_count(path),
                "sha256": sha256_file(path),
            }
        )

    checksum_text = "".join(
        f"{entry['sha256']}  {entry['logical_name']}\n" for entry in entries
    )
    root_hash = hashlib.sha256(checksum_text.encode("utf-8")).hexdigest()
    release_name = f"{args.dataset_id}-{args.version}"
    manifest = {
        "schema": "nio.dataset-release.v1",
        "release_name": release_name,
        "dataset_id": args.dataset_id,
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_root_sha256": root_hash,
        "metadata": metadata,
        "files": entries,
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

    rows = "\n".join(
        f"| `{e['logical_name']}` | {e['lines']:,} | {e['bytes']:,} | `{e['sha256']}` |"
        for e in entries
    )
    metadata_rows = "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(metadata.items()))
    release_md = f"""# {release_name}

This directory is an immutable dataset release record. The data files remain
on `io`; this release stores their content identities and provenance.

## Release identity

- root SHA-256: `{root_hash}`
- schema: `nio.dataset-release.v1`

## Provenance

{metadata_rows}

## Members

| Logical file | Lines | Bytes | SHA-256 |
|---|---:|---:|---|
{rows}

Changing any member, scorer, threshold, ordering, seed, source snapshot, or
normalization rule requires a new release version. Never overwrite this record.
"""
    (output / "RELEASE.md").write_text(release_md, encoding="utf-8")
    print(json.dumps({"release_name": release_name, "root_sha256": root_hash}))


if __name__ == "__main__":
    main()
