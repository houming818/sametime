#!/usr/bin/env python3
"""Build a traceable CPU-only catalogue of TreeHeap ARA evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


METRIC_WORDS = (
    "nll", "ppl", "perplex", "bleu", "repeat", "entropy", "variance",
    "cosine", "overlap", "coverage", "accuracy", "loss", "grad_norm",
    "nonempty", "margin", "exact", "retrieval", "js", "kl",
)
TEXT_SUFFIXES = {".json", ".jsonl", ".txt"}
SKIP_NAMES = {"observer.sqlite3", "latest.json", "summary.json"}
DREAM_HEADER = re.compile(r"^\[(\d+)\]\s+(\S+)\s*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def connect_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            format TEXT NOT NULL,
            records INTEGER NOT NULL,
            parse_errors INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metrics (
            file_path TEXT NOT NULL,
            record_no INTEGER NOT NULL,
            json_path TEXT NOT NULL,
            name TEXT NOT NULL,
            value REAL NOT NULL,
            step INTEGER,
            stage TEXT,
            PRIMARY KEY (file_path, record_no, json_path)
        );
        CREATE TABLE IF NOT EXISTS generations (
            file_path TEXT NOT NULL,
            record_no INTEGER NOT NULL,
            json_path TEXT NOT NULL,
            direction TEXT,
            source TEXT NOT NULL,
            reference TEXT,
            generation TEXT NOT NULL,
            step INTEGER,
            output_chars INTEGER NOT NULL,
            char_diversity REAL NOT NULL,
            adjacent_repeat_rate REAL NOT NULL,
            PRIMARY KEY (file_path, record_no, json_path)
        );
        CREATE TABLE IF NOT EXISTS findings (
            severity TEXT NOT NULL,
            code TEXT NOT NULL,
            file_path TEXT NOT NULL,
            location TEXT NOT NULL,
            detail TEXT NOT NULL,
            PRIMARY KEY (code, file_path, location, detail)
        );
        """
    )
    return connection


def json_location(parts: Iterable[str]) -> str:
    return "$" + "".join(f"[{part}]" if part.isdigit() else f".{part}" for part in parts)


def context_value(stack: list[dict[str, Any]], key: str) -> Any:
    for item in reversed(stack):
        if key in item:
            return item[key]
    return None


def repeat_rate(text: str) -> float:
    compact = [character for character in text if not character.isspace()]
    if len(compact) < 2:
        return 0.0
    return sum(a == b for a, b in zip(compact, compact[1:])) / (len(compact) - 1)


def insert_generation(
    connection: sqlite3.Connection,
    file_path: str,
    record_no: int,
    location: str,
    payload: dict[str, Any],
    stack: list[dict[str, Any]],
) -> None:
    source = payload.get("source")
    generation = payload.get("generation", payload.get("dream"))
    if not isinstance(source, str) or not isinstance(generation, str):
        return
    reference = payload.get("reference")
    direction = payload.get("direction", payload.get("kind", context_value(stack, "direction")))
    step = payload.get("step", context_value(stack, "step"))
    compact = [character for character in generation if not character.isspace()]
    diversity = len(set(compact)) / len(compact) if compact else 0.0
    connection.execute(
        """INSERT OR REPLACE INTO generations
        (file_path, record_no, json_path, direction, source, reference,
         generation, step, output_chars, char_diversity, adjacent_repeat_rate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            file_path, record_no, location,
            str(direction) if direction is not None else None,
            source, reference if isinstance(reference, str) else None,
            generation, int(step) if isinstance(step, (int, float)) else None,
            len(generation), diversity, repeat_rate(generation),
        ),
    )


def walk_json(
    connection: sqlite3.Connection,
    file_path: str,
    record_no: int,
    value: Any,
    parts: list[str] | None = None,
    stack: list[dict[str, Any]] | None = None,
) -> None:
    parts = parts or []
    stack = stack or []
    if isinstance(value, dict):
        current_stack = stack + [value]
        insert_generation(
            connection, file_path, record_no, json_location(parts), value, current_stack
        )
        nll = value.get("nll")
        ppl = value.get("ppl")
        if isinstance(nll, (int, float)) and isinstance(ppl, (int, float)) and nll < 20:
            expected = math.exp(float(nll))
            relative = abs(float(ppl) - expected) / max(expected, 1e-12)
            if relative > 0.02:
                connection.execute(
                    "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
                    ("warning", "NLL_PPL_MISMATCH", file_path, json_location(parts),
                     f"nll={nll}, ppl={ppl}, exp(nll)={expected:.6g}, relative_error={relative:.3g}"),
                )
        for key, child in value.items():
            walk_json(connection, file_path, record_no, child, parts + [str(key)], current_stack)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            walk_json(connection, file_path, record_no, child, parts + [str(index)], stack)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        name = parts[-1].lower() if parts else "value"
        if not math.isfinite(float(value)):
            connection.execute(
                "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
                ("error", "NONFINITE", file_path, json_location(parts), repr(value)),
            )
            return
        if any(word in name for word in METRIC_WORDS):
            step = context_value(stack, "step")
            stage = context_value(stack, "stage")
            connection.execute(
                """INSERT OR REPLACE INTO metrics
                (file_path, record_no, json_path, name, value, step, stage)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_path, record_no, json_location(parts), name, float(value),
                    int(step) if isinstance(step, (int, float)) else None,
                    str(stage) if stage is not None else None,
                ),
            )


def parse_dream_text(connection: sqlite3.Connection, relative: str, text: str) -> int:
    step_match = re.search(r"TreeHeap dreams at step\s+(\d+)", text)
    step = int(step_match.group(1)) if step_match else None
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        header = DREAM_HEADER.match(line.strip())
        if header:
            if current:
                rows.append(current)
            current = {"record": int(header.group(1)), "direction": header.group(2)}
            continue
        if current is None:
            continue
        for prefix, key in (("SOURCE:", "source"), ("REFERENCE:", "reference"), ("DREAM:", "generation")):
            if line.startswith(prefix):
                current[key] = line[len(prefix):].strip()
                break
    if current:
        rows.append(current)
    for row in rows:
        row["step"] = step
        insert_generation(
            connection, relative, row["record"], f"$[{row['record']}]", row, [row]
        )
    return len(rows)


def parse_file(connection: sqlite3.Connection, root: Path, path: Path) -> tuple[int, int]:
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    digest = sha256_file(path)
    cached = connection.execute(
        "SELECT mtime_ns, bytes, sha256, records, parse_errors FROM files WHERE path=?",
        (relative,),
    ).fetchone()
    if cached and cached[:3] == (stat.st_mtime_ns, stat.st_size, digest):
        return int(cached[3]), int(cached[4])

    connection.execute("DELETE FROM metrics WHERE file_path=?", (relative,))
    connection.execute("DELETE FROM generations WHERE file_path=?", (relative,))
    connection.execute("DELETE FROM findings WHERE file_path=?", (relative,))
    records = 0
    errors = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            payload = json.loads(text)
            records = 1
            walk_json(connection, relative, records, payload)
        elif path.suffix == ".jsonl":
            for line_no, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    errors += 1
                    connection.execute(
                        "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
                        ("warning", "JSONL_PARSE", relative, f"line:{line_no}", str(error)),
                    )
                    continue
                records += 1
                walk_json(connection, relative, line_no, payload)
        else:
            records = parse_dream_text(connection, relative, text)
    except (OSError, json.JSONDecodeError) as error:
        errors += 1
        connection.execute(
            "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
            ("error", "FILE_PARSE", relative, "$", str(error)),
        )
    connection.execute(
        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?, ?)",
        (relative, stat.st_size, stat.st_mtime_ns, digest, path.suffix[1:], records, errors),
    )
    return records, errors


def source_insensitivity_findings(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM findings WHERE code='GENERATION_REUSE'")
    groups = connection.execute(
        """SELECT file_path, generation, COUNT(DISTINCT source) AS sources
        FROM generations
        WHERE length(trim(generation)) >= 12
        GROUP BY file_path, generation
        HAVING sources >= 2"""
    ).fetchall()
    for file_path, generation, sources in groups:
        detail = f"same generation reused for {sources} distinct sources: {generation[:160]}"
        connection.execute(
            "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
            ("warning", "GENERATION_REUSE", file_path, "$", detail),
        )
    connection.execute("DELETE FROM findings WHERE code='HIGH_ADJACENT_REPETITION'")
    rows = connection.execute(
        """SELECT file_path, record_no, json_path, adjacent_repeat_rate, generation
        FROM generations WHERE output_chars >= 12 AND adjacent_repeat_rate >= 0.20"""
    ).fetchall()
    for file_path, record_no, location, rate, generation in rows:
        connection.execute(
            "INSERT OR IGNORE INTO findings VALUES (?, ?, ?, ?, ?)",
            ("warning", "HIGH_ADJACENT_REPETITION", file_path,
             f"{location}/record:{record_no}", f"rate={rate:.3f}: {generation[:160]}"),
        )


def scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def build_summary(connection: sqlite3.Connection, started: float, root: Path) -> dict[str, Any]:
    metric_counts = dict(connection.execute(
        "SELECT name, COUNT(*) FROM metrics GROUP BY name ORDER BY COUNT(*) DESC LIMIT 30"
    ).fetchall())
    finding_counts = dict(connection.execute(
        "SELECT code, COUNT(*) FROM findings GROUP BY code ORDER BY code"
    ).fetchall())
    generation_rates = [row[0] for row in connection.execute(
        "SELECT adjacent_repeat_rate FROM generations"
    ).fetchall()]
    return {
        "observer": "TREEHEAP-OBSERVER-EVIDENCE-MINING-C01",
        "generated_at": utc_now(),
        "input_root": str(root.resolve()),
        "elapsed_seconds": time.monotonic() - started,
        "files": scalar(connection, "SELECT COUNT(*) FROM files"),
        "source_bytes": scalar(connection, "SELECT COALESCE(SUM(bytes),0) FROM files"),
        "parsed_records": scalar(connection, "SELECT COALESCE(SUM(records),0) FROM files"),
        "parse_errors": scalar(connection, "SELECT COALESCE(SUM(parse_errors),0) FROM files"),
        "metrics": scalar(connection, "SELECT COUNT(*) FROM metrics"),
        "generations": scalar(connection, "SELECT COUNT(*) FROM generations"),
        "findings": scalar(connection, "SELECT COUNT(*) FROM findings"),
        "metric_counts": metric_counts,
        "finding_counts": finding_counts,
        "generation_adjacent_repeat": {
            "mean": statistics.fmean(generation_rates) if generation_rates else None,
            "max": max(generation_rates) if generation_rates else None,
        },
    }


def write_report(connection: sqlite3.Connection, summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# TreeHeap Observer C01 Report",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Inventory",
        "",
        f"- Source files: {summary['files']}",
        f"- Source bytes: {summary['source_bytes']}",
        f"- Parsed records: {summary['parsed_records']}",
        f"- Parse errors: {summary['parse_errors']}",
        f"- Metrics: {summary['metrics']}",
        f"- Generations: {summary['generations']}",
        f"- Findings: {summary['findings']}",
        "",
        "## Metric Catalogue",
        "",
        "| Metric | Records |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | {count} |" for name, count in summary["metric_counts"].items())
    lines.extend(["", "## Findings", "", "| Severity | Code | Source | Detail |", "|---|---|---|---|"])
    for severity, code, file_path, location, detail in connection.execute(
        """SELECT severity, code, file_path, location, detail FROM findings
        ORDER BY CASE severity WHEN 'error' THEN 0 ELSE 1 END, code, file_path LIMIT 200"""
    ):
        safe_detail = detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {severity} | `{code}` | `{file_path}:{location}` | {safe_detail} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def discover(root: Path, output: Path) -> list[Path]:
    output_resolved = output.resolve()
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name in SKIP_NAMES and output_resolved in path.resolve().parents:
            continue
        paths.append(path)
    return sorted(paths)


def scan(root: Path, output: Path) -> dict[str, Any]:
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    connection = connect_database(output / "observer.sqlite3")
    paths = discover(root, output)
    for index, path in enumerate(paths, 1):
        parse_file(connection, root, path)
        if index % 250 == 0:
            connection.commit()
    source_insensitivity_findings(connection)
    connection.commit()
    summary = build_summary(connection, started, root)
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "latest.json", summary)
    write_report(connection, summary, output / "report.md")
    connection.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=0, help="seconds between scans; 0 runs once")
    parser.add_argument("--cycles", type=int, default=1, help="0 means unlimited")
    args = parser.parse_args()
    cycle = 0
    run_log = args.output_dir / "run.jsonl"
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        summary = scan(args.input_root, args.output_dir)
        summary["cycle"] = cycle
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        if args.interval <= 0 or (args.cycles and cycle >= args.cycles):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
