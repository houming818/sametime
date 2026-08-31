# TreeHeap Observer C01: Evidence Mining on `ni`

Status: preregistered engineering probe

## Purpose

TreeHeap experiments already produce `summary.json`, `wake_report.json`,
`trace.jsonl`, and fixed Dreams.  These artifacts are individually useful but
are difficult to compare across a long research history.  C01 asks whether a
CPU-only observer can turn the existing evidence into a queryable catalogue
without loading a checkpoint or consuming GPU time.

This is an observation tool.  It does not train TreeHeap and it cannot upgrade
a scientific claim by itself.

## Input contract

The input is a metadata-only copy of `ara/`.  The copy may contain Markdown,
JSON, JSONL, and Dreams text files.  Model checkpoints, tensors, datasets,
archives, and other binary artifacts are excluded.

Every source file is identified by its relative path, byte length, modification
time, and SHA-256 digest.  Malformed inputs remain visible as parse findings;
they are not silently discarded.

## Outputs

The observer writes:

- `observer.sqlite3`: source files, normalized metrics, generations, findings;
- `summary.json`: machine-readable inventory and aggregate statistics;
- `report.md`: a compact human-readable audit;
- `run.jsonl`: one record for every scan cycle;
- `latest.json`: atomic heartbeat for remote inspection.

The database stores evidence paths and JSON paths so every extracted value can
be traced back to its source.

## Measurements

### Evidence coverage

- files discovered and parsed;
- JSON/JSONL records parsed or rejected;
- metric and generation records extracted;
- distinct evidence families represented.

### Generation diagnostics

For every discovered `source`/`generation` or `SOURCE`/`DREAM` pair:

- output length;
- character diversity;
- adjacent token/character repetition proxy;
- longest repeated substring proxy;
- exact generation reused for different sources in the same artifact.

These are diagnostic signals, not semantic quality scores.

### Numerical consistency

- non-finite values;
- `PPL` inconsistent with `exp(NLL)` when both are in the same object;
- explicit repetition rates above the preregistered warning threshold `0.20`;
- duplicate-generation groups spanning multiple distinct sources.

## Prediction

1. The metadata-only corpus can be indexed on the current `ni` host with at
   most two CPU workers and less than 4 GiB resident memory.
2. The observer will recover both numerical metrics and human-readable
   generations from multiple historical evidence formats.
3. At least one historical run will contain a measurable repetition or
   source-insensitivity warning already visible to a human reviewer.

## Falsification and stop rules

C01 fails operationally if any of the following occurs:

- source evidence is modified by the observer;
- peak RSS exceeds 4 GiB;
- sustained host load exceeds 3.0 for three consecutive 60-second checks;
- more than 5% of JSON/JSONL records fail parsing for reasons other than a
  file being concurrently written;
- a reported finding cannot be traced to a source path and JSON/text location.

The process must be stopped on repeated I/O errors, rapidly growing memory,
or a control-plane service regression on `ni`.

## Interpretation boundary

A warning says that an artifact deserves review.  It does not prove that the
model architecture is wrong, that a checkpoint has collapsed globally, or
that one experiment is better than another.  Semantic conclusions remain a
human and task-specific evaluation step.
