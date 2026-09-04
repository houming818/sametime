#!/usr/bin/env python3
"""CPU tests for resumable sequential repeat-corpus iteration."""
from __future__ import annotations

import tempfile
from pathlib import Path

import s3_epoch_repeat_scaling as repeat


class FakeSentencePiece:
    def encode(self, text, out_type=int):
        return [1 + (ord(char) % 13) for char in text]


def collect(path, start_line=0, start_byte=0, flip=0):
    return list(repeat.iter_sequential_batches(
        path, FakeSentencePiece(), {"en2zh": 101, "zh2en": 102}, 2,
        batch_size=2, start_line=start_line, start_byte=start_byte,
        end_line=6, excluded=set(), direction_flip=flip,
    ))


def test_resume_has_no_duplicate_or_gap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pairs.tsv"
        path.write_text("".join(f"中文{i}\tenglish {i}\n" for i in range(6)), encoding="utf-8")
        full = collect(path)
        first_cursor, first_byte, first_batch, _ = full[0]
        resumed = collect(path, first_cursor, first_byte)
        full_ids = [row[4] for item in full for row in item[2]]
        resumed_ids = [row[4] for item in resumed for row in item[2]]
        assert [row[4] for row in first_batch] + resumed_ids == full_ids
        assert full_ids == list(range(6))


def test_direction_flip_is_exact() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pairs.tsv"
        path.write_text("".join(f"中文{i}\tenglish {i}\n" for i in range(6)), encoding="utf-8")
        native = [row[2] for item in collect(path, flip=0) for row in item[2]]
        flipped = [row[2] for item in collect(path, flip=1) for row in item[2]]
        assert len(native) == len(flipped)
        assert all(left != right for left, right in zip(native, flipped))


if __name__ == "__main__":
    test_resume_has_no_duplicate_or_gap()
    test_direction_flip_is_exact()
    print("epoch repeat scaling CPU tests passed")

