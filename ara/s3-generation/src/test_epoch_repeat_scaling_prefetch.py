#!/usr/bin/env python3
"""CPU smoke tests for the bounded input-prefetch adapter."""
from __future__ import annotations

import s3_epoch_repeat_scaling_prefetch as prefetch


def test_order_and_cardinality() -> None:
    expected = list(range(37))
    assert list(prefetch.buffered(iter(expected), 3)) == expected
    assert list(prefetch.buffered(iter(expected), 0)) == expected


def test_producer_error_propagates() -> None:
    def broken():
        yield 1
        raise RuntimeError("expected producer failure")

    iterator = prefetch.buffered(broken(), 2)
    assert next(iterator) == 1
    try:
        next(iterator)
    except RuntimeError as error:
        assert "expected producer failure" in str(error)
    else:
        raise AssertionError("producer failure was swallowed")


def test_argument_is_removed() -> None:
    argv, capacity = prefetch.take_prefetch_argument(
        ["runner", "--mode", "formal", "--prefetch-batches", "2"]
    )
    assert argv == ["runner", "--mode", "formal"]
    assert capacity == 2


if __name__ == "__main__":
    test_order_and_cardinality()
    test_producer_error_propagates()
    test_argument_is_removed()
    print("epoch repeat prefetch CPU tests passed")
