#!/usr/bin/env python3
"""Run epoch-repeat training with an optional bounded input-prefetch queue."""
from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Iterable, Iterator

_END = object()


def buffered(items: Iterable, capacity: int) -> Iterator:
    """Overlap CPU parsing/tokenization with GPU work without changing order."""
    if capacity <= 0:
        yield from items
        return

    work: queue.Queue = queue.Queue(maxsize=capacity)

    def produce() -> None:
        try:
            for item in items:
                work.put((True, item))
        except BaseException as error:  # Propagate producer failures to the caller.
            work.put((False, error))
        finally:
            work.put((True, _END))

    thread = threading.Thread(target=produce, name="treeheap-input-prefetch", daemon=True)
    thread.start()
    while True:
        ok, value = work.get()
        if value is _END:
            break
        if not ok:
            raise value
        yield value
    thread.join()


def take_prefetch_argument(argv: list[str]) -> tuple[list[str], int]:
    cleaned = [argv[0]]
    capacity = 0
    index = 1
    while index < len(argv):
        if argv[index] == "--prefetch-batches":
            if index + 1 >= len(argv):
                raise SystemExit("--prefetch-batches requires an integer")
            capacity = int(argv[index + 1])
            index += 2
        else:
            cleaned.append(argv[index])
            index += 1
    if capacity < 0:
        raise SystemExit("--prefetch-batches must be non-negative")
    return cleaned, capacity


def main() -> None:
    import s3_epoch_repeat_scaling as repeat

    argv, capacity = take_prefetch_argument(sys.argv)
    if capacity:
        original = repeat.iter_sequential_batches

        def prefetched(*args, **kwargs):
            return buffered(original(*args, **kwargs), capacity)

        repeat.iter_sequential_batches = prefetched
    sys.argv = argv
    repeat.main()


if __name__ == "__main__":
    main()
