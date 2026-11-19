"""Find the first candidate satisfying a counterexample predicate."""

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def find_counterexample(candidates: Iterable[T], predicate: Callable[[T], bool], limit: int | None = None) -> T | None:
    """Return the first candidate matching ``predicate``, or ``None``.

    ``limit`` bounds how many candidates are checked.
    """
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None")
    # Check one candidate at a time and stop as soon as one matches.
    for index, candidate in enumerate(candidates):
        if limit is not None and index >= limit:
            break
        if predicate(candidate):
            return candidate
    return None
