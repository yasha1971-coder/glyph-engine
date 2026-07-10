#!/usr/bin/env python3

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    offset: int


def compare_suffixes(
    documents: Sequence[bytes],
    left: Coordinate,
    right: Coordinate,
) -> int:
    """Canonical GLYPH V1 suffix comparator."""

    if left == right:
        return 0

    a = documents[left.doc_id]
    b = documents[right.doc_id]

    ia = left.offset
    ib = right.offset

    while ia < len(a) and ib < len(b):
        av = a[ia]
        bv = b[ib]

        if av < bv:
            return -1
        if av > bv:
            return 1

        ia += 1
        ib += 1

    a_ended = ia == len(a)
    b_ended = ib == len(b)

    if a_ended and not b_ended:
        return -1

    if b_ended and not a_ended:
        return 1

    # Both suffixes ended at the same comparison depth.
    # END(doc_id) order is the deterministic final tie-break.
    if left.doc_id < right.doc_id:
        return -1
    if left.doc_id > right.doc_id:
        return 1

    # Same document and identical remaining bytes imply the same offset.
    # Reaching this point for distinct coordinates is an invariant failure.
    raise AssertionError(
        f"distinct suffix coordinates compare equal: {left} vs {right}"
    )


def canonical_order(documents: Sequence[bytes]) -> list[Coordinate]:
    coordinates = [
        Coordinate(doc_id, offset)
        for doc_id, document in enumerate(documents)
        for offset in range(len(document))
    ]

    return sorted(
        coordinates,
        key=functools.cmp_to_key(
            lambda a, b: compare_suffixes(documents, a, b)
        ),
    )


def assert_strict_total_order(documents: Sequence[bytes]) -> list[Coordinate]:
    order = canonical_order(documents)

    assert len(order) == sum(len(document) for document in documents)
    assert len(set(order)) == len(order)

    for i, left in enumerate(order):
        for j, right in enumerate(order):
            result = compare_suffixes(documents, left, right)

            if i < j:
                assert result < 0, (left, right, result)
            elif i > j:
                assert result > 0, (left, right, result)
            else:
                assert result == 0, (left, right, result)

    return order


def coords(order: Sequence[Coordinate]) -> list[list[int]]:
    return [[coordinate.doc_id, coordinate.offset] for coordinate in order]


def main() -> int:
    fixtures: list[tuple[str, list[bytes], list[list[int]] | None]] = [
        (
            "aa",
            [b"aa"],
            [[0, 1], [0, 0]],
        ),
        (
            "nul_nul",
            [b"\x00\x00"],
            [[0, 1], [0, 0]],
        ),
        (
            "ff_nul",
            [b"\xff\x00"],
            [[0, 1], [0, 0]],
        ),
        (
            "duplicate_docs",
            [b"a", b"a"],
            [[0, 0], [1, 0]],
        ),
        (
            "prefix_docs",
            [b"ab", b"abc"],
            [[0, 0], [1, 0], [0, 1], [1, 1], [1, 2]],
        ),
        (
            "periodic",
            [b"abab"],
            None,
        ),
        (
            "full_alphabet",
            [bytes(range(256))],
            None,
        ),
        (
            "reverse_alphabet",
            [bytes(reversed(range(256)))],
            None,
        ),
        (
            "empty_and_nonempty_docs",
            [b"", b"\x00", b""],
            [[1, 0]],
        ),
        (
            "duplicate_binary_docs",
            [b"\x00\xff", b"\x00\xff"],
            [[0, 0], [1, 0], [0, 1], [1, 1]],
        ),
    ]

    results = []

    for name, documents, expected in fixtures:
        first = assert_strict_total_order(documents)
        second = assert_strict_total_order(documents)

        actual = coords(first)

        assert actual == coords(second), f"non-deterministic order: {name}"

        if expected is not None:
            assert actual == expected, {
                "fixture": name,
                "expected": expected,
                "actual": actual,
            }

        results.append(
            {
                "fixture": name,
                "document_count": len(documents),
                "suffix_count": len(actual),
                "order": actual,
                "ok": True,
            }
        )

    print(
        json.dumps(
            {
                "ok": True,
                "proof_obligation": "P1",
                "spec": "GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1",
                "fixtures": results,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
