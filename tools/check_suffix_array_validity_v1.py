#!/usr/bin/env python3

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Sequence


SPEC = "GLYPH_SUFFIX_ARRAY_VALIDITY_V1"
DEPENDENCY = "GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1"


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    offset: int


class SuffixArrayValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def compare_suffixes(
    documents: Sequence[bytes],
    left: Coordinate,
    right: Coordinate,
) -> int:
    """
    Independent implementation of the P1 canonical suffix comparator.

    Order:
      unsigned byte lexicographic
      shorter suffix first
      doc_id tie-break when suffix bytes end together
    """

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

    if left.doc_id < right.doc_id:
        return -1

    if left.doc_id > right.doc_id:
        return 1

    raise AssertionError(
        f"distinct coordinates compare equal: {left} vs {right}"
    )


def expected_coordinate_set(
    documents: Sequence[bytes],
) -> set[Coordinate]:
    return {
        Coordinate(doc_id, offset)
        for doc_id, document in enumerate(documents)
        for offset in range(len(document))
    }


def canonical_suffix_array(
    documents: Sequence[bytes],
) -> list[Coordinate]:
    coordinates = list(expected_coordinate_set(documents))

    return sorted(
        coordinates,
        key=functools.cmp_to_key(
            lambda left, right: compare_suffixes(
                documents,
                left,
                right,
            )
        ),
    )


def validate_coordinate(
    documents: Sequence[bytes],
    coordinate: Coordinate,
    position: int,
) -> None:
    if coordinate.doc_id < 0:
        raise SuffixArrayValidationError(
            "invalid_document_id",
            f"SA[{position}] has negative document ID: {coordinate}",
        )

    if coordinate.doc_id >= len(documents):
        raise SuffixArrayValidationError(
            "invalid_document_id",
            (
                f"SA[{position}] document ID is outside corpus: "
                f"{coordinate}"
            ),
        )

    document_length = len(documents[coordinate.doc_id])

    if coordinate.offset < 0:
        raise SuffixArrayValidationError(
            "negative_offset",
            f"SA[{position}] has negative offset: {coordinate}",
        )

    if coordinate.offset >= document_length:
        raise SuffixArrayValidationError(
            "offset_out_of_range",
            (
                f"SA[{position}] offset is outside document length "
                f"{document_length}: {coordinate}"
            ),
        )


def validate_suffix_array(
    documents: Sequence[bytes],
    candidate: Sequence[Coordinate],
) -> dict:
    expected = expected_coordinate_set(documents)
    expected_length = len(expected)

    if len(candidate) != expected_length:
        raise SuffixArrayValidationError(
            "length_mismatch",
            (
                f"candidate length {len(candidate)} != "
                f"expected suffix count {expected_length}"
            ),
        )

    for position, coordinate in enumerate(candidate):
        validate_coordinate(documents, coordinate, position)

    seen: set[Coordinate] = set()

    for position, coordinate in enumerate(candidate):
        if coordinate in seen:
            raise SuffixArrayValidationError(
                "duplicate_coordinate",
                f"duplicate coordinate at SA[{position}]: {coordinate}",
            )

        seen.add(coordinate)

    missing = expected - seen
    unexpected = seen - expected

    if missing or unexpected:
        raise SuffixArrayValidationError(
            "permutation_mismatch",
            (
                f"missing={sorted(missing)} "
                f"unexpected={sorted(unexpected)}"
            ),
        )

    for position in range(len(candidate) - 1):
        left = candidate[position]
        right = candidate[position + 1]

        comparison = compare_suffixes(documents, left, right)

        if comparison >= 0:
            raise SuffixArrayValidationError(
                "not_strictly_increasing",
                (
                    f"SA inversion at positions {position}, "
                    f"{position + 1}: {left} !< {right}"
                ),
            )

    return {
        "ok": True,
        "suffix_count": expected_length,
        "document_count": len(documents),
        "empty_document_count": sum(
            1 for document in documents if not document
        ),
    }


def coordinate_list(
    pairs: Sequence[tuple[int, int]],
) -> list[Coordinate]:
    return [
        Coordinate(doc_id, offset)
        for doc_id, offset in pairs
    ]


def coordinates_as_json(
    coordinates: Sequence[Coordinate],
) -> list[list[int]]:
    return [
        [coordinate.doc_id, coordinate.offset]
        for coordinate in coordinates
    ]


def expect_rejection(
    name: str,
    documents: Sequence[bytes],
    candidate: Sequence[Coordinate],
    expected_codes: set[str],
) -> dict:
    try:
        validate_suffix_array(documents, candidate)
    except SuffixArrayValidationError as error:
        if error.code not in expected_codes:
            raise AssertionError(
                {
                    "mutation": name,
                    "expected_codes": sorted(expected_codes),
                    "actual_code": error.code,
                    "message": error.message,
                }
            ) from error

        return {
            "mutation": name,
            "rejected": True,
            "error_code": error.code,
            "message": error.message,
        }

    raise AssertionError(
        f"mutation unexpectedly accepted: {name}"
    )


def run_positive_fixtures() -> list[dict]:
    fixtures: list[tuple[str, list[bytes]]] = [
        ("empty_corpus", []),
        ("one_empty_document", [b""]),
        ("aa", [b"aa"]),
        ("nul_nul", [b"\x00\x00"]),
        ("ff_nul", [b"\xff\x00"]),
        ("duplicate_documents", [b"a", b"a"]),
        ("prefix_documents", [b"ab", b"abc"]),
        ("periodic", [b"abababab"]),
        ("full_alphabet", [bytes(range(256))]),
        (
            "duplicate_binary_documents",
            [b"\x00\xff", b"\x00\xff"],
        ),
        (
            "empty_mixed_with_nonempty",
            [b"", b"\x00", b"", b"\xff\x00"],
        ),
    ]

    results = []

    for name, documents in fixtures:
        first = canonical_suffix_array(documents)
        second = canonical_suffix_array(documents)

        if first != second:
            raise AssertionError(
                f"non-deterministic canonical SA: {name}"
            )

        validation = validate_suffix_array(
            documents,
            first,
        )

        results.append(
            {
                "fixture": name,
                **validation,
                "canonical_sa": coordinates_as_json(first),
            }
        )

    return results


def run_negative_fixtures() -> list[dict]:
    documents = [
        b"aba",
        b"\x00\xff",
        b"",
        b"aba",
    ]

    canonical = canonical_suffix_array(documents)

    results = []

    results.append(
        expect_rejection(
            "missing_coordinate",
            documents,
            canonical[:-1],
            {"length_mismatch"},
        )
    )

    duplicate = list(canonical)
    duplicate[-1] = duplicate[0]

    results.append(
        expect_rejection(
            "duplicate_coordinate",
            documents,
            duplicate,
            {"duplicate_coordinate"},
        )
    )

    invalid_doc = list(canonical)
    invalid_doc[0] = Coordinate(len(documents), 0)

    results.append(
        expect_rejection(
            "invalid_document_id",
            documents,
            invalid_doc,
            {"invalid_document_id"},
        )
    )

    negative_doc = list(canonical)
    negative_doc[0] = Coordinate(-1, 0)

    results.append(
        expect_rejection(
            "negative_document_id",
            documents,
            negative_doc,
            {"invalid_document_id"},
        )
    )

    negative_offset = list(canonical)
    negative_offset[0] = Coordinate(0, -1)

    results.append(
        expect_rejection(
            "negative_offset",
            documents,
            negative_offset,
            {"negative_offset"},
        )
    )

    offset_at_length = list(canonical)
    offset_at_length[0] = Coordinate(
        0,
        len(documents[0]),
    )

    results.append(
        expect_rejection(
            "offset_equal_document_length",
            documents,
            offset_at_length,
            {"offset_out_of_range"},
        )
    )

    offset_beyond_length = list(canonical)
    offset_beyond_length[0] = Coordinate(
        1,
        len(documents[1]) + 1,
    )

    results.append(
        expect_rejection(
            "offset_beyond_document_length",
            documents,
            offset_beyond_length,
            {"offset_out_of_range"},
        )
    )

    empty_document_coordinate = list(canonical)
    empty_document_coordinate[0] = Coordinate(2, 0)

    results.append(
        expect_rejection(
            "coordinate_inside_empty_document",
            documents,
            empty_document_coordinate,
            {"offset_out_of_range"},
        )
    )

    adjacent_inversion = list(canonical)

    if len(adjacent_inversion) < 2:
        raise AssertionError(
            "negative fixture corpus unexpectedly too small"
        )

    adjacent_inversion[0], adjacent_inversion[1] = (
        adjacent_inversion[1],
        adjacent_inversion[0],
    )

    results.append(
        expect_rejection(
            "adjacent_inversion",
            documents,
            adjacent_inversion,
            {"not_strictly_increasing"},
        )
    )

    reversed_candidate = list(reversed(canonical))

    results.append(
        expect_rejection(
            "reversed_suffix_array",
            documents,
            reversed_candidate,
            {"not_strictly_increasing"},
        )
    )

    rotated_candidate = canonical[1:] + canonical[:1]

    results.append(
        expect_rejection(
            "rotated_suffix_array",
            documents,
            rotated_candidate,
            {"not_strictly_increasing"},
        )
    )

    return results


def main() -> int:
    positive = run_positive_fixtures()
    negative = run_negative_fixtures()

    output = {
        "ok": True,
        "proof_obligation": "P2",
        "spec": SPEC,
        "depends_on": DEPENDENCY,
        "positive_fixture_count": len(positive),
        "negative_mutation_count": len(negative),
        "positive_fixtures": positive,
        "negative_mutations": negative,
    }

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
