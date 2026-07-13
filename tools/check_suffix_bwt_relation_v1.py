#!/usr/bin/env python3

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import Any, Sequence


SPEC = "GLYPH_SUFFIX_BWT_RELATION_V1"
PROOF_OBLIGATION = "P3"

DEPENDENCIES = [
    "GLYPH_VIRTUAL_SENTINEL_TOTAL_ORDER_V1",
    "GLYPH_SUFFIX_ARRAY_VALIDITY_V1",
]


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    offset: int


@dataclass(frozen=True)
class ByteToken:
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.value, int):
            raise TypeError("byte token value must be an integer")

        if self.value < 0 or self.value > 255:
            raise ValueError(
                f"byte token outside unsigned-byte range: {self.value}"
            )


@dataclass(frozen=True)
class VirtualSentinelToken:
    doc_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, int):
            raise TypeError("sentinel doc_id must be an integer")

        if self.doc_id < 0:
            raise ValueError(
                f"sentinel doc_id must be non-negative: {self.doc_id}"
            )


Token = ByteToken | VirtualSentinelToken


class BWTValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def compare_suffixes(
    documents: Sequence[bytes],
    left: Coordinate,
    right: Coordinate,
) -> int:
    if left == right:
        return 0

    left_document = documents[left.doc_id]
    right_document = documents[right.doc_id]

    left_index = left.offset
    right_index = right.offset

    while (
        left_index < len(left_document)
        and right_index < len(right_document)
    ):
        left_byte = left_document[left_index]
        right_byte = right_document[right_index]

        if left_byte < right_byte:
            return -1

        if left_byte > right_byte:
            return 1

        left_index += 1
        right_index += 1

    left_ended = left_index == len(left_document)
    right_ended = right_index == len(right_document)

    if left_ended and not right_ended:
        return -1

    if right_ended and not left_ended:
        return 1

    if left.doc_id < right.doc_id:
        return -1

    if left.doc_id > right.doc_id:
        return 1

    raise AssertionError(
        f"distinct coordinates compare equal: {left} vs {right}"
    )


def canonical_suffix_array(
    documents: Sequence[bytes],
) -> list[Coordinate]:
    coordinates = [
        Coordinate(doc_id, offset)
        for doc_id, document in enumerate(documents)
        for offset in range(len(document) + 1)
    ]

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


def validate_suffix_array(
    documents: Sequence[bytes],
    suffix_array: Sequence[Coordinate],
) -> None:
    expected = {
        Coordinate(doc_id, offset)
        for doc_id, document in enumerate(documents)
        for offset in range(len(document) + 1)
    }

    if len(suffix_array) != len(expected):
        raise BWTValidationError(
            "invalid_suffix_array_length",
            (
                f"SA length {len(suffix_array)} != "
                f"expected {len(expected)}"
            ),
        )

    if len(set(suffix_array)) != len(suffix_array):
        raise BWTValidationError(
            "invalid_suffix_array_duplicate",
            "SA contains duplicate coordinates",
        )

    if set(suffix_array) != expected:
        raise BWTValidationError(
            "invalid_suffix_array_permutation",
            "SA is not the complete coordinate permutation",
        )

    for position in range(len(suffix_array) - 1):
        left = suffix_array[position]
        right = suffix_array[position + 1]

        if compare_suffixes(documents, left, right) >= 0:
            raise BWTValidationError(
                "invalid_suffix_array_order",
                (
                    f"SA inversion at {position}: "
                    f"{left} !< {right}"
                ),
            )


def expected_token_for_coordinate(
    documents: Sequence[bytes],
    coordinate: Coordinate,
) -> Token:
    document = documents[coordinate.doc_id]

    if coordinate.offset == 0:
        return VirtualSentinelToken(coordinate.doc_id)

    return ByteToken(document[coordinate.offset - 1])


def canonical_suffix_bwt(
    documents: Sequence[bytes],
    suffix_array: Sequence[Coordinate],
) -> list[Token]:
    validate_suffix_array(documents, suffix_array)

    return [
        expected_token_for_coordinate(
            documents,
            coordinate,
        )
        for coordinate in suffix_array
    ]


def token_to_json(token: Token) -> dict[str, Any]:
    if isinstance(token, ByteToken):
        return {
            "kind": "byte",
            "value": token.value,
        }

    if isinstance(token, VirtualSentinelToken):
        return {
            "kind": "virtual_sentinel",
            "doc_id": token.doc_id,
        }

    raise TypeError(f"unknown token type: {type(token)!r}")


def validate_token_shape(
    token: object,
    position: int,
) -> None:
    if isinstance(token, ByteToken):
        if token.value < 0 or token.value > 255:
            raise BWTValidationError(
                "invalid_byte_token",
                (
                    f"BWT[{position}] has invalid byte value: "
                    f"{token.value}"
                ),
            )

        return

    if isinstance(token, VirtualSentinelToken):
        if token.doc_id < 0:
            raise BWTValidationError(
                "invalid_sentinel_token",
                (
                    f"BWT[{position}] has invalid sentinel doc_id: "
                    f"{token.doc_id}"
                ),
            )

        return

    raise BWTValidationError(
        "unknown_token_kind",
        (
            f"BWT[{position}] has unsupported token type: "
            f"{type(token).__name__}"
        ),
    )


def validate_suffix_bwt(
    documents: Sequence[bytes],
    suffix_array: Sequence[Coordinate],
    candidate_bwt: Sequence[object],
) -> dict[str, Any]:
    validate_suffix_array(documents, suffix_array)

    if len(candidate_bwt) != len(suffix_array):
        raise BWTValidationError(
            "length_mismatch",
            (
                f"BWT length {len(candidate_bwt)} != "
                f"SA length {len(suffix_array)}"
            ),
        )

    expected = canonical_suffix_bwt(
        documents,
        suffix_array,
    )

    for position, token in enumerate(candidate_bwt):
        validate_token_shape(token, position)

        coordinate = suffix_array[position]
        expected_token = expected[position]

        if token != expected_token:
            if coordinate.offset == 0:
                if isinstance(token, ByteToken):
                    raise BWTValidationError(
                        "document_start_requires_virtual_sentinel",
                        (
                            f"BWT[{position}] for SA coordinate "
                            f"{coordinate} used byte {token.value:#04x} "
                            f"instead of virtual sentinel"
                        ),
                    )

                if isinstance(token, VirtualSentinelToken):
                    raise BWTValidationError(
                        "wrong_virtual_sentinel_document",
                        (
                            f"BWT[{position}] for SA coordinate "
                            f"{coordinate} used sentinel for document "
                            f"{token.doc_id}"
                        ),
                    )

            if coordinate.offset > 0:
                if isinstance(token, VirtualSentinelToken):
                    raise BWTValidationError(
                        "sentinel_at_nonzero_offset",
                        (
                            f"BWT[{position}] for SA coordinate "
                            f"{coordinate} used virtual sentinel"
                        ),
                    )

                if isinstance(token, ByteToken):
                    raise BWTValidationError(
                        "wrong_predecessor_byte",
                        (
                            f"BWT[{position}] for SA coordinate "
                            f"{coordinate} used {token.value:#04x}; "
                            f"expected "
                            f"{expected_token.value:#04x}"
                        ),
                    )

            raise BWTValidationError(
                "token_mismatch",
                (
                    f"BWT[{position}] mismatch for coordinate "
                    f"{coordinate}"
                ),
            )

    document_ids = set(range(len(documents)))

    sentinel_counts = {
        doc_id: 0
        for doc_id in range(len(documents))
    }

    for token in candidate_bwt:
        if isinstance(token, VirtualSentinelToken):
            if token.doc_id >= len(documents):
                raise BWTValidationError(
                    "wrong_virtual_sentinel_document",
                    (
                        "virtual sentinel references document "
                        f"{token.doc_id}, but corpus has "
                        f"{len(documents)} documents"
                    ),
                )

            sentinel_counts[token.doc_id] += 1

    for doc_id, document in enumerate(documents):
        expected_count = 1
        actual_count = sentinel_counts[doc_id]

        if actual_count != expected_count:
            raise BWTValidationError(
                "sentinel_multiplicity_mismatch",
                (
                    f"document {doc_id}: sentinel count "
                    f"{actual_count} != expected {expected_count}"
                ),
            )

    return {
        "ok": True,
        "document_count": len(documents),
        "suffix_count": len(suffix_array),
        "bwt_token_count": len(candidate_bwt),
        "document_sentinel_count": len(document_ids),
        "virtual_sentinel_count": sum(
            sentinel_counts.values()
        ),
    }


def expect_rejection(
    name: str,
    documents: Sequence[bytes],
    suffix_array: Sequence[Coordinate],
    candidate_bwt: Sequence[object],
    expected_codes: set[str],
) -> dict[str, Any]:
    try:
        validate_suffix_bwt(
            documents,
            suffix_array,
            candidate_bwt,
        )
    except BWTValidationError as error:
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


def run_positive_fixtures() -> list[dict[str, Any]]:
    fixtures: list[tuple[str, list[bytes]]] = [
        ("empty_corpus", []),
        ("one_empty_document", [b""]),
        ("one_byte_zero", [b"\x00"]),
        ("one_byte_ff", [b"\xff"]),
        ("zero_ff", [b"\x00\xff"]),
        ("aba", [b"aba"]),
        ("periodic", [b"abababab"]),
        ("duplicate_documents", [b"same", b"same"]),
        ("prefix_documents", [b"ab", b"abc"]),
        (
            "multiple_binary_documents",
            [b"\x00\xff", b"\xff\x00"],
        ),
        (
            "empty_mixed",
            [b"", b"\x00", b"", b"abc"],
        ),
        ("full_alphabet", [bytes(range(256))]),
    ]

    results: list[dict[str, Any]] = []

    for name, documents in fixtures:
        suffix_array = canonical_suffix_array(documents)
        first_bwt = canonical_suffix_bwt(
            documents,
            suffix_array,
        )
        second_bwt = canonical_suffix_bwt(
            documents,
            suffix_array,
        )

        if first_bwt != second_bwt:
            raise AssertionError(
                f"non-deterministic BWT fixture: {name}"
            )

        validation = validate_suffix_bwt(
            documents,
            suffix_array,
            first_bwt,
        )

        results.append(
            {
                "fixture": name,
                **validation,
                "suffix_array": [
                    [coordinate.doc_id, coordinate.offset]
                    for coordinate in suffix_array
                ],
                "bwt": [
                    token_to_json(token)
                    for token in first_bwt
                ],
            }
        )

    return results


def replace_at(
    values: Sequence[object],
    position: int,
    replacement: object,
) -> list[object]:
    output = list(values)
    output[position] = replacement
    return output


def find_row(
    suffix_array: Sequence[Coordinate],
    coordinate: Coordinate,
) -> int:
    try:
        return suffix_array.index(coordinate)
    except ValueError as error:
        raise AssertionError(
            f"coordinate not present in SA: {coordinate}"
        ) from error


def run_negative_fixtures() -> list[dict[str, Any]]:
    documents = [
        b"aba",
        b"\x00\xff",
        b"",
        b"XYZ",
    ]

    suffix_array = canonical_suffix_array(documents)
    canonical_bwt = canonical_suffix_bwt(
        documents,
        suffix_array,
    )

    results: list[dict[str, Any]] = []

    results.append(
        expect_rejection(
            "missing_token",
            documents,
            suffix_array,
            canonical_bwt[:-1],
            {"length_mismatch"},
        )
    )

    results.append(
        expect_rejection(
            "extra_token",
            documents,
            suffix_array,
            list(canonical_bwt) + [ByteToken(0)],
            {"length_mismatch"},
        )
    )

    start_row_doc0 = find_row(
        suffix_array,
        Coordinate(0, 0),
    )

    rotation_bwt = replace_at(
        canonical_bwt,
        start_row_doc0,
        ByteToken(documents[0][-1]),
    )

    results.append(
        expect_rejection(
            "rotation_bwt_wraparound",
            documents,
            suffix_array,
            rotation_bwt,
            {"document_start_requires_virtual_sentinel"},
        )
    )

    zero_sentinel = replace_at(
        canonical_bwt,
        start_row_doc0,
        ByteToken(0),
    )

    results.append(
        expect_rejection(
            "zero_byte_used_as_sentinel",
            documents,
            suffix_array,
            zero_sentinel,
            {"document_start_requires_virtual_sentinel"},
        )
    )

    wrong_doc_sentinel = replace_at(
        canonical_bwt,
        start_row_doc0,
        VirtualSentinelToken(1),
    )

    results.append(
        expect_rejection(
            "wrong_document_sentinel",
            documents,
            suffix_array,
            wrong_doc_sentinel,
            {"wrong_virtual_sentinel_document"},
        )
    )

    start_row_doc3 = find_row(
        suffix_array,
        Coordinate(3, 0),
    )

    previous_physical_document_byte = replace_at(
        canonical_bwt,
        start_row_doc3,
        ByteToken(documents[1][-1]),
    )

    results.append(
        expect_rejection(
            "previous_document_predecessor",
            documents,
            suffix_array,
            previous_physical_document_byte,
            {"document_start_requires_virtual_sentinel"},
        )
    )

    nonzero_coordinate = next(
        coordinate
        for coordinate in suffix_array
        if coordinate.offset > 0
    )

    nonzero_row = find_row(
        suffix_array,
        nonzero_coordinate,
    )

    sentinel_at_nonzero = replace_at(
        canonical_bwt,
        nonzero_row,
        VirtualSentinelToken(nonzero_coordinate.doc_id),
    )

    results.append(
        expect_rejection(
            "sentinel_at_nonzero_offset",
            documents,
            suffix_array,
            sentinel_at_nonzero,
            {"sentinel_at_nonzero_offset"},
        )
    )

    expected_nonzero_token = canonical_bwt[nonzero_row]

    if not isinstance(expected_nonzero_token, ByteToken):
        raise AssertionError(
            "non-zero suffix unexpectedly maps to sentinel"
        )

    wrong_byte_value = (
        expected_nonzero_token.value + 1
    ) % 256

    wrong_predecessor = replace_at(
        canonical_bwt,
        nonzero_row,
        ByteToken(wrong_byte_value),
    )

    results.append(
        expect_rejection(
            "wrong_predecessor_byte",
            documents,
            suffix_array,
            wrong_predecessor,
            {"wrong_predecessor_byte"},
        )
    )

    malformed_token = replace_at(
        canonical_bwt,
        nonzero_row,
        {
            "kind": "byte",
            "value": expected_nonzero_token.value,
        },
    )

    results.append(
        expect_rejection(
            "malformed_untyped_token_object",
            documents,
            suffix_array,
            malformed_token,
            {"unknown_token_kind"},
        )
    )

    unknown_token = replace_at(
        canonical_bwt,
        nonzero_row,
        "not-a-token",
    )

    results.append(
        expect_rejection(
            "unknown_token_kind",
            documents,
            suffix_array,
            unknown_token,
            {"unknown_token_kind"},
        )
    )

    shared_sentinel = [
        (
            VirtualSentinelToken(0)
            if isinstance(token, VirtualSentinelToken)
            else token
        )
        for token in canonical_bwt
    ]

    results.append(
        expect_rejection(
            "collapsed_shared_sentinel_identity",
            documents,
            suffix_array,
            shared_sentinel,
            {
                "wrong_virtual_sentinel_document",
                "sentinel_multiplicity_mismatch",
            },
        )
    )

    return results


def main() -> int:
    positive = run_positive_fixtures()
    negative = run_negative_fixtures()

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "spec": SPEC,
        "dependencies": DEPENDENCIES,
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
