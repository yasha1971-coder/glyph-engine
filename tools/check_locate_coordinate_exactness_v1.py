#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence


PROOF_OBLIGATION = "P7"
FORMAT = "GLYPH_LOCATE_COORDINATE_EXACTNESS_V1"


class LocateError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    kind: str
    value: int

    @staticmethod
    def sentinel(doc_id: int) -> "Token":
        return Token("sentinel", doc_id)

    @staticmethod
    def byte(value: int) -> "Token":
        if not 0 <= value <= 255:
            raise LocateError(f"byte outside range: {value}")
        return Token("byte", value)

    def key(self) -> tuple[int, int]:
        if self.kind == "sentinel":
            return (0, self.value)
        if self.kind == "byte":
            return (1, self.value)
        raise LocateError(f"unknown token kind: {self.kind}")


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    doc_offset: int

    def as_list(self) -> list[int]:
        return [self.doc_id, self.doc_offset]


def suffix_key(
    documents: Sequence[bytes],
    coordinate: Coordinate,
) -> tuple[tuple[int, int], ...]:
    data = documents[coordinate.doc_id]

    return tuple(
        Token.byte(value).key()
        for value in data[coordinate.doc_offset:]
    ) + (
        Token.sentinel(coordinate.doc_id).key(),
    )


def build_sa(
    documents: Sequence[bytes],
) -> list[Coordinate]:
    rows = [
        Coordinate(doc_id, offset)
        for doc_id, data in enumerate(documents)
        for offset in range(len(data))
    ]

    rows.sort(
        key=lambda coordinate: suffix_key(
            documents,
            coordinate,
        )
    )

    return rows


def build_bwt(
    documents: Sequence[bytes],
    sa: Sequence[Coordinate],
) -> list[Token]:
    result: list[Token] = []

    for coordinate in sa:
        if coordinate.doc_offset == 0:
            result.append(
                Token.sentinel(coordinate.doc_id)
            )
        else:
            result.append(
                Token.byte(
                    documents[coordinate.doc_id][
                        coordinate.doc_offset - 1
                    ]
                )
            )

    return result


def symbols(
    bwt: Sequence[Token],
) -> list[Token]:
    return sorted(set(bwt), key=Token.key)


def frequencies(
    bwt: Sequence[Token],
    alphabet: Sequence[Token],
) -> dict[Token, int]:
    return {
        symbol: sum(token == symbol for token in bwt)
        for symbol in alphabet
    }


def build_c(
    alphabet: Sequence[Token],
    freq: dict[Token, int],
) -> dict[Token, int]:
    total = 0
    result: dict[Token, int] = {}

    for symbol in alphabet:
        result[symbol] = total
        total += freq[symbol]

    return result


def rank(
    bwt: Sequence[Token],
    symbol: Token,
    position: int,
) -> int:
    if not 0 <= position <= len(bwt):
        raise LocateError(
            f"rank position outside range: {position}"
        )

    return sum(
        token == symbol
        for token in bwt[:position]
    )


def c_for_symbol(
    symbol: Token,
    alphabet: Sequence[Token],
    freq: dict[Token, int],
) -> int:
    return sum(
        freq[item]
        for item in alphabet
        if item.key() < symbol.key()
    )


def backward_search(
    bwt: Sequence[Token],
    pattern: bytes,
) -> tuple[int, int]:
    if not pattern:
        raise LocateError("EMPTY_PATTERN")

    alphabet = symbols(bwt)
    freq = frequencies(bwt, alphabet)
    c_array = build_c(alphabet, freq)

    l = 0
    r = len(bwt)

    for value in reversed(pattern):
        symbol = Token.byte(value)
        c_value = c_array.get(
            symbol,
            c_for_symbol(symbol, alphabet, freq),
        )

        l = c_value + rank(bwt, symbol, l)
        r = c_value + rank(bwt, symbol, r)

        if l >= r:
            return (l, l)

    return (l, r)


def naive_coordinates(
    documents: Sequence[bytes],
    pattern: bytes,
) -> list[Coordinate]:
    if not pattern:
        raise LocateError("EMPTY_PATTERN")

    result: list[Coordinate] = []

    for doc_id, data in enumerate(documents):
        if len(pattern) > len(data):
            continue

        for offset in range(
            len(data) - len(pattern) + 1
        ):
            if data[
                offset : offset + len(pattern)
            ] == pattern:
                result.append(
                    Coordinate(doc_id, offset)
                )

    return sorted(result)


def byte_check_coordinate(
    documents: Sequence[bytes],
    pattern: bytes,
    coordinate: Coordinate,
) -> bool:
    if not 0 <= coordinate.doc_id < len(documents):
        return False

    data = documents[coordinate.doc_id]

    if coordinate.doc_offset < 0:
        return False

    end = coordinate.doc_offset + len(pattern)

    if end > len(data):
        return False

    return data[
        coordinate.doc_offset:end
    ] == pattern


def full_locate(
    documents: Sequence[bytes],
    pattern: bytes,
) -> dict:
    if not pattern:
        raise LocateError("EMPTY_PATTERN")

    sa = build_sa(documents)
    bwt = build_bwt(documents, sa)
    l, r = backward_search(bwt, pattern)

    coordinates = sorted(sa[l:r])
    expected = naive_coordinates(
        documents,
        pattern,
    )

    if coordinates != expected:
        raise LocateError(
            "full locate differs from naive oracle"
        )

    if len(coordinates) != r - l:
        raise LocateError(
            "full locate size differs from FM count"
        )

    if len(set(coordinates)) != len(coordinates):
        raise LocateError(
            "duplicate coordinate in full locate"
        )

    if coordinates != sorted(coordinates):
        raise LocateError(
            "coordinates are not canonically ordered"
        )

    byte_check = all(
        byte_check_coordinate(
            documents,
            pattern,
            coordinate,
        )
        for coordinate in coordinates
    )

    if not byte_check:
        raise LocateError(
            "full locate byte check failed"
        )

    return {
        "fm_interval": [l, r],
        "match_count": r - l,
        "coordinates": [
            coordinate.as_list()
            for coordinate in coordinates
        ],
        "returned_count": len(coordinates),
        "bounded": False,
        "offsets_complete": True,
        "byte_check": True,
    }


def bounded_locate(
    documents: Sequence[bytes],
    pattern: bytes,
    max_offsets: int,
) -> dict:
    if max_offsets < 0:
        raise LocateError(
            "max_offsets must be non-negative"
        )

    full = full_locate(documents, pattern)
    full_coordinates = full["coordinates"]

    coordinates = full_coordinates[:max_offsets]
    returned_count = len(coordinates)
    match_count = full["match_count"]
    bounded = returned_count < match_count

    return {
        "fm_interval": list(full["fm_interval"]),
        "match_count": match_count,
        "coordinates": coordinates,
        "returned_count": returned_count,
        "max_offsets": max_offsets,
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
    }


def validate_result(
    documents: Sequence[bytes],
    pattern: bytes,
    result: dict,
    *,
    max_offsets: int | None,
) -> None:
    expected_full = full_locate(
        documents,
        pattern,
    )

    expected_coordinates = expected_full["coordinates"]
    expected_count = expected_full["match_count"]
    expected_interval = expected_full["fm_interval"]

    if result.get("fm_interval") != expected_interval:
        raise LocateError("FM interval mismatch")

    if result.get("match_count") != expected_count:
        raise LocateError("match_count mismatch")

    coordinates_raw = result.get("coordinates")

    if not isinstance(coordinates_raw, list):
        raise LocateError("coordinates must be a list")

    coordinates: list[Coordinate] = []

    for item in coordinates_raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(
                isinstance(value, int)
                and not isinstance(value, bool)
                for value in item
            )
        ):
            raise LocateError(
                "invalid coordinate representation"
            )

        coordinates.append(
            Coordinate(item[0], item[1])
        )

    if len(set(coordinates)) != len(coordinates):
        raise LocateError("duplicate coordinates")

    if coordinates != sorted(coordinates):
        raise LocateError(
            "non-canonical coordinate order"
        )

    for coordinate in coordinates:
        if not byte_check_coordinate(
            documents,
            pattern,
            coordinate,
        ):
            raise LocateError(
                "coordinate byte check failed"
            )

    if result.get("byte_check") is not True:
        raise LocateError(
            "byte_check must be true"
        )

    if max_offsets is None:
        expected_returned = expected_count
        expected_bounded = False
        expected_complete = True
        expected_result_coordinates = expected_coordinates
    else:
        if max_offsets < 0:
            raise LocateError(
                "negative max_offsets"
            )

        expected_returned = min(
            expected_count,
            max_offsets,
        )
        expected_bounded = (
            expected_returned < expected_count
        )
        expected_complete = not expected_bounded
        expected_result_coordinates = (
            expected_coordinates[:expected_returned]
        )

        if result.get("max_offsets") != max_offsets:
            raise LocateError(
                "max_offsets metadata mismatch"
            )

    if coordinates_raw != expected_result_coordinates:
        raise LocateError(
            "coordinates differ from canonical expected result"
        )

    if result.get("returned_count") != expected_returned:
        raise LocateError(
            "returned_count mismatch"
        )

    if result.get("returned_count") != len(coordinates_raw):
        raise LocateError(
            "returned_count differs from coordinate length"
        )

    if result.get("bounded") is not expected_bounded:
        raise LocateError(
            "bounded flag mismatch"
        )

    if (
        result.get("offsets_complete")
        is not expected_complete
    ):
        raise LocateError(
            "offsets_complete mismatch"
        )


def query_bounds(match_count: int) -> list[int]:
    candidates = {
        0,
        1,
        match_count,
        match_count + 1,
    }

    if match_count > 0:
        candidates.add(match_count - 1)

    return sorted(
        value
        for value in candidates
        if value >= 0
    )


def validate_query(
    documents: Sequence[bytes],
    pattern: bytes,
) -> dict:
    full = full_locate(
        documents,
        pattern,
    )

    validate_result(
        documents,
        pattern,
        full,
        max_offsets=None,
    )

    bounded_results = []

    for max_offsets in query_bounds(
        full["match_count"]
    ):
        result = bounded_locate(
            documents,
            pattern,
            max_offsets,
        )

        validate_result(
            documents,
            pattern,
            result,
            max_offsets=max_offsets,
        )

        bounded_results.append(result)

    return {
        "pattern_hex": pattern.hex(),
        "full": full,
        "bounded_results": bounded_results,
    }


def unique_patterns(
    patterns: Sequence[bytes],
) -> list[bytes]:
    result: list[bytes] = []
    seen: set[bytes] = set()

    for pattern in patterns:
        if pattern and pattern not in seen:
            seen.add(pattern)
            result.append(pattern)

    return result


def fixture_patterns(
    documents: Sequence[bytes],
) -> list[bytes]:
    patterns = [
        bytes([value])
        for value in range(256)
    ]

    patterns.extend(
        [
            b"\x00\xff",
            b"\xff\x00",
            b"\x00\xff\x00",
            b"aa",
            b"aaa",
            b"ab",
            b"aba",
            b"same",
            b"not-present",
            bytes(range(16)),
        ]
    )

    for data in documents:
        if not data:
            continue

        patterns.append(data)
        patterns.append(data[:1])
        patterns.append(data[-1:])

        if len(data) >= 2:
            patterns.append(data[:2])
            patterns.append(data[-2:])

        if len(data) >= 3:
            patterns.append(data[:3])
            patterns.append(data[-3:])

    return unique_patterns(patterns)


def fixture(
    name: str,
    documents: Sequence[bytes],
) -> dict:
    queries = [
        validate_query(documents, pattern)
        for pattern in fixture_patterns(documents)
    ]

    return {
        "fixture": name,
        "document_count": len(documents),
        "total_bytes": sum(
            len(document)
            for document in documents
        ),
        "query_count": len(queries),
        "queries": queries,
    }


def expect_failure(
    name: str,
    fn,
) -> dict:
    try:
        fn()
    except (
        LocateError,
        AssertionError,
        ValueError,
        IndexError,
    ) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise AssertionError(
        f"mutation unexpectedly accepted: {name}"
    )


def mutated_result(
    base: dict,
    **changes,
) -> dict:
    result = {
        key: (
            list(value)
            if isinstance(value, list)
            else value
        )
        for key, value in base.items()
    }

    result.update(changes)
    return result


def main() -> int:
    fixtures = [
        ("empty_corpus", []),
        ("empty_documents", [b"", b""]),
        ("single_ascii", [b"banana"]),
        ("repeated", [b"aaaaaa"]),
        ("periodic", [b"abababab"]),
        ("binary", [b"\x00\xff\x00\xff"]),
        ("multiple_documents", [b"ab", b"cd", b"banana"]),
        ("duplicates", [b"same", b"same"]),
        ("prefix_documents", [b"a", b"ab", b"abc"]),
        ("full_alphabet", [bytes(range(256))]),
        (
            "mixed",
            [
                b"",
                b"\x00abc\xff",
                b"abc",
                b"\xff\x00",
            ],
        ),
    ]

    positive = [
        fixture(name, documents)
        for name, documents in fixtures
    ]

    boundary_documents = [b"ab", b"cd"]
    boundary = full_locate(
        boundary_documents,
        b"bc",
    )

    if boundary["match_count"] != 0:
        raise AssertionError(
            "cross-document match returned"
        )

    duplicate_result = full_locate(
        [b"same", b"same"],
        b"same",
    )

    if duplicate_result["coordinates"] != [
        [0, 0],
        [1, 0],
    ]:
        raise AssertionError(
            "duplicate-document coordinates collapsed"
        )

    mutation_documents = [
        b"banana",
        b"banana",
        b"\x00\xff\x00",
    ]
    mutation_pattern = b"ana"

    base_full = full_locate(
        mutation_documents,
        mutation_pattern,
    )

    if base_full["match_count"] < 4:
        raise AssertionError(
            "mutation fixture needs multiple matches"
        )

    missing_coordinates = (
        base_full["coordinates"][:-1]
    )

    fabricated_coordinates = list(
        base_full["coordinates"]
    ) + [[0, 999]]

    duplicate_coordinates = list(
        base_full["coordinates"]
    ) + [base_full["coordinates"][0]]

    reversed_coordinates = list(
        reversed(base_full["coordinates"])
    )

    wrong_doc_coordinates = list(
        base_full["coordinates"]
    )
    wrong_doc_coordinates[0] = [99, 0]

    wrong_offset_coordinates = list(
        base_full["coordinates"]
    )
    wrong_offset_coordinates[0] = [0, 0]

    bounded_one = bounded_locate(
        mutation_documents,
        mutation_pattern,
        1,
    )

    wrong_prefix = mutated_result(
        bounded_one,
        coordinates=[
            base_full["coordinates"][-1]
        ],
    )

    mutations = [
        expect_failure(
            "empty_query_accepted",
            lambda: full_locate(
                mutation_documents,
                b"",
            ),
        ),
        expect_failure(
            "negative_max_offsets",
            lambda: bounded_locate(
                mutation_documents,
                mutation_pattern,
                -1,
            ),
        ),
        expect_failure(
            "missing_coordinate",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=missing_coordinates,
                    returned_count=len(
                        missing_coordinates
                    ),
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "fabricated_coordinate",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=fabricated_coordinates,
                    returned_count=len(
                        fabricated_coordinates
                    ),
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "duplicate_coordinate",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=duplicate_coordinates,
                    returned_count=len(
                        duplicate_coordinates
                    ),
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "wrong_doc_id",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=wrong_doc_coordinates,
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "wrong_doc_offset",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=wrong_offset_coordinates,
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "wrong_match_count",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    match_count=(
                        base_full["match_count"] + 1
                    ),
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "wrong_returned_count",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    returned_count=0,
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "wrong_bounded_flag",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    bounded_one,
                    bounded=False,
                ),
                max_offsets=1,
            ),
        ),
        expect_failure(
            "wrong_offsets_complete",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    bounded_one,
                    offsets_complete=True,
                ),
                max_offsets=1,
            ),
        ),
        expect_failure(
            "bounded_not_canonical_prefix",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                wrong_prefix,
                max_offsets=1,
            ),
        ),
        expect_failure(
            "unstable_coordinate_order",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    coordinates=reversed_coordinates,
                ),
                max_offsets=None,
            ),
        ),
        expect_failure(
            "byte_check_false",
            lambda: validate_result(
                mutation_documents,
                mutation_pattern,
                mutated_result(
                    base_full,
                    byte_check=False,
                ),
                max_offsets=None,
            ),
        ),
    ]

    total_queries = sum(
        item["query_count"]
        for item in positive
    )

    total_bounded_cases = sum(
        len(query["bounded_results"])
        for item in positive
        for query in item["queries"]
    )

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "positive_fixture_count": len(positive),
        "query_count": total_queries,
        "bounded_case_count": total_bounded_cases,
        "mutation_count": len(mutations),
        "canonical_coordinate": [
            "doc_id",
            "doc_offset",
        ],
        "canonical_order": [
            "doc_id_ascending",
            "doc_offset_ascending",
        ],
        "cross_document_match_rejected": True,
        "duplicate_documents_preserved": True,
        "byte_zero_supported": True,
        "byte_ff_supported": True,
        "boundary_result": boundary,
        "duplicate_result": duplicate_result,
        "positive_fixtures": positive,
        "mutations": mutations,
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
