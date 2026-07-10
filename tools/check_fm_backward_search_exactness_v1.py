#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence


PROOF_OBLIGATION = "P6"
FORMAT = "GLYPH_FM_BACKWARD_SEARCH_EXACTNESS_V1"


class SearchError(ValueError):
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
            raise SearchError(f"byte outside range: {value}")
        return Token("byte", value)

    def key(self) -> tuple[int, int]:
        if self.kind == "sentinel":
            return (0, self.value)
        if self.kind == "byte":
            return (1, self.value)
        raise SearchError(f"unknown token kind: {self.kind}")

    def display(self) -> str:
        if self.kind == "sentinel":
            return f"VIRTUAL_SENTINEL({self.value})"
        return f"BYTE(0x{self.value:02X})"


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    doc_offset: int


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


def symbols_for_search(
    bwt: Sequence[Token],
) -> list[Token]:
    return sorted(set(bwt), key=Token.key)


def frequencies(
    bwt: Sequence[Token],
    symbols: Sequence[Token],
) -> dict[Token, int]:
    return {
        symbol: sum(token == symbol for token in bwt)
        for symbol in symbols
    }


def build_c(
    symbols: Sequence[Token],
    freq: dict[Token, int],
) -> dict[Token, int]:
    total = 0
    result: dict[Token, int] = {}

    for symbol in symbols:
        result[symbol] = total
        total += freq[symbol]

    return result


def rank(
    bwt: Sequence[Token],
    symbol: Token,
    position: int,
) -> int:
    if not 0 <= position <= len(bwt):
        raise SearchError(
            f"rank position outside range: {position}"
        )

    return sum(
        token == symbol
        for token in bwt[:position]
    )


def c_for_byte(
    symbol: Token,
    symbols: Sequence[Token],
    freq: dict[Token, int],
) -> int:
    return sum(
        freq[token]
        for token in symbols
        if token.key() < symbol.key()
    )


def backward_search(
    bwt: Sequence[Token],
    pattern: bytes,
    *,
    inclusive_rank: bool = False,
    process_forward: bool = False,
    initial_l: int = 0,
    initial_r: int | None = None,
) -> tuple[int, int]:
    if not pattern:
        raise SearchError("EMPTY_PATTERN")

    symbols = symbols_for_search(bwt)
    freq = frequencies(bwt, symbols)
    c_array = build_c(symbols, freq)

    if initial_r is None:
        initial_r = len(bwt)

    l = initial_l
    r = initial_r

    sequence = pattern if process_forward else reversed(pattern)

    for value in sequence:
        symbol = Token.byte(value)

        c_value = c_array.get(
            symbol,
            c_for_byte(symbol, symbols, freq),
        )

        if inclusive_rank:
            l = c_value + rank(
                bwt,
                symbol,
                min(l + 1, len(bwt)),
            )
            r = c_value + rank(
                bwt,
                symbol,
                min(r + 1, len(bwt)),
            )
        else:
            l = c_value + rank(bwt, symbol, l)
            r = c_value + rank(bwt, symbol, r)

        if l >= r:
            return (l, l)

    return (l, r)


def naive_occurrences(
    documents: Sequence[bytes],
    pattern: bytes,
) -> list[Coordinate]:
    if not pattern:
        raise SearchError("EMPTY_PATTERN")

    results: list[Coordinate] = []

    for doc_id, data in enumerate(documents):
        if len(pattern) > len(data):
            continue

        for offset in range(
            0,
            len(data) - len(pattern) + 1,
        ):
            if data[
                offset : offset + len(pattern)
            ] == pattern:
                results.append(
                    Coordinate(doc_id, offset)
                )

    return sorted(results)


def concatenated_naive_occurrences(
    documents: Sequence[bytes],
    pattern: bytes,
) -> int:
    joined = b"".join(documents)

    if not pattern:
        raise SearchError("EMPTY_PATTERN")

    return sum(
        joined[offset : offset + len(pattern)]
        == pattern
        for offset in range(
            max(0, len(joined) - len(pattern) + 1)
        )
    )


def row_has_prefix(
    documents: Sequence[bytes],
    coordinate: Coordinate,
    pattern: bytes,
) -> bool:
    data = documents[coordinate.doc_id]

    return (
        coordinate.doc_offset + len(pattern)
        <= len(data)
        and
        data[
            coordinate.doc_offset :
            coordinate.doc_offset + len(pattern)
        ]
        == pattern
    )


def validate_query(
    documents: Sequence[bytes],
    pattern: bytes,
) -> dict:
    sa = build_sa(documents)
    bwt = build_bwt(documents, sa)

    l, r = backward_search(bwt, pattern)

    if not 0 <= l <= r <= len(sa):
        raise AssertionError(
            f"invalid FM interval: {(l, r)}"
        )

    fm_coordinates = sorted(sa[l:r])
    naive_coordinates = naive_occurrences(
        documents,
        pattern,
    )

    if fm_coordinates != naive_coordinates:
        raise AssertionError(
            {
                "pattern_hex": pattern.hex(),
                "interval": [l, r],
                "fm_coordinates": [
                    [item.doc_id, item.doc_offset]
                    for item in fm_coordinates
                ],
                "naive_coordinates": [
                    [item.doc_id, item.doc_offset]
                    for item in naive_coordinates
                ],
            }
        )

    for row in range(l, r):
        if not row_has_prefix(
            documents,
            sa[row],
            pattern,
        ):
            raise AssertionError(
                f"non-matching row inside interval: {row}"
            )

    matching_rows = [
        row
        for row, coordinate in enumerate(sa)
        if row_has_prefix(
            documents,
            coordinate,
            pattern,
        )
    ]

    if matching_rows:
        expected_rows = list(
            range(
                matching_rows[0],
                matching_rows[-1] + 1,
            )
        )

        if matching_rows != expected_rows:
            raise AssertionError(
                "matching SA rows are not contiguous"
            )

        if (l, r) != (
            matching_rows[0],
            matching_rows[-1] + 1,
        ):
            raise AssertionError(
                "FM interval does not equal matching rows"
            )
    elif l != r:
        raise AssertionError(
            "empty match set must have empty interval"
        )

    for row, coordinate in enumerate(sa):
        is_match = row_has_prefix(
            documents,
            coordinate,
            pattern,
        )

        inside = l <= row < r

        if is_match != inside:
            raise AssertionError(
                "interval inclusion/exclusion mismatch"
            )

    return {
        "pattern_hex": pattern.hex(),
        "pattern_length": len(pattern),
        "interval": [l, r],
        "match_count": r - l,
        "coordinates": [
            [item.doc_id, item.doc_offset]
            for item in fm_coordinates
        ],
    }


def unique_queries(
    queries: Sequence[bytes],
) -> list[bytes]:
    seen: set[bytes] = set()
    result: list[bytes] = []

    for query in queries:
        if query not in seen:
            seen.add(query)
            result.append(query)

    return result


def fixture_queries(
    documents: Sequence[bytes],
) -> list[bytes]:
    queries: list[bytes] = [
        bytes([value])
        for value in range(256)
    ]

    queries.extend(
        [
            b"\x00\x00",
            b"\x00\xff",
            b"\xff\x00",
            b"\xff\xff",
            b"aa",
            b"aaa",
            b"ab",
            b"ba",
            b"aba",
            b"banana",
            b"not-present",
            bytes(range(16)),
            bytes(range(255, 239, -1)),
        ]
    )

    for data in documents:
        if not data:
            continue

        queries.append(data)
        queries.append(data[:1])
        queries.append(data[-1:])

        if len(data) >= 2:
            queries.append(data[:2])
            queries.append(data[-2:])

        if len(data) >= 3:
            middle = len(data) // 2
            start = max(
                0,
                min(
                    middle,
                    len(data) - 3,
                ),
            )
            queries.append(data[start : start + 3])

    return unique_queries(queries)


def fixture(
    name: str,
    documents: Sequence[bytes],
) -> dict:
    query_results = [
        validate_query(documents, pattern)
        for pattern in fixture_queries(documents)
    ]

    return {
        "fixture": name,
        "document_count": len(documents),
        "total_bytes": sum(map(len, documents)),
        "query_count": len(query_results),
        "queries": query_results,
    }


def expect_failure(
    name: str,
    fn,
) -> dict:
    try:
        fn()
    except (
        AssertionError,
        SearchError,
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


def main() -> int:
    fixtures = [
        ("empty_corpus", []),
        ("empty_documents", [b"", b""]),
        ("ascii", [b"banana bandana"]),
        ("repeated", [b"aaaaaa"]),
        ("periodic", [b"abababab"]),
        ("binary", [b"\x00\xff\x00\xff"]),
        ("multiple_documents", [b"ab", b"cd", b"banana"]),
        ("duplicate_documents", [b"same", b"same"]),
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
    boundary_pattern = b"bc"

    boundary_result = validate_query(
        boundary_documents,
        boundary_pattern,
    )

    if boundary_result["match_count"] != 0:
        raise AssertionError(
            "cross-document pattern unexpectedly matched"
        )

    concatenated_count = concatenated_naive_occurrences(
        boundary_documents,
        boundary_pattern,
    )

    if concatenated_count != 1:
        raise AssertionError(
            "boundary-control concatenated oracle must see one match"
        )

    empty_query_rejected = expect_failure(
        "empty_query",
        lambda: validate_query([b"abc"], b""),
    )

    mutation_documents = [b"banana", b"\x00\xff"]
    mutation_sa = build_sa(mutation_documents)
    mutation_bwt = build_bwt(
        mutation_documents,
        mutation_sa,
    )

    canonical_interval = backward_search(
        mutation_bwt,
        b"ana",
    )

    mutations = [
        empty_query_rejected,
        expect_failure(
            "inclusive_rank_backward_search",
            lambda: (
                backward_search(
                    mutation_bwt,
                    b"ana",
                    inclusive_rank=True,
                )
                == canonical_interval
            )
            or (_ for _ in ()).throw(
                AssertionError(
                    "inclusive rank changed interval"
                )
            ),
        ),
        expect_failure(
            "pattern_processed_forward",
            lambda: (
                backward_search(
                    mutation_bwt,
                    b"ana",
                    process_forward=True,
                )
                == canonical_interval
            )
            or (_ for _ in ()).throw(
                AssertionError(
                    "forward processing changed interval"
                )
            ),
        ),
        expect_failure(
            "wrong_initial_right",
            lambda: (
                backward_search(
                    mutation_bwt,
                    b"ana",
                    initial_r=max(
                        0,
                        len(mutation_bwt) - 1,
                    ),
                )
                == canonical_interval
            )
            or (_ for _ in ()).throw(
                AssertionError(
                    "wrong initial interval changed result"
                )
            ),
        ),
        expect_failure(
            "cross_document_naive_oracle",
            lambda: (
                concatenated_naive_occurrences(
                    boundary_documents,
                    boundary_pattern,
                )
                ==
                len(
                    naive_occurrences(
                        boundary_documents,
                        boundary_pattern,
                    )
                )
            )
            or (_ for _ in ()).throw(
                AssertionError(
                    "concatenated oracle disagrees with document-local oracle"
                )
            ),
        ),
    ]

    total_queries = sum(
        item["query_count"]
        for item in positive
    )

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "positive_fixture_count": len(positive),
        "query_count": total_queries,
        "mutation_count": len(mutations),
        "all_single_byte_queries_tested": True,
        "empty_query_rejected": True,
        "cross_document_match_rejected": True,
        "cross_document_control_concatenated_count":
            concatenated_count,
        "boundary_result": boundary_result,
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
