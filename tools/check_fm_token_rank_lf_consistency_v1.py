#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence


PROOF_OBLIGATION = "P5"
FORMAT = "GLYPH_FM_TOKEN_RANK_LF_CONSISTENCY_V1"


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
            raise ValueError(f"byte outside range: {value}")
        return Token("byte", value)

    def key(self) -> tuple[int, int]:
        if self.kind == "sentinel":
            return (0, self.value)
        if self.kind == "byte":
            return (1, self.value)
        raise ValueError(f"unknown token kind: {self.kind}")

    def display(self) -> str:
        if self.kind == "sentinel":
            return f"VIRTUAL_SENTINEL({self.value})"
        return f"BYTE(0x{self.value:02X})"


@dataclass(frozen=True)
class Coordinate:
    doc_id: int
    doc_offset: int


class ValidationError(ValueError):
    pass


def suffixes(
    documents: Sequence[bytes],
) -> list[Coordinate]:
    rows: list[Coordinate] = []

    for doc_id, data in enumerate(documents):
        for offset in range(len(data) + 1):
            rows.append(Coordinate(doc_id, offset))

    def suffix_key(coord: Coordinate):
        data = documents[coord.doc_id]
        suffix = tuple(
            Token.byte(value).key()
            for value in data[coord.doc_offset:]
        )
        return suffix + (Token.sentinel(coord.doc_id).key(),)

    rows.sort(key=suffix_key)
    return rows


def build_bwt(
    documents: Sequence[bytes],
    sa: Sequence[Coordinate],
) -> list[Token]:
    result: list[Token] = []

    for coord in sa:
        if coord.doc_offset == 0:
            result.append(Token.sentinel(coord.doc_id))
        else:
            result.append(
                Token.byte(
                    documents[coord.doc_id][coord.doc_offset - 1]
                )
            )

    return result


def alphabet(tokens: Iterable[Token]) -> list[Token]:
    return sorted(set(tokens), key=Token.key)


def frequencies(
    tokens: Sequence[Token],
    symbols: Sequence[Token],
) -> dict[Token, int]:
    return {
        symbol: sum(token == symbol for token in tokens)
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
    tokens: Sequence[Token],
    symbol: Token,
    position: int,
) -> int:
    if not 0 <= position <= len(tokens):
        raise ValidationError(
            f"rank position outside range: {position}"
        )

    return sum(
        token == symbol
        for token in tokens[:position]
    )


def lf_mapping(
    tokens: Sequence[Token],
    c_array: dict[Token, int],
) -> list[int]:
    return [
        c_array[token] + rank(tokens, token, index)
        for index, token in enumerate(tokens)
    ]


def validate(
    documents: Sequence[bytes],
    *,
    override_bwt: Sequence[Token] | None = None,
    override_c: dict[Token, int] | None = None,
    inclusive_rank: bool = False,
) -> dict:
    sa = suffixes(documents)
    expected_bwt = build_bwt(documents, sa)
    bwt = list(expected_bwt if override_bwt is None else override_bwt)

    if bwt != expected_bwt:
        raise ValidationError("BWT token sequence mismatch")

    symbols = alphabet(bwt)
    freq = frequencies(bwt, symbols)
    expected_c = build_c(symbols, freq)
    c_array = expected_c if override_c is None else override_c

    if c_array != expected_c:
        raise ValidationError("C array mismatch")

    for symbol in symbols:
        if rank(bwt, symbol, 0) != 0:
            raise ValidationError("rank(symbol, 0) must be zero")

        if rank(bwt, symbol, len(bwt)) != freq[symbol]:
            raise ValidationError("terminal rank != frequency")

        previous = 0
        for position in range(len(bwt)):
            current = rank(bwt, symbol, position + 1)
            expected_increment = int(bwt[position] == symbol)

            if current != previous + expected_increment:
                raise ValidationError("rank recurrence mismatch")

            previous = current

    if inclusive_rank:
        lf = [
            c_array[token] + rank(bwt, token, index + 1)
            for index, token in enumerate(bwt)
        ]
    else:
        lf = lf_mapping(bwt, c_array)

    n = len(bwt)

    if any(target < 0 or target >= n for target in lf):
        raise ValidationError("LF target outside BWT range")

    if sorted(lf) != list(range(n)):
        raise ValidationError("LF is not a permutation")

    first_column = sorted(bwt, key=Token.key)

    for row, target in enumerate(lf):
        if first_column[target] != bwt[row]:
            raise ValidationError(
                "first-column token mismatch under LF"
            )

    sentinel_counts = {
        doc_id: sum(
            token == Token.sentinel(doc_id)
            for token in bwt
        )
        for doc_id, data in enumerate(documents)
    }

    for doc_id, count in sentinel_counts.items():
        if count != 1:
            raise ValidationError(
                f"sentinel frequency for doc {doc_id} is {count}"
            )

    return {
        "ok": True,
        "rows": len(sa),
        "bwt_length": len(bwt),
        "symbols": [
            symbol.display()
            for symbol in symbols
        ],
        "frequencies": {
            symbol.display(): freq[symbol]
            for symbol in symbols
        },
        "c_array": {
            symbol.display(): c_array[symbol]
            for symbol in symbols
        },
        "lf": lf,
    }


def expect_failure(
    name: str,
    fn,
) -> dict:
    try:
        fn()
    except (ValidationError, ValueError) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise AssertionError(
        f"mutation unexpectedly accepted: {name}"
    )


def fixture(
    name: str,
    documents: Sequence[bytes],
) -> dict:
    first = validate(documents)
    second = validate(documents)

    if first != second:
        raise AssertionError(
            f"non-deterministic P5 result: {name}"
        )

    return {
        "fixture": name,
        **first,
    }


def main() -> int:
    fixtures = [
        ("empty_corpus", []),
        ("single_ascii", [b"banana"]),
        ("single_zero", [b"\x00"]),
        ("single_ff", [b"\xff"]),
        ("zero_ff", [b"\x00\xff"]),
        ("alternating_zero_ff", [b"\x00\xff\x00\xff"]),
        ("repeated_bytes", [b"aaaaaa"]),
        ("periodic", [b"abababab"]),
        ("multiple_documents", [b"abc", b"\x00\xff", b"banana"]),
        ("equal_documents", [b"same", b"same"]),
        ("prefix_documents", [b"ab", b"abc", b"a"]),
        ("full_alphabet", [bytes(range(256))]),
        ("with_empty_document", [b"abc", b"", b"\x00"]),
    ]

    positive = [
        fixture(name, documents)
        for name, documents in fixtures
    ]

    mutation_docs = [b"\x00A", b"BC"]
    mutation_sa = suffixes(mutation_docs)
    canonical_bwt = build_bwt(mutation_docs, mutation_sa)

    zero_index = next(
        index
        for index, token in enumerate(canonical_bwt)
        if token == Token.byte(0x00)
    )

    sentinel_index = next(
        index
        for index, token in enumerate(canonical_bwt)
        if token.kind == "sentinel"
    )

    wrong_zero = list(canonical_bwt)
    wrong_zero[zero_index] = Token.sentinel(0)

    wrong_sentinel = list(canonical_bwt)
    wrong_sentinel[sentinel_index] = Token.byte(0x00)

    wrong_doc_sentinel = list(canonical_bwt)
    wrong_doc_sentinel[sentinel_index] = Token.sentinel(99)

    symbols = alphabet(canonical_bwt)
    freq = frequencies(canonical_bwt, symbols)
    canonical_c = build_c(symbols, freq)

    wrong_c = dict(canonical_c)
    first_symbol = symbols[0]
    wrong_c[first_symbol] += 1

    mutations = [
        expect_failure(
            "zero_byte_replaced_by_sentinel",
            lambda: validate(
                mutation_docs,
                override_bwt=wrong_zero,
            ),
        ),
        expect_failure(
            "sentinel_replaced_by_zero_byte",
            lambda: validate(
                mutation_docs,
                override_bwt=wrong_sentinel,
            ),
        ),
        expect_failure(
            "wrong_sentinel_doc_id",
            lambda: validate(
                mutation_docs,
                override_bwt=wrong_doc_sentinel,
            ),
        ),
        expect_failure(
            "incorrect_c_array",
            lambda: validate(
                mutation_docs,
                override_c=wrong_c,
            ),
        ),
        expect_failure(
            "inclusive_rank_in_lf",
            lambda: validate(
                mutation_docs,
                inclusive_rank=True,
            ),
        ),
    ]

    zero_token = Token.byte(0)
    sentinel_zero = Token.sentinel(0)
    ff_token = Token.byte(255)

    if zero_token == sentinel_zero:
        raise AssertionError(
            "BYTE(0x00) aliases virtual sentinel"
        )

    if not sentinel_zero.key() < zero_token.key():
        raise AssertionError(
            "virtual sentinel must sort before BYTE(0x00)"
        )

    if not Token.byte(254).key() < ff_token.key():
        raise AssertionError(
            "BYTE(0xFF) ordering invalid"
        )

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "positive_fixture_count": len(positive),
        "mutation_count": len(mutations),
        "byte_zero_distinct_from_sentinel": True,
        "byte_ff_preserved": True,
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
