#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


PROOF_OBLIGATION = "P8"
FORMAT = "GLYPH_BINARY_SAFE_QUERY_TRANSPORT_V1"


class TransportError(ValueError):
    pass


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_query_hex(data: bytes) -> str:
    if not data:
        raise TransportError("EMPTY_QUERY")

    return data.hex()


def decode_query_hex(value: Any) -> bytes:
    if not isinstance(value, str):
        raise TransportError("query_hex must be a string")

    if value == "":
        raise TransportError("EMPTY_QUERY")

    if value != value.lower():
        raise TransportError("query_hex must be lowercase")

    if value.startswith("0x"):
        raise TransportError("query_hex must not use 0x prefix")

    if any(character.isspace() for character in value):
        raise TransportError("query_hex must not contain whitespace")

    if len(value) % 2 != 0:
        raise TransportError("query_hex must have even length")

    allowed = set("0123456789abcdef")

    if any(character not in allowed for character in value):
        raise TransportError("query_hex contains non-hex character")

    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise TransportError("invalid query_hex") from error

    if not decoded:
        raise TransportError("EMPTY_QUERY")

    if decoded.hex() != value:
        raise TransportError("query_hex is not canonical")

    return decoded


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    doc_offset: int

    def to_json(self) -> list[int]:
        return [self.doc_id, self.doc_offset]


def validate_coordinate(
    documents: list[bytes],
    query: bytes,
    coordinate: Coordinate,
) -> bool:
    if coordinate.doc_id < 0:
        return False

    if coordinate.doc_id >= len(documents):
        return False

    if coordinate.doc_offset < 0:
        return False

    document = documents[coordinate.doc_id]
    end = coordinate.doc_offset + len(query)

    if end > len(document):
        return False

    candidate = document[coordinate.doc_offset:end]

    return (
        len(candidate) == len(query)
        and candidate == query
    )


def naive_coordinates(
    documents: list[bytes],
    query: bytes,
) -> list[Coordinate]:
    if not query:
        raise TransportError("EMPTY_QUERY")

    result: list[Coordinate] = []

    for doc_id, document in enumerate(documents):
        if len(query) > len(document):
            continue

        for offset in range(
            len(document) - len(query) + 1
        ):
            candidate = document[
                offset : offset + len(query)
            ]

            if (
                len(candidate) == len(query)
                and candidate == query
            ):
                result.append(
                    Coordinate(doc_id, offset)
                )

    return sorted(result)


def make_artifact(
    documents: list[bytes],
    query: bytes,
    *,
    query_display: str | None = None,
) -> dict:
    query_hex = encode_query_hex(query)
    coordinates = naive_coordinates(documents, query)

    artifact = {
        "artifact_version":
            "GLYPH_BINARY_QUERY_ARTIFACT_V1",
        "query_hex": query_hex,
        "query_length_bytes": len(query),
        "query_sha256": sha256_hex(query),
        "match_count": len(coordinates),
        "coordinates": [
            coordinate.to_json()
            for coordinate in coordinates
        ],
        "byte_check": True,
    }

    if query_display is not None:
        artifact["query_display"] = query_display

    return artifact


def parse_coordinate(value: Any) -> Coordinate:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(
            isinstance(item, int)
            and not isinstance(item, bool)
            for item in value
        )
    ):
        raise TransportError("invalid coordinate")

    return Coordinate(value[0], value[1])


def replay_artifact(
    documents: list[bytes],
    artifact: dict,
) -> dict:
    query = decode_query_hex(
        artifact.get("query_hex")
    )

    declared_length = artifact.get(
        "query_length_bytes"
    )

    if (
        not isinstance(declared_length, int)
        or isinstance(declared_length, bool)
        or declared_length <= 0
    ):
        raise TransportError(
            "invalid query_length_bytes"
        )

    if declared_length != len(query):
        raise TransportError(
            "query_length_bytes mismatch"
        )

    declared_hash = artifact.get("query_sha256")

    if declared_hash != sha256_hex(query):
        raise TransportError("query_sha256 mismatch")

    coordinates_raw = artifact.get("coordinates")

    if not isinstance(coordinates_raw, list):
        raise TransportError(
            "coordinates must be a list"
        )

    coordinates = [
        parse_coordinate(value)
        for value in coordinates_raw
    ]

    if coordinates != sorted(coordinates):
        raise TransportError(
            "coordinates not canonical"
        )

    if len(coordinates) != len(set(coordinates)):
        raise TransportError(
            "duplicate coordinates"
        )

    for coordinate in coordinates:
        if not validate_coordinate(
            documents,
            query,
            coordinate,
        ):
            raise TransportError(
                "coordinate byte check failed"
            )

    expected = naive_coordinates(
        documents,
        query,
    )

    if coordinates != expected:
        raise TransportError(
            "coordinates differ from byte oracle"
        )

    declared_count = artifact.get("match_count")

    if declared_count != len(expected):
        raise TransportError("match_count mismatch")

    if artifact.get("byte_check") is not True:
        raise TransportError(
            "byte_check must be true"
        )

    return {
        "ok": True,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256": sha256_hex(query),
        "match_count": len(expected),
        "coordinates": [
            coordinate.to_json()
            for coordinate in expected
        ],
        "byte_check": True,
    }


def json_round_trip(value: dict) -> dict:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    decoded = json.loads(encoded)

    if not isinstance(decoded, dict):
        raise TransportError(
            "JSON round trip did not return object"
        )

    return decoded


def clone_artifact(
    artifact: dict,
    **changes: Any,
) -> dict:
    cloned = json.loads(json.dumps(artifact))
    cloned.update(changes)
    return cloned


def expect_failure(name: str, fn) -> dict:
    try:
        fn()
    except (
        TransportError,
        AssertionError,
        ValueError,
        TypeError,
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


def fixture_queries() -> list[bytes]:
    return [
        b"\x00",
        b"\xff",
        b"\x00\xff",
        b"\xff\x00",
        b"\x00\x00",
        b"\xff\xff",
        b"A\x00B",
        b"\x00\xff\x00A\xff",
        b"\x80\x81\xfe\xff",
        b"\n\r\x00\xff",
        bytes(range(256)),
    ]


def main() -> int:
    all_bytes = bytes(range(256))

    documents = [
        b"",
        b"prefix\x00suffix",
        b"\xff\x00\xff\x00",
        b"A\x00B",
        all_bytes,
        b"\x00\xff\x00A\xff",
        b"\x80\x81\xfe\xff",
        b"\n\r\x00\xff",
    ]

    fixtures = []

    for query in fixture_queries():
        artifact = make_artifact(
            documents,
            query,
            query_display="non-authoritative",
        )

        round_tripped = json_round_trip(artifact)
        replay = replay_artifact(
            documents,
            round_tripped,
        )

        if replay["query_hex"] != query.hex():
            raise AssertionError(
                "query changed during round trip"
            )

        if replay["query_length_bytes"] != len(query):
            raise AssertionError(
                "query length changed"
            )

        if replay["query_sha256"] != sha256_hex(query):
            raise AssertionError(
                "query hash changed"
            )

        fixtures.append(
            {
                "query_hex": query.hex(),
                "query_length_bytes": len(query),
                "query_sha256": sha256_hex(query),
                "match_count": replay["match_count"],
                "json_round_trip_ok": True,
                "replay_ok": True,
            }
        )

    mutation_documents = [
        b"A\x00B",
        b"\x00\xff\x00A\xff",
    ]
    mutation_query = b"A\x00B"

    base = make_artifact(
        mutation_documents,
        mutation_query,
        query_display="A",
    )

    changed_source = list(mutation_documents)
    changed_source[0] = b"A\x00C"

    truncated_hex = mutation_query.split(
        b"\x00",
        1,
    )[0].hex()

    mutations = [
        expect_failure(
            "uppercase_hex",
            lambda: decode_query_hex("00FF"),
        ),
        expect_failure(
            "odd_length_hex",
            lambda: decode_query_hex("0"),
        ),
        expect_failure(
            "whitespace_hex",
            lambda: decode_query_hex("00 ff"),
        ),
        expect_failure(
            "hex_prefix",
            lambda: decode_query_hex("0x00"),
        ),
        expect_failure(
            "invalid_hex_digit",
            lambda: decode_query_hex("00gg"),
        ),
        expect_failure(
            "empty_hex",
            lambda: decode_query_hex(""),
        ),
        expect_failure(
            "wrong_query_length",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_length_bytes=2,
                ),
            ),
        ),
        expect_failure(
            "wrong_query_sha256",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_sha256="0" * 64,
                ),
            ),
        ),
        expect_failure(
            "nul_truncated_query",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_hex=truncated_hex,
                ),
            ),
        ),
        expect_failure(
            "hash_of_hex_text",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_sha256=sha256_hex(
                        base["query_hex"].encode(
                            "ascii"
                        )
                    ),
                ),
            ),
        ),
        expect_failure(
            "changed_coordinate",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    coordinates=[[0, 1]],
                ),
            ),
        ),
        expect_failure(
            "changed_source_byte",
            lambda: replay_artifact(
                changed_source,
                base,
            ),
        ),
        expect_failure(
            "false_byte_check",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    byte_check=False,
                ),
            ),
        ),
        expect_failure(
            "display_text_used_as_query",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_hex=base[
                        "query_display"
                    ].encode("utf-8").hex(),
                ),
            ),
        ),
        expect_failure(
            "trailing_byte_omitted",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_hex=b"A\x00".hex(),
                ),
            ),
        ),
        expect_failure(
            "zero_byte_removed",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_hex=b"AB".hex(),
                    query_length_bytes=2,
                    query_sha256=sha256_hex(
                        b"AB"
                    ),
                ),
            ),
        ),
        expect_failure(
            "ff_byte_replaced",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    make_artifact(
                        mutation_documents,
                        b"\x00\xff",
                    ),
                    query_hex=b"\x00\xfe".hex(),
                ),
            ),
        ),
        expect_failure(
            "noncanonical_json_query_hex",
            lambda: replay_artifact(
                mutation_documents,
                clone_artifact(
                    base,
                    query_hex="410042\n",
                ),
            ),
        ),
    ]

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "authoritative_query_field": "query_hex",
        "query_hash_preimage": "decoded_query_bytes",
        "empty_query_rejected": True,
        "all_256_bytes_round_trip": True,
        "embedded_nul_round_trip": True,
        "byte_ff_round_trip": True,
        "json_round_trip_exact": True,
        "explicit_length_byte_check": True,
        "fixture_count": len(fixtures),
        "mutation_count": len(mutations),
        "fixtures": fixtures,
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
