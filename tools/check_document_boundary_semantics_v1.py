#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence


PROOF_OBLIGATION = "P9"
FORMAT = "GLYPH_DOCUMENT_BOUNDARY_SEMANTICS_V1"
BOUNDARY_POLICY = "DOCUMENT_LOCAL_MATCHES_ONLY_V1"


class BoundaryError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    doc_offset: int

    def as_json(self) -> list[int]:
        return [self.doc_id, self.doc_offset]


@dataclass(frozen=True)
class Boundary:
    doc_id: int
    global_start: int
    byte_length: int
    global_end: int


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_identity(documents: Sequence[bytes]) -> str:
    h = hashlib.sha256()
    h.update(b"GLYPH_CORPUS_IDENTITY_P9_V1\x00")
    h.update(len(documents).to_bytes(8, "big"))

    for doc_id, document in enumerate(documents):
        h.update(doc_id.to_bytes(8, "big"))
        h.update(len(document).to_bytes(8, "big"))
        h.update(hashlib.sha256(document).digest())

    return h.hexdigest()


def boundary_table(
    documents: Sequence[bytes],
) -> list[Boundary]:
    result: list[Boundary] = []
    cursor = 0

    for doc_id, document in enumerate(documents):
        result.append(
            Boundary(
                doc_id=doc_id,
                global_start=cursor,
                byte_length=len(document),
                global_end=cursor + len(document),
            )
        )
        cursor += len(document)

    return result


def physical_concatenation(
    documents: Sequence[bytes],
) -> bytes:
    return b"".join(documents)


def naive_document_local_coordinates(
    documents: Sequence[bytes],
    query: bytes,
) -> list[Coordinate]:
    if not query:
        raise BoundaryError("EMPTY_QUERY")

    result: list[Coordinate] = []

    for doc_id, document in enumerate(documents):
        if len(query) > len(document):
            continue

        for offset in range(
            len(document) - len(query) + 1
        ):
            if (
                document[offset:offset + len(query)]
                == query
            ):
                result.append(
                    Coordinate(doc_id, offset)
                )

    return sorted(result)


def physical_global_offsets(
    documents: Sequence[bytes],
    query: bytes,
) -> list[int]:
    if not query:
        raise BoundaryError("EMPTY_QUERY")

    corpus = physical_concatenation(documents)
    result: list[int] = []

    if len(query) > len(corpus):
        return result

    for offset in range(
        len(corpus) - len(query) + 1
    ):
        if corpus[offset:offset + len(query)] == query:
            result.append(offset)

    return result


def map_global_span(
    boundaries: Sequence[Boundary],
    global_offset: int,
    query_length: int,
) -> Coordinate:
    if global_offset < 0:
        raise BoundaryError("negative global offset")

    if query_length <= 0:
        raise BoundaryError("invalid query length")

    global_end = global_offset + query_length

    for boundary in boundaries:
        if (
            global_offset >= boundary.global_start
            and global_offset < boundary.global_end
            and global_end <= boundary.global_end
        ):
            return Coordinate(
                boundary.doc_id,
                global_offset - boundary.global_start,
            )

    raise BoundaryError(
        "global span crosses or lies outside document boundary"
    )


def safe_coordinates_from_global_offsets(
    documents: Sequence[bytes],
    query: bytes,
    offsets: Sequence[int],
) -> list[Coordinate]:
    boundaries = boundary_table(documents)
    result: list[Coordinate] = []

    for offset in offsets:
        try:
            coordinate = map_global_span(
                boundaries,
                offset,
                len(query),
            )
        except BoundaryError:
            continue

        result.append(coordinate)

    return sorted(result)


def validate_coordinate(
    documents: Sequence[bytes],
    query: bytes,
    coordinate: Coordinate,
) -> None:
    if coordinate.doc_id < 0:
        raise BoundaryError("negative doc_id")

    if coordinate.doc_id >= len(documents):
        raise BoundaryError("doc_id outside corpus")

    if coordinate.doc_offset < 0:
        raise BoundaryError("negative doc_offset")

    document = documents[coordinate.doc_id]
    end = coordinate.doc_offset + len(query)

    if coordinate.doc_offset >= len(document):
        raise BoundaryError(
            "coordinate starts at or after document end"
        )

    if end > len(document):
        raise BoundaryError(
            "coordinate crosses document boundary"
        )

    candidate = document[
        coordinate.doc_offset:end
    ]

    if len(candidate) != len(query):
        raise BoundaryError(
            "candidate length mismatch"
        )

    if candidate != query:
        raise BoundaryError(
            "coordinate bytes do not match query"
        )


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
        raise BoundaryError(
            "invalid coordinate representation"
        )

    return Coordinate(value[0], value[1])


def make_artifact(
    documents: Sequence[bytes],
    query: bytes,
    *,
    max_offsets: int | None = None,
) -> dict:
    if not query:
        raise BoundaryError("EMPTY_QUERY")

    if max_offsets is not None and max_offsets < 0:
        raise BoundaryError("negative max_offsets")

    full_coordinates = (
        naive_document_local_coordinates(
            documents,
            query,
        )
    )

    if max_offsets is None:
        returned = full_coordinates
    else:
        returned = full_coordinates[:max_offsets]

    bounded = len(returned) < len(full_coordinates)

    artifact = {
        "artifact_version":
            "GLYPH_DOCUMENT_LOCAL_EVIDENCE_V1",
        "corpus_id": corpus_identity(documents),
        "document_count": len(documents),
        "document_lengths": [
            len(document)
            for document in documents
        ],
        "document_boundary_policy": BOUNDARY_POLICY,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256": sha256_hex(query),
        "match_count": len(full_coordinates),
        "coordinates": [
            coordinate.as_json()
            for coordinate in returned
        ],
        "returned_count": len(returned),
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
    }

    if max_offsets is not None:
        artifact["max_offsets"] = max_offsets

    return artifact


def decode_query(artifact: dict) -> bytes:
    query_hex = artifact.get("query_hex")

    if not isinstance(query_hex, str):
        raise BoundaryError("query_hex missing")

    if not query_hex:
        raise BoundaryError("EMPTY_QUERY")

    if query_hex != query_hex.lower():
        raise BoundaryError("noncanonical query_hex")

    if len(query_hex) % 2:
        raise BoundaryError("odd query_hex length")

    try:
        query = bytes.fromhex(query_hex)
    except ValueError as error:
        raise BoundaryError("invalid query_hex") from error

    if query.hex() != query_hex:
        raise BoundaryError("noncanonical query_hex")

    if not query:
        raise BoundaryError("EMPTY_QUERY")

    return query


def replay_artifact(
    documents: Sequence[bytes],
    artifact: dict,
) -> dict:
    if (
        artifact.get("document_boundary_policy")
        != BOUNDARY_POLICY
    ):
        raise BoundaryError(
            "document boundary policy mismatch"
        )

    if artifact.get("corpus_id") != corpus_identity(
        documents
    ):
        raise BoundaryError("corpus identity mismatch")

    if artifact.get("document_count") != len(
        documents
    ):
        raise BoundaryError("document count mismatch")

    expected_lengths = [
        len(document)
        for document in documents
    ]

    if (
        artifact.get("document_lengths")
        != expected_lengths
    ):
        raise BoundaryError(
            "document lengths mismatch"
        )

    query = decode_query(artifact)

    if (
        artifact.get("query_length_bytes")
        != len(query)
    ):
        raise BoundaryError(
            "query length mismatch"
        )

    if (
        artifact.get("query_sha256")
        != sha256_hex(query)
    ):
        raise BoundaryError(
            "query hash mismatch"
        )

    expected_full = (
        naive_document_local_coordinates(
            documents,
            query,
        )
    )

    if (
        artifact.get("match_count")
        != len(expected_full)
    ):
        raise BoundaryError(
            "match count differs from document-local oracle"
        )

    raw_coordinates = artifact.get("coordinates")

    if not isinstance(raw_coordinates, list):
        raise BoundaryError(
            "coordinates must be a list"
        )

    coordinates = [
        parse_coordinate(value)
        for value in raw_coordinates
    ]

    if coordinates != sorted(coordinates):
        raise BoundaryError(
            "coordinates not canonically ordered"
        )

    if len(coordinates) != len(set(coordinates)):
        raise BoundaryError(
            "duplicate coordinates"
        )

    for coordinate in coordinates:
        validate_coordinate(
            documents,
            query,
            coordinate,
        )

    if artifact.get("byte_check") is not True:
        raise BoundaryError(
            "byte_check must be true"
        )

    max_offsets = artifact.get("max_offsets")

    if max_offsets is None:
        expected_returned = expected_full
    else:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
        ):
            raise BoundaryError(
                "invalid max_offsets"
            )

        expected_returned = expected_full[:max_offsets]

    if coordinates != expected_returned:
        raise BoundaryError(
            "coordinates differ from document-local oracle"
        )

    expected_returned_count = len(expected_returned)
    expected_bounded = (
        expected_returned_count
        < len(expected_full)
    )

    if (
        artifact.get("returned_count")
        != expected_returned_count
    ):
        raise BoundaryError(
            "returned_count mismatch"
        )

    if artifact.get("bounded") is not expected_bounded:
        raise BoundaryError(
            "bounded flag mismatch"
        )

    if (
        artifact.get("offsets_complete")
        is not (not expected_bounded)
    ):
        raise BoundaryError(
            "offsets_complete mismatch"
        )

    return {
        "ok": True,
        "document_boundary_policy": BOUNDARY_POLICY,
        "match_count": len(expected_full),
        "coordinates": [
            coordinate.as_json()
            for coordinate in expected_returned
        ],
        "returned_count": expected_returned_count,
        "bounded": expected_bounded,
        "offsets_complete": not expected_bounded,
        "byte_check": True,
    }


def clone(value: dict, **changes: Any) -> dict:
    copied = json.loads(json.dumps(value))
    copied.update(changes)
    return copied


def expect_failure(name: str, fn) -> dict:
    try:
        fn()
    except (
        BoundaryError,
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


def fixture(
    name: str,
    documents: Sequence[bytes],
    queries: Sequence[bytes],
) -> dict:
    results = []

    for query in queries:
        expected = (
            naive_document_local_coordinates(
                documents,
                query,
            )
        )

        physical = physical_global_offsets(
            documents,
            query,
        )

        safe = safe_coordinates_from_global_offsets(
            documents,
            query,
            physical,
        )

        if safe != expected:
            raise AssertionError(
                f"boundary filtering mismatch in {name}"
            )

        artifact = make_artifact(
            documents,
            query,
        )
        replay = replay_artifact(
            documents,
            artifact,
        )

        if replay["match_count"] != len(expected):
            raise AssertionError(
                f"count mismatch in {name}"
            )

        bounded_limits = {
            0,
            1,
            len(expected),
            len(expected) + 1,
        }

        if expected:
            bounded_limits.add(
                len(expected) - 1
            )

        bounded_results = []

        for limit in sorted(bounded_limits):
            bounded = make_artifact(
                documents,
                query,
                max_offsets=limit,
            )

            replayed = replay_artifact(
                documents,
                bounded,
            )

            bounded_results.append(
                {
                    "max_offsets": limit,
                    "returned_count":
                        replayed["returned_count"],
                    "bounded":
                        replayed["bounded"],
                    "offsets_complete":
                        replayed[
                            "offsets_complete"
                        ],
                }
            )

        results.append(
            {
                "query_hex": query.hex(),
                "document_local_count":
                    len(expected),
                "document_local_coordinates": [
                    coordinate.as_json()
                    for coordinate in expected
                ],
                "physical_match_count":
                    len(physical),
                "physical_global_offsets":
                    physical,
                "safe_mapped_coordinates": [
                    coordinate.as_json()
                    for coordinate in safe
                ],
                "replay_ok": True,
                "bounded_results":
                    bounded_results,
            }
        )

    return {
        "fixture": name,
        "document_count": len(documents),
        "document_lengths": [
            len(document)
            for document in documents
        ],
        "query_count": len(results),
        "results": results,
    }


def main() -> int:
    full_alphabet = bytes(range(256))

    fixtures = [
        fixture(
            "ascii_cross_boundary",
            [b"ab", b"cd"],
            [b"bc", b"ab", b"cd"],
        ),
        fixture(
            "nul_ff_boundary",
            [b"\x00", b"\xff"],
            [
                b"\x00\xff",
                b"\x00",
                b"\xff",
            ],
        ),
        fixture(
            "ff_nul_boundary",
            [b"\xff", b"\x00"],
            [
                b"\xff\x00",
                b"\xff",
                b"\x00",
            ],
        ),
        fixture(
            "empty_middle_document",
            [b"ab", b"", b"cd"],
            [b"bc", b"ab", b"cd"],
        ),
        fixture(
            "duplicate_documents",
            [b"same", b"same"],
            [b"same", b"am"],
        ),
        fixture(
            "prefix_documents",
            [b"a", b"ab", b"abc"],
            [b"ab", b"abc", b"aa"],
        ),
        fixture(
            "one_byte_documents",
            [b"a", b"b", b"c"],
            [b"ab", b"bc", b"a", b"b"],
        ),
        fixture(
            "repeated_boundary",
            [b"aaa", b"aaa"],
            [b"aaaa", b"aaa", b"aa"],
        ),
        fixture(
            "newline_boundary",
            [b"line\n", b"\rnext"],
            [b"\n\r", b"line\n", b"\rnext"],
        ),
        fixture(
            "alphabet_split",
            [
                full_alphabet[:128],
                full_alphabet[128:],
            ],
            [
                bytes([127, 128]),
                bytes([126, 127]),
                bytes([128, 129]),
            ],
        ),
        fixture(
            "binary_documents",
            [
                b"\x00abc\xff",
                b"\xffabc\x00",
            ],
            [
                b"\xff\xff",
                b"\x00a",
                b"c\xff",
                b"\xffa",
                b"c\x00",
            ],
        ),
    ]

    direct_cross_checks = []

    for documents, query in [
        ([b"ab", b"cd"], b"bc"),
        ([b"\x00", b"\xff"], b"\x00\xff"),
        ([b"\xff", b"\x00"], b"\xff\x00"),
        ([b"a", b"b"], b"ab"),
        ([b"aaa", b"aaa"], b"aaaa"),
        ([b"line\n", b"\rnext"], b"\n\r"),
        (
            [full_alphabet[:128], full_alphabet[128:]],
            bytes([127, 128]),
        ),
    ]:
        local = naive_document_local_coordinates(
            documents,
            query,
        )
        physical = physical_global_offsets(
            documents,
            query,
        )

        if local:
            raise AssertionError(
                "cross-only fixture unexpectedly local"
            )

        if not physical:
            raise AssertionError(
                "cross-only fixture missing physical match"
            )

        safe = safe_coordinates_from_global_offsets(
            documents,
            query,
            physical,
        )

        if safe:
            raise AssertionError(
                "cross-boundary physical match survived"
            )

        direct_cross_checks.append(
            {
                "query_hex": query.hex(),
                "physical_offsets": physical,
                "document_local_count": 0,
                "safe_count": 0,
            }
        )

    mutation_documents = [b"ab", b"cd"]
    mutation_query = b"bc"

    valid_zero = make_artifact(
        mutation_documents,
        mutation_query,
    )

    duplicate_documents = [b"same", b"same"]
    duplicate_query = b"same"
    duplicate_artifact = make_artifact(
        duplicate_documents,
        duplicate_query,
    )

    reordered_documents = [
        duplicate_documents[1],
        duplicate_documents[0],
    ]

    altered_length_documents = [
        b"same",
        b"same!",
    ]

    valid_local_documents = [b"banana"]
    valid_local_query = b"ana"
    valid_local_artifact = make_artifact(
        valid_local_documents,
        valid_local_query,
        max_offsets=1,
    )

    mutations = [
        expect_failure(
            "cross_document_count_included",
            lambda: replay_artifact(
                mutation_documents,
                clone(
                    valid_zero,
                    match_count=1,
                ),
            ),
        ),
        expect_failure(
            "cross_document_coordinate_included",
            lambda: replay_artifact(
                mutation_documents,
                clone(
                    valid_zero,
                    match_count=1,
                    coordinates=[[0, 1]],
                    returned_count=1,
                    bounded=False,
                    offsets_complete=True,
                ),
            ),
        ),
        expect_failure(
            "global_span_crosses_boundary",
            lambda: map_global_span(
                boundary_table(
                    mutation_documents
                ),
                1,
                2,
            ),
        ),
        expect_failure(
            "wrong_document_coordinate",
            lambda: replay_artifact(
                [b"abc", b"zzz"],
                clone(
                    make_artifact(
                        [b"abc", b"zzz"],
                        b"abc",
                    ),
                    coordinates=[[1, 0]],
                ),
            ),
        ),
        expect_failure(
            "coordinate_starts_at_document_end",
            lambda: validate_coordinate(
                [b"abc"],
                b"a",
                Coordinate(0, 3),
            ),
        ),
        expect_failure(
            "coordinate_ends_after_document_end",
            lambda: validate_coordinate(
                [b"abc"],
                b"bcx",
                Coordinate(0, 1),
            ),
        ),
        expect_failure(
            "missing_boundary_policy",
            lambda: replay_artifact(
                mutation_documents,
                {
                    key: value
                    for key, value
                    in valid_zero.items()
                    if key
                    != "document_boundary_policy"
                },
            ),
        ),
        expect_failure(
            "wrong_boundary_policy",
            lambda: replay_artifact(
                mutation_documents,
                clone(
                    valid_zero,
                    document_boundary_policy=
                        "PHYSICAL_CONCATENATION",
                ),
            ),
        ),
        expect_failure(
            "wrong_document_local_count",
            lambda: replay_artifact(
                valid_local_documents,
                clone(
                    valid_local_artifact,
                    match_count=999,
                ),
            ),
        ),
        expect_failure(
            "stored_byte_check_not_trusted",
            lambda: replay_artifact(
                [b"banXna"],
                valid_local_artifact,
            ),
        ),
        expect_failure(
            "physical_concatenation_used_as_oracle",
            lambda: replay_artifact(
                mutation_documents,
                clone(
                    valid_zero,
                    match_count=1,
                    coordinates=[[0, 1]],
                    returned_count=1,
                ),
            ),
        ),
        expect_failure(
            "duplicate_document_occurrence_collapsed",
            lambda: replay_artifact(
                duplicate_documents,
                clone(
                    duplicate_artifact,
                    match_count=1,
                    coordinates=[[0, 0]],
                    returned_count=1,
                ),
            ),
        ),
        expect_failure(
            "empty_document_id_shift",
            lambda: replay_artifact(
                [b"ab", b"", b"cd"],
                make_artifact(
                    [b"ab", b"cd"],
                    b"cd",
                ),
            ),
        ),
        expect_failure(
            "document_order_mutation",
            lambda: replay_artifact(
                reordered_documents,
                duplicate_artifact,
            ),
        ),
        expect_failure(
            "document_length_mutation",
            lambda: replay_artifact(
                altered_length_documents,
                duplicate_artifact,
            ),
        ),
        expect_failure(
            "bounded_prefix_from_invalid_full_set",
            lambda: replay_artifact(
                valid_local_documents,
                clone(
                    valid_local_artifact,
                    coordinates=[[0, 3]],
                ),
            ),
        ),
    ]

    total_queries = sum(
        item["query_count"]
        for item in fixtures
    )

    total_physical_cross_only = sum(
        1
        for item in direct_cross_checks
        if item["physical_offsets"]
        and item["document_local_count"] == 0
    )

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "authoritative_semantics":
            "document_local_matches_only",
        "fixture_count": len(fixtures),
        "query_count": total_queries,
        "cross_only_fixture_count":
            total_physical_cross_only,
        "mutation_count": len(mutations),
        "count_equals_document_local_oracle":
            True,
        "locate_equals_document_local_oracle":
            True,
        "artifact_boundary_policy_bound":
            True,
        "replay_recomputes_byte_spans":
            True,
        "binary_boundaries_supported":
            True,
        "duplicate_documents_preserved":
            True,
        "fixtures": fixtures,
        "cross_only_checks":
            direct_cross_checks,
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
