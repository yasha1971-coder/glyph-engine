#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable

PROOF = "P10"
FORMAT = "GLYPH_REPLAY_DETERMINISM_V1"
ARTIFACT_VERSION = "GLYPH_DETERMINISTIC_EVIDENCE_V1"
BOUNDARY_POLICY = "DOCUMENT_LOCAL_MATCHES_ONLY_V1"

FORBIDDEN_AUTHORITATIVE_FIELDS = {
    "artifact_sha256",
    "created_at",
    "timestamp",
    "hostname",
    "pid",
    "cwd",
    "source_path",
    "absolute_path",
    "temporary_path",
    "duration_ms",
    "random_nonce",
}


class DeterminismError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class Coordinate:
    doc_id: int
    doc_offset: int

    def as_json(self) -> list[int]:
        return [self.doc_id, self.doc_offset]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DeterminismError("non-canonical JSON value") from exc
    return text.encode("utf-8")


def corpus_id(documents: list[bytes]) -> str:
    h = hashlib.sha256()
    h.update(b"GLYPH_P10_CORPUS_V1\x00")
    h.update(len(documents).to_bytes(8, "big"))

    for doc_id, document in enumerate(documents):
        h.update(doc_id.to_bytes(8, "big"))
        h.update(len(document).to_bytes(8, "big"))
        h.update(hashlib.sha256(document).digest())

    return h.hexdigest()


def coordinates(
    documents: list[bytes],
    query: bytes,
) -> list[Coordinate]:
    if not query:
        raise DeterminismError("EMPTY_QUERY")

    result: list[Coordinate] = []

    for doc_id, document in enumerate(documents):
        if len(query) > len(document):
            continue

        for offset in range(len(document) - len(query) + 1):
            if document[offset:offset + len(query)] == query:
                result.append(Coordinate(doc_id, offset))

    result.sort()
    return result


def validate_payload(payload: dict[str, Any]) -> None:
    forbidden = set(payload) & FORBIDDEN_AUTHORITATIVE_FIELDS
    if forbidden:
        raise DeterminismError(
            "observational fields in authoritative payload: "
            + ",".join(sorted(forbidden))
        )

    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise DeterminismError("artifact_version mismatch")

    if payload.get("format") != FORMAT:
        raise DeterminismError("format mismatch")

    if payload.get("document_boundary_policy") != BOUNDARY_POLICY:
        raise DeterminismError("boundary policy mismatch")

    query_hex = payload.get("query_hex")
    if (
        not isinstance(query_hex, str)
        or not query_hex
        or query_hex != query_hex.lower()
        or len(query_hex) % 2
    ):
        raise DeterminismError("invalid query_hex")

    try:
        query = bytes.fromhex(query_hex)
    except ValueError as exc:
        raise DeterminismError("invalid query_hex") from exc

    if query.hex() != query_hex:
        raise DeterminismError("noncanonical query_hex")

    if payload.get("query_length_bytes") != len(query):
        raise DeterminismError("query length mismatch")

    if payload.get("query_sha256") != sha256(query):
        raise DeterminismError("query hash mismatch")

    raw = payload.get("coordinates")
    if not isinstance(raw, list):
        raise DeterminismError("coordinates must be a list")

    parsed: list[Coordinate] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in item
            )
        ):
            raise DeterminismError("invalid coordinate")
        parsed.append(Coordinate(item[0], item[1]))

    if parsed != sorted(parsed):
        raise DeterminismError("coordinates are not canonical")

    if len(parsed) != len(set(parsed)):
        raise DeterminismError("duplicate coordinates")

    if payload.get("returned_count") != len(parsed):
        raise DeterminismError("returned_count mismatch")

    match_count = payload.get("match_count")
    if (
        not isinstance(match_count, int)
        or isinstance(match_count, bool)
        or match_count < len(parsed)
    ):
        raise DeterminismError("invalid match_count")

    bounded = len(parsed) < match_count

    if payload.get("bounded") is not bounded:
        raise DeterminismError("bounded mismatch")

    if payload.get("offsets_complete") is not (not bounded):
        raise DeterminismError("offsets_complete mismatch")

    if payload.get("byte_check") is not True:
        raise DeterminismError("byte_check must be true")

    canonical_json(payload)


def generate(
    documents: list[bytes],
    query: bytes,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    if not query:
        raise DeterminismError("EMPTY_QUERY")

    if max_offsets is not None and (
        not isinstance(max_offsets, int)
        or isinstance(max_offsets, bool)
        or max_offsets < 0
    ):
        raise DeterminismError("invalid max_offsets")

    full = coordinates(documents, query)
    returned = full if max_offsets is None else full[:max_offsets]
    bounded = len(returned) < len(full)

    payload: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "format": FORMAT,
        "corpus_id": corpus_id(documents),
        "document_count": len(documents),
        "document_lengths": [len(document) for document in documents],
        "document_boundary_policy": BOUNDARY_POLICY,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256": sha256(query),
        "match_count": len(full),
        "coordinates": [coordinate.as_json() for coordinate in returned],
        "returned_count": len(returned),
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
        "coordinate_order": "doc_id_then_doc_offset_ascending",
    }

    if max_offsets is not None:
        payload["max_offsets"] = max_offsets

    validate_payload(payload)
    return payload


def envelope(
    payload: dict[str, Any],
    observational: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_payload(payload)

    result: dict[str, Any] = {
        "authoritative": copy.deepcopy(payload),
        "artifact_sha256": sha256(canonical_json(payload)),
    }

    if observational is not None:
        result["observational"] = copy.deepcopy(observational)

    return result


def replay(
    documents: list[bytes],
    value: dict[str, Any],
) -> dict[str, Any]:
    payload = value.get("authoritative")
    if not isinstance(payload, dict):
        raise DeterminismError("authoritative payload missing")

    validate_payload(payload)

    actual_digest = sha256(canonical_json(payload))
    if value.get("artifact_sha256") != actual_digest:
        raise DeterminismError("artifact digest mismatch")

    query = bytes.fromhex(payload["query_hex"])
    expected = generate(
        documents,
        query,
        payload.get("max_offsets"),
    )

    if payload != expected:
        raise DeterminismError("independent replay mismatch")

    return {
        "ok": True,
        "artifact_sha256": actual_digest,
        "canonical_authoritative_hex":
            canonical_json(payload).hex(),
    }


def reorder(value: Any, rng: random.Random) -> Any:
    if isinstance(value, dict):
        items = list(value.items())
        rng.shuffle(items)
        return {key: reorder(item, rng) for key, item in items}

    if isinstance(value, list):
        return [reorder(item, rng) for item in value]

    return value


def expect_failure(name: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        operation()
    except (DeterminismError, AssertionError, ValueError, TypeError) as exc:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(exc),
        }

    raise AssertionError(f"mutation accepted: {name}")


def run_fixture(
    name: str,
    documents: list[bytes],
    query: bytes,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    payloads = [
        generate(documents, query, max_offsets)
        for _ in range(3)
    ]

    serialized = [canonical_json(payload) for payload in payloads]
    digests = [sha256(item) for item in serialized]

    assert serialized[0] == serialized[1] == serialized[2]
    assert digests[0] == digests[1] == digests[2]

    reordered = reorder(payloads[0], random.Random(20260712))
    assert canonical_json(reordered) == serialized[0]

    first = envelope(
        payloads[0],
        {
            "timestamp": "2026-07-12T00:00:00Z",
            "hostname": "host-a",
            "cwd": "/tmp/a",
        },
    )
    second = envelope(
        payloads[0],
        {
            "timestamp": "2099-01-01T00:00:00Z",
            "hostname": "host-b",
            "cwd": "/different/path",
        },
    )

    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert replay(documents, first) == replay(documents, second)

    return {
        "fixture": name,
        "query_hex": query.hex(),
        "artifact_sha256": digests[0],
        "generation_count": 3,
        "byte_identical": True,
        "key_order_independent": True,
        "observational_metadata_excluded": True,
        "replay_ok": True,
    }


def main() -> int:
    alphabet = bytes(range(256))

    fixtures = [
        run_fixture("ascii", [b"banana"], b"ana"),
        run_fixture("embedded_nul", [b"A\x00B", b"A\x00B"], b"A\x00B"),
        run_fixture("byte_ff", [b"\xff\x00\xff"], b"\xff"),
        run_fixture("invalid_utf8", [b"\x80\x81\xfe\xff"], b"\x81\xfe"),
        run_fixture("empty_documents", [b"", b"abc", b""], b"abc"),
        run_fixture("duplicate_documents", [b"same", b"same"], b"same"),
        run_fixture("zero_match", [b"abc"], b"xyz"),
        run_fixture("bounded_locate", [b"aaaaaa"], b"aa", 2),
        run_fixture("all_bytes", [alphabet], alphabet),
        run_fixture("repeated_runs", [b"\x00" * 16], b"\x00\x00"),
    ]

    documents = [b"banana"]
    payload = generate(documents, b"ana")
    valid = envelope(payload)

    def modified_payload(**changes: Any) -> dict[str, Any]:
        changed = copy.deepcopy(payload)
        changed.update(changes)
        return envelope(changed)

    unsorted = copy.deepcopy(payload)
    unsorted["coordinates"] = list(reversed(unsorted["coordinates"]))

    duplicate = copy.deepcopy(payload)
    duplicate["coordinates"] = [[0, 1], [0, 1]]
    duplicate["returned_count"] = 2

    mutations = [
        expect_failure(
            "corpus_identity",
            lambda: replay(
                documents,
                modified_payload(corpus_id="0" * 64),
            ),
        ),
        expect_failure(
            "query_hex",
            lambda: replay(
                documents,
                modified_payload(query_hex=b"anb".hex()),
            ),
        ),
        expect_failure(
            "query_length",
            lambda: replay(
                documents,
                modified_payload(query_length_bytes=2),
            ),
        ),
        expect_failure(
            "query_hash",
            lambda: replay(
                documents,
                modified_payload(query_sha256="0" * 64),
            ),
        ),
        expect_failure(
            "match_count",
            lambda: replay(
                documents,
                modified_payload(match_count=999),
            ),
        ),
        expect_failure(
            "coordinate",
            lambda: replay(
                documents,
                modified_payload(coordinates=[[0, 0], [0, 3]]),
            ),
        ),
        expect_failure(
            "unsorted_coordinates",
            lambda: validate_payload(unsorted),
        ),
        expect_failure(
            "duplicate_coordinates",
            lambda: validate_payload(duplicate),
        ),
        expect_failure(
            "boundary_policy",
            lambda: validate_payload(
                {**payload, "document_boundary_policy": "WRONG"}
            ),
        ),
        expect_failure(
            "artifact_version",
            lambda: validate_payload(
                {**payload, "artifact_version": "V2"}
            ),
        ),
        expect_failure(
            "digest",
            lambda: replay(
                documents,
                {**valid, "artifact_sha256": "0" * 64},
            ),
        ),
        expect_failure(
            "timestamp_authoritative",
            lambda: validate_payload(
                {**payload, "timestamp": "now"}
            ),
        ),
        expect_failure(
            "absolute_path_authoritative",
            lambda: validate_payload(
                {**payload, "absolute_path": "/tmp/a"}
            ),
        ),
        expect_failure(
            "random_nonce_authoritative",
            lambda: validate_payload(
                {**payload, "random_nonce": "123"}
            ),
        ),
        expect_failure(
            "changed_document_order",
            lambda: replay(
                [b"other", b"banana"],
                envelope(generate([b"banana", b"other"], b"ana")),
            ),
        ),
        expect_failure(
            "changed_source_bytes",
            lambda: replay([b"banXna"], valid),
        ),
        expect_failure(
            "false_byte_check",
            lambda: validate_payload(
                {**payload, "byte_check": False}
            ),
        ),
        expect_failure(
            "changed_max_offsets",
            lambda: replay(
                documents,
                envelope(
                    {
                        **generate(documents, b"ana", 1),
                        "max_offsets": 2,
                    }
                ),
            ),
        ),
    ]

    output = {
        "ok": True,
        "proof_obligation": PROOF,
        "format": FORMAT,
        "artifact_version": ARTIFACT_VERSION,
        "fixture_count": len(fixtures),
        "mutation_count": len(mutations),
        "repeated_generation_byte_identical": True,
        "canonical_json_key_order_independent": True,
        "artifact_digest_authoritative_only": True,
        "observational_metadata_excluded": True,
        "timestamps_excluded": True,
        "absolute_paths_excluded": True,
        "canonical_coordinate_order": True,
        "fixtures": fixtures,
        "mutations": mutations,
    }

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
