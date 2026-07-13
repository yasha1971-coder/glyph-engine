#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

sys.path.insert(0, str(TOOLS))

from glyph_binary_runtime_evidence_v1 import (  # noqa: E402
    ARTIFACT_VERSION,
    EvidenceError,
    canonical_json_bytes,
    ensure_binaries,
    make_artifact,
    replay_artifact,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_BINARY_RUNTIME_EVIDENCE_V1.json"
)


class GateError(RuntimeError):
    pass


def artifact_sha256(
    artifact: dict[str, Any],
) -> str:
    return hashlib.sha256(
        canonical_json_bytes(artifact)
    ).hexdigest()


def verify_deterministic_artifact(
    documents: Sequence[bytes],
    query: bytes,
    max_offsets: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    first = make_artifact(
        documents,
        query.hex(),
        max_offsets,
    )

    second = make_artifact(
        documents,
        query.hex(),
        max_offsets,
    )

    first_bytes = canonical_json_bytes(first)
    second_bytes = canonical_json_bytes(second)

    if first_bytes != second_bytes:
        raise GateError(
            "identical runtime builds produced "
            "different artifacts"
        )

    replay = replay_artifact(
        first,
        documents,
    )

    if replay.get("ok") is not True:
        raise GateError(
            "runtime evidence replay failed"
        )

    return first, replay


def expect_replay_failure(
    name: str,
    artifact: dict[str, Any],
    documents: Sequence[bytes],
) -> dict[str, Any]:
    try:
        replay_artifact(
            artifact,
            documents,
        )
    except (
        EvidenceError,
        ValueError,
        TypeError,
        IndexError,
        KeyError,
    ) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise GateError(
        f"mutation unexpectedly accepted: {name}"
    )


def fixture(
    name: str,
    documents: Sequence[bytes],
    query: bytes,
    max_offsets: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, replay = (
        verify_deterministic_artifact(
            documents,
            query,
            max_offsets,
        )
    )

    return (
        {
            "fixture": name,
            "document_count": len(documents),
            "document_lengths": [
                len(document)
                for document in documents
            ],
            "query_hex": query.hex(),
            "max_offsets": max_offsets,
            "match_count":
                artifact["match_count"],
            "returned_count":
                artifact["returned_count"],
            "coordinates":
                artifact["coordinates"],
            "bounded": artifact["bounded"],
            "artifact_sha256":
                artifact_sha256(artifact),
            "replay_ok": replay["ok"],
            "deterministic": True,
        },
        artifact,
    )


def main() -> int:
    ensure_binaries()

    all_bytes = bytes(range(256))

    cases = [
        (
            "embedded_nul",
            [b"A\x00B", b"\x00A", b""],
            b"\x00",
            None,
        ),
        (
            "cross_document_only",
            [b"ab", b"cd"],
            b"bc",
            None,
        ),
        (
            "duplicate_documents",
            [b"same", b"same"],
            b"same",
            None,
        ),
        (
            "all_256_split_boundary",
            [
                all_bytes[:128],
                all_bytes[128:],
            ],
            b"\x7f\x80",
            None,
        ),
        (
            "bounded_repeated_matches",
            [b"aaaa", b"aa", b"aaaaaa"],
            b"aa",
            3,
        ),
        (
            "binary_multidoc",
            [
                b"\x00\xff\x00",
                b"\xff\x00",
                b"\x80\x81\xfe\xff",
            ],
            b"\xff\x00",
            None,
        ),
    ]

    fixtures: list[dict[str, Any]] = []
    artifacts: dict[str, dict[str, Any]] = {}

    for (
        name,
        documents,
        query,
        max_offsets,
    ) in cases:
        result, artifact = fixture(
            name,
            documents,
            query,
            max_offsets,
        )

        fixtures.append(result)
        artifacts[name] = artifact

    base_documents = [
        b"A\x00B",
        b"\xffA\x00B",
    ]

    base = make_artifact(
        base_documents,
        b"A\x00B".hex(),
        None,
    )

    mutations = []

    changed_documents = list(base_documents)
    changed_documents[0] = b"A\x00C"

    mutations.append(
        expect_replay_failure(
            "source_document_byte_changed",
            base,
            changed_documents,
        )
    )

    mutations.append(
        expect_replay_failure(
            "document_order_changed",
            base,
            list(reversed(base_documents)),
        )
    )

    altered = copy.deepcopy(base)
    altered["query_hex"] = b"A\x00".hex()
    mutations.append(
        expect_replay_failure(
            "query_trailing_byte_omitted",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["query_length_bytes"] = 2
    mutations.append(
        expect_replay_failure(
            "query_length_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["query_sha256"] = "0" * 64
    mutations.append(
        expect_replay_failure(
            "query_sha256_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["match_count"] += 1
    mutations.append(
        expect_replay_failure(
            "match_count_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["coordinates"] = [[0, 1]]
    altered["returned_count"] = 1
    mutations.append(
        expect_replay_failure(
            "coordinate_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["runtime_format_version"] = 2
    mutations.append(
        expect_replay_failure(
            "runtime_format_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["corpus_id"] = "0" * 64
    mutations.append(
        expect_replay_failure(
            "corpus_identity_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["documents"][0]["index"]["sa"][
        "sha256"
    ] = "0" * 64

    mutations.append(
        expect_replay_failure(
            "sa_commitment_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["documents"][1]["index"]["bwt"][
        "size_bytes"
    ] += 1

    mutations.append(
        expect_replay_failure(
            "bwt_size_changed",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["byte_check"] = False

    mutations.append(
        expect_replay_failure(
            "byte_check_false",
            altered,
            base_documents,
        )
    )

    altered = copy.deepcopy(base)
    altered["unexpected_field"] = "forbidden"

    mutations.append(
        expect_replay_failure(
            "unexpected_artifact_field",
            altered,
            base_documents,
        )
    )

    bounded_artifact = artifacts[
        "bounded_repeated_matches"
    ]

    altered = copy.deepcopy(bounded_artifact)
    altered["max_offsets"] = 2

    mutations.append(
        expect_replay_failure(
            "bounded_limit_changed",
            altered,
            [
                b"aaaa",
                b"aa",
                b"aaaaaa",
            ],
        )
    )

    if not all(
        mutation["rejected"] is True
        for mutation in mutations
    ):
        raise GateError(
            "runtime evidence mutation gate failed"
        )

    output = {
        "ok": True,
        "format":
            "GLYPH_BINARY_RUNTIME_EVIDENCE_GATE_V1",
        "artifact_version": ARTIFACT_VERSION,
        "runtime_profile":
            "GLYPH_BINARY_RUNTIME_V1",
        "count_path_conformant": True,
        "locate_path_conformant": True,
        "multidoc_path_conformant": True,
        "runtime_evidence_conformant": True,
        "runtime_conformant": False,
        "ordered_corpus_identity_bound": True,
        "query_identity_bound": True,
        "runtime_index_hashes_bound": True,
        "deterministic_artifact_verified": True,
        "independent_runtime_replay_verified": True,
        "all_coordinates_byte_checked": True,
        "fixture_count": len(fixtures),
        "mutation_count": len(mutations),
        "fixtures": fixtures,
        "mutations": mutations,
        "remaining_runtime_work": [
            "self-contained binary runtime bundle",
            "portable replay without external source paths",
            "top-level proof-graph integration",
            "full runtime conformance closure",
        ],
        "non_claims": [
            "Runtime evidence replay still requires separately supplied source documents.",
            "This is not yet a self-contained portable runtime bundle.",
            "Full runtime conformance is not yet established.",
        ],
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_bytes(
        canonical_json_bytes(output)
    )

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
