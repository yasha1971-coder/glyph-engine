#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_BINARY_RUNTIME_MULTIDOC_V1.json"
)

TARGETS = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
    "query_fm_locate_binary_v1",
]


class GateError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
        check=False,
    )

    if expect_success and result.returncode != 0:
        raise GateError(
            f"command failed: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result


def configure_and_build() -> None:
    run(["cmake", "-S", ".", "-B", "build"])

    run([
        "cmake",
        "--build",
        "build",
        "--target",
        *TARGETS,
        "-j2",
    ])


def build_document_index(
    directory: Path,
    document: bytes,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "corpus": directory / "corpus.bin",
        "sa": directory / "sa.binary_v1",
        "bwt": directory / "bwt.binary_v1",
        "fm": directory / "fm.binary_v1",
    }

    paths["corpus"].write_bytes(document)

    run([
        str(BUILD / "build_sa_binary_v1"),
        str(paths["corpus"]),
        str(paths["sa"]),
    ])

    run([
        str(BUILD / "build_bwt_binary_v1"),
        str(paths["corpus"]),
        str(paths["sa"]),
        str(paths["bwt"]),
    ])

    run([
        str(BUILD / "build_fm_binary_v1"),
        str(paths["bwt"]),
        str(paths["fm"]),
        "32",
    ])

    return paths


def build_corpus_indexes(
    directory: Path,
    documents: list[bytes],
) -> list[dict[str, Path]]:
    return [
        build_document_index(
            directory / f"doc_{doc_id:04d}",
            document,
        )
        for doc_id, document in enumerate(documents)
    ]


def naive_coordinates(
    documents: list[bytes],
    query: bytes,
) -> list[list[int]]:
    if not query:
        raise GateError("EMPTY_QUERY")

    coordinates: list[list[int]] = []

    for doc_id, document in enumerate(documents):
        if len(query) > len(document):
            continue

        for offset in range(
            len(document) - len(query) + 1
        ):
            if (
                document[
                    offset:offset + len(query)
                ]
                == query
            ):
                coordinates.append([doc_id, offset])

    return coordinates


def physical_concat_count(
    documents: list[bytes],
    query: bytes,
) -> int:
    joined = b"".join(documents)

    if len(query) > len(joined):
        return 0

    return sum(
        joined[offset:offset + len(query)] == query
        for offset in range(
            len(joined) - len(query) + 1
        )
    )


def query_document(
    paths: dict[str, Path],
    query: bytes,
) -> dict[str, Any]:
    result = run([
        str(BUILD / "query_fm_locate_binary_v1"),
        str(paths["fm"]),
        str(paths["bwt"]),
        str(paths["sa"]),
        str(paths["corpus"]),
        query.hex(),
    ])

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            f"invalid document locate JSON: "
            f"{result.stdout}"
        ) from error

    return parsed


def aggregate_query(
    documents: list[bytes],
    indexes: list[dict[str, Path]],
    query: bytes,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    if not query:
        raise GateError("EMPTY_QUERY")

    if len(documents) != len(indexes):
        raise GateError(
            "document/index count mismatch"
        )

    coordinates: list[list[int]] = []
    document_results = []

    for doc_id, (document, paths) in enumerate(
        zip(documents, indexes)
    ):
        local = query_document(paths, query)

        expected_local = [
            coordinate[1]
            for coordinate in naive_coordinates(
                [document],
                query,
            )
        ]

        if local.get("offsets") != expected_local:
            raise GateError({
                "doc_id": doc_id,
                "query_hex": query.hex(),
                "expected_offsets": expected_local,
                "actual": local,
            })

        if local.get("match_count") != len(
            expected_local
        ):
            raise GateError(
                "document-local count mismatch"
            )

        for offset in local["offsets"]:
            end = offset + len(query)

            if document[offset:end] != query:
                raise GateError(
                    "document-local byte check failed"
                )

            coordinates.append([doc_id, offset])

        document_results.append({
            "doc_id": doc_id,
            "match_count": local["match_count"],
            "offsets": local["offsets"],
            "byte_check": local["byte_check"],
        })

    coordinates.sort()

    complete_coordinates = list(coordinates)
    match_count = len(complete_coordinates)

    if max_offsets is None:
        returned = complete_coordinates
    else:
        if max_offsets < 0:
            raise GateError("negative max_offsets")

        returned = complete_coordinates[:max_offsets]

    bounded = len(returned) < match_count

    return {
        "ok": True,
        "format":
            "GLYPH_QUERY_MULTIDOC_BINARY_V1",
        "runtime_profile":
            "GLYPH_BINARY_RUNTIME_V1",
        "document_count": len(documents),
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "match_count": match_count,
        "returned_count": len(returned),
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
        "coordinates": returned,
        "document_results": document_results,
    }


def validate_aggregate(
    documents: list[bytes],
    indexes: list[dict[str, Path]],
    query: bytes,
) -> dict[str, Any]:
    expected = naive_coordinates(documents, query)

    full = aggregate_query(
        documents,
        indexes,
        query,
    )

    if full["coordinates"] != expected:
        raise GateError({
            "query_hex": query.hex(),
            "expected": expected,
            "actual": full,
        })

    if full["match_count"] != len(expected):
        raise GateError(
            "multi-document match count mismatch"
        )

    bounds = sorted({
        0,
        1,
        max(0, len(expected) - 1),
        len(expected),
        len(expected) + 1,
    })

    bounded_results = []

    for max_offsets in bounds:
        result = aggregate_query(
            documents,
            indexes,
            query,
            max_offsets,
        )

        expected_returned = expected[:max_offsets]
        expected_bounded = (
            len(expected_returned) < len(expected)
        )

        if (
            result["coordinates"]
            != expected_returned
        ):
            raise GateError(
                "bounded coordinate prefix mismatch"
            )

        if (
            result["bounded"]
            is not expected_bounded
        ):
            raise GateError(
                "bounded flag mismatch"
            )

        if (
            result["offsets_complete"]
            is not (not expected_bounded)
        ):
            raise GateError(
                "offsets_complete mismatch"
            )

        bounded_results.append({
            "max_offsets": max_offsets,
            "returned_count":
                result["returned_count"],
            "bounded": result["bounded"],
        })

    return {
        "query_hex": query.hex(),
        "match_count": full["match_count"],
        "coordinates": full["coordinates"],
        "bounded_results": bounded_results,
    }


def validate_fixture(
    work: Path,
    name: str,
    documents: list[bytes],
    queries: list[bytes],
) -> dict[str, Any]:
    indexes = build_corpus_indexes(
        work / name,
        documents,
    )

    results = [
        validate_aggregate(
            documents,
            indexes,
            query,
        )
        for query in queries
    ]

    return {
        "fixture": name,
        "document_count": len(documents),
        "document_lengths": [
            len(document)
            for document in documents
        ],
        "document_sha256": [
            hashlib.sha256(document).hexdigest()
            for document in documents
        ],
        "query_count": len(results),
        "queries": results,
    }


def validate_cross_only(
    work: Path,
    name: str,
    documents: list[bytes],
    query: bytes,
) -> dict[str, Any]:
    indexes = build_corpus_indexes(
        work / f"cross_{name}",
        documents,
    )

    physical_count = physical_concat_count(
        documents,
        query,
    )

    local_coordinates = naive_coordinates(
        documents,
        query,
    )

    result = aggregate_query(
        documents,
        indexes,
        query,
    )

    if physical_count <= 0:
        raise GateError(
            "cross-only control has no physical match"
        )

    if local_coordinates:
        raise GateError(
            "cross-only query unexpectedly local"
        )

    if result["match_count"] != 0:
        raise GateError(
            "cross-document match survived runtime"
        )

    if result["coordinates"]:
        raise GateError(
            "cross-document coordinate returned"
        )

    return {
        "fixture": name,
        "query_hex": query.hex(),
        "physical_concat_count": physical_count,
        "document_local_count": 0,
        "runtime_count": result["match_count"],
        "coordinates": result["coordinates"],
        "rejected_by_construction": True,
    }


def main() -> int:
    configure_and_build()

    with tempfile.TemporaryDirectory(
        prefix="glyph-binary-multidoc-v1-"
    ) as temporary:
        work = Path(temporary)

        all_bytes = bytes(range(256))

        fixtures = [
            validate_fixture(
                work,
                "ascii_boundaries",
                [b"ab", b"cd"],
                [
                    b"ab",
                    b"cd",
                    b"bc",
                    b"a",
                    b"d",
                    b"abcd",
                ],
            ),
            validate_fixture(
                work,
                "empty_document_ids",
                [b"ab", b"", b"cd"],
                [
                    b"ab",
                    b"cd",
                    b"a",
                    b"\x00",
                ],
            ),
            validate_fixture(
                work,
                "duplicate_documents",
                [b"same", b"same"],
                [
                    b"same",
                    b"am",
                    b"s",
                    b"not-present",
                ],
            ),
            validate_fixture(
                work,
                "binary_documents",
                [
                    b"\x00\xff",
                    b"\x00A",
                    b"\xff\x00",
                ],
                [
                    b"\x00",
                    b"\xff",
                    b"\x00\xff",
                    b"\xff\x00",
                    b"\x00A",
                ],
            ),
            validate_fixture(
                work,
                "all_256_split",
                [
                    all_bytes[:128],
                    all_bytes[128:],
                ],
                [
                    b"\x00",
                    b"\x7f",
                    b"\x80",
                    b"\xff",
                    b"\x7f\x80",
                    all_bytes,
                ],
            ),
            validate_fixture(
                work,
                "repeated_matches",
                [
                    b"aaaa",
                    b"aa",
                    b"",
                    b"aaaaaa",
                ],
                [
                    b"a",
                    b"aa",
                    b"aaa",
                    b"aaaa",
                    b"aaaaaa",
                ],
            ),
        ]

        cross_only_checks = [
            validate_cross_only(
                work,
                "ascii_bc",
                [b"ab", b"cd"],
                b"bc",
            ),
            validate_cross_only(
                work,
                "nul_ff_nul",
                [b"\x00\xff", b"\x00A"],
                b"\xff\x00",
            ),
            validate_cross_only(
                work,
                "all_256_split_boundary",
                [
                    all_bytes[:128],
                    all_bytes[128:],
                ],
                b"\x7f\x80",
            ),
        ]

        original_documents = [
            b"left-document",
            b"right-document",
        ]
        reordered_documents = list(
            reversed(original_documents)
        )
        query = b"right"

        original_indexes = build_corpus_indexes(
            work / "order_original",
            original_documents,
        )
        reordered_indexes = build_corpus_indexes(
            work / "order_reordered",
            reordered_documents,
        )

        original = aggregate_query(
            original_documents,
            original_indexes,
            query,
        )
        reordered = aggregate_query(
            reordered_documents,
            reordered_indexes,
            query,
        )

        if original["coordinates"] == reordered[
            "coordinates"
        ]:
            raise GateError(
                "document reorder did not alter doc_id"
            )

        wrong_corpus_paths = build_document_index(
            work / "wrong_corpus_index",
            b"abc",
        )
        wrong_corpus = work / "wrong-corpus.bin"
        wrong_corpus.write_bytes(b"xyz")

        wrong_corpus_result = run(
            [
                str(
                    BUILD
                    / "query_fm_locate_binary_v1"
                ),
                str(wrong_corpus_paths["fm"]),
                str(wrong_corpus_paths["bwt"]),
                str(wrong_corpus_paths["sa"]),
                str(wrong_corpus),
                b"a".hex(),
            ],
            expect_success=False,
        )

        if wrong_corpus_result.returncode == 0:
            raise GateError(
                "wrong corpus accepted by document runtime"
            )

        mutations = [
            {
                "mutation":
                    "document_order_changes_coordinates",
                "rejected": True,
                "original_coordinates":
                    original["coordinates"],
                "reordered_coordinates":
                    reordered["coordinates"],
            },
            {
                "mutation":
                    "wrong_corpus_for_index",
                "rejected": True,
                "exit_code":
                    wrong_corpus_result.returncode,
            },
            {
                "mutation":
                    "physical_cross_document_matches",
                "rejected": all(
                    item["runtime_count"] == 0
                    for item in cross_only_checks
                ),
            },
        ]

        if not all(
            mutation["rejected"] is True
            for mutation in mutations
        ):
            raise GateError(
                "multi-document mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_BINARY_RUNTIME_MULTIDOC_V1",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "count_path_conformant": True,
            "locate_path_conformant": True,
            "multidoc_path_conformant": True,
            "runtime_conformant": False,
            "index_topology":
                "one_independent_index_per_document",
            "document_boundary_policy":
                "NO_PHYSICAL_DOCUMENT_CONCATENATION",
            "coordinate_model":
                "(document_id, document_offset)",
            "source_byte_domain":
                "0x00..0xFF",
            "logical_sentinel": 256,
            "fixture_count": len(fixtures),
            "query_count": sum(
                fixture["query_count"]
                for fixture in fixtures
            ),
            "cross_only_fixture_count":
                len(cross_only_checks),
            "mutation_count": len(mutations),
            "cross_document_matches_structurally_impossible":
                True,
            "empty_document_ids_preserved": True,
            "duplicate_documents_preserved": True,
            "bounded_global_coordinate_prefix_verified":
                True,
            "fixtures": fixtures,
            "cross_only_checks":
                cross_only_checks,
            "mutations": mutations,
            "remaining_runtime_work": [
                "committed multi-document corpus identity",
                "runtime evidence artifact",
                "deterministic runtime replay",
                "self-contained runtime bundle",
                "proof-graph integration",
            ],
            "non_claims": [
                "Multi-document search is not yet a committed evidence artifact.",
                "Runtime artifact replay is not yet established.",
                "Full runtime conformance is not yet established.",
            ],
        }

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUT.write_text(
            json.dumps(
                output,
                indent=2,
                sort_keys=True,
            )
            + "\n"
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
