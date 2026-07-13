#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

ARTIFACT_VERSION = "GLYPH_BINARY_RUNTIME_EVIDENCE_V1"
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"
RUNTIME_FORMAT_VERSION = 1
INDEX_TOPOLOGY = "one_independent_index_per_document"
BOUNDARY_POLICY = "NO_PHYSICAL_DOCUMENT_CONCATENATION"

TARGETS = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
    "query_fm_locate_binary_v1",
]

TOP_LEVEL_KEYS = {
    "artifact_version",
    "runtime_profile",
    "runtime_format_version",
    "index_topology",
    "document_boundary_policy",
    "corpus_identity_version",
    "corpus_id",
    "document_count",
    "documents",
    "query_hex",
    "query_length_bytes",
    "query_sha256",
    "max_offsets",
    "match_count",
    "coordinates",
    "returned_count",
    "bounded",
    "offsets_complete",
    "byte_check",
}


class EvidenceError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(
            lambda: source.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


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
        timeout=300,
        check=False,
    )

    if expect_success and result.returncode != 0:
        raise EvidenceError(
            f"command failed: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result


def ensure_binaries() -> None:
    run(["cmake", "-S", ".", "-B", "build"])

    run([
        "cmake",
        "--build",
        "build",
        "--target",
        *TARGETS,
        "-j2",
    ])

    for target in TARGETS:
        path = BUILD / target

        if not path.is_file():
            raise EvidenceError(
                f"missing runtime binary: {path}"
            )


def parse_query_hex(value: Any) -> bytes:
    if not isinstance(value, str):
        raise EvidenceError(
            "query_hex must be a string"
        )

    if value == "":
        raise EvidenceError("EMPTY_QUERY")

    if value != value.lower():
        raise EvidenceError(
            "query_hex must be lowercase"
        )

    if len(value) % 2 != 0:
        raise EvidenceError(
            "query_hex must have even length"
        )

    allowed = set("0123456789abcdef")

    if any(character not in allowed for character in value):
        raise EvidenceError(
            "query_hex contains invalid character"
        )

    query = bytes.fromhex(value)

    if not query:
        raise EvidenceError("EMPTY_QUERY")

    if query.hex() != value:
        raise EvidenceError(
            "query_hex is not canonical"
        )

    return query


def corpus_identity(documents: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00"
    )
    digest.update(
        len(documents).to_bytes(8, "big")
    )

    for doc_id, document in enumerate(documents):
        digest.update(doc_id.to_bytes(8, "big"))
        digest.update(
            len(document).to_bytes(8, "big")
        )
        digest.update(
            hashlib.sha256(document).digest()
        )

    return digest.hexdigest()


def build_document_index(
    directory: Path,
    document: bytes,
) -> dict[str, Path]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def index_commitment(
    paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    formats = {
        "sa": "GLYPH_SA_BINARY_V1",
        "bwt": "GLYPH_BWT_BINARY_V1",
        "fm": "GLYPH_FM_BINARY_V1",
    }

    return {
        role: {
            "format": formats[role],
            "size_bytes": paths[role].stat().st_size,
            "sha256": sha256_file(paths[role]),
        }
        for role in ("sa", "bwt", "fm")
    }


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
        raise EvidenceError(
            f"runtime returned invalid JSON: "
            f"{result.stdout}"
        ) from error

    if parsed.get("ok") is not True:
        raise EvidenceError(
            "runtime query result is not ok"
        )

    if parsed.get("byte_check") is not True:
        raise EvidenceError(
            "runtime byte_check is not true"
        )

    return parsed


def aggregate_runtime(
    documents: Sequence[bytes],
    indexes: Sequence[dict[str, Path]],
    query: bytes,
    max_offsets: int | None,
) -> dict[str, Any]:
    if len(documents) != len(indexes):
        raise EvidenceError(
            "document/index count mismatch"
        )

    coordinates: list[list[int]] = []
    match_count = 0

    for doc_id, (document, paths) in enumerate(
        zip(documents, indexes)
    ):
        local = query_document(paths, query)

        local_offsets = local.get("offsets")

        if not isinstance(local_offsets, list):
            raise EvidenceError(
                "runtime offsets must be a list"
            )

        if local.get("match_count") != len(
            local_offsets
        ):
            raise EvidenceError(
                "runtime local count mismatch"
            )

        for offset in local_offsets:
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset < 0
            ):
                raise EvidenceError(
                    "invalid runtime offset"
                )

            end = offset + len(query)

            if end > len(document):
                raise EvidenceError(
                    "runtime offset crosses document end"
                )

            if document[offset:end] != query:
                raise EvidenceError(
                    "runtime offset failed byte check"
                )

            coordinates.append([doc_id, offset])

        match_count += local["match_count"]

    coordinates.sort()

    if len(coordinates) != match_count:
        raise EvidenceError(
            "aggregate coordinate count mismatch"
        )

    if len({
        (item[0], item[1])
        for item in coordinates
    }) != len(coordinates):
        raise EvidenceError(
            "duplicate aggregate coordinate"
        )

    if max_offsets is None:
        returned = coordinates
    else:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
        ):
            raise EvidenceError(
                "invalid max_offsets"
            )

        returned = coordinates[:max_offsets]

    bounded = len(returned) < match_count

    return {
        "match_count": match_count,
        "coordinates": returned,
        "returned_count": len(returned),
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
    }


def make_artifact(
    documents: Sequence[bytes],
    query_hex: str,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    query = parse_query_hex(query_hex)

    with tempfile.TemporaryDirectory(
        prefix="glyph-runtime-evidence-v1-"
    ) as temporary:
        work = Path(temporary)

        indexes = [
            build_document_index(
                work / f"doc_{doc_id:08d}",
                document,
            )
            for doc_id, document
            in enumerate(documents)
        ]

        document_records = [
            {
                "doc_id": doc_id,
                "byte_length": len(document),
                "sha256": sha256_bytes(document),
                "index": index_commitment(
                    indexes[doc_id]
                ),
            }
            for doc_id, document
            in enumerate(documents)
        ]

        result = aggregate_runtime(
            documents,
            indexes,
            query,
            max_offsets,
        )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "runtime_format_version":
            RUNTIME_FORMAT_VERSION,
        "index_topology": INDEX_TOPOLOGY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "corpus_identity_version":
            "GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1",
        "corpus_id": corpus_identity(documents),
        "document_count": len(documents),
        "documents": document_records,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256": sha256_bytes(query),
        "max_offsets": max_offsets,
        **result,
    }

    return artifact


def validate_document_record(
    value: Any,
    doc_id: int,
    document: bytes,
) -> None:
    if not isinstance(value, dict):
        raise EvidenceError(
            "document record must be an object"
        )

    if set(value) != {
        "doc_id",
        "byte_length",
        "sha256",
        "index",
    }:
        raise EvidenceError(
            "document record fields mismatch"
        )

    if value.get("doc_id") != doc_id:
        raise EvidenceError(
            "document ID mismatch"
        )

    if value.get("byte_length") != len(document):
        raise EvidenceError(
            "document length mismatch"
        )

    if value.get("sha256") != sha256_bytes(document):
        raise EvidenceError(
            "document SHA256 mismatch"
        )

    index = value.get("index")

    if not isinstance(index, dict):
        raise EvidenceError(
            "document index commitment missing"
        )

    if set(index) != {"sa", "bwt", "fm"}:
        raise EvidenceError(
            "index commitment roles mismatch"
        )


def replay_artifact(
    artifact: dict[str, Any],
    documents: Sequence[bytes],
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise EvidenceError(
            "artifact must be an object"
        )

    if set(artifact) != TOP_LEVEL_KEYS:
        missing = sorted(
            TOP_LEVEL_KEYS - set(artifact)
        )
        extra = sorted(
            set(artifact) - TOP_LEVEL_KEYS
        )

        raise EvidenceError(
            f"artifact fields mismatch; "
            f"missing={missing}; extra={extra}"
        )

    constants = {
        "artifact_version": ARTIFACT_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "runtime_format_version":
            RUNTIME_FORMAT_VERSION,
        "index_topology": INDEX_TOPOLOGY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "corpus_identity_version":
            "GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1",
    }

    for key, expected in constants.items():
        if artifact.get(key) != expected:
            raise EvidenceError(
                f"artifact constant mismatch: {key}"
            )

    if artifact.get("document_count") != len(
        documents
    ):
        raise EvidenceError(
            "document count mismatch"
        )

    records = artifact.get("documents")

    if (
        not isinstance(records, list)
        or len(records) != len(documents)
    ):
        raise EvidenceError(
            "document records mismatch"
        )

    for doc_id, document in enumerate(documents):
        validate_document_record(
            records[doc_id],
            doc_id,
            document,
        )

    expected_corpus_id = corpus_identity(
        documents
    )

    if artifact.get("corpus_id") != expected_corpus_id:
        raise EvidenceError(
            "corpus identity mismatch"
        )

    query = parse_query_hex(
        artifact.get("query_hex")
    )

    if (
        artifact.get("query_length_bytes")
        != len(query)
    ):
        raise EvidenceError(
            "query length mismatch"
        )

    if (
        artifact.get("query_sha256")
        != sha256_bytes(query)
    ):
        raise EvidenceError(
            "query SHA256 mismatch"
        )

    max_offsets = artifact.get("max_offsets")

    if max_offsets is not None and (
        not isinstance(max_offsets, int)
        or isinstance(max_offsets, bool)
        or max_offsets < 0
    ):
        raise EvidenceError(
            "invalid max_offsets"
        )

    with tempfile.TemporaryDirectory(
        prefix="glyph-runtime-replay-v1-"
    ) as temporary:
        work = Path(temporary)

        indexes = [
            build_document_index(
                work / f"doc_{doc_id:08d}",
                document,
            )
            for doc_id, document
            in enumerate(documents)
        ]

        for doc_id, paths in enumerate(indexes):
            rebuilt = index_commitment(paths)
            committed = records[doc_id]["index"]

            if rebuilt != committed:
                raise EvidenceError(
                    f"runtime index commitment mismatch "
                    f"for document {doc_id}"
                )

        replayed = aggregate_runtime(
            documents,
            indexes,
            query,
            max_offsets,
        )

    for field in (
        "match_count",
        "coordinates",
        "returned_count",
        "bounded",
        "offsets_complete",
        "byte_check",
    ):
        if artifact.get(field) != replayed[field]:
            raise EvidenceError(
                f"replay result mismatch: {field}"
            )

    if artifact.get("byte_check") is not True:
        raise EvidenceError(
            "byte_check must be true"
        )

    return {
        "ok": True,
        "replay_version":
            "GLYPH_BINARY_RUNTIME_EVIDENCE_REPLAY_V1",
        "artifact_version": ARTIFACT_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "corpus_id": expected_corpus_id,
        "document_count": len(documents),
        "query_hex": query.hex(),
        "match_count": replayed["match_count"],
        "coordinates": replayed["coordinates"],
        "returned_count":
            replayed["returned_count"],
        "bounded": replayed["bounded"],
        "offsets_complete":
            replayed["offsets_complete"],
        "byte_check": True,
        "artifact_sha256": sha256_bytes(
            canonical_json_bytes(artifact)
        ),
    }


def read_documents(paths: Sequence[str]) -> list[bytes]:
    return [
        Path(path).read_bytes()
        for path in paths
    ]


def command_make(args: argparse.Namespace) -> int:
    ensure_binaries()

    documents = read_documents(args.document)

    artifact = make_artifact(
        documents,
        args.query_hex,
        args.max_offsets,
    )

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_bytes(
        canonical_json_bytes(artifact)
    )

    print(json.dumps({
        "ok": True,
        "mode": "make",
        "artifact": str(output_path),
        "artifact_version": ARTIFACT_VERSION,
        "artifact_sha256": sha256_file(
            output_path
        ),
        "document_count": len(documents),
        "match_count":
            artifact["match_count"],
        "returned_count":
            artifact["returned_count"],
    }, indent=2, sort_keys=True))

    return 0


def command_replay(args: argparse.Namespace) -> int:
    ensure_binaries()

    artifact_path = Path(
        args.artifact
    ).resolve()

    artifact = json.loads(
        artifact_path.read_text()
    )

    documents = read_documents(args.document)

    result = replay_artifact(
        artifact,
        documents,
    )

    if args.out is not None:
        output_path = Path(args.out).resolve()
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        output_path.write_bytes(
            canonical_json_bytes(result)
        )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    make_parser = subparsers.add_parser("make")
    make_parser.add_argument(
        "--document",
        action="append",
        required=True,
    )
    make_parser.add_argument(
        "--query-hex",
        required=True,
    )
    make_parser.add_argument(
        "--max-offsets",
        type=int,
    )
    make_parser.add_argument(
        "--out",
        required=True,
    )
    make_parser.set_defaults(
        handler=command_make
    )

    replay_parser = subparsers.add_parser(
        "replay"
    )
    replay_parser.add_argument(
        "--artifact",
        required=True,
    )
    replay_parser.add_argument(
        "--document",
        action="append",
        required=True,
    )
    replay_parser.add_argument("--out")
    replay_parser.set_defaults(
        handler=command_replay
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
