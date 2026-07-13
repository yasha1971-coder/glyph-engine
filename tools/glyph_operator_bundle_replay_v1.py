#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

BUNDLE_VERSION = "GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1"
BUNDLE_MANIFEST_VERSION = (
    "GLYPH_OPERATOR_EVIDENCE_BUNDLE_MANIFEST_V1"
)
BUNDLE_COMPLETE_VERSION = (
    "GLYPH_OPERATOR_EVIDENCE_BUNDLE_COMPLETE_V1"
)

SOURCE_MANIFEST_VERSION = (
    "GLYPH_OPERATOR_CORPUS_MANIFEST_V1"
)
SOURCE_COMPLETE_VERSION = (
    "GLYPH_OPERATOR_BUILD_COMPLETE_V1"
)
RUNTIME_MANIFEST_VERSION = (
    "GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1"
)
RUNTIME_COMPLETE_VERSION = (
    "GLYPH_OPERATOR_RUNTIME_INDEX_COMPLETE_V1"
)
QUERY_RESULT_VERSION = (
    "GLYPH_OPERATOR_QUERY_RESULT_V1"
)

RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"
INDEX_TOPOLOGY = "one_independent_index_per_document"
BOUNDARY_POLICY = "NO_PHYSICAL_DOCUMENT_CONCATENATION"

MANIFEST_NAME = "bundle_manifest_v1.json"
COMPLETE_NAME = "BUNDLE_COMPLETE_V1.json"

SOURCE_MANIFEST_PATH = (
    "source/source_manifest_v1.json"
)
SOURCE_COMPLETE_PATH = (
    "source/BUILD_COMPLETE_V1.json"
)
RUNTIME_MANIFEST_PATH = (
    "runtime/runtime_manifest_v1.json"
)
RUNTIME_COMPLETE_PATH = (
    "runtime/RUNTIME_BUILD_COMPLETE_V1.json"
)
ARTIFACT_PATH = "artifact/query_result_v1.json"
QUERY_PATH = "query/query.bin"
REPLAY_PATH = "replay.py"

LOGICAL_SENTINEL = 256
ALPHABET_SIZE = 257
CHECKPOINT_STEP = 32

BUNDLE_MANIFEST_KEYS = {
    "bundle_manifest_version",
    "bundle_version",
    "runtime_profile",
    "construction_status",
    "artifact_version",
    "corpus_id",
    "source_manifest_id",
    "runtime_index_id",
    "query_result_id",
    "document_count",
    "file_count",
    "files",
    "bundle_root_sha256",
    "external_dependencies",
    "network_dependency_required",
    "repository_dependency_required",
}

FILE_KEYS = {
    "path",
    "role",
    "size_bytes",
    "sha256",
    "executable",
}

BUNDLE_COMPLETE_KEYS = {
    "complete_version",
    "bundle_manifest_sha256",
    "bundle_root_sha256",
    "query_result_id",
    "runtime_index_id",
    "corpus_id",
    "file_count",
}

SOURCE_MANIFEST_KEYS = {
    "manifest_version",
    "runtime_profile",
    "corpus_identity_version",
    "source_manifest_identity_version",
    "construction_status",
    "document_count",
    "total_source_bytes",
    "corpus_id",
    "source_manifest_id",
    "documents",
}

SOURCE_DOCUMENT_KEYS = {
    "doc_id",
    "relative_path_bytes_hex",
    "display_path",
    "document_type",
    "byte_length",
    "sha256",
    "snapshot_path",
}

SOURCE_COMPLETE_KEYS = {
    "complete_version",
    "manifest_sha256",
    "corpus_id",
    "source_manifest_id",
    "document_count",
}

RUNTIME_MANIFEST_KEYS = {
    "manifest_version",
    "runtime_profile",
    "construction_status",
    "index_topology",
    "logical_sentinel",
    "alphabet_size",
    "checkpoint_step",
    "source_manifest_name",
    "source_manifest_sha256",
    "corpus_id",
    "source_manifest_id",
    "document_count",
    "total_source_bytes",
    "total_runtime_bytes",
    "runtime_binaries",
    "documents",
    "runtime_index_id",
}

RUNTIME_BINARY_KEYS = {
    "name",
    "size_bytes",
    "sha256",
}

RUNTIME_DOCUMENT_KEYS = {
    "doc_id",
    "source_snapshot_path",
    "source_byte_length",
    "source_sha256",
    "index_directory",
    "sa",
    "bwt",
    "fm",
}

RUNTIME_ARTIFACT_KEYS = {
    "path",
    "format",
    "size_bytes",
    "sha256",
}

RUNTIME_COMPLETE_KEYS = {
    "complete_version",
    "runtime_manifest_sha256",
    "runtime_index_id",
    "corpus_id",
    "source_manifest_id",
    "document_count",
    "total_runtime_bytes",
}

COUNT_KEYS = {
    "ok",
    "format",
    "query_hex",
    "query_length_bytes",
    "interval",
    "count",
    "alphabet_size",
    "logical_sentinel",
}

LOCATE_KEYS = {
    "ok",
    "format",
    "runtime_profile",
    "document_count",
    "query_hex",
    "query_length_bytes",
    "interval",
    "match_count",
    "returned_count",
    "bounded",
    "offsets_complete",
    "byte_check",
    "offsets",
    "coordinates",
    "alphabet_size",
    "logical_sentinel",
}


class ReplayError(RuntimeError):
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


def require_uint(value: Any, field: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ReplayError(
            f"invalid unsigned integer: {field}"
        )

    return value


def validate_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ReplayError(
            f"invalid SHA256: {field}"
        )

    return value


def safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayError(
            "bundle path must be non-empty string"
        )

    if "\\" in value:
        raise ReplayError(
            "backslash forbidden in bundle path"
        )

    path = PurePosixPath(value)

    if path.is_absolute():
        raise ReplayError(
            "absolute bundle path forbidden"
        )

    if any(
        component in ("", ".", "..")
        for component in path.parts
    ):
        raise ReplayError(
            "unsafe bundle path component"
        )

    if path.as_posix() != value:
        raise ReplayError(
            "noncanonical bundle path"
        )

    return value


def physical_path(
    bundle: Path,
    relative: str,
) -> Path:
    safe_relative_path(relative)
    return bundle.joinpath(
        *PurePosixPath(relative).parts
    )


def load_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise ReplayError(
            f"cannot load JSON: {path.name}"
        ) from error

    if not isinstance(value, dict):
        raise ReplayError(
            f"JSON root must be object: {path.name}"
        )

    if canonical_json_bytes(value) != raw:
        raise ReplayError(
            f"JSON is not canonical: {path.name}"
        )

    return value


def file_identity(
    status: os.stat_result,
) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
    )


def bundle_tree_files(bundle: Path) -> set[str]:
    result: set[str] = set()

    for root, directories, filenames in os.walk(
        bundle,
        topdown=True,
        followlinks=False,
    ):
        root_path = Path(root)

        for directory in list(directories):
            path = root_path / directory
            status = os.lstat(path)

            if (
                stat.S_ISLNK(status.st_mode)
                or not stat.S_ISDIR(status.st_mode)
            ):
                raise ReplayError(
                    "bundle directory must be real: "
                    + str(path.relative_to(bundle))
                )

        for filename in filenames:
            path = root_path / filename
            status = os.lstat(path)

            if (
                stat.S_ISLNK(status.st_mode)
                or not stat.S_ISREG(status.st_mode)
            ):
                raise ReplayError(
                    "bundle payload must be regular: "
                    + str(path.relative_to(bundle))
                )

            relative = (
                path.relative_to(bundle).as_posix()
            )
            safe_relative_path(relative)
            result.add(relative)

    return result


def bundle_root_identity(
    entries: Sequence[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"GLYPH_OPERATOR_EVIDENCE_BUNDLE_ROOT_V1\x00"
    )
    digest.update(
        len(entries).to_bytes(8, "big")
    )

    for entry in entries:
        path = safe_relative_path(
            entry["path"]
        ).encode("utf-8")
        role = entry["role"].encode("ascii")

        digest.update(
            len(path).to_bytes(8, "big")
        )
        digest.update(path)
        digest.update(
            len(role).to_bytes(8, "big")
        )
        digest.update(role)
        digest.update(
            entry["size_bytes"].to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            bytes.fromhex(entry["sha256"])
        )
        digest.update(
            b"\x01"
            if entry["executable"]
            else b"\x00"
        )

    return digest.hexdigest()


def source_corpus_identity(
    documents: Sequence[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00"
    )
    digest.update(
        len(documents).to_bytes(8, "big")
    )

    for document in documents:
        digest.update(
            document["doc_id"].to_bytes(8, "big")
        )
        digest.update(
            document["byte_length"].to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            bytes.fromhex(document["sha256"])
        )

    return digest.hexdigest()


def source_manifest_identity(
    documents: Sequence[dict[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"GLYPH_OPERATOR_CORPUS_MANIFEST_V1\x00"
    )
    digest.update(
        len(documents).to_bytes(8, "big")
    )

    for document in documents:
        raw_path = bytes.fromhex(
            document["relative_path_bytes_hex"]
        )

        digest.update(
            document["doc_id"].to_bytes(8, "big")
        )
        digest.update(
            len(raw_path).to_bytes(8, "big")
        )
        digest.update(raw_path)
        digest.update(
            document["byte_length"].to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            bytes.fromhex(document["sha256"])
        )

    return digest.hexdigest()


def runtime_index_identity(
    manifest: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        b"GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1\x00"
    )

    for field in (
        "source_manifest_sha256",
        "corpus_id",
        "source_manifest_id",
    ):
        digest.update(
            bytes.fromhex(
                validate_sha256(
                    manifest[field],
                    field,
                )
            )
        )

    topology = manifest[
        "index_topology"
    ].encode("ascii")

    digest.update(
        len(topology).to_bytes(8, "big")
    )
    digest.update(topology)
    digest.update(
        manifest["logical_sentinel"].to_bytes(
            4,
            "big",
        )
    )
    digest.update(
        manifest["alphabet_size"].to_bytes(
            4,
            "big",
        )
    )
    digest.update(
        manifest["checkpoint_step"].to_bytes(
            4,
            "big",
        )
    )

    binaries = manifest["runtime_binaries"]
    digest.update(
        len(binaries).to_bytes(8, "big")
    )

    for record in binaries:
        name = record["name"].encode("ascii")

        digest.update(
            len(name).to_bytes(8, "big")
        )
        digest.update(name)
        digest.update(
            record["size_bytes"].to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            bytes.fromhex(record["sha256"])
        )

    documents = manifest["documents"]
    digest.update(
        len(documents).to_bytes(8, "big")
    )

    for document in documents:
        digest.update(
            document["doc_id"].to_bytes(8, "big")
        )
        digest.update(
            document[
                "source_byte_length"
            ].to_bytes(8, "big")
        )
        digest.update(
            bytes.fromhex(
                document["source_sha256"]
            )
        )

        for role in ("sa", "bwt", "fm"):
            artifact = document[role]
            format_bytes = artifact[
                "format"
            ].encode("ascii")

            digest.update(
                len(format_bytes).to_bytes(
                    8,
                    "big",
                )
            )
            digest.update(format_bytes)
            digest.update(
                artifact["size_bytes"].to_bytes(
                    8,
                    "big",
                )
            )
            digest.update(
                bytes.fromhex(
                    artifact["sha256"]
                )
            )

    return digest.hexdigest()


def per_document_digest(
    results: list[dict[str, Any]],
) -> str:
    authoritative = [
        {
            "doc_id": item["doc_id"],
            "interval": item["interval"],
            "match_count": item["match_count"],
            "returned_count":
                item["returned_count"],
        }
        for item in results
    ]

    return sha256_bytes(
        canonical_json_bytes(authoritative)
    )


def query_result_identity(
    result: dict[str, Any],
) -> str:
    preimage = {
        "result_version":
            result["result_version"],
        "runtime_profile":
            result["runtime_profile"],
        "corpus_id": result["corpus_id"],
        "source_manifest_id":
            result["source_manifest_id"],
        "runtime_index_id":
            result["runtime_index_id"],
        "source_manifest_sha256":
            result["source_manifest_sha256"],
        "runtime_manifest_sha256":
            result["runtime_manifest_sha256"],
        "query_binary_commitments":
            result["query_binary_commitments"],
        "query_hex": result["query_hex"],
        "query_length_bytes":
            result["query_length_bytes"],
        "query_sha256":
            result["query_sha256"],
        "max_offsets": result["max_offsets"],
        "documents_queried":
            result["documents_queried"],
        "per_document_results_sha256":
            result[
                "per_document_results_sha256"
            ],
        "match_count": result["match_count"],
        "coordinates": [
            item["coordinate"]
            for item in result["coordinates"]
        ],
        "returned_count":
            result["returned_count"],
        "bounded": result["bounded"],
        "offsets_complete":
            result["offsets_complete"],
    }

    return sha256_bytes(
        canonical_json_bytes(preimage)
    )


def verify_declared_file(
    bundle: Path,
    entry: dict[str, Any],
) -> None:
    path = physical_path(
        bundle,
        entry["path"],
    )

    status = os.lstat(path)

    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
    ):
        raise ReplayError(
            "declared payload is not regular: "
            + entry["path"]
        )

    if status.st_size != entry["size_bytes"]:
        raise ReplayError(
            "payload size mismatch: "
            + entry["path"]
        )

    if sha256_file(path) != entry["sha256"]:
        raise ReplayError(
            "payload SHA256 mismatch: "
            + entry["path"]
        )

    executable = bool(
        status.st_mode
        & (
            stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )
    )

    if executable is not entry["executable"]:
        raise ReplayError(
            "payload executable flag mismatch: "
            + entry["path"]
        )


def verify_source(
    bundle: Path,
) -> tuple[dict[str, Any], str]:
    manifest_path = bundle / SOURCE_MANIFEST_PATH
    complete_path = bundle / SOURCE_COMPLETE_PATH

    manifest = load_canonical_json(
        manifest_path
    )
    complete = load_canonical_json(
        complete_path
    )

    if set(manifest) != SOURCE_MANIFEST_KEYS:
        raise ReplayError(
            "source manifest fields mismatch"
        )

    constants = {
        "manifest_version":
            SOURCE_MANIFEST_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "corpus_identity_version":
            "GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1",
        "source_manifest_identity_version":
            "GLYPH_OPERATOR_CORPUS_MANIFEST_V1",
        "construction_status": "complete",
    }

    for field, expected in constants.items():
        if manifest.get(field) != expected:
            raise ReplayError(
                "source manifest constant mismatch: "
                + field
            )

    documents = manifest.get("documents")

    if (
        not isinstance(documents, list)
        or manifest["document_count"]
        != len(documents)
    ):
        raise ReplayError(
            "source document count mismatch"
        )

    raw_paths: list[bytes] = []
    total_bytes = 0

    for doc_id, document in enumerate(documents):
        if (
            not isinstance(document, dict)
            or set(document)
            != SOURCE_DOCUMENT_KEYS
        ):
            raise ReplayError(
                "source document fields mismatch"
            )

        if document["doc_id"] != doc_id:
            raise ReplayError(
                "source doc_id order mismatch"
            )

        if document["document_type"] != "regular_file":
            raise ReplayError(
                "source document type mismatch"
            )

        raw_hex = document[
            "relative_path_bytes_hex"
        ]

        if (
            not isinstance(raw_hex, str)
            or raw_hex != raw_hex.lower()
            or len(raw_hex) % 2
        ):
            raise ReplayError(
                "invalid source path hex"
            )

        try:
            raw_path = bytes.fromhex(raw_hex)
        except ValueError as error:
            raise ReplayError(
                "invalid source path hex"
            ) from error

        if (
            not raw_path
            or raw_path.startswith(b"/")
            or b"\x00" in raw_path
            or any(
                component in (b"", b".", b"..")
                for component
                in raw_path.split(b"/")
            )
        ):
            raise ReplayError(
                "unsafe source path bytes"
            )

        if (
            document["display_path"]
            != raw_path.decode(
                "utf-8",
                errors="backslashreplace",
            )
        ):
            raise ReplayError(
                "source display path mismatch"
            )

        raw_paths.append(raw_path)

        expected_snapshot = (
            f"documents/doc_{doc_id:08d}.bin"
        )

        if (
            document["snapshot_path"]
            != expected_snapshot
        ):
            raise ReplayError(
                "source snapshot path mismatch"
            )

        length = require_uint(
            document["byte_length"],
            "source byte length",
        )
        declared_hash = validate_sha256(
            document["sha256"],
            "source document",
        )

        payload = (
            bundle
            / "source"
            / expected_snapshot
        )

        status = os.lstat(payload)

        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
        ):
            raise ReplayError(
                "source payload must be regular"
            )

        if status.st_size != length:
            raise ReplayError(
                "source payload size mismatch"
            )

        if sha256_file(payload) != declared_hash:
            raise ReplayError(
                "source payload SHA256 mismatch"
            )

        total_bytes += length

    if raw_paths != sorted(raw_paths):
        raise ReplayError(
            "source paths not canonically ordered"
        )

    if len(set(raw_paths)) != len(raw_paths):
        raise ReplayError(
            "duplicate source path"
        )

    if manifest["total_source_bytes"] != total_bytes:
        raise ReplayError(
            "source total byte count mismatch"
        )

    corpus_id = source_corpus_identity(
        documents
    )
    source_manifest_id = (
        source_manifest_identity(documents)
    )

    if manifest["corpus_id"] != corpus_id:
        raise ReplayError(
            "source corpus identity mismatch"
        )

    if (
        manifest["source_manifest_id"]
        != source_manifest_id
    ):
        raise ReplayError(
            "source manifest identity mismatch"
        )

    manifest_hash = sha256_file(
        manifest_path
    )

    expected_complete = {
        "complete_version":
            SOURCE_COMPLETE_VERSION,
        "manifest_sha256": manifest_hash,
        "corpus_id": corpus_id,
        "source_manifest_id":
            source_manifest_id,
        "document_count": len(documents),
    }

    if (
        set(complete) != SOURCE_COMPLETE_KEYS
        or complete != expected_complete
    ):
        raise ReplayError(
            "source complete marker mismatch"
        )

    return manifest, manifest_hash


def runtime_payload_path(
    bundle: Path,
    original_path: str,
) -> Path:
    prefix = "runtime_index_v1/"

    if not original_path.startswith(prefix):
        raise ReplayError(
            "runtime artifact path prefix mismatch"
        )

    relative = original_path[len(prefix):]

    return bundle / "runtime" / relative


def verify_runtime(
    bundle: Path,
    source: dict[str, Any],
    source_manifest_hash: str,
) -> tuple[dict[str, Any], str]:
    manifest_path = bundle / RUNTIME_MANIFEST_PATH
    complete_path = bundle / RUNTIME_COMPLETE_PATH

    manifest = load_canonical_json(
        manifest_path
    )
    complete = load_canonical_json(
        complete_path
    )

    if set(manifest) != RUNTIME_MANIFEST_KEYS:
        raise ReplayError(
            "runtime manifest fields mismatch"
        )

    constants = {
        "manifest_version":
            RUNTIME_MANIFEST_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "construction_status": "complete",
        "index_topology":
            INDEX_TOPOLOGY,
        "logical_sentinel":
            LOGICAL_SENTINEL,
        "alphabet_size":
            ALPHABET_SIZE,
        "checkpoint_step":
            CHECKPOINT_STEP,
        "source_manifest_name":
            "source_manifest_v1.json",
    }

    for field, expected in constants.items():
        if manifest.get(field) != expected:
            raise ReplayError(
                "runtime constant mismatch: "
                + field
            )

    if (
        manifest["source_manifest_sha256"]
        != source_manifest_hash
    ):
        raise ReplayError(
            "runtime source-manifest hash mismatch"
        )

    if manifest["corpus_id"] != source["corpus_id"]:
        raise ReplayError(
            "runtime corpus ID mismatch"
        )

    if (
        manifest["source_manifest_id"]
        != source["source_manifest_id"]
    ):
        raise ReplayError(
            "runtime source-manifest ID mismatch"
        )

    binaries = manifest["runtime_binaries"]

    if not isinstance(binaries, list):
        raise ReplayError(
            "runtime binaries must be a list"
        )

    binary_names = []

    for record in binaries:
        if (
            not isinstance(record, dict)
            or set(record) != RUNTIME_BINARY_KEYS
        ):
            raise ReplayError(
                "runtime binary fields mismatch"
            )

        name = record["name"]

        if not isinstance(name, str) or not name:
            raise ReplayError(
                "invalid runtime binary name"
            )

        binary_names.append(name)

        size = require_uint(
            record["size_bytes"],
            "runtime binary size",
        )
        declared_hash = validate_sha256(
            record["sha256"],
            "runtime binary",
        )

        path = bundle / "bin" / name
        status = os.lstat(path)

        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
            or not os.access(path, os.X_OK)
        ):
            raise ReplayError(
                "runtime binary unavailable: "
                + name
            )

        if (
            status.st_size != size
            or sha256_file(path) != declared_hash
        ):
            raise ReplayError(
                "runtime binary commitment mismatch: "
                + name
            )

    if binary_names != sorted(binary_names):
        raise ReplayError(
            "runtime binary order mismatch"
        )

    documents = manifest["documents"]

    if (
        not isinstance(documents, list)
        or len(documents)
        != source["document_count"]
        or manifest["document_count"]
        != len(documents)
    ):
        raise ReplayError(
            "runtime document count mismatch"
        )

    total_runtime_bytes = 0

    expected_formats = {
        "sa": "GLYPH_SA_BINARY_V1",
        "bwt": "GLYPH_BWT_BINARY_V1",
        "fm": "GLYPH_FM_BINARY_V1",
    }

    for doc_id, document in enumerate(documents):
        if (
            not isinstance(document, dict)
            or set(document)
            != RUNTIME_DOCUMENT_KEYS
        ):
            raise ReplayError(
                "runtime document fields mismatch"
            )

        if document["doc_id"] != doc_id:
            raise ReplayError(
                "runtime doc_id mismatch"
            )

        source_document = source[
            "documents"
        ][doc_id]

        if (
            document["source_snapshot_path"]
            != source_document["snapshot_path"]
            or document["source_byte_length"]
            != source_document["byte_length"]
            or document["source_sha256"]
            != source_document["sha256"]
        ):
            raise ReplayError(
                "runtime source binding mismatch"
            )

        expected_directory = (
            "runtime_index_v1/"
            f"documents/doc_{doc_id:08d}"
        )

        if (
            document["index_directory"]
            != expected_directory
        ):
            raise ReplayError(
                "runtime index-directory mismatch"
            )

        for role in ("sa", "bwt", "fm"):
            artifact = document[role]

            if (
                not isinstance(artifact, dict)
                or set(artifact)
                != RUNTIME_ARTIFACT_KEYS
            ):
                raise ReplayError(
                    "runtime artifact fields mismatch"
                )

            expected_path = (
                expected_directory
                + f"/{role}.bin"
            )

            if (
                artifact["path"]
                != expected_path
                or artifact["format"]
                != expected_formats[role]
            ):
                raise ReplayError(
                    "runtime artifact metadata mismatch"
                )

            size = require_uint(
                artifact["size_bytes"],
                "runtime artifact size",
            )
            declared_hash = validate_sha256(
                artifact["sha256"],
                "runtime artifact",
            )

            path = runtime_payload_path(
                bundle,
                artifact["path"],
            )
            status = os.lstat(path)

            if (
                stat.S_ISLNK(status.st_mode)
                or not stat.S_ISREG(
                    status.st_mode
                )
            ):
                raise ReplayError(
                    "runtime artifact must be regular"
                )

            if (
                status.st_size != size
                or sha256_file(path)
                != declared_hash
            ):
                raise ReplayError(
                    "runtime artifact commitment mismatch"
                )

            total_runtime_bytes += size

    if (
        manifest["total_source_bytes"]
        != source["total_source_bytes"]
        or manifest["total_runtime_bytes"]
        != total_runtime_bytes
    ):
        raise ReplayError(
            "runtime byte totals mismatch"
        )

    runtime_index_id = (
        runtime_index_identity(manifest)
    )

    if (
        manifest["runtime_index_id"]
        != runtime_index_id
    ):
        raise ReplayError(
            "runtime index identity mismatch"
        )

    manifest_hash = sha256_file(
        manifest_path
    )

    expected_complete = {
        "complete_version":
            RUNTIME_COMPLETE_VERSION,
        "runtime_manifest_sha256":
            manifest_hash,
        "runtime_index_id":
            runtime_index_id,
        "corpus_id":
            manifest["corpus_id"],
        "source_manifest_id":
            manifest["source_manifest_id"],
        "document_count":
            manifest["document_count"],
        "total_runtime_bytes":
            total_runtime_bytes,
    }

    if (
        set(complete) != RUNTIME_COMPLETE_KEYS
        or complete != expected_complete
    ):
        raise ReplayError(
            "runtime complete marker mismatch"
        )

    return manifest, manifest_hash


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )

    if result.returncode != 0:
        raise ReplayError(
            "bundled runtime command failed: "
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout[-4000:]
            + "\nstderr:\n"
            + result.stderr[-4000:]
        )

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReplayError(
            "bundled runtime returned invalid JSON"
        ) from error

    if not isinstance(value, dict):
        raise ReplayError(
            "bundled runtime JSON is not an object"
        )

    return value


def validate_interval(
    value: Any,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
    ):
        raise ReplayError(
            "invalid FM interval"
        )

    left = require_uint(value[0], "interval left")
    right = require_uint(value[1], "interval right")

    if left > right:
        raise ReplayError(
            "reversed FM interval"
        )

    return left, right


def validate_count(
    value: dict[str, Any],
    query: bytes,
) -> dict[str, Any]:
    if set(value) != COUNT_KEYS:
        raise ReplayError(
            "count result fields mismatch"
        )

    constants = {
        "ok": True,
        "format": "GLYPH_QUERY_BINARY_V1",
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "alphabet_size": 257,
        "logical_sentinel": 256,
    }

    for field, expected in constants.items():
        if value.get(field) != expected:
            raise ReplayError(
                "count result mismatch: "
                + field
            )

    left, right = validate_interval(
        value["interval"]
    )
    count = require_uint(
        value["count"],
        "count",
    )

    if count != right - left:
        raise ReplayError(
            "count differs from interval"
        )

    return {
        "interval": [left, right],
        "count": count,
    }


def validate_locate(
    value: dict[str, Any],
    query: bytes,
    count: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    expected_keys = set(LOCATE_KEYS)

    if limit is not None:
        expected_keys.add("max_offsets")

    if set(value) != expected_keys:
        raise ReplayError(
            "locate result fields mismatch"
        )

    constants = {
        "ok": True,
        "format":
            "GLYPH_QUERY_LOCATE_BINARY_V1",
        "runtime_profile":
            RUNTIME_PROFILE,
        "document_count": 1,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "alphabet_size": 257,
        "logical_sentinel": 256,
        "byte_check": True,
    }

    for field, expected in constants.items():
        if value.get(field) != expected:
            raise ReplayError(
                "locate result mismatch: "
                + field
            )

    left, right = validate_interval(
        value["interval"]
    )

    if [left, right] != count["interval"]:
        raise ReplayError(
            "count/locate interval mismatch"
        )

    match_count = require_uint(
        value["match_count"],
        "locate match count",
    )

    if match_count != count["count"]:
        raise ReplayError(
            "count/locate count mismatch"
        )

    offsets = value["offsets"]

    if (
        not isinstance(offsets, list)
        or any(
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
            for offset in offsets
        )
    ):
        raise ReplayError(
            "invalid locate offsets"
        )

    if (
        offsets != sorted(offsets)
        or len(set(offsets)) != len(offsets)
    ):
        raise ReplayError(
            "locate offsets not canonical"
        )

    returned_count = require_uint(
        value["returned_count"],
        "returned count",
    )

    expected_returned = (
        match_count
        if limit is None
        else min(limit, match_count)
    )

    if (
        returned_count != len(offsets)
        or returned_count != expected_returned
    ):
        raise ReplayError(
            "locate returned count mismatch"
        )

    bounded = (
        returned_count < match_count
    )

    if (
        value["bounded"] is not bounded
        or value["offsets_complete"]
        is not (not bounded)
    ):
        raise ReplayError(
            "locate bounded metadata mismatch"
        )

    if (
        limit is not None
        and value["max_offsets"] != limit
    ):
        raise ReplayError(
            "locate max_offsets mismatch"
        )

    if (
        value["coordinates"]
        != [[0, offset] for offset in offsets]
    ):
        raise ReplayError(
            "locate coordinates mismatch"
        )

    return {
        "interval": [left, right],
        "match_count": match_count,
        "offsets": offsets,
        "returned_count": returned_count,
    }


def verify_query_binary_commitments(
    bundle: Path,
    commitments: Any,
) -> list[dict[str, Any]]:
    if not isinstance(commitments, list):
        raise ReplayError(
            "query commitments must be list"
        )

    names = []

    for record in commitments:
        if (
            not isinstance(record, dict)
            or set(record)
            != RUNTIME_BINARY_KEYS
        ):
            raise ReplayError(
                "query binary fields mismatch"
            )

        name = record["name"]
        names.append(name)

        path = bundle / "bin" / name
        status = os.lstat(path)

        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
            or not os.access(path, os.X_OK)
        ):
            raise ReplayError(
                "query binary unavailable: "
                + name
            )

        if (
            status.st_size
            != require_uint(
                record["size_bytes"],
                "query binary size",
            )
            or sha256_file(path)
            != validate_sha256(
                record["sha256"],
                "query binary",
            )
        ):
            raise ReplayError(
                "query binary commitment mismatch: "
                + name
            )

    expected = [
        "query_fm_binary_v1",
        "query_fm_locate_binary_v1",
    ]

    if names != expected:
        raise ReplayError(
            "query binary set mismatch"
        )

    return commitments


def byte_check(
    path: Path,
    query: bytes,
    offsets: Sequence[int],
) -> None:
    data = path.read_bytes()

    for offset in offsets:
        if (
            offset > len(data)
            or len(query) > len(data) - offset
            or data[
                offset:offset + len(query)
            ] != query
        ):
            raise ReplayError(
                "independent byte check failed"
            )


def replay_query(
    bundle: Path,
    source: dict[str, Any],
    source_manifest_hash: str,
    runtime: dict[str, Any],
    runtime_manifest_hash: str,
) -> dict[str, Any]:
    artifact_path = bundle / ARTIFACT_PATH
    artifact = load_canonical_json(
        artifact_path
    )
    query = (bundle / QUERY_PATH).read_bytes()

    if not query:
        raise ReplayError("EMPTY_QUERY")

    if (
        artifact.get("result_version")
        != QUERY_RESULT_VERSION
        or artifact.get("query_hex")
        != query.hex()
        or artifact.get("query_length_bytes")
        != len(query)
        or artifact.get("query_sha256")
        != sha256_bytes(query)
    ):
        raise ReplayError(
            "query identity mismatch"
        )

    if (
        artifact.get("corpus_id")
        != source["corpus_id"]
        or artifact.get("source_manifest_id")
        != source["source_manifest_id"]
        or artifact.get("runtime_index_id")
        != runtime["runtime_index_id"]
        or artifact.get("source_manifest_sha256")
        != source_manifest_hash
        or artifact.get("runtime_manifest_sha256")
        != runtime_manifest_hash
    ):
        raise ReplayError(
            "artifact corpus/runtime binding mismatch"
        )

    commitments = (
        verify_query_binary_commitments(
            bundle,
            artifact.get(
                "query_binary_commitments"
            ),
        )
    )

    max_offsets = artifact.get("max_offsets")

    if max_offsets is not None:
        require_uint(max_offsets, "max_offsets")

    count_binary = (
        bundle / "bin/query_fm_binary_v1"
    )
    locate_binary = (
        bundle
        / "bin/query_fm_locate_binary_v1"
    )

    coordinates: list[dict[str, Any]] = []
    document_results: list[dict[str, Any]] = []
    matched_documents: list[dict[str, Any]] = []
    total_match_count = 0

    for doc_id, (
        source_record,
        runtime_record,
    ) in enumerate(
        zip(
            source["documents"],
            runtime["documents"],
        )
    ):
        source_path = (
            bundle
            / "source"
            / source_record["snapshot_path"]
        )

        fm_path = runtime_payload_path(
            bundle,
            runtime_record["fm"]["path"],
        )
        bwt_path = runtime_payload_path(
            bundle,
            runtime_record["bwt"]["path"],
        )
        sa_path = runtime_payload_path(
            bundle,
            runtime_record["sa"]["path"],
        )

        count_raw = run_json([
            str(count_binary),
            str(fm_path),
            str(bwt_path),
            query.hex(),
        ])
        count = validate_count(
            count_raw,
            query,
        )

        local_limit = (
            None
            if max_offsets is None
            else max(
                max_offsets - len(coordinates),
                0,
            )
        )

        command = [
            str(locate_binary),
            str(fm_path),
            str(bwt_path),
            str(sa_path),
            str(source_path),
            query.hex(),
        ]

        if local_limit is not None:
            command.append(str(local_limit))

        locate = validate_locate(
            run_json(command),
            query,
            count,
            local_limit,
        )

        byte_check(
            source_path,
            query,
            locate["offsets"],
        )

        total_match_count += count["count"]

        document_results.append({
            "doc_id": doc_id,
            "interval": count["interval"],
            "match_count": count["count"],
            "returned_count":
                locate["returned_count"],
        })

        if count["count"] > 0:
            matched_documents.append({
                "doc_id": doc_id,
                "relative_path_bytes_hex":
                    source_record[
                        "relative_path_bytes_hex"
                    ],
                "display_path":
                    source_record["display_path"],
                "source_sha256":
                    source_record["sha256"],
                "interval":
                    count["interval"],
                "match_count":
                    count["count"],
                "returned_count":
                    locate["returned_count"],
            })

        for offset in locate["offsets"]:
            coordinates.append({
                "doc_id": doc_id,
                "doc_offset": offset,
                "coordinate": [
                    doc_id,
                    offset,
                ],
                "relative_path_bytes_hex":
                    source_record[
                        "relative_path_bytes_hex"
                    ],
                "display_path":
                    source_record["display_path"],
                "source_sha256":
                    source_record["sha256"],
                "byte_check": True,
            })

    numeric = [
        tuple(item["coordinate"])
        for item in coordinates
    ]

    if (
        numeric != sorted(numeric)
        or len(set(numeric)) != len(numeric)
    ):
        raise ReplayError(
            "global coordinates not canonical"
        )

    returned_count = len(coordinates)
    bounded = (
        returned_count < total_match_count
    )

    expected_returned = (
        total_match_count
        if max_offsets is None
        else min(
            max_offsets,
            total_match_count,
        )
    )

    if returned_count != expected_returned:
        raise ReplayError(
            "global bounded count mismatch"
        )

    result = {
        "ok": True,
        "result_version":
            QUERY_RESULT_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "index_topology":
            INDEX_TOPOLOGY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "corpus_id": source["corpus_id"],
        "source_manifest_id":
            source["source_manifest_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "source_manifest_sha256":
            source_manifest_hash,
        "runtime_manifest_sha256":
            runtime_manifest_hash,
        "query_binary_commitments":
            commitments,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256":
            sha256_bytes(query),
        "max_offsets": max_offsets,
        "document_count":
            source["document_count"],
        "documents_queried":
            len(document_results),
        "matched_document_count":
            len(matched_documents),
        "per_document_results_sha256":
            per_document_digest(
                document_results
            ),
        "matched_documents":
            matched_documents,
        "match_count":
            total_match_count,
        "coordinates": coordinates,
        "returned_count":
            returned_count,
        "bounded": bounded,
        "offsets_complete":
            not bounded,
        "byte_check": True,
        "display_paths_authoritative":
            False,
        "original_source_directory_used":
            False,
    }

    result["query_result_id"] = (
        query_result_identity(result)
    )

    if result != artifact:
        raise ReplayError(
            "replayed query result differs "
            "from committed artifact"
        )

    return result


def verify_bundle(
    bundle: Path,
) -> dict[str, Any]:
    bundle = bundle.absolute()

    try:
        root_status = os.lstat(bundle)
    except OSError as error:
        raise ReplayError(
            "bundle directory unavailable"
        ) from error

    if (
        stat.S_ISLNK(root_status.st_mode)
        or not stat.S_ISDIR(root_status.st_mode)
    ):
        raise ReplayError(
            "bundle path must be real directory"
        )

    manifest_path = bundle / MANIFEST_NAME
    complete_path = bundle / COMPLETE_NAME

    manifest = load_canonical_json(
        manifest_path
    )
    complete = load_canonical_json(
        complete_path
    )

    if set(manifest) != BUNDLE_MANIFEST_KEYS:
        raise ReplayError(
            "bundle manifest fields mismatch"
        )

    constants = {
        "bundle_manifest_version":
            BUNDLE_MANIFEST_VERSION,
        "bundle_version":
            BUNDLE_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "construction_status": "complete",
        "artifact_version":
            QUERY_RESULT_VERSION,
        "external_dependencies": [],
        "network_dependency_required":
            False,
        "repository_dependency_required":
            False,
    }

    for field, expected in constants.items():
        if manifest.get(field) != expected:
            raise ReplayError(
                "bundle manifest constant mismatch: "
                + field
            )

    files = manifest.get("files")

    if not isinstance(files, list):
        raise ReplayError(
            "bundle files must be list"
        )

    if manifest["file_count"] != len(files):
        raise ReplayError(
            "bundle file_count mismatch"
        )

    paths: list[str] = []

    for entry in files:
        if (
            not isinstance(entry, dict)
            or set(entry) != FILE_KEYS
        ):
            raise ReplayError(
                "bundle file record mismatch"
            )

        path = safe_relative_path(
            entry["path"]
        )
        paths.append(path)

        if (
            not isinstance(entry["role"], str)
            or not entry["role"]
        ):
            raise ReplayError(
                "invalid bundle file role"
            )

        require_uint(
            entry["size_bytes"],
            "bundle file size",
        )
        validate_sha256(
            entry["sha256"],
            "bundle payload",
        )

        if not isinstance(
            entry["executable"],
            bool,
        ):
            raise ReplayError(
                "invalid executable flag"
            )

    if paths != sorted(paths):
        raise ReplayError(
            "bundle files not canonically ordered"
        )

    if len(set(paths)) != len(paths):
        raise ReplayError(
            "duplicate bundle path"
        )

    actual_files = bundle_tree_files(bundle)
    expected_files = (
        set(paths)
        | {
            MANIFEST_NAME,
            COMPLETE_NAME,
        }
    )

    if actual_files != expected_files:
        raise ReplayError(
            "bundle exact coverage mismatch; "
            f"missing={sorted(expected_files - actual_files)}; "
            f"extra={sorted(actual_files - expected_files)}"
        )

    for entry in files:
        verify_declared_file(
            bundle,
            entry,
        )

    bundle_root = bundle_root_identity(
        files
    )

    if (
        manifest["bundle_root_sha256"]
        != bundle_root
    ):
        raise ReplayError(
            "bundle root SHA256 mismatch"
        )

    manifest_hash = sha256_file(
        manifest_path
    )

    expected_complete = {
        "complete_version":
            BUNDLE_COMPLETE_VERSION,
        "bundle_manifest_sha256":
            manifest_hash,
        "bundle_root_sha256":
            bundle_root,
        "query_result_id":
            manifest["query_result_id"],
        "runtime_index_id":
            manifest["runtime_index_id"],
        "corpus_id":
            manifest["corpus_id"],
        "file_count":
            manifest["file_count"],
    }

    if (
        set(complete) != BUNDLE_COMPLETE_KEYS
        or complete != expected_complete
    ):
        raise ReplayError(
            "bundle complete marker mismatch"
        )

    source, source_hash = verify_source(
        bundle
    )
    runtime, runtime_hash = verify_runtime(
        bundle,
        source,
        source_hash,
    )
    result = replay_query(
        bundle,
        source,
        source_hash,
        runtime,
        runtime_hash,
    )

    bindings = {
        "corpus_id": source["corpus_id"],
        "source_manifest_id":
            source["source_manifest_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "query_result_id":
            result["query_result_id"],
        "document_count":
            source["document_count"],
    }

    for field, expected in bindings.items():
        if manifest.get(field) != expected:
            raise ReplayError(
                "bundle binding mismatch: "
                + field
            )

    return {
        "ok": True,
        "bundle_version":
            BUNDLE_VERSION,
        "bundle_manifest_version":
            BUNDLE_MANIFEST_VERSION,
        "bundle_root_sha256":
            bundle_root,
        "bundle_manifest_sha256":
            manifest_hash,
        "corpus_id":
            source["corpus_id"],
        "source_manifest_id":
            source["source_manifest_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "query_result_id":
            result["query_result_id"],
        "document_count":
            source["document_count"],
        "match_count":
            result["match_count"],
        "returned_count":
            result["returned_count"],
        "bounded": result["bounded"],
        "payload_hashes_verified": True,
        "exact_manifest_coverage_verified":
            True,
        "source_manifest_verified": True,
        "runtime_manifest_verified": True,
        "compiled_query_replay_verified":
            True,
        "byte_check_verified": True,
        "repository_dependency_required":
            False,
        "network_dependency_required":
            False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        required=True,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        result = verify_bundle(
            Path(args.bundle)
        )

        print(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "bundle_version":
                        BUNDLE_VERSION,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
