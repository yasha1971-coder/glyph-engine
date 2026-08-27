#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_index_v1 import (  # noqa: E402
    INDEX_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
)
from glyph_operator_query_v1 import (  # noqa: E402
    execute_operator_query,
)


MAX_U64 = (2**64) - 1

ROOT_VERSION = "GLYPH_COMPOSITION_ROOT_V1"
ROOT_PUBLICATION_STATUS = "COMPLETE"
RESULT_VERSION = (
    "GLYPH_COMPOSITION_REFERENCE_RESULT_V1"
)
REPLAY_VERSION = (
    "GLYPH_COMPOSITION_INDEPENDENT_REPLAY_V1"
)

COMPOSITION_POLICY = (
    "ORDERED_CONTIGUOUS_RUNTIME_UNITS_V1"
)
COVERAGE_POLICY = "ALL_ROOT_BLOCKS_REQUIRED_V1"
BOUNDARY_POLICY = "DOCUMENT_LOCAL_MATCHES_ONLY_V1"
RESULT_IDENTITY_VERSION = (
    "GLYPH_COMPOSITION_REFERENCE_"
    "RESULT_IDENTITY_V1"
)

ROOT_KEYS = {
    "format",
    "publication_status",
    "global_document_count",
    "block_count",
    "runtime_corpus_id",
    "source_manifest_id",
    "blocks",
    "composition_root_id",
}

ROOT_BLOCK_KEYS = {
    "block_ordinal",
    "block_document_count",
    "runtime_index_id",
    "runtime_manifest_sha256",
}

RESULT_KEYS = {
    "ok",
    "format",
    "runtime_corpus_id",
    "source_manifest_id",
    "composition_root_id",
    "query_hex",
    "query_length_bytes",
    "query_sha256",
    "max_offsets",
    "match_count",
    "returned_count",
    "bounded",
    "offsets_complete",
    "coordinates",
    "expected_blocks",
    "verified_blocks",
    "queried_blocks",
    "composition_policy",
    "coverage_policy",
    "document_boundary_policy",
    "composition_result_id",
}


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedBlock:
    ordinal: int
    start: int
    end: int
    corpus: Path
    runtime_index_id: str
    runtime_manifest_sha256: str

    @property
    def document_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class VerifiedDocument:
    global_doc_id: int
    path_bytes: bytes
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class VerifiedRoot:
    blocks: tuple[VerifiedBlock, ...]
    documents: tuple[VerifiedDocument, ...]
    document_count: int
    runtime_corpus_id: str
    source_manifest_id: str
    composition_root_id: str


def fail(
    error_class: str,
    message: str,
) -> ReplayError:
    return ReplayError(
        f"{error_class}: {message}"
    )


def canonical_json_bytes(
    value: Any,
    *,
    ensure_ascii: bool = False,
) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_u64(
    value: Any,
    field: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > MAX_U64
    ):
        raise fail(
            "COMPOSITION_E_LIMIT",
            f"invalid u64: {field}",
        )

    return value


def u64_be(
    value: Any,
    field: str,
) -> bytes:
    return require_u64(
        value,
        field,
    ).to_bytes(8, "big")


def checked_add(
    left: int,
    right: int,
    field: str,
) -> int:
    return require_u64(
        left + right,
        field,
    )


def require_sha256(
    value: Any,
    field: str,
) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            f"invalid SHA256: {field}",
        )

    return bytes.fromhex(value)


def load_canonical_object(
    path: Path,
    label: str,
    *,
    ensure_ascii: bool = False,
) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
    ):
        raise fail(
            "COMPOSITION_E_VERIFY",
            f"{label} is unavailable",
        )

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8")
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise fail(
            "COMPOSITION_E_VERIFY",
            f"invalid {label}",
        ) from error

    if not isinstance(value, dict):
        raise fail(
            "COMPOSITION_E_VERIFY",
            f"{label} is not an object",
        )

    if raw != canonical_json_bytes(
        value,
        ensure_ascii=ensure_ascii,
    ):
        raise fail(
            "COMPOSITION_E_VERIFY",
            f"{label} is not canonical JSON",
        )

    return value


def safe_relative_path(
    value: Any,
    field: str,
) -> Path:
    if (
        not isinstance(value, str)
        or value == ""
        or "\\" in value
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            f"invalid relative path: {field}",
        )

    path = Path(value)

    if (
        path.is_absolute()
        or any(
            part in ("", ".", "..")
            for part in path.parts
        )
        or path.as_posix() != value
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            f"non-canonical relative path: {field}",
        )

    return path


def runtime_corpus_id(
    documents: Sequence[VerifiedDocument],
) -> str:
    preimage = bytearray(
        b"GLYPH_BINARY_RUNTIME_"
        b"CORPUS_IDENTITY_V1\x00"
    )
    preimage.extend(
        u64_be(
            len(documents),
            "document_count",
        )
    )

    for expected_doc_id, document in enumerate(
        documents
    ):
        if document.global_doc_id != expected_doc_id:
            raise fail(
                "COMPOSITION_E_IDENTITY",
                "non-canonical global doc_id",
            )

        preimage.extend(
            u64_be(
                expected_doc_id,
                "global_doc_id",
            )
        )
        preimage.extend(
            u64_be(
                len(document.payload),
                "byte_length",
            )
        )
        preimage.extend(
            require_sha256(
                document.sha256,
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def source_manifest_id(
    documents: Sequence[VerifiedDocument],
) -> str:
    preimage = bytearray(
        b"GLYPH_OPERATOR_"
        b"CORPUS_MANIFEST_V1\x00"
    )
    preimage.extend(
        u64_be(
            len(documents),
            "document_count",
        )
    )

    for expected_doc_id, document in enumerate(
        documents
    ):
        if document.global_doc_id != expected_doc_id:
            raise fail(
                "COMPOSITION_E_IDENTITY",
                "non-canonical global doc_id",
            )

        preimage.extend(
            u64_be(
                expected_doc_id,
                "global_doc_id",
            )
        )
        preimage.extend(
            u64_be(
                len(document.path_bytes),
                "path_length",
            )
        )
        preimage.extend(document.path_bytes)
        preimage.extend(
            u64_be(
                len(document.payload),
                "byte_length",
            )
        )
        preimage.extend(
            require_sha256(
                document.sha256,
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def composition_root_id(
    manifest: dict[str, Any],
) -> str:
    preimage = bytearray(
        ROOT_VERSION.encode("ascii")
        + b"\x00"
    )
    preimage.extend(
        require_sha256(
            manifest.get("runtime_corpus_id"),
            "runtime_corpus_id",
        )
    )
    preimage.extend(
        require_sha256(
            manifest.get("source_manifest_id"),
            "source_manifest_id",
        )
    )
    preimage.extend(
        u64_be(
            manifest.get(
                "global_document_count"
            ),
            "global_document_count",
        )
    )
    preimage.extend(
        u64_be(
            manifest.get("block_count"),
            "block_count",
        )
    )

    records = manifest.get("blocks")

    if not isinstance(records, list):
        raise fail(
            "COMPOSITION_E_ROOT_INVALID",
            "root blocks are not a list",
        )

    for expected, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != ROOT_BLOCK_KEYS
        ):
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "invalid root block record",
            )

        ordinal = require_u64(
            record.get("block_ordinal"),
            "block_ordinal",
        )

        if ordinal != expected:
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "non-canonical block ordinal",
            )

        preimage.extend(
            u64_be(ordinal, "block_ordinal")
        )
        preimage.extend(
            u64_be(
                record.get(
                    "block_document_count"
                ),
                "block_document_count",
            )
        )
        preimage.extend(
            require_sha256(
                record.get("runtime_index_id"),
                "runtime_index_id",
            )
        )
        preimage.extend(
            require_sha256(
                record.get(
                    "runtime_manifest_sha256"
                ),
                "runtime_manifest_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def decode_path_hex(value: Any) -> bytes:
    if (
        not isinstance(value, str)
        or value != value.lower()
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "invalid relative path hex",
        )

    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "invalid relative path hex",
        ) from error

    if decoded.hex() != value:
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "non-canonical relative path hex",
        )

    return decoded


def load_available_blocks(
    block_paths: Sequence[Path],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    available: dict[
        str,
        tuple[Path, dict[str, Any]],
    ] = {}

    for block_path in block_paths:
        if block_path.is_symlink():
            raise fail(
                "COMPOSITION_E_COVERAGE",
                "runtime unit must not be a symlink",
            )

        corpus = block_path.resolve()

        if (
            not corpus.is_dir()
        ):
            raise fail(
                "COMPOSITION_E_COVERAGE",
                "runtime unit is unavailable",
            )

        try:
            verify_runtime_index(
                corpus,
                require_current_binaries=True,
                rebuild=False,
            )
        except Exception as error:
            raise fail(
                "COMPOSITION_E_VERIFY",
                "runtime unit verification failed",
            ) from error

        runtime_manifest = load_canonical_object(
            corpus
            / RUNTIME_INDEX_DIRECTORY
            / INDEX_MANIFEST_NAME,
            "runtime manifest",
            ensure_ascii=True,
        )
        runtime_id = runtime_manifest.get(
            "runtime_index_id"
        )
        require_sha256(
            runtime_id,
            "runtime_index_id",
        )

        if runtime_id in available:
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "duplicate available runtime unit",
            )

        available[runtime_id] = (
            corpus,
            runtime_manifest,
        )

    return available


def verify_root(
    root_path: Path,
    block_paths: Sequence[Path],
) -> VerifiedRoot:
    manifest = load_canonical_object(
        root_path,
        "composition root",
    )

    if set(manifest) != ROOT_KEYS:
        raise fail(
            "COMPOSITION_E_ROOT_INVALID",
            "composition root key mismatch",
        )

    if manifest.get("format") != ROOT_VERSION:
        raise fail(
            "COMPOSITION_E_VERSION",
            "unsupported composition root",
        )

    if (
        manifest.get("publication_status")
        != ROOT_PUBLICATION_STATUS
    ):
        raise fail(
            "COMPOSITION_E_ROOT_INVALID",
            "composition root is incomplete",
        )

    document_count = require_u64(
        manifest.get("global_document_count"),
        "global_document_count",
    )
    block_count = require_u64(
        manifest.get("block_count"),
        "block_count",
    )
    records = manifest.get("blocks")

    if (
        block_count == 0
        or not isinstance(records, list)
        or len(records) != block_count
    ):
        raise fail(
            "COMPOSITION_E_ROOT_INVALID",
            "invalid root block count",
        )

    committed_root_id = manifest.get(
        "composition_root_id"
    )
    require_sha256(
        committed_root_id,
        "composition_root_id",
    )

    if composition_root_id(manifest) != committed_root_id:
        raise fail(
            "COMPOSITION_E_ROOT_MISMATCH",
            "composition root identity mismatch",
        )

    available = load_available_blocks(
        block_paths
    )
    verified_blocks: list[VerifiedBlock] = []
    documents: list[VerifiedDocument] = []
    seen_paths: set[bytes] = set()
    seen_runtime_ids: set[str] = set()
    global_doc_base = 0

    for expected_ordinal, record in enumerate(
        records
    ):
        if (
            not isinstance(record, dict)
            or set(record) != ROOT_BLOCK_KEYS
        ):
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "invalid root block record",
            )

        ordinal = require_u64(
            record.get("block_ordinal"),
            "block_ordinal",
        )
        block_document_count = require_u64(
            record.get("block_document_count"),
            "block_document_count",
        )
        runtime_id = record.get(
            "runtime_index_id"
        )
        runtime_manifest_sha256 = record.get(
            "runtime_manifest_sha256"
        )

        if ordinal != expected_ordinal:
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "non-canonical block ordinal",
            )

        if block_document_count == 0:
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "empty runtime unit",
            )

        require_sha256(
            runtime_id,
            "runtime_index_id",
        )
        require_sha256(
            runtime_manifest_sha256,
            "runtime_manifest_sha256",
        )

        if runtime_id in seen_runtime_ids:
            raise fail(
                "COMPOSITION_E_ROOT_INVALID",
                "duplicate runtime index identity",
            )

        seen_runtime_ids.add(runtime_id)
        physical = available.get(runtime_id)

        if physical is None:
            raise fail(
                "COMPOSITION_E_COVERAGE",
                "required runtime unit is unavailable",
            )

        corpus, runtime_manifest = physical
        runtime_manifest_path = (
            corpus
            / RUNTIME_INDEX_DIRECTORY
            / INDEX_MANIFEST_NAME
        )

        if (
            sha256_file(runtime_manifest_path)
            != runtime_manifest_sha256
        ):
            raise fail(
                "COMPOSITION_E_IDENTITY",
                "runtime manifest commitment mismatch",
            )

        source_manifest = load_canonical_object(
            corpus / SOURCE_MANIFEST_NAME,
            "source manifest",
            ensure_ascii=True,
        )

        if (
            source_manifest.get("corpus_id")
            != runtime_manifest.get("corpus_id")
            or source_manifest.get(
                "source_manifest_id"
            )
            != runtime_manifest.get(
                "source_manifest_id"
            )
        ):
            raise fail(
                "COMPOSITION_E_IDENTITY",
                "source/runtime identity mismatch",
            )

        source_records = source_manifest.get(
            "documents"
        )
        runtime_records = runtime_manifest.get(
            "documents"
        )

        if (
            not isinstance(source_records, list)
            or not isinstance(runtime_records, list)
            or len(source_records)
            != block_document_count
            or len(runtime_records)
            != block_document_count
        ):
            raise fail(
                "COMPOSITION_E_IDENTITY",
                "block document count mismatch",
            )

        range_end = checked_add(
            global_doc_base,
            block_document_count,
            "global_document_range_end",
        )

        verified_blocks.append(
            VerifiedBlock(
                ordinal=ordinal,
                start=global_doc_base,
                end=range_end,
                corpus=corpus,
                runtime_index_id=runtime_id,
                runtime_manifest_sha256=(
                    runtime_manifest_sha256
                ),
            )
        )

        for local_doc_id, (
            source_record,
            runtime_record,
        ) in enumerate(
            zip(source_records, runtime_records)
        ):
            if (
                not isinstance(source_record, dict)
                or not isinstance(runtime_record, dict)
                or source_record.get("doc_id")
                != local_doc_id
                or runtime_record.get("doc_id")
                != local_doc_id
            ):
                raise fail(
                    "COMPOSITION_E_IDENTITY",
                    "non-canonical local document",
                )

            path_bytes = decode_path_hex(
                source_record.get(
                    "relative_path_bytes_hex"
                )
            )

            if path_bytes in seen_paths:
                raise fail(
                    "COMPOSITION_E_IDENTITY",
                    "duplicate global source path",
                )

            seen_paths.add(path_bytes)
            byte_length = require_u64(
                source_record.get("byte_length"),
                "source_byte_length",
            )
            source_sha256 = source_record.get(
                "sha256"
            )
            require_sha256(
                source_sha256,
                "source_sha256",
            )

            if (
                runtime_record.get(
                    "source_byte_length"
                )
                != byte_length
                or runtime_record.get(
                    "source_sha256"
                )
                != source_sha256
            ):
                raise fail(
                    "COMPOSITION_E_IDENTITY",
                    "source/runtime document mismatch",
                )

            snapshot_relative = safe_relative_path(
                source_record.get("snapshot_path"),
                "snapshot_path",
            )
            snapshot_path = corpus / snapshot_relative

            if (
                not snapshot_path.is_file()
                or snapshot_path.is_symlink()
            ):
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "source snapshot unavailable",
                )

            payload = snapshot_path.read_bytes()

            if (
                len(payload) != byte_length
                or hashlib.sha256(
                    payload
                ).hexdigest() != source_sha256
            ):
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "source snapshot commitment mismatch",
                )

            documents.append(
                VerifiedDocument(
                    global_doc_id=(
                        global_doc_base
                        + local_doc_id
                    ),
                    path_bytes=path_bytes,
                    payload=payload,
                    sha256=source_sha256,
                )
            )

        global_doc_base = range_end

    if (
        global_doc_base != document_count
        or len(documents) != document_count
    ):
        raise fail(
            "COMPOSITION_E_COVERAGE",
            "global document coverage mismatch",
        )

    committed_runtime_id = manifest.get(
        "runtime_corpus_id"
    )
    committed_source_id = manifest.get(
        "source_manifest_id"
    )

    require_sha256(
        committed_runtime_id,
        "runtime_corpus_id",
    )
    require_sha256(
        committed_source_id,
        "source_manifest_id",
    )

    if runtime_corpus_id(documents) != committed_runtime_id:
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "global runtime corpus identity mismatch",
        )

    if source_manifest_id(documents) != committed_source_id:
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "global source manifest identity mismatch",
        )

    return VerifiedRoot(
        blocks=tuple(verified_blocks),
        documents=tuple(documents),
        document_count=document_count,
        runtime_corpus_id=committed_runtime_id,
        source_manifest_id=committed_source_id,
        composition_root_id=committed_root_id,
    )


def decode_query(
    result: dict[str, Any],
) -> bytes:
    query_hex = result.get("query_hex")

    if (
        not isinstance(query_hex, str)
        or query_hex == ""
        or query_hex != query_hex.lower()
        or len(query_hex) % 2 != 0
        or any(
            character not in "0123456789abcdef"
            for character in query_hex
        )
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "invalid query identity",
        )

    query = bytes.fromhex(query_hex)

    if (
        not query
        or query.hex() != query_hex
        or result.get("query_length_bytes")
        != len(query)
        or result.get("query_sha256")
        != hashlib.sha256(query).hexdigest()
    ):
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "query identity mismatch",
        )

    return query


def oracle_coordinates(
    documents: Sequence[VerifiedDocument],
    query: bytes,
) -> list[list[int]]:
    coordinates: list[list[int]] = []

    for document in documents:
        payload = document.payload

        if len(query) > len(payload):
            continue

        for offset in range(
            len(payload) - len(query) + 1
        ):
            if (
                payload[offset:offset + len(query)]
                == query
            ):
                coordinates.append([
                    document.global_doc_id,
                    offset,
                ])

    return coordinates


def query_runtime_units(
    root: VerifiedRoot,
    query: bytes,
) -> list[list[int]]:
    coordinates: list[list[int]] = []

    for block in root.blocks:
        try:
            result = execute_operator_query(
                block.corpus,
                query,
            )
        except Exception as error:
            raise fail(
                "COMPOSITION_E_RUNTIME",
                "runtime query failed",
            ) from error

        local_coordinates = result.get(
            "coordinates"
        )

        if (
            result.get("match_count")
            != result.get("returned_count")
            or result.get("bounded") is not False
            or result.get("offsets_complete") is not True
            or result.get("byte_check") is not True
            or not isinstance(local_coordinates, list)
        ):
            raise fail(
                "COMPOSITION_E_VERIFY",
                "incomplete block replay result",
            )

        block_coordinates: list[list[int]] = []

        for item in local_coordinates:
            if not isinstance(item, dict):
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "invalid local coordinate record",
                )

            coordinate = item.get("coordinate")

            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
            ):
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "invalid local coordinate",
                )

            local_doc_id = require_u64(
                coordinate[0],
                "local_doc_id",
            )
            doc_offset = require_u64(
                coordinate[1],
                "doc_offset",
            )

            if local_doc_id >= block.document_count:
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "local doc_id out of range",
                )

            global_doc_id = checked_add(
                block.start,
                local_doc_id,
                "global_doc_id",
            )
            document = root.documents[
                global_doc_id
            ]

            if (
                document.payload[
                    doc_offset:doc_offset + len(query)
                ]
                != query
            ):
                raise fail(
                    "COMPOSITION_E_VERIFY",
                    "runtime coordinate byte-check failed",
                )

            block_coordinates.append([
                global_doc_id,
                doc_offset,
            ])

        if block_coordinates != sorted(
            block_coordinates
        ):
            raise fail(
                "COMPOSITION_E_VERIFY",
                "non-canonical block coordinates",
            )

        if len(block_coordinates) != result.get(
            "match_count"
        ):
            raise fail(
                "COMPOSITION_E_VERIFY",
                "block count/coordinate mismatch",
            )

        coordinates.extend(block_coordinates)

    if coordinates != sorted(coordinates):
        raise fail(
            "COMPOSITION_E_VERIFY",
            "non-canonical global coordinates",
        )

    return coordinates


def composition_result_id(
    result: dict[str, Any],
) -> str:
    payload = dict(result)
    payload.pop("composition_result_id", None)
    preimage = (
        RESULT_IDENTITY_VERSION.encode("ascii")
        + b"\x00"
        + canonical_json_bytes(payload)
    )
    return hashlib.sha256(
        preimage
    ).hexdigest()


def require_coverage(
    result: dict[str, Any],
    block_count: int,
) -> None:
    expected = list(range(block_count))

    for field in (
        "expected_blocks",
        "verified_blocks",
        "queried_blocks",
    ):
        if result.get(field) != expected:
            raise fail(
                "COMPOSITION_E_COVERAGE",
                f"incomplete coverage: {field}",
            )


def verify_result(
    root: VerifiedRoot,
    result_path: Path,
) -> dict[str, Any]:
    result = load_canonical_object(
        result_path,
        "composition result",
    )

    if set(result) != RESULT_KEYS:
        raise fail(
            "COMPOSITION_E_VERIFY",
            "composition result key mismatch",
        )

    if (
        result.get("ok") is not True
        or result.get("format")
        != RESULT_VERSION
    ):
        raise fail(
            "COMPOSITION_E_VERSION",
            "unsupported composition result",
        )

    identity_checks = {
        "runtime_corpus_id":
            root.runtime_corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "composition_root_id":
            root.composition_root_id,
    }

    for field, expected in identity_checks.items():
        require_sha256(
            result.get(field),
            field,
        )

        if result.get(field) != expected:
            raise fail(
                "COMPOSITION_E_IDENTITY",
                f"result identity mismatch: {field}",
            )

    query = decode_query(result)
    max_offsets = result.get("max_offsets")

    if max_offsets is not None:
        max_offsets = require_u64(
            max_offsets,
            "max_offsets",
        )

    require_coverage(
        result,
        len(root.blocks),
    )

    policy_checks = {
        "composition_policy":
            COMPOSITION_POLICY,
        "coverage_policy":
            COVERAGE_POLICY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
    }

    for field, expected in policy_checks.items():
        if result.get(field) != expected:
            raise fail(
                "COMPOSITION_E_VERIFY",
                f"policy mismatch: {field}",
            )

    runtime_coordinates = query_runtime_units(
        root,
        query,
    )
    independent_coordinates = oracle_coordinates(
        root.documents,
        query,
    )

    if runtime_coordinates != independent_coordinates:
        raise fail(
            "COMPOSITION_E_VERIFY",
            "runtime/source oracle mismatch",
        )

    expected_returned = (
        independent_coordinates
        if max_offsets is None
        else independent_coordinates[:max_offsets]
    )
    expected_match_count = len(
        independent_coordinates
    )
    expected_returned_count = len(
        expected_returned
    )
    expected_bounded = (
        expected_returned_count
        < expected_match_count
    )

    if (
        require_u64(
            result.get("match_count"),
            "match_count",
        )
        != expected_match_count
        or require_u64(
            result.get("returned_count"),
            "returned_count",
        )
        != expected_returned_count
        or result.get("coordinates")
        != expected_returned
        or result.get("bounded")
        is not expected_bounded
        or result.get("offsets_complete")
        is not (not expected_bounded)
    ):
        raise fail(
            "COMPOSITION_E_VERIFY",
            "composition result replay mismatch",
        )

    for global_doc_id, doc_offset in expected_returned:
        document = root.documents[global_doc_id]

        if (
            document.payload[
                doc_offset:doc_offset + len(query)
            ]
            != query
        ):
            raise fail(
                "COMPOSITION_E_VERIFY",
                "returned coordinate byte-check failed",
            )

    committed_result_id = result.get(
        "composition_result_id"
    )
    require_sha256(
        committed_result_id,
        "composition_result_id",
    )

    if composition_result_id(result) != committed_result_id:
        raise fail(
            "COMPOSITION_E_IDENTITY",
            "composition result identity mismatch",
        )

    return {
        "ok": True,
        "format": REPLAY_VERSION,
        "composition_root_id":
            root.composition_root_id,
        "composition_result_id":
            committed_result_id,
        "runtime_corpus_id":
            root.runtime_corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "block_count":
            len(root.blocks),
        "document_count":
            root.document_count,
        "query_sha256":
            hashlib.sha256(query).hexdigest(),
        "match_count":
            expected_match_count,
        "returned_count":
            expected_returned_count,
        "bounded":
            expected_bounded,
        "complete_block_coverage": True,
        "runtime_query_replayed": True,
        "independent_source_oracle_replayed": True,
        "returned_coordinates_byte_checked": True,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently replay a GLYPH "
            "Composition V1 reference result."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Composition root manifest.",
    )
    parser.add_argument(
        "--result",
        type=Path,
        required=True,
        help="Canonical composition result.",
    )
    parser.add_argument(
        "--block",
        type=Path,
        action="append",
        required=True,
        help=(
            "Available operator runtime unit; "
            "repeat once per unit."
        ),
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    root = verify_root(
        arguments.root.absolute(),
        arguments.block,
    )
    summary = verify_result(
        root,
        arguments.result.absolute(),
    )

    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
    )
    print(
        "GLYPH COMPOSITION INDEPENDENT "
        "REPLAY OK"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "COMPOSITION_INDEPENDENT_"
                        "REPLAY_FAILURE"
                    ),
                    "error_type":
                        type(error).__name__,
                    "message": str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
