#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
INDEPENDENT_REPLAY = (
    TOOLS
    / "replay_composition_reference_v1.py"
)
INDEPENDENT_REPLAY_MARKER = (
    "GLYPH COMPOSITION INDEPENDENT "
    "REPLAY OK"
)

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_index_v1 import (  # noqa: E402
    INDEX_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    build_runtime_index,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    build_snapshot,
    load_canonical_json,
)
from glyph_operator_query_v1 import (  # noqa: E402
    execute_operator_query,
)

MAX_U64 = (2**64) - 1

ROOT_VERSION = "GLYPH_COMPOSITION_ROOT_V1"
RESULT_VERSION = (
    "GLYPH_COMPOSITION_REFERENCE_RESULT_V1"
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


NORMATIVE_MUTATION_REQUIREMENTS = (
    {
        "id": "M01",
        "description":
            "one required block removed",
        "status": "EXACT",
        "tests": (
            "missing_required_block",
        ),
    },
    {
        "id": "M02",
        "description":
            "one valid block substituted",
        "status": "EXACT",
        "tests": (
            "valid_block_substituted",
        ),
    },
    {
        "id": "M03",
        "description":
            "block order changed without "
            "updating the root",
        "status": "EXACT",
        "tests": (
            "reordered_block_records",
        ),
    },
    {
        "id": "M04",
        "description":
            "one block entry duplicated",
        "status": "EXACT",
        "tests": (
            "duplicated_block_identity",
        ),
    },
    {
        "id": "M05",
        "description":
            "one runtime manifest byte changed",
        "status": "EXACT",
        "tests": (
            "runtime_manifest_byte_changed",
        ),
    },
    {
        "id": "M06",
        "description":
            "one runtime manifest hash changed",
        "status": "EXACT",
        "tests": (
            "changed_runtime_manifest_commitment",
        ),
    },
    {
        "id": "M07",
        "description":
            "one runtime_index_id changed",
        "status": "EXACT",
        "tests": (
            "runtime_index_id_changed",
        ),
    },
    {
        "id": "M08",
        "description":
            "global runtime corpus ID changed",
        "status": "EXACT",
        "tests": (
            "changed_global_runtime_corpus_id",
        ),
    },
    {
        "id": "M09",
        "description":
            "global source manifest ID changed",
        "status": "EXACT",
        "tests": (
            "changed_global_source_manifest_id",
        ),
    },
    {
        "id": "M10",
        "description":
            "composition root ID changed",
        "status": "EXACT",
        "tests": (
            "changed_composition_root_id",
        ),
    },
    {
        "id": "M11",
        "description":
            "one source document byte changed",
        "status": "EXACT",
        "tests": (
            "source_document_byte_changed",
        ),
    },
    {
        "id": "M12",
        "description":
            "one block result omitted while "
            "coverage is claimed",
        "status": "EXACT",
        "tests": (
            "block_result_omitted_with_"
            "claimed_coverage",
        ),
    },
    {
        "id": "M13",
        "description":
            "incomplete coverage represented "
            "as zero matches",
        "status": "EXACT",
        "tests": (
            "partial_coverage_represented_as_zero",
        ),
    },
    {
        "id": "M14",
        "description":
            "merged coordinates reordered",
        "status": "EXACT",
        "tests": (
            "reordered_coordinates",
        ),
    },
    {
        "id": "M15",
        "description":
            "local-to-global document base changed",
        "status": "EXACT",
        "tests": (
            "wrong_global_doc_id",
        ),
    },
    {
        "id": "M16",
        "description":
            "integer overflow attempted",
        "status": "EXACT",
        "tests": (
            "global_match_count_"
            "overflow_attempted",
        ),
    },
    {
        "id": "M17",
        "description":
            "max_offsets applied independently "
            "per block",
        "status": "EXACT",
        "tests": (
            "max_offsets_applied_"
            "independently_per_block",
        ),
    },
    {
        "id": "M18",
        "description":
            "unsupported root version",
        "status": "EXACT",
        "tests": (
            "unsupported_root_format",
        ),
    },
    {
        "id": "M19",
        "description":
            "unsupported runtime profile",
        "status": "EXACT",
        "tests": (
            "unsupported_runtime_profile",
        ),
    },
    {
        "id": "M20",
        "description":
            "replay attempted against "
            "a different root",
        "status": "EXACT",
        "tests": (
            "replay_against_different_root",
        ),
    },
    {
        "id": "M21",
        "description":
            "document order changed",
        "status": "EXACT",
        "tests": (
            "document_order_changed",
        ),
    },
    {
        "id": "M22",
        "description":
            "empty document removed",
        "status": "EXACT",
        "tests": (
            "empty_document_removed",
        ),
    },
    {
        "id": "M23",
        "description":
            "duplicate document deduplicated",
        "status": "EXACT",
        "tests": (
            "duplicate_document_deduplicated",
        ),
    },
    {
        "id": "M24",
        "description":
            "physical concatenation used "
            "as the matching oracle",
        "status": "EXACT",
        "tests": (
            "physical_concatenation_"
            "cross_document",
            "physical_concatenation_"
            "cross_block",
        ),
    },
    {
        "id": "M25",
        "description":
            "stored byte-check success trusted "
            "without recomputation",
        "status": "EXACT",
        "tests": (
            "stored_byte_check_success_"
            "without_recomputation",
        ),
    },
)

ADDITIONAL_MUTATION_TESTS = (
    "incomplete_publication",
    "missing_verified_block",
    "missing_queried_block",
    "incorrect_complete_match_count",
    "incorrect_global_max_offsets_prefix",
    "false_bounded_completeness_flags",
    "incorrect_returned_count",
    "changed_composition_result_id",
)


ROOT_PUBLICATION_STATUS = "COMPLETE"

ROOT_MANIFEST_KEYS = {
    "format",
    "publication_status",
    "global_document_count",
    "block_count",
    "runtime_corpus_id",
    "source_manifest_id",
    "blocks",
    "composition_root_id",
}

ROOT_BLOCK_RECORD_KEYS = {
    "block_ordinal",
    "block_document_count",
    "runtime_index_id",
    "runtime_manifest_sha256",
}


class CompositionError(RuntimeError):
    pass


def root_error(
    error_class: str,
    message: str,
) -> CompositionError:
    return CompositionError(
        f"{error_class}: {message}"
    )



def result_error(
    error_class: str,
    message: str,
) -> CompositionError:
    return CompositionError(
        f"{error_class}: {message}"
    )



def artifact_error(
    error_class: str,
    message: str,
) -> CompositionError:
    return CompositionError(
        f"{error_class}: {message}"
    )



def aggregation_error(
    error_class: str,
    message: str,
) -> CompositionError:
    return CompositionError(
        f"{error_class}: {message}"
    )


@dataclass(frozen=True)
class Document:
    path: str
    data: bytes

    @property
    def path_bytes(self) -> bytes:
        return self.path.encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            self.data
        ).hexdigest()


@dataclass(frozen=True)
class Block:
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
class Root:
    name: str
    blocks: tuple[Block, ...]
    document_count: int
    corpus_id: str
    source_manifest_id: str
    composition_root_id: str


def canonical_json_bytes(
    value: Any,
) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def u64(
    value: int,
    field: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > MAX_U64
    ):
        raise CompositionError(
            f"invalid u64: {field}"
        )

    return value


def u64_be(
    value: int,
    field: str,
) -> bytes:
    return u64(
        value,
        field,
    ).to_bytes(
        8,
        "big",
    )


def checked_add(
    left: int,
    right: int,
    field: str,
) -> int:
    return u64(
        left + right,
        field,
    )



def checked_add_aggregation(
    left: int,
    right: int,
    field: str,
) -> int:
    try:
        return checked_add(
            left,
            right,
            field,
        )

    except CompositionError as error:
        raise aggregation_error(
            "COMPOSITION_E_LIMIT",
            f"aggregation integer overflow "
            f"or invalid value: {field}",
        ) from error


def raw_sha256(
    value: Any,
    field: str,
) -> bytes:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
    ):
        raise CompositionError(
            f"invalid sha256: {field}"
        )

    try:
        raw = bytes.fromhex(value)
    except ValueError as error:
        raise CompositionError(
            f"invalid sha256: {field}"
        ) from error

    if len(raw) != 32:
        raise CompositionError(
            f"invalid sha256 length: {field}"
        )

    return raw


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def runtime_corpus_id(
    documents: Sequence[Document],
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

    for doc_id, document in enumerate(
        documents
    ):
        preimage.extend(
            u64_be(
                doc_id,
                "doc_id",
            )
        )
        preimage.extend(
            u64_be(
                len(document.data),
                "byte_length",
            )
        )
        preimage.extend(
            raw_sha256(
                document.sha256,
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def source_manifest_id(
    documents: Sequence[Document],
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

    for doc_id, document in enumerate(
        documents
    ):
        path_bytes = document.path_bytes

        preimage.extend(
            u64_be(
                doc_id,
                "doc_id",
            )
        )
        preimage.extend(
            u64_be(
                len(path_bytes),
                "path_length",
            )
        )
        preimage.extend(path_bytes)
        preimage.extend(
            u64_be(
                len(document.data),
                "byte_length",
            )
        )
        preimage.extend(
            raw_sha256(
                document.sha256,
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def composition_root_id(
    corpus_id: str,
    manifest_id: str,
    document_count: int,
    blocks: Sequence[Block],
) -> str:
    preimage = bytearray(
        ROOT_VERSION.encode("ascii")
        + b"\x00"
    )

    preimage.extend(
        raw_sha256(
            corpus_id,
            "corpus_id",
        )
    )
    preimage.extend(
        raw_sha256(
            manifest_id,
            "source_manifest_id",
        )
    )
    preimage.extend(
        u64_be(
            document_count,
            "document_count",
        )
    )
    preimage.extend(
        u64_be(
            len(blocks),
            "block_count",
        )
    )

    for expected, block in enumerate(
        blocks
    ):
        if block.ordinal != expected:
            raise CompositionError(
                "non-canonical block ordinal"
            )

        preimage.extend(
            u64_be(
                block.ordinal,
                "block_ordinal",
            )
        )
        preimage.extend(
            u64_be(
                block.document_count,
                "block_document_count",
            )
        )
        preimage.extend(
            raw_sha256(
                block.runtime_index_id,
                "runtime_index_id",
            )
        )
        preimage.extend(
            raw_sha256(
                block.runtime_manifest_sha256,
                "runtime_manifest_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def make_root_manifest(
    root: Root,
) -> dict[str, Any]:
    return {
        "format": ROOT_VERSION,
        "publication_status":
            ROOT_PUBLICATION_STATUS,
        "global_document_count":
            root.document_count,
        "block_count":
            len(root.blocks),
        "runtime_corpus_id":
            root.corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "blocks": [
            {
                "block_ordinal":
                    block.ordinal,
                "block_document_count":
                    block.document_count,
                "runtime_index_id":
                    block.runtime_index_id,
                "runtime_manifest_sha256":
                    block.runtime_manifest_sha256,
            }
            for block in root.blocks
        ],
        "composition_root_id":
            root.composition_root_id,
    }


def recompute_root_identity_from_manifest(
    manifest: dict[str, Any],
) -> str:
    preimage = bytearray(
        ROOT_VERSION.encode("ascii")
        + b"\x00"
    )

    preimage.extend(
        raw_sha256(
            manifest.get(
                "runtime_corpus_id"
            ),
            "runtime_corpus_id",
        )
    )
    preimage.extend(
        raw_sha256(
            manifest.get(
                "source_manifest_id"
            ),
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
            manifest.get(
                "block_count"
            ),
            "block_count",
        )
    )

    blocks = manifest.get("blocks")

    if not isinstance(blocks, list):
        raise CompositionError(
            "root blocks are not a list"
        )

    for expected, record in enumerate(
        blocks
    ):
        if not isinstance(record, dict):
            raise CompositionError(
                "root block record "
                "is not an object"
            )

        ordinal = u64(
            record.get(
                "block_ordinal"
            ),
            "block_ordinal",
        )

        if ordinal != expected:
            raise CompositionError(
                "non-canonical root "
                "block ordinal"
            )

        preimage.extend(
            u64_be(
                ordinal,
                "block_ordinal",
            )
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
            raw_sha256(
                record.get(
                    "runtime_index_id"
                ),
                "runtime_index_id",
            )
        )
        preimage.extend(
            raw_sha256(
                record.get(
                    "runtime_manifest_sha256"
                ),
                "runtime_manifest_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def runtime_corpus_id_from_records(
    records: Sequence[dict[str, Any]],
) -> str:
    preimage = bytearray(
        b"GLYPH_BINARY_RUNTIME_"
        b"CORPUS_IDENTITY_V1\x00"
    )

    preimage.extend(
        u64_be(
            len(records),
            "global_document_count",
        )
    )

    for global_doc_id, record in enumerate(
        records
    ):
        preimage.extend(
            u64_be(
                global_doc_id,
                "global_doc_id",
            )
        )
        preimage.extend(
            u64_be(
                record["byte_length"],
                "byte_length",
            )
        )
        preimage.extend(
            raw_sha256(
                record["sha256"],
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def source_manifest_id_from_records(
    records: Sequence[dict[str, Any]],
) -> str:
    preimage = bytearray(
        b"GLYPH_OPERATOR_"
        b"CORPUS_MANIFEST_V1\x00"
    )

    preimage.extend(
        u64_be(
            len(records),
            "global_document_count",
        )
    )

    for global_doc_id, record in enumerate(
        records
    ):
        path_bytes = record["path_bytes"]

        if not isinstance(path_bytes, bytes):
            raise CompositionError(
                "path identity is not bytes"
            )

        preimage.extend(
            u64_be(
                global_doc_id,
                "global_doc_id",
            )
        )
        preimage.extend(
            u64_be(
                len(path_bytes),
                "path_length",
            )
        )
        preimage.extend(path_bytes)
        preimage.extend(
            u64_be(
                record["byte_length"],
                "byte_length",
            )
        )
        preimage.extend(
            raw_sha256(
                record["sha256"],
                "document_sha256",
            )
        )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def flatten_verified_identity_records(
    blocks: Sequence[Block],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    seen_paths: set[bytes] = set()

    for block in blocks:
        verify_physical_runtime_block(
            block
        )

        source_manifest_path = (
            block.corpus
            / SOURCE_MANIFEST_NAME
        )

        runtime_manifest_path = (
            block.corpus
            / RUNTIME_INDEX_DIRECTORY
            / INDEX_MANIFEST_NAME
        )

        actual_runtime_sha256 = (
            sha256_file(
                runtime_manifest_path
            )
        )

        if (
            actual_runtime_sha256
            != block.runtime_manifest_sha256
        ):
            raise artifact_error(
                "COMPOSITION_E_VERIFY",
                "runtime manifest hash "
                "does not match root",
            )

        source_manifest = (
            load_canonical_json(
                source_manifest_path
            )
        )

        runtime_manifest = (
            load_canonical_json(
                runtime_manifest_path
            )
        )

        if (
            runtime_manifest.get(
                "runtime_index_id"
            )
            != block.runtime_index_id
        ):
            raise artifact_error(
                "COMPOSITION_E_IDENTITY",
                "runtime index identity "
                "does not match root",
            )

        for identity_field in (
            "corpus_id",
            "source_manifest_id",
        ):
            if (
                source_manifest.get(
                    identity_field
                )
                != runtime_manifest.get(
                    identity_field
                )
            ):
                raise CompositionError(
                    "source/runtime identity "
                    f"mismatch: {identity_field}"
                )

        source_records = (
            source_manifest.get(
                "documents"
            )
        )

        runtime_records = (
            runtime_manifest.get(
                "documents"
            )
        )

        if (
            not isinstance(
                source_records,
                list,
            )
            or not isinstance(
                runtime_records,
                list,
            )
            or len(source_records)
            != block.document_count
            or len(runtime_records)
            != block.document_count
        ):
            raise CompositionError(
                "block document count "
                "does not match root"
            )

        for local_doc_id, (
            source_record,
            runtime_record,
        ) in enumerate(
            zip(
                source_records,
                runtime_records,
            )
        ):
            if (
                not isinstance(
                    source_record,
                    dict,
                )
                or not isinstance(
                    runtime_record,
                    dict,
                )
            ):
                raise CompositionError(
                    "invalid document record"
                )

            if (
                source_record.get(
                    "doc_id"
                )
                != local_doc_id
                or runtime_record.get(
                    "doc_id"
                )
                != local_doc_id
            ):
                raise CompositionError(
                    "non-canonical local doc_id"
                )

            path_hex = source_record.get(
                "relative_path_bytes_hex"
            )

            if (
                not isinstance(
                    path_hex,
                    str,
                )
                or path_hex
                != path_hex.lower()
            ):
                raise CompositionError(
                    "invalid relative path hex"
                )

            try:
                path_bytes = bytes.fromhex(
                    path_hex
                )
            except ValueError as error:
                raise CompositionError(
                    "invalid relative path hex"
                ) from error

            if path_bytes.hex() != path_hex:
                raise CompositionError(
                    "non-canonical path hex"
                )

            if path_bytes in seen_paths:
                raise CompositionError(
                    "duplicate global "
                    "source path"
                )

            seen_paths.add(path_bytes)

            byte_length = u64(
                source_record.get(
                    "byte_length"
                ),
                "source_byte_length",
            )

            source_sha256 = (
                source_record.get(
                    "sha256"
                )
            )

            raw_sha256(
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
                raise CompositionError(
                    "source/runtime document "
                    "commitment mismatch"
                )

            snapshot_relative = (
                source_record.get(
                    "snapshot_path"
                )
            )

            if not isinstance(
                snapshot_relative,
                str,
            ):
                raise CompositionError(
                    "invalid snapshot path"
                )

            snapshot_path = (
                block.corpus
                / snapshot_relative
            )

            payload = snapshot_path.read_bytes()

            if (
                len(payload) != byte_length
                or hashlib.sha256(
                    payload
                ).hexdigest()
                != source_sha256
            ):
                raise artifact_error(
                    "COMPOSITION_E_VERIFY",
                    "snapshot commitment mismatch",
                )

            flattened.append({
                "path_bytes":
                    path_bytes,
                "byte_length":
                    byte_length,
                "sha256":
                    source_sha256,
            })

    return flattened


def validate_root_manifest(
    manifest: dict[str, Any],
    available_blocks: Sequence[Block],
    *,
    name: str,
) -> Root:
    if not isinstance(manifest, dict):
        raise CompositionError(
            "root manifest is not an object"
        )

    if set(manifest) != ROOT_MANIFEST_KEYS:
        raise root_error(
            "COMPOSITION_E_ROOT_INVALID",
            "root manifest key mismatch",
        )

    if manifest.get("format") != ROOT_VERSION:
        raise root_error(
            "COMPOSITION_E_VERSION",
            "unsupported root format",
        )

    if (
        manifest.get(
            "publication_status"
        )
        != ROOT_PUBLICATION_STATUS
    ):
        raise root_error(
            "COMPOSITION_E_ROOT_INVALID",
            "root is not complete",
        )

    document_count = u64(
        manifest.get(
            "global_document_count"
        ),
        "global_document_count",
    )

    block_count = u64(
        manifest.get(
            "block_count"
        ),
        "block_count",
    )

    if block_count == 0:
        raise CompositionError(
            "empty composition root"
        )

    runtime_id = manifest.get(
        "runtime_corpus_id"
    )
    source_id = manifest.get(
        "source_manifest_id"
    )
    committed_root_id = manifest.get(
        "composition_root_id"
    )

    raw_sha256(
        runtime_id,
        "runtime_corpus_id",
    )
    raw_sha256(
        source_id,
        "source_manifest_id",
    )
    raw_sha256(
        committed_root_id,
        "composition_root_id",
    )

    records = manifest.get("blocks")

    if (
        not isinstance(records, list)
        or len(records) != block_count
    ):
        raise CompositionError(
            "root block_count mismatch"
        )

    available_by_id: dict[
        str,
        Block,
    ] = {}

    for block in available_blocks:
        if (
            block.runtime_index_id
            in available_by_id
        ):
            raise CompositionError(
                "duplicate available "
                "runtime unit"
            )

        available_by_id[
            block.runtime_index_id
        ] = block

    verified_blocks: list[Block] = []
    seen_runtime_ids: set[str] = set()
    global_doc_base = 0

    for expected_ordinal, record in enumerate(
        records
    ):
        if (
            not isinstance(record, dict)
            or set(record)
            != ROOT_BLOCK_RECORD_KEYS
        ):
            raise CompositionError(
                "root block record "
                "key mismatch"
            )

        ordinal = u64(
            record.get(
                "block_ordinal"
            ),
            "block_ordinal",
        )

        if ordinal != expected_ordinal:
            raise root_error(
                "COMPOSITION_E_ROOT_INVALID",
                "non-canonical root block ordinal",
            )

        block_document_count = u64(
            record.get(
                "block_document_count"
            ),
            "block_document_count",
        )

        if block_document_count == 0:
            raise CompositionError(
                "empty runtime unit"
            )

        block_runtime_id = (
            record.get(
                "runtime_index_id"
            )
        )

        block_manifest_sha256 = (
            record.get(
                "runtime_manifest_sha256"
            )
        )

        raw_sha256(
            block_runtime_id,
            "runtime_index_id",
        )
        raw_sha256(
            block_manifest_sha256,
            "runtime_manifest_sha256",
        )

        if (
            block_runtime_id
            in seen_runtime_ids
        ):
            raise root_error(
                "COMPOSITION_E_ROOT_INVALID",
                "duplicate runtime_index_id in root",
            )

        seen_runtime_ids.add(
            block_runtime_id
        )

        physical = available_by_id.get(
            block_runtime_id
        )

        if physical is None:
            raise root_error(
                "COMPOSITION_E_COVERAGE",
                "required runtime unit is unavailable",
            )

        if (
            physical.runtime_manifest_sha256
            != block_manifest_sha256
        ):
            raise root_error(
                "COMPOSITION_E_IDENTITY",
                "available runtime manifest "
                "does not match root",
            )

        range_end = checked_add(
            global_doc_base,
            block_document_count,
            "global_document_range_end",
        )

        verified_blocks.append(
            Block(
                ordinal=ordinal,
                start=global_doc_base,
                end=range_end,
                corpus=physical.corpus,
                runtime_index_id=(
                    block_runtime_id
                ),
                runtime_manifest_sha256=(
                    block_manifest_sha256
                ),
            )
        )

        global_doc_base = range_end

    if global_doc_base != document_count:
        raise CompositionError(
            "global document coverage "
            "mismatch"
        )

    recomputed_root_id = (
        recompute_root_identity_from_manifest(
            manifest
        )
    )

    if recomputed_root_id != committed_root_id:
        raise root_error(
            "COMPOSITION_E_ROOT_MISMATCH",
            "composition root identity mismatch",
        )

    flattened = (
        flatten_verified_identity_records(
            verified_blocks
        )
    )

    if len(flattened) != document_count:
        raise CompositionError(
            "flattened document count "
            "mismatch"
        )

    recomputed_runtime_id = (
        runtime_corpus_id_from_records(
            flattened
        )
    )

    if recomputed_runtime_id != runtime_id:
        raise root_error(
            "COMPOSITION_E_IDENTITY",
            "global runtime corpus identity mismatch",
        )

    recomputed_source_id = (
        source_manifest_id_from_records(
            flattened
        )
    )

    if recomputed_source_id != source_id:
        raise root_error(
            "COMPOSITION_E_IDENTITY",
            "global source manifest identity mismatch",
        )

    return Root(
        name=name,
        blocks=tuple(
            verified_blocks
        ),
        document_count=document_count,
        corpus_id=runtime_id,
        source_manifest_id=source_id,
        composition_root_id=(
            committed_root_id
        ),
    )


def serialize_and_validate_root(
    work: Path,
    root: Root,
) -> Root:
    manifest = make_root_manifest(
        root
    )

    if (
        recompute_root_identity_from_manifest(
            manifest
        )
        != root.composition_root_id
    ):
        raise CompositionError(
            "builder root identity "
            "self-check failed"
        )

    manifest_path = (
        work
        / (
            f"{root.name}-"
            "composition-root-v1.json"
        )
    )

    serialized = canonical_json_bytes(
        manifest
    )

    manifest_path.write_bytes(
        serialized
    )

    loaded = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    if canonical_json_bytes(
        loaded
    ) != serialized:
        raise CompositionError(
            "root manifest canonical "
            "serialization mismatch"
        )

    verified = validate_root_manifest(
        loaded,
        root.blocks,
        name=root.name,
    )

    if make_root_manifest(
        verified
    ) != loaded:
        raise CompositionError(
            "verified root view differs "
            "from manifest"
        )

    return verified


def clone_json(
    value: Any,
) -> Any:
    return json.loads(
        json.dumps(value)
    )


def expect_root_failure(
    name: str,
    expected_error_class: str,
    function,
) -> dict[str, Any]:
    try:
        function()

    except CompositionError as error:
        message = str(error)
        prefix = expected_error_class + ":"

        if not message.startswith(prefix):
            raise CompositionError(
                f"{name}: expected "
                f"{expected_error_class}, "
                f"received {message}"
            ) from error

        return {
            "mutation": name,
            "expected_error_class":
                expected_error_class,
            "rejected": True,
        }

    raise CompositionError(
        f"mutation unexpectedly accepted: {name}"
    )


def validate_root_mutations(
    root: Root,
) -> list[dict[str, Any]]:
    baseline = make_root_manifest(root)
    mutations: list[dict[str, Any]] = []

    def validate(
        manifest: dict[str, Any],
        blocks: Sequence[Block] | None = None,
    ) -> Root:
        return validate_root_manifest(
            manifest,
            root.blocks
            if blocks is None
            else blocks,
            name=root.name + "-mutation",
        )

    unsupported = clone_json(baseline)
    unsupported["format"] = (
        "GLYPH_COMPOSITION_ROOT_V2"
    )

    mutations.append(
        expect_root_failure(
            "unsupported_root_format",
            "COMPOSITION_E_VERSION",
            lambda: validate(unsupported),
        )
    )

    incomplete = clone_json(baseline)
    incomplete["publication_status"] = (
        "INCOMPLETE"
    )

    mutations.append(
        expect_root_failure(
            "incomplete_publication",
            "COMPOSITION_E_ROOT_INVALID",
            lambda: validate(incomplete),
        )
    )

    changed_root = clone_json(baseline)
    changed_root["composition_root_id"] = (
        "0" * 64
    )

    mutations.append(
        expect_root_failure(
            "changed_composition_root_id",
            "COMPOSITION_E_ROOT_MISMATCH",
            lambda: validate(changed_root),
        )
    )

    reordered = clone_json(baseline)

    reordered["blocks"][0], reordered["blocks"][1] = (
        reordered["blocks"][1],
        reordered["blocks"][0],
    )

    mutations.append(
        expect_root_failure(
            "reordered_block_records",
            "COMPOSITION_E_ROOT_INVALID",
            lambda: validate(reordered),
        )
    )

    duplicated = clone_json(baseline)

    duplicated_record = clone_json(
        duplicated["blocks"][0]
    )

    duplicated_record["block_ordinal"] = 1
    duplicated["blocks"][1] = duplicated_record

    mutations.append(
        expect_root_failure(
            "duplicated_block_identity",
            "COMPOSITION_E_ROOT_INVALID",
            lambda: validate(duplicated),
        )
    )

    mutations.append(
        expect_root_failure(
            "missing_required_block",
            "COMPOSITION_E_COVERAGE",
            lambda: validate(
                clone_json(baseline),
                root.blocks[:-1],
            ),
        )
    )

    changed_manifest = clone_json(baseline)

    changed_manifest["blocks"][1][
        "runtime_manifest_sha256"
    ] = "0" * 64

    changed_manifest[
        "composition_root_id"
    ] = recompute_root_identity_from_manifest(
        changed_manifest
    )

    mutations.append(
        expect_root_failure(
            "changed_runtime_manifest_commitment",
            "COMPOSITION_E_IDENTITY",
            lambda: validate(changed_manifest),
        )
    )

    changed_corpus = clone_json(baseline)

    changed_corpus[
        "runtime_corpus_id"
    ] = "0" * 64

    changed_corpus[
        "composition_root_id"
    ] = recompute_root_identity_from_manifest(
        changed_corpus
    )

    mutations.append(
        expect_root_failure(
            "changed_global_runtime_corpus_id",
            "COMPOSITION_E_IDENTITY",
            lambda: validate(changed_corpus),
        )
    )

    changed_source = clone_json(baseline)

    changed_source[
        "source_manifest_id"
    ] = "0" * 64

    changed_source[
        "composition_root_id"
    ] = recompute_root_identity_from_manifest(
        changed_source
    )

    mutations.append(
        expect_root_failure(
            "changed_global_source_manifest_id",
            "COMPOSITION_E_IDENTITY",
            lambda: validate(changed_source),
        )
    )

    if len(mutations) != 9:
        raise CompositionError(
            "root mutation count mismatch"
        )

    if not all(
        item["rejected"] is True
        for item in mutations
    ):
        raise CompositionError(
            "root mutation gate failed"
        )

    return mutations


def verify_physical_runtime_block(
    block: Block,
) -> dict[str, Any]:
    try:
        return verify_runtime_index(
            block.corpus,
            require_current_binaries=True,
            rebuild=False,
        )

    except Exception as error:
        message = str(error)

        if (
            "runtime manifest constant "
            "mismatch: runtime_profile"
            in message
        ):
            error_class = (
                "COMPOSITION_E_VERSION"
            )

        elif (
            "runtime index identity mismatch"
            in message
        ):
            error_class = (
                "COMPOSITION_E_IDENTITY"
            )

        else:
            error_class = (
                "COMPOSITION_E_VERIFY"
            )

        raise artifact_error(
            error_class,
            "physical runtime verification "
            f"failed: {message}",
        ) from error


def clone_block_for_mutation(
    work: Path,
    block: Block,
    label: str,
) -> Block:
    destination = (
        work
        / (
            "artifact-mutation-"
            f"{label}-"
            f"block-{block.ordinal:02d}"
        )
    )

    if destination.exists():
        raise CompositionError(
            "artifact mutation destination "
            "already exists"
        )

    shutil.copytree(
        block.corpus,
        destination,
    )

    return Block(
        ordinal=block.ordinal,
        start=block.start,
        end=block.end,
        corpus=destination,
        runtime_index_id=(
            block.runtime_index_id
        ),
        runtime_manifest_sha256=(
            block.runtime_manifest_sha256
        ),
    )


def replace_available_block(
    root: Root,
    ordinal: int,
    replacement: Block,
) -> tuple[Block, ...]:
    if not (
        0 <= ordinal < len(root.blocks)
    ):
        raise CompositionError(
            "replacement ordinal out of range"
        )

    if replacement.ordinal != ordinal:
        raise CompositionError(
            "replacement ordinal mismatch"
        )

    blocks = list(root.blocks)
    blocks[ordinal] = replacement

    return tuple(blocks)


def root_manifest_with_replacement(
    root: Root,
    replacement: Block,
) -> dict[str, Any]:
    manifest = make_root_manifest(
        root
    )

    ordinal = replacement.ordinal

    record = manifest["blocks"][
        ordinal
    ]

    if (
        record["block_document_count"]
        != replacement.document_count
    ):
        raise CompositionError(
            "replacement document count "
            "mismatch"
        )

    record["runtime_index_id"] = (
        replacement.runtime_index_id
    )

    record["runtime_manifest_sha256"] = (
        replacement.runtime_manifest_sha256
    )

    manifest["composition_root_id"] = (
        recompute_root_identity_from_manifest(
            manifest
        )
    )

    return manifest


def block_view_from_mutated_corpus(
    original: Block,
    *,
    corpus: Path,
    runtime_index_id: str | None = None,
    runtime_manifest_sha256: (
        str | None
    ) = None,
) -> Block:
    return Block(
        ordinal=original.ordinal,
        start=original.start,
        end=original.end,
        corpus=corpus,
        runtime_index_id=(
            original.runtime_index_id
            if runtime_index_id is None
            else runtime_index_id
        ),
        runtime_manifest_sha256=(
            original.runtime_manifest_sha256
            if runtime_manifest_sha256
            is None
            else runtime_manifest_sha256
        ),
    )


def mutate_runtime_manifest_digit(
    runtime_manifest_path: Path,
) -> None:
    original = runtime_manifest_path.read_bytes()
    mutated = bytearray(original)

    marker = b'"total_runtime_bytes":'

    start = mutated.find(marker)

    if start < 0:
        raise CompositionError(
            "total_runtime_bytes marker "
            "not found"
        )

    cursor = start + len(marker)

    while (
        cursor < len(mutated)
        and mutated[cursor]
        in b" \t\r\n"
    ):
        cursor += 1

    digit_start = cursor

    while (
        cursor < len(mutated)
        and chr(mutated[cursor]).isdigit()
    ):
        cursor += 1

    if cursor == digit_start:
        raise CompositionError(
            "total_runtime_bytes value "
            "is not numeric"
        )

    digit_index = cursor - 1
    current = mutated[digit_index]

    mutated[digit_index] = (
        ord("0")
        if current != ord("0")
        else ord("1")
    )

    changed = sum(
        left != right
        for left, right in zip(
            original,
            mutated,
        )
    )

    if (
        len(original) != len(mutated)
        or changed != 1
    ):
        raise CompositionError(
            "runtime manifest mutation "
            "was not exactly one byte"
        )

    runtime_manifest_path.write_bytes(
        mutated
    )


def expect_artifact_failure(
    name: str,
    expected_error_class: str,
    function,
) -> dict[str, Any]:
    try:
        function()

    except CompositionError as error:
        message = str(error)
        prefix = expected_error_class + ":"

        if not message.startswith(prefix):
            raise CompositionError(
                f"{name}: expected "
                f"{expected_error_class}, "
                f"received {message}"
            ) from error

        return {
            "mutation": name,
            "expected_error_class":
                expected_error_class,
            "rejected": True,
        }

    raise CompositionError(
        "artifact mutation unexpectedly "
        f"accepted: {name}"
    )


def validate_artifact_integrity_mutations(
    work: Path,
    root: Root,
    documents: Sequence[Document],
) -> list[dict[str, Any]]:
    baseline_manifest = make_root_manifest(
        root
    )

    validate_root_manifest(
        clone_json(baseline_manifest),
        root.blocks,
        name=root.name + "-artifact-baseline",
    )

    target = root.blocks[1]

    mutations: list[dict[str, Any]] = []

    # M02 — substitute one complete, independently valid block.
    alternative_documents = list(
        documents
    )

    alternative_documents[3] = Document(
        "31-substitute-a.bin",
        b"substitute-A",
    )

    alternative_documents[4] = Document(
        "41-substitute-b.bin",
        b"substitute-B",
    )

    alternative_documents[5] = Document(
        "51-substitute-c.bin",
        b"substitute-C",
    )

    substitute = build_block(
        work,
        "artifact-valid-substitute",
        target.ordinal,
        target.start,
        target.end,
        alternative_documents,
    )

    substitute_manifest = (
        root_manifest_with_replacement(
            root,
            substitute,
        )
    )

    substitute_blocks = (
        replace_available_block(
            root,
            target.ordinal,
            substitute,
        )
    )

    mutations.append(
        expect_artifact_failure(
            "valid_block_substituted",
            "COMPOSITION_E_IDENTITY",
            lambda: validate_root_manifest(
                substitute_manifest,
                substitute_blocks,
                name=(
                    root.name
                    + "-valid-substitute"
                ),
            ),
        )
    )

    # M05 — alter exactly one raw byte in the runtime manifest.
    byte_changed = clone_block_for_mutation(
        work,
        target,
        "runtime-manifest-byte",
    )

    byte_changed_runtime_path = (
        byte_changed.corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )

    mutate_runtime_manifest_digit(
        byte_changed_runtime_path
    )

    mutations.append(
        expect_artifact_failure(
            "runtime_manifest_byte_changed",
            "COMPOSITION_E_VERIFY",
            lambda: validate_root_manifest(
                clone_json(
                    baseline_manifest
                ),
                replace_available_block(
                    root,
                    target.ordinal,
                    byte_changed,
                ),
                name=(
                    root.name
                    + "-manifest-byte"
                ),
            ),
        )
    )

    # M07 — forge only runtime_index_id in the physical manifest.
    runtime_id_changed = (
        clone_block_for_mutation(
            work,
            target,
            "runtime-index-id",
        )
    )

    runtime_id_path = (
        runtime_id_changed.corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )

    runtime_id_manifest = (
        load_canonical_json(
            runtime_id_path
        )
    )

    forged_runtime_id = (
        "0" * 64
        if runtime_id_manifest[
            "runtime_index_id"
        ] != "0" * 64
        else "1" * 64
    )

    runtime_id_manifest[
        "runtime_index_id"
    ] = forged_runtime_id

    runtime_id_path.write_bytes(
        canonical_json_bytes(
            runtime_id_manifest
        )
    )

    runtime_id_changed = (
        block_view_from_mutated_corpus(
            target,
            corpus=(
                runtime_id_changed.corpus
            ),
            runtime_index_id=(
                forged_runtime_id
            ),
            runtime_manifest_sha256=(
                sha256_file(
                    runtime_id_path
                )
            ),
        )
    )

    runtime_id_root_manifest = (
        root_manifest_with_replacement(
            root,
            runtime_id_changed,
        )
    )

    mutations.append(
        expect_artifact_failure(
            "runtime_index_id_changed",
            "COMPOSITION_E_IDENTITY",
            lambda: validate_root_manifest(
                runtime_id_root_manifest,
                replace_available_block(
                    root,
                    target.ordinal,
                    runtime_id_changed,
                ),
                name=(
                    root.name
                    + "-runtime-id"
                ),
            ),
        )
    )

    # M11 — mutate one byte in a committed source snapshot.
    source_changed = clone_block_for_mutation(
        work,
        target,
        "source-document-byte",
    )

    source_manifest = load_canonical_json(
        source_changed.corpus
        / SOURCE_MANIFEST_NAME
    )

    source_path: Path | None = None

    for record in source_manifest[
        "documents"
    ]:
        if record["byte_length"] > 0:
            source_path = (
                source_changed.corpus
                / record["snapshot_path"]
            )
            break

    if source_path is None:
        raise CompositionError(
            "no non-empty source document "
            "available for mutation"
        )

    source_payload = bytearray(
        source_path.read_bytes()
    )

    source_payload[0] ^= 0x01

    source_path.write_bytes(
        source_payload
    )

    mutations.append(
        expect_artifact_failure(
            "source_document_byte_changed",
            "COMPOSITION_E_VERIFY",
            lambda: validate_root_manifest(
                clone_json(
                    baseline_manifest
                ),
                replace_available_block(
                    root,
                    target.ordinal,
                    source_changed,
                ),
                name=(
                    root.name
                    + "-source-byte"
                ),
            ),
        )
    )

    # M19 — change the actual runtime profile.
    unsupported_profile = (
        clone_block_for_mutation(
            work,
            target,
            "unsupported-runtime-profile",
        )
    )

    unsupported_runtime_path = (
        unsupported_profile.corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )

    unsupported_manifest = (
        load_canonical_json(
            unsupported_runtime_path
        )
    )

    unsupported_manifest[
        "runtime_profile"
    ] = "GLYPH_BINARY_RUNTIME_V2"

    unsupported_runtime_path.write_bytes(
        canonical_json_bytes(
            unsupported_manifest
        )
    )

    unsupported_profile = (
        block_view_from_mutated_corpus(
            target,
            corpus=(
                unsupported_profile.corpus
            ),
            runtime_manifest_sha256=(
                sha256_file(
                    unsupported_runtime_path
                )
            ),
        )
    )

    unsupported_root_manifest = (
        root_manifest_with_replacement(
            root,
            unsupported_profile,
        )
    )

    mutations.append(
        expect_artifact_failure(
            "unsupported_runtime_profile",
            "COMPOSITION_E_VERSION",
            lambda: validate_root_manifest(
                unsupported_root_manifest,
                replace_available_block(
                    root,
                    target.ordinal,
                    unsupported_profile,
                ),
                name=(
                    root.name
                    + "-runtime-profile"
                ),
            ),
        )
    )

    if len(mutations) != 5:
        raise CompositionError(
            "artifact mutation count mismatch"
        )

    if not all(
        item["rejected"] is True
        for item in mutations
    ):
        raise CompositionError(
            "artifact mutation gate failed"
        )

    # Prove that mutation copies did not alter the baseline root.
    validate_root_manifest(
        clone_json(baseline_manifest),
        root.blocks,
        name=root.name + "-artifact-postcheck",
    )

    return mutations


def require_claimed_complete_results(
    root: Root,
    results: dict[int, dict[str, Any]],
    claimed_blocks: Sequence[int],
) -> None:
    expected = list(
        range(len(root.blocks))
    )

    claimed = list(
        claimed_blocks
    )

    if claimed != expected:
        raise aggregation_error(
            "COMPOSITION_E_COVERAGE",
            "claimed block coverage is not "
            "canonical and complete",
        )

    actual = sorted(
        results
    )

    if actual != expected:
        missing = [
            ordinal
            for ordinal in expected
            if ordinal not in results
        ]

        unexpected = [
            ordinal
            for ordinal in actual
            if ordinal not in expected
        ]

        raise aggregation_error(
            "COMPOSITION_E_COVERAGE",
            "claimed complete coverage differs "
            "from actual block results; "
            f"missing={missing}; "
            f"unexpected={unexpected}",
        )


def compose_from_claimed_coverage(
    root: Root,
    query: bytes,
    results: dict[int, dict[str, Any]],
    claimed_blocks: Sequence[int],
    max_offsets: int | None,
) -> dict[str, Any]:
    require_claimed_complete_results(
        root,
        results,
        claimed_blocks,
    )

    return compose_full(
        root,
        query,
        results,
        max_offsets,
    )


def compose_with_independent_block_limits(
    root: Root,
    query: bytes,
    max_offsets: int,
) -> dict[str, Any]:
    u64(
        max_offsets,
        "max_offsets",
    )

    total = 0
    coordinates: list[list[int]] = []

    for block in root.blocks:
        # Deliberately reproduce the forbidden
        # algorithm: each block receives the full
        # global limit instead of a shrinking
        # remaining budget.
        local = execute_operator_query(
            block.corpus,
            query,
            max_offsets=max_offsets,
        )

        local_count = u64(
            local["match_count"],
            "block_match_count",
        )

        total = checked_add_aggregation(
            total,
            local_count,
            "global_match_count",
        )

        for item in local["coordinates"]:
            local_doc_id, doc_offset = (
                item["coordinate"]
            )

            coordinates.append([
                checked_add(
                    block.start,
                    local_doc_id,
                    "global_doc_id",
                ),
                doc_offset,
            ])

    if len(coordinates) <= max_offsets:
        raise CompositionError(
            "independent-limit fixture did "
            "not exceed the global limit"
        )

    candidate = make_result(
        root,
        query,
        max_offsets,
        total,
        coordinates,
    )

    # The strict validator must reject this candidate
    # because it is not the canonical global prefix.
    return serialize_and_validate_result(
        root,
        query,
        candidate,
    )


def expect_aggregation_failure(
    name: str,
    expected_error_class: str,
    function,
) -> dict[str, Any]:
    try:
        function()

    except CompositionError as error:
        message = str(error)
        prefix = expected_error_class + ":"

        if not message.startswith(prefix):
            raise CompositionError(
                f"{name}: expected "
                f"{expected_error_class}, "
                f"received {message}"
            ) from error

        return {
            "mutation": name,
            "expected_error_class":
                expected_error_class,
            "rejected": True,
        }

    raise CompositionError(
        "aggregation mutation unexpectedly "
        f"accepted: {name}"
    )


def validate_aggregation_limit_mutations(
    root: Root,
) -> list[dict[str, Any]]:
    query = b"a"

    expected_ordinals = list(
        range(len(root.blocks))
    )

    complete_results = run_full_results(
        root,
        query,
        expected_ordinals,
    )

    mutations: list[dict[str, Any]] = []

    # M12 — claim complete coverage while one actual
    # block result is absent.
    omitted_results = dict(
        complete_results
    )

    omitted_ordinal = 1

    if omitted_ordinal not in omitted_results:
        raise CompositionError(
            "M12 fixture block result missing "
            "before mutation"
        )

    omitted_results.pop(
        omitted_ordinal
    )

    mutations.append(
        expect_aggregation_failure(
            "block_result_omitted_with_"
            "claimed_coverage",
            "COMPOSITION_E_COVERAGE",
            lambda: compose_from_claimed_coverage(
                root,
                query,
                omitted_results,
                expected_ordinals,
                None,
            ),
        )
    )

    # M16 — overflow the real global count
    # aggregation path before coordinate validation.
    overflow_results: dict[
        int,
        dict[str, Any],
    ] = {}

    overflow_counts = [
        MAX_U64,
        1,
    ] + [
        0
        for _ in range(
            max(
                0,
                len(root.blocks) - 2,
            )
        )
    ]

    if len(overflow_counts) != len(
        root.blocks
    ):
        raise CompositionError(
            "overflow fixture block-count "
            "mismatch"
        )

    for ordinal, count in enumerate(
        overflow_counts
    ):
        overflow_results[ordinal] = {
            "match_count": count,
            "coordinates": [],
        }

    mutations.append(
        expect_aggregation_failure(
            "global_match_count_"
            "overflow_attempted",
            "COMPOSITION_E_LIMIT",
            lambda: compose_full(
                root,
                query,
                overflow_results,
                None,
            ),
        )
    )

    # M17 — apply the same full max_offsets value
    # independently to every block.
    mutations.append(
        expect_aggregation_failure(
            "max_offsets_applied_"
            "independently_per_block",
            "COMPOSITION_E_VERIFY",
            lambda:
                compose_with_independent_block_limits(
                    root,
                    query,
                    1,
                ),
        )
    )

    if len(mutations) != 3:
        raise CompositionError(
            "aggregation mutation count mismatch"
        )

    if not all(
        item["rejected"] is True
        for item in mutations
    ):
        raise CompositionError(
            "aggregation mutation gate failed"
        )

    # Positive control: the correct remaining-budget
    # implementation must still succeed.
    correct = compose_two_phase(
        root,
        query,
        1,
    )

    if (
        correct["returned_count"] != 1
        or correct["max_offsets"] != 1
        or correct["bounded"] is not True
    ):
        raise CompositionError(
            "correct global-budget control failed"
        )

    return mutations


def build_mutation_traceability(
    root_mutations: Sequence[dict[str, Any]],
    result_mutations: Sequence[dict[str, Any]],
    artifact_mutations: Sequence[
        dict[str, Any]
    ],
    aggregation_mutations: Sequence[
        dict[str, Any]
    ],
    document_model_mutations: Sequence[
        dict[str, Any]
    ],
    replay_mutations: Sequence[
        dict[str, Any]
    ],
    byte_check_replay_mutations: Sequence[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    implemented: dict[str, str] = {}

    for item in (
        list(root_mutations)
        + list(result_mutations)
        + list(artifact_mutations)
        + list(aggregation_mutations)
        + list(document_model_mutations)
        + list(replay_mutations)
        + list(byte_check_replay_mutations)
    ):
        if not isinstance(item, dict):
            raise CompositionError(
                "mutation result is not an object"
            )

        if set(item) != {
            "mutation",
            "expected_error_class",
            "rejected",
        }:
            raise CompositionError(
                "mutation result key mismatch"
            )

        name = item["mutation"]
        error_class = item[
            "expected_error_class"
        ]

        if (
            not isinstance(name, str)
            or not name
            or not isinstance(
                error_class,
                str,
            )
            or not error_class
            or item["rejected"] is not True
        ):
            raise CompositionError(
                "invalid mutation result record"
            )

        if name in implemented:
            raise CompositionError(
                "duplicate implemented "
                f"mutation test: {name}"
            )

        implemented[name] = error_class

    expected_ids = [
        f"M{number:02d}"
        for number in range(1, 26)
    ]

    actual_ids = [
        item["id"]
        for item
        in NORMATIVE_MUTATION_REQUIREMENTS
    ]

    if actual_ids != expected_ids:
        raise CompositionError(
            "normative mutation IDs are "
            "not exactly M01 through M25"
        )

    allowed_statuses = {
        "EXACT",
        "SUPPORTING",
        "OPEN",
    }

    referenced_tests: set[str] = set()
    requirements: list[
        dict[str, Any]
    ] = []

    for requirement in (
        NORMATIVE_MUTATION_REQUIREMENTS
    ):
        if set(requirement) != {
            "id",
            "description",
            "status",
            "tests",
        }:
            raise CompositionError(
                "traceability requirement "
                "key mismatch"
            )

        requirement_id = requirement["id"]
        description = requirement[
            "description"
        ]
        status = requirement["status"]
        tests = list(requirement["tests"])

        if (
            not isinstance(
                description,
                str,
            )
            or not description
            or status not in allowed_statuses
        ):
            raise CompositionError(
                "invalid traceability "
                f"requirement: {requirement_id}"
            )

        if status in {
            "EXACT",
            "SUPPORTING",
        } and not tests:
            raise CompositionError(
                f"{requirement_id}: "
                "non-open requirement "
                "has no tests"
            )

        if status == "OPEN" and tests:
            raise CompositionError(
                f"{requirement_id}: "
                "open requirement has tests"
            )

        enriched_tests = []

        for test_name in tests:
            if test_name not in implemented:
                raise CompositionError(
                    f"{requirement_id}: "
                    "unknown implemented test: "
                    f"{test_name}"
                )

            if test_name in referenced_tests:
                raise CompositionError(
                    "mutation test mapped more "
                    f"than once: {test_name}"
                )

            referenced_tests.add(
                test_name
            )

            enriched_tests.append({
                "name": test_name,
                "expected_error_class":
                    implemented[test_name],
            })

        requirements.append({
            "id": requirement_id,
            "description": description,
            "status": status,
            "tests": enriched_tests,
        })

    additional_tests = []

    for test_name in (
        ADDITIONAL_MUTATION_TESTS
    ):
        if test_name not in implemented:
            raise CompositionError(
                "unknown additional mutation "
                f"test: {test_name}"
            )

        if test_name in referenced_tests:
            raise CompositionError(
                "additional mutation test also "
                f"mapped normatively: {test_name}"
            )

        referenced_tests.add(test_name)

        additional_tests.append({
            "name": test_name,
            "expected_error_class":
                implemented[test_name],
        })

    if set(implemented) != referenced_tests:
        missing = sorted(
            set(implemented)
            - referenced_tests
        )

        undeclared = sorted(
            referenced_tests
            - set(implemented)
        )

        raise CompositionError(
            "mutation traceability coverage "
            f"mismatch; missing={missing}; "
            f"undeclared={undeclared}"
        )

    exact_ids = [
        item["id"]
        for item in requirements
        if item["status"] == "EXACT"
    ]

    supporting_ids = [
        item["id"]
        for item in requirements
        if item["status"] == "SUPPORTING"
    ]

    open_ids = [
        item["id"]
        for item in requirements
        if item["status"] == "OPEN"
    ]

    exact_count = len(exact_ids)
    supporting_count = len(
        supporting_ids
    )
    open_count = len(open_ids)

    if (
        exact_count,
        supporting_count,
        open_count,
    ) != (25, 0, 0):
        raise CompositionError(
            "unexpected traceability "
            "baseline counts"
        )

    not_exact_count = (
        supporting_count + open_count
    )

    closure_complete = (
        exact_count
        == len(requirements)
    )

    return {
        "ok": True,
        "format":
            "GLYPH_COMPOSITION_"
            "MUTATION_TRACEABILITY_V1",
        "normative_requirement_count":
            len(requirements),
        "implemented_test_count":
            len(implemented),
        "normatively_mapped_test_count":
            sum(
                len(item["tests"])
                for item in requirements
            ),
        "additional_test_count":
            len(additional_tests),
        "normative_exact_count":
            exact_count,
        "normative_supporting_count":
            supporting_count,
        "normative_open_count":
            open_count,
        "normative_not_exact_count":
            not_exact_count,
        "normative_closure_complete":
            closure_complete,
        "exact_requirement_ids":
            exact_ids,
        "supporting_requirement_ids":
            supporting_ids,
        "open_requirement_ids":
            open_ids,
        "requirements":
            requirements,
        "additional_tests":
            additional_tests,
    }


def fixture_documents() -> list[Document]:
    return [
        Document(
            "00-empty.bin",
            b"",
        ),
        Document(
            "10-alpha.bin",
            b"alpha LEFT",
        ),
        Document(
            "20-edge-a.bin",
            b"EDGE-A",
        ),
        Document(
            "30-edge-b.bin",
            b"B-EDGE shared\x00\xff",
        ),
        Document(
            "40-shared.bin",
            b"shared middle",
        ),
        Document(
            "50-dup-a.bin",
            b"dup",
        ),
        Document(
            "60-dup-b.bin",
            b"dup",
        ),
        Document(
            "70-tail.bin",
            b"tail shared",
        ),
        Document(
            "80-omega.bin",
            b"omega",
        ),
    ]


def write_tree(
    root: Path,
    documents: Sequence[Document],
) -> None:
    root.mkdir(parents=True)

    for document in documents:
        path = root / document.path

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(
            document.data
        )


def validate_source_manifest(
    manifest: dict[str, Any],
    expected: Sequence[Document],
) -> None:
    records = manifest.get(
        "documents"
    )

    if (
        not isinstance(records, list)
        or len(records) != len(expected)
    ):
        raise CompositionError(
            "source manifest count mismatch"
        )

    for local_id, (
        record,
        document,
    ) in enumerate(
        zip(records, expected)
    ):
        required = {
            "doc_id":
                local_id,
            "relative_path_bytes_hex":
                document.path_bytes.hex(),
            "byte_length":
                len(document.data),
            "sha256":
                document.sha256,
        }

        for field, value in (
            required.items()
        ):
            if record.get(field) != value:
                raise CompositionError(
                    "source manifest mismatch: "
                    f"{field}"
                )

    if manifest.get("corpus_id") != (
        runtime_corpus_id(expected)
    ):
        raise CompositionError(
            "block corpus_id mismatch"
        )

    if (
        manifest.get(
            "source_manifest_id"
        )
        != source_manifest_id(expected)
    ):
        raise CompositionError(
            "block source_manifest_id "
            "mismatch"
        )


def validate_runtime_manifest(
    manifest: dict[str, Any],
    expected: Sequence[Document],
) -> None:
    records = manifest.get(
        "documents"
    )

    if (
        not isinstance(records, list)
        or len(records) != len(expected)
    ):
        raise CompositionError(
            "runtime manifest count mismatch"
        )

    for local_id, (
        record,
        document,
    ) in enumerate(
        zip(records, expected)
    ):
        required = {
            "doc_id":
                local_id,
            "source_byte_length":
                len(document.data),
            "source_sha256":
                document.sha256,
        }

        for field, value in (
            required.items()
        ):
            if record.get(field) != value:
                raise CompositionError(
                    "runtime manifest mismatch: "
                    f"{field}"
                )

    if manifest.get("corpus_id") != (
        runtime_corpus_id(expected)
    ):
        raise CompositionError(
            "runtime corpus_id mismatch"
        )

    if (
        manifest.get(
            "source_manifest_id"
        )
        != source_manifest_id(expected)
    ):
        raise CompositionError(
            "runtime source_manifest_id "
            "mismatch"
        )


def build_block(
    work: Path,
    partition: str,
    ordinal: int,
    start: int,
    end: int,
    documents: Sequence[Document],
) -> Block:
    if not (
        0 <= start < end
        <= len(documents)
    ):
        raise CompositionError(
            "invalid block range"
        )

    expected = list(
        documents[start:end]
    )

    source = (
        work
        / f"{partition}-"
        f"source-{ordinal:02d}"
    )

    corpus = (
        work
        / f"{partition}-"
        f"corpus-{ordinal:02d}"
    )

    write_tree(
        source,
        expected,
    )

    build_snapshot(
        source,
        corpus,
    )

    build_runtime_index(
        corpus
    )

    verify_runtime_index(
        corpus,
        require_current_binaries=True,
        rebuild=False,
    )

    source_manifest = (
        load_canonical_json(
            corpus
            / SOURCE_MANIFEST_NAME
        )
    )

    runtime_path = (
        corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )

    runtime_manifest = (
        load_canonical_json(
            runtime_path
        )
    )

    validate_source_manifest(
        source_manifest,
        expected,
    )

    validate_runtime_manifest(
        runtime_manifest,
        expected,
    )

    runtime_id = runtime_manifest.get(
        "runtime_index_id"
    )

    raw_sha256(
        runtime_id,
        "runtime_index_id",
    )

    if (
        runtime_manifest.get(
            "document_count"
        )
        != len(expected)
    ):
        raise CompositionError(
            "runtime document_count "
            "mismatch"
        )

    return Block(
        ordinal=ordinal,
        start=start,
        end=end,
        corpus=corpus,
        runtime_index_id=runtime_id,
        runtime_manifest_sha256=(
            sha256_file(runtime_path)
        ),
    )


def build_root(
    work: Path,
    name: str,
    ranges: Sequence[
        tuple[int, int]
    ],
    documents: Sequence[Document],
    corpus_id: str,
    manifest_id: str,
) -> Root:
    blocks: list[Block] = []
    next_start = 0

    for ordinal, (
        start,
        end,
    ) in enumerate(ranges):
        if start != next_start:
            raise CompositionError(
                "non-contiguous block ranges"
            )

        blocks.append(
            build_block(
                work,
                name,
                ordinal,
                start,
                end,
                documents,
            )
        )

        next_start = end

    if next_start != len(documents):
        raise CompositionError(
            "incomplete block coverage"
        )

    runtime_ids = [
        block.runtime_index_id
        for block in blocks
    ]

    if (
        len(runtime_ids)
        != len(set(runtime_ids))
    ):
        raise CompositionError(
            "duplicate runtime_index_id"
        )

    root_id = composition_root_id(
        corpus_id,
        manifest_id,
        len(documents),
        blocks,
    )

    provisional = Root(
        name=name,
        blocks=tuple(blocks),
        document_count=len(documents),
        corpus_id=corpus_id,
        source_manifest_id=manifest_id,
        composition_root_id=root_id,
    )

    return serialize_and_validate_root(
        work,
        provisional,
    )


def naive_coordinates(
    documents: Sequence[Document],
    query: bytes,
) -> list[list[int]]:
    if not query:
        raise CompositionError(
            "empty query"
        )

    result: list[list[int]] = []

    for doc_id, document in enumerate(
        documents
    ):
        if len(query) > len(document.data):
            continue

        for offset in range(
            len(document.data)
            - len(query)
            + 1
        ):
            if (
                document.data[
                    offset:
                    offset + len(query)
                ]
                == query
            ):
                result.append([
                    doc_id,
                    offset,
                ])

    return result


def validate_straddles(
    documents: Sequence[Document],
    roots: Sequence[Root],
) -> None:
    cross_document = (
        b"\x00\xffshared"
    )
    cross_block = b"dupdup"

    if any(
        cross_document in document.data
        for document in documents
    ):
        raise CompositionError(
            "cross-document control "
            "exists locally"
        )

    if (
        cross_document
        not in (
            documents[3].data
            + documents[4].data
        )
    ):
        raise CompositionError(
            "cross-document physical "
            "control missing"
        )

    if any(
        cross_block in document.data
        for document in documents
    ):
        raise CompositionError(
            "cross-block control "
            "exists locally"
        )

    if (
        cross_block
        not in (
            documents[5].data
            + documents[6].data
        )
    ):
        raise CompositionError(
            "cross-block physical "
            "control missing"
        )

    for root in roots:
        if not any(
            block.start <= 3
            and block.end >= 5
            for block in root.blocks
        ):
            raise CompositionError(
                "cross-document control "
                "not inside one block"
            )

        if not any(
            left.end == 6
            and right.start == 6
            for left, right in zip(
                root.blocks,
                root.blocks[1:],
            )
        ):
            raise CompositionError(
                "cross-block control "
                "not at block boundary"
            )


def changed_document_model_result(
    root: Root,
    documents: Sequence[Document],
    query: bytes,
) -> dict[str, Any]:
    coordinates = naive_coordinates(
        documents,
        query,
    )

    candidate = make_result(
        root,
        query,
        None,
        len(coordinates),
        coordinates,
    )

    candidate["runtime_corpus_id"] = (
        runtime_corpus_id(documents)
    )

    candidate["source_manifest_id"] = (
        source_manifest_id(documents)
    )

    refresh_composition_result_id(
        candidate
    )

    return serialize_and_validate_result(
        root,
        query,
        candidate,
    )


def physical_concatenation_coordinates(
    documents: Sequence[Document],
    query: bytes,
) -> list[list[int]]:
    if not query:
        raise CompositionError(
            "empty physical oracle query"
        )

    payload = b"".join(
        document.data
        for document in documents
    )

    offsets = [
        offset
        for offset in range(
            max(
                0,
                len(payload) - len(query) + 1,
            )
        )
        if payload[
            offset:
            offset + len(query)
        ] == query
    ]

    coordinates: list[list[int]] = []

    for offset in offsets:
        document_start = 0

        for doc_id, document in enumerate(
            documents
        ):
            document_end = (
                document_start
                + len(document.data)
            )

            if (
                document_start
                <= offset
                < document_end
            ):
                coordinates.append([
                    doc_id,
                    offset - document_start,
                ])
                break

            document_start = document_end

        else:
            raise CompositionError(
                "physical oracle offset "
                "has no starting document"
            )

    return coordinates


def validate_document_model_mutations(
    root: Root,
    documents: Sequence[Document],
) -> list[dict[str, Any]]:
    baseline = list(documents)

    if len(baseline) != 9:
        raise CompositionError(
            "document-model fixture count mismatch"
        )

    baseline_runtime_id = runtime_corpus_id(
        baseline
    )

    baseline_manifest_id = source_manifest_id(
        baseline
    )

    mutations: list[dict[str, Any]] = []

    # M21 — reorder the same document objects.
    reordered = list(baseline)

    reordered[1], reordered[2] = (
        reordered[2],
        reordered[1],
    )

    order_query = b"alpha LEFT"

    if (
        reordered[1] is not baseline[2]
        or reordered[2] is not baseline[1]
        or runtime_corpus_id(reordered)
        == baseline_runtime_id
        or source_manifest_id(reordered)
        == baseline_manifest_id
        or naive_coordinates(
            baseline,
            order_query,
        ) != [[1, 0]]
        or naive_coordinates(
            reordered,
            order_query,
        ) != [[2, 0]]
    ):
        raise CompositionError(
            "document-order mutation control failed"
        )

    mutations.append(
        expect_result_failure(
            "document_order_changed",
            "COMPOSITION_E_IDENTITY",
            lambda: changed_document_model_result(
                root,
                reordered,
                order_query,
            ),
        )
    )

    # M22 — removing an empty document leaves the
    # physical bytes unchanged but renumbers documents.
    empty_removed = baseline[1:]

    if (
        baseline[0].data != b""
        or len(empty_removed)
        != len(baseline) - 1
        or b"".join(
            document.data
            for document in empty_removed
        )
        != b"".join(
            document.data
            for document in baseline
        )
        or runtime_corpus_id(empty_removed)
        == baseline_runtime_id
        or source_manifest_id(empty_removed)
        == baseline_manifest_id
        or naive_coordinates(
            empty_removed,
            order_query,
        ) != [[0, 0]]
    ):
        raise CompositionError(
            "empty-document mutation control failed"
        )

    mutations.append(
        expect_result_failure(
            "empty_document_removed",
            "COMPOSITION_E_IDENTITY",
            lambda: changed_document_model_result(
                root,
                empty_removed,
                order_query,
            ),
        )
    )

    # M23 — byte-identical documents remain distinct
    # occurrences and later doc_ids must not collapse.
    deduplicated = (
        baseline[:6]
        + baseline[7:]
    )

    duplicate_query = b"dup"
    later_query = b"tail shared"

    if (
        baseline[5].data
        != baseline[6].data
        or baseline[5].path
        == baseline[6].path
        or len(deduplicated)
        != len(baseline) - 1
        or runtime_corpus_id(deduplicated)
        == baseline_runtime_id
        or source_manifest_id(deduplicated)
        == baseline_manifest_id
        or naive_coordinates(
            baseline,
            duplicate_query,
        ) != [
            [5, 0],
            [6, 0],
        ]
        or naive_coordinates(
            deduplicated,
            duplicate_query,
        ) != [[5, 0]]
        or naive_coordinates(
            baseline,
            later_query,
        ) != [[7, 0]]
        or naive_coordinates(
            deduplicated,
            later_query,
        ) != [[6, 0]]
    ):
        raise CompositionError(
            "duplicate-document mutation control failed"
        )

    mutations.append(
        expect_result_failure(
            "duplicate_document_deduplicated",
            "COMPOSITION_E_IDENTITY",
            lambda: changed_document_model_result(
                root,
                deduplicated,
                duplicate_query,
            ),
        )
    )

    # M24 — deliberately use the forbidden physical
    # concatenation oracle for both boundary classes.
    boundary_cases = (
        (
            "physical_concatenation_"
            "cross_document",
            b"\x00\xffshared",
            [[3, 13]],
        ),
        (
            "physical_concatenation_"
            "cross_block",
            b"dupdup",
            [[5, 0]],
        ),
    )

    for name, query, wrong_coordinates in (
        boundary_cases
    ):
        actual_wrong = (
            physical_concatenation_coordinates(
                baseline,
                query,
            )
        )

        correct = compose_full(
            root,
            query,
            run_full_results(
                root,
                query,
                list(
                    range(len(root.blocks))
                ),
            ),
            None,
        )

        if (
            actual_wrong != wrong_coordinates
            or naive_coordinates(
                baseline,
                query,
            ) != []
            or correct["match_count"] != 0
            or correct["returned_count"] != 0
            or correct["coordinates"] != []
        ):
            raise CompositionError(
                "physical-concatenation mutation "
                "control failed"
            )

        candidate = make_result(
            root,
            query,
            None,
            len(actual_wrong),
            actual_wrong,
        )

        mutations.append(
            expect_result_failure(
                name,
                "COMPOSITION_E_VERIFY",
                lambda candidate=candidate,
                query=query:
                    serialize_and_validate_result(
                        root,
                        query,
                        candidate,
                    ),
            )
        )

    if len(mutations) != 5:
        raise CompositionError(
            "document-model mutation count mismatch"
        )

    if not all(
        item["rejected"] is True
        for item in mutations
    ):
        raise CompositionError(
            "document-model mutation gate failed"
        )

    return mutations


def validate_full_block_result(
    block: Block,
    result: dict[str, Any],
) -> None:
    if (
        result.get("match_count")
        != result.get("returned_count")
    ):
        raise CompositionError(
            "full block result "
            "unexpectedly bounded"
        )

    if (
        result.get("bounded") is not False
        or result.get(
            "offsets_complete"
        ) is not True
    ):
        raise CompositionError(
            "full block completeness "
            "mismatch"
        )

    for item in result.get(
        "coordinates",
        [],
    ):
        coordinate = item.get(
            "coordinate"
        )

        if (
            not isinstance(
                coordinate,
                list,
            )
            or len(coordinate) != 2
        ):
            raise CompositionError(
                "invalid local coordinate"
            )

        local_doc_id, doc_offset = (
            coordinate
        )

        if not (
            isinstance(local_doc_id, int)
            and not isinstance(
                local_doc_id,
                bool,
            )
            and 0 <= local_doc_id
            < block.document_count
        ):
            raise CompositionError(
                "invalid local doc_id"
            )

        u64(
            doc_offset,
            "doc_offset",
        )


def run_full_results(
    root: Root,
    query: bytes,
    execution_order: Sequence[int],
) -> dict[int, dict[str, Any]]:
    expected = list(
        range(len(root.blocks))
    )

    if sorted(execution_order) != expected:
        raise CompositionError(
            "execution order is not "
            "a permutation"
        )

    results: dict[
        int,
        dict[str, Any],
    ] = {}

    for ordinal in execution_order:
        block = root.blocks[ordinal]

        result = execute_operator_query(
            block.corpus,
            query,
        )

        validate_full_block_result(
            block,
            result,
        )

        results[ordinal] = result

    if sorted(results) != expected:
        raise CompositionError(
            "incomplete result coverage"
        )

    return results


def composition_result_id(
    result: dict[str, Any],
) -> str:
    if not isinstance(result, dict):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "result is not an object",
        )

    payload = dict(result)

    payload.pop(
        "composition_result_id",
        None,
    )

    preimage = (
        RESULT_IDENTITY_VERSION.encode(
            "ascii"
        )
        + b"\x00"
        + canonical_json_bytes(payload)
    )

    return hashlib.sha256(
        preimage
    ).hexdigest()


def require_result_u64(
    value: Any,
    field: str,
) -> int:
    try:
        return u64(
            value,
            field,
        )

    except CompositionError as error:
        raise result_error(
            "COMPOSITION_E_LIMIT",
            f"invalid result integer: {field}",
        ) from error


def validate_block_coverage_list(
    value: Any,
    field: str,
    block_count: int,
) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != block_count
    ):
        raise result_error(
            "COMPOSITION_E_COVERAGE",
            f"invalid coverage list: {field}",
        )

    for expected, item in enumerate(value):
        if (
            not isinstance(item, int)
            or isinstance(item, bool)
            or item != expected
        ):
            raise result_error(
                "COMPOSITION_E_COVERAGE",
                f"non-canonical coverage: {field}",
            )

    return list(value)


def naive_coordinates_from_root(
    root: Root,
    query: bytes,
) -> list[list[int]]:
    if not query:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "empty query",
        )

    coordinates: list[list[int]] = []
    next_global_doc_id = 0

    for block in root.blocks:
        if block.start != next_global_doc_id:
            raise result_error(
                "COMPOSITION_E_IDENTITY",
                "non-contiguous verified root",
            )

        source_manifest = load_canonical_json(
            block.corpus
            / SOURCE_MANIFEST_NAME
        )

        records = source_manifest.get(
            "documents"
        )

        if (
            not isinstance(records, list)
            or len(records)
            != block.document_count
        ):
            raise result_error(
                "COMPOSITION_E_IDENTITY",
                "source-manifest document "
                "count mismatch",
            )

        for local_doc_id, record in enumerate(
            records
        ):
            if (
                not isinstance(record, dict)
                or record.get("doc_id")
                != local_doc_id
            ):
                raise result_error(
                    "COMPOSITION_E_IDENTITY",
                    "non-canonical source doc_id",
                )

            snapshot_relative = record.get(
                "snapshot_path"
            )

            if not isinstance(
                snapshot_relative,
                str,
            ):
                raise result_error(
                    "COMPOSITION_E_IDENTITY",
                    "invalid snapshot path",
                )

            payload = (
                block.corpus
                / snapshot_relative
            ).read_bytes()

            byte_length = require_result_u64(
                record.get("byte_length"),
                "source_byte_length",
            )

            source_sha256 = record.get(
                "sha256"
            )

            try:
                raw_sha256(
                    source_sha256,
                    "source_sha256",
                )

            except CompositionError as error:
                raise result_error(
                    "COMPOSITION_E_IDENTITY",
                    "invalid source identity",
                ) from error

            if (
                len(payload) != byte_length
                or hashlib.sha256(
                    payload
                ).hexdigest()
                != source_sha256
            ):
                raise result_error(
                    "COMPOSITION_E_VERIFY",
                    "source snapshot commitment "
                    "mismatch",
                )

            global_doc_id = checked_add(
                block.start,
                local_doc_id,
                "global_doc_id",
            )

            if global_doc_id != (
                next_global_doc_id
                + local_doc_id
            ):
                raise result_error(
                    "COMPOSITION_E_IDENTITY",
                    "global document mapping "
                    "mismatch",
                )

            if len(query) <= len(payload):
                for offset in range(
                    len(payload)
                    - len(query)
                    + 1
                ):
                    if (
                        payload[
                            offset:
                            offset + len(query)
                        ]
                        == query
                    ):
                        coordinates.append([
                            global_doc_id,
                            offset,
                        ])

        next_global_doc_id = block.end

    if next_global_doc_id != (
        root.document_count
    ):
        raise result_error(
            "COMPOSITION_E_IDENTITY",
            "verified root coverage mismatch",
        )

    return coordinates


def validate_composed_result(
    root: Root,
    query: bytes,
    result: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "composition result is not "
            "an object",
        )

    if set(result) != RESULT_KEYS:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "composition result key "
            "mismatch",
        )

    if result.get("ok") is not True:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "successful result must have "
            "ok=true",
        )

    if result.get("format") != (
        RESULT_VERSION
    ):
        raise result_error(
            "COMPOSITION_E_VERSION",
            "unsupported composition "
            "result format",
        )

    identity_checks = {
        "runtime_corpus_id":
            root.corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "composition_root_id":
            root.composition_root_id,
    }

    for field, expected in (
        identity_checks.items()
    ):
        actual = result.get(field)

        try:
            raw_sha256(
                actual,
                field,
            )

        except CompositionError as error:
            raise result_error(
                "COMPOSITION_E_IDENTITY",
                f"invalid identity field: {field}",
            ) from error

        if actual != expected:
            raise result_error(
                "COMPOSITION_E_IDENTITY",
                f"result identity mismatch: {field}",
            )

    if not isinstance(query, bytes) or not query:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "invalid reference query",
        )

    expected_query_hex = query.hex()
    expected_query_sha256 = hashlib.sha256(
        query
    ).hexdigest()

    if (
        result.get("query_hex")
        != expected_query_hex
        or result.get(
            "query_length_bytes"
        )
        != len(query)
        or result.get(
            "query_sha256"
        )
        != expected_query_sha256
    ):
        raise result_error(
            "COMPOSITION_E_IDENTITY",
            "query identity mismatch",
        )

    max_offsets = result.get(
        "max_offsets"
    )

    if max_offsets is not None:
        max_offsets = require_result_u64(
            max_offsets,
            "max_offsets",
        )

    match_count = require_result_u64(
        result.get("match_count"),
        "match_count",
    )

    returned_count = require_result_u64(
        result.get("returned_count"),
        "returned_count",
    )

    coordinates_raw = result.get(
        "coordinates"
    )

    if not isinstance(
        coordinates_raw,
        list,
    ):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "coordinates are not a list",
        )

    coordinates: list[list[int]] = []

    for item in coordinates_raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
        ):
            raise result_error(
                "COMPOSITION_E_VERIFY",
                "invalid coordinate shape",
            )

        doc_id = require_result_u64(
            item[0],
            "coordinate_doc_id",
        )

        doc_offset = require_result_u64(
            item[1],
            "coordinate_doc_offset",
        )

        if doc_id >= root.document_count:
            raise result_error(
                "COMPOSITION_E_VERIFY",
                "coordinate doc_id out "
                "of range",
            )

        coordinates.append([
            doc_id,
            doc_offset,
        ])

    if coordinates != sorted(
        coordinates
    ):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "coordinates are not canonical",
        )

    if len({
        (item[0], item[1])
        for item in coordinates
    }) != len(coordinates):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "duplicate coordinate",
        )

    if returned_count != len(
        coordinates
    ):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "returned_count mismatch",
        )

    expected_coordinates = (
        naive_coordinates_from_root(
            root,
            query,
        )
    )

    if match_count != len(
        expected_coordinates
    ):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "complete match_count mismatch",
        )

    expected_returned = (
        expected_coordinates
        if max_offsets is None
        else expected_coordinates[
            :max_offsets
        ]
    )

    if coordinates != expected_returned:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "result is not the canonical "
            "global prefix",
        )

    expected_bounded = (
        len(expected_returned)
        < len(expected_coordinates)
    )

    expected_complete = (
        not expected_bounded
    )

    if (
        result.get("bounded")
        is not expected_bounded
        or result.get(
            "offsets_complete"
        )
        is not expected_complete
    ):
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "bounded/completeness flag "
            "mismatch",
        )

    block_count = len(root.blocks)

    expected_blocks = (
        validate_block_coverage_list(
            result.get(
                "expected_blocks"
            ),
            "expected_blocks",
            block_count,
        )
    )

    verified_blocks = (
        validate_block_coverage_list(
            result.get(
                "verified_blocks"
            ),
            "verified_blocks",
            block_count,
        )
    )

    queried_blocks = (
        validate_block_coverage_list(
            result.get(
                "queried_blocks"
            ),
            "queried_blocks",
            block_count,
        )
    )

    if not (
        expected_blocks
        == verified_blocks
        == queried_blocks
    ):
        raise result_error(
            "COMPOSITION_E_COVERAGE",
            "incomplete block coverage",
        )

    policy_checks = {
        "composition_policy":
            COMPOSITION_POLICY,
        "coverage_policy":
            COVERAGE_POLICY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
    }

    for field, expected in (
        policy_checks.items()
    ):
        if result.get(field) != expected:
            raise result_error(
                "COMPOSITION_E_VERIFY",
                f"policy mismatch: {field}",
            )

    committed_result_id = result.get(
        "composition_result_id"
    )

    try:
        raw_sha256(
            committed_result_id,
            "composition_result_id",
        )

    except CompositionError as error:
        raise result_error(
            "COMPOSITION_E_IDENTITY",
            "invalid composition result "
            "identity",
        ) from error

    recomputed_result_id = (
        composition_result_id(result)
    )

    if (
        recomputed_result_id
        != committed_result_id
    ):
        raise result_error(
            "COMPOSITION_E_IDENTITY",
            "composition result identity "
            "mismatch",
        )

    return dict(result)


def serialize_and_validate_result(
    root: Root,
    query: bytes,
    result: dict[str, Any],
) -> dict[str, Any]:
    serialized = canonical_json_bytes(
        result
    )

    loaded = json.loads(
        serialized.decode("utf-8")
    )

    if canonical_json_bytes(
        loaded
    ) != serialized:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "composition result canonical "
            "serialization mismatch",
        )

    verified = validate_composed_result(
        root,
        query,
        loaded,
    )

    if canonical_json_bytes(
        verified
    ) != serialized:
        raise result_error(
            "COMPOSITION_E_VERIFY",
            "verified result differs from "
            "serialized result",
        )

    return verified


def refresh_composition_result_id(
    result: dict[str, Any],
) -> dict[str, Any]:
    result["composition_result_id"] = (
        composition_result_id(result)
    )

    return result


def expect_result_failure(
    name: str,
    expected_error_class: str,
    function,
) -> dict[str, Any]:
    try:
        function()

    except CompositionError as error:
        message = str(error)
        prefix = expected_error_class + ":"

        if not message.startswith(prefix):
            raise CompositionError(
                f"{name}: expected "
                f"{expected_error_class}, "
                f"received {message}"
            ) from error

        return {
            "mutation": name,
            "expected_error_class":
                expected_error_class,
            "rejected": True,
        }

    raise CompositionError(
        f"mutation unexpectedly accepted: {name}"
    )


def validate_result_mutations(
    root: Root,
    positive_query: bytes,
    positive_result: dict[str, Any],
    zero_query: bytes,
    zero_result: dict[str, Any],
) -> list[dict[str, Any]]:
    validate_composed_result(
        root,
        positive_query,
        positive_result,
    )

    validate_composed_result(
        root,
        zero_query,
        zero_result,
    )

    if (
        positive_result.get("max_offsets")
        is None
        or positive_result.get(
            "match_count"
        ) <= positive_result.get(
            "returned_count"
        )
    ):
        raise CompositionError(
            "positive mutation baseline "
            "must be bounded"
        )

    if (
        zero_result.get("match_count") != 0
        or zero_result.get(
            "returned_count"
        ) != 0
        or zero_result.get(
            "coordinates"
        ) != []
    ):
        raise CompositionError(
            "zero mutation baseline "
            "is not empty"
        )

    mutations: list[dict[str, Any]] = []

    def validate_positive(
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return validate_composed_result(
            root,
            positive_query,
            candidate,
        )

    def validate_zero(
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return validate_composed_result(
            root,
            zero_query,
            candidate,
        )

    missing_verified = clone_json(
        positive_result
    )

    missing_verified[
        "verified_blocks"
    ] = missing_verified[
        "verified_blocks"
    ][:-1]

    refresh_composition_result_id(
        missing_verified
    )

    mutations.append(
        expect_result_failure(
            "missing_verified_block",
            "COMPOSITION_E_COVERAGE",
            lambda: validate_positive(
                missing_verified
            ),
        )
    )

    missing_queried = clone_json(
        positive_result
    )

    missing_queried[
        "queried_blocks"
    ] = missing_queried[
        "queried_blocks"
    ][:-1]

    refresh_composition_result_id(
        missing_queried
    )

    mutations.append(
        expect_result_failure(
            "missing_queried_block",
            "COMPOSITION_E_COVERAGE",
            lambda: validate_positive(
                missing_queried
            ),
        )
    )

    partial_zero = clone_json(
        zero_result
    )

    partial_zero[
        "verified_blocks"
    ] = partial_zero[
        "verified_blocks"
    ][:-1]

    partial_zero[
        "queried_blocks"
    ] = partial_zero[
        "queried_blocks"
    ][:-1]

    refresh_composition_result_id(
        partial_zero
    )

    mutations.append(
        expect_result_failure(
            "partial_coverage_represented_as_zero",
            "COMPOSITION_E_COVERAGE",
            lambda: validate_zero(
                partial_zero
            ),
        )
    )

    wrong_count = clone_json(
        positive_result
    )

    wrong_count["match_count"] += 1

    refresh_composition_result_id(
        wrong_count
    )

    mutations.append(
        expect_result_failure(
            "incorrect_complete_match_count",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                wrong_count
            ),
        )
    )

    reordered = clone_json(
        positive_result
    )

    if len(reordered["coordinates"]) < 2:
        raise CompositionError(
            "insufficient coordinates "
            "for reorder mutation"
        )

    (
        reordered["coordinates"][0],
        reordered["coordinates"][1],
    ) = (
        reordered["coordinates"][1],
        reordered["coordinates"][0],
    )

    refresh_composition_result_id(
        reordered
    )

    mutations.append(
        expect_result_failure(
            "reordered_coordinates",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                reordered
            ),
        )
    )

    wrong_doc_id = clone_json(
        positive_result
    )

    if not wrong_doc_id["coordinates"]:
        raise CompositionError(
            "missing coordinate for "
            "doc_id mutation"
        )

    wrong_doc_id["coordinates"][0] = [
        0,
        0,
    ]

    refresh_composition_result_id(
        wrong_doc_id
    )

    mutations.append(
        expect_result_failure(
            "wrong_global_doc_id",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                wrong_doc_id
            ),
        )
    )

    wrong_prefix = clone_json(
        positive_result
    )

    expected_coordinates = (
        naive_coordinates_from_root(
            root,
            positive_query,
        )
    )

    prefix_length = len(
        positive_result["coordinates"]
    )

    if len(expected_coordinates) <= prefix_length:
        raise CompositionError(
            "insufficient oracle coordinates "
            "for prefix mutation"
        )

    wrong_prefix["coordinates"] = (
        expected_coordinates[
            1:
            1 + prefix_length
        ]
    )

    wrong_prefix["returned_count"] = len(
        wrong_prefix["coordinates"]
    )

    refresh_composition_result_id(
        wrong_prefix
    )

    mutations.append(
        expect_result_failure(
            "incorrect_global_max_offsets_prefix",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                wrong_prefix
            ),
        )
    )

    wrong_flags = clone_json(
        positive_result
    )

    wrong_flags["bounded"] = False
    wrong_flags["offsets_complete"] = True

    refresh_composition_result_id(
        wrong_flags
    )

    mutations.append(
        expect_result_failure(
            "false_bounded_completeness_flags",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                wrong_flags
            ),
        )
    )

    wrong_returned_count = clone_json(
        positive_result
    )

    wrong_returned_count[
        "returned_count"
    ] += 1

    refresh_composition_result_id(
        wrong_returned_count
    )

    mutations.append(
        expect_result_failure(
            "incorrect_returned_count",
            "COMPOSITION_E_VERIFY",
            lambda: validate_positive(
                wrong_returned_count
            ),
        )
    )

    changed_result_id = clone_json(
        positive_result
    )

    changed_result_id[
        "composition_result_id"
    ] = "0" * 64

    mutations.append(
        expect_result_failure(
            "changed_composition_result_id",
            "COMPOSITION_E_IDENTITY",
            lambda: validate_positive(
                changed_result_id
            ),
        )
    )

    if len(mutations) != 10:
        raise CompositionError(
            "result mutation count mismatch"
        )

    if not all(
        item["rejected"] is True
        for item in mutations
    ):
        raise CompositionError(
            "result mutation gate failed"
        )

    return mutations


def make_result(
    root: Root,
    query: bytes,
    max_offsets: int | None,
    match_count: int,
    coordinates: list[list[int]],
) -> dict[str, Any]:
    returned_count = len(coordinates)

    expected = list(
        range(len(root.blocks))
    )

    result = {
        "ok": True,
        "format": RESULT_VERSION,
        "runtime_corpus_id":
            root.corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "composition_root_id":
            root.composition_root_id,
        "query_hex":
            query.hex(),
        "query_length_bytes":
            len(query),
        "query_sha256":
            hashlib.sha256(
                query
            ).hexdigest(),
        "max_offsets":
            max_offsets,
        "match_count":
            match_count,
        "returned_count":
            returned_count,
        "bounded":
            returned_count < match_count,
        "offsets_complete":
            returned_count == match_count,
        "coordinates":
            coordinates,
        "expected_blocks":
            expected,
        "verified_blocks":
            expected,
        "queried_blocks":
            expected,
        "composition_policy":
            COMPOSITION_POLICY,
        "coverage_policy":
            COVERAGE_POLICY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
    }

    result["composition_result_id"] = (
        composition_result_id(result)
    )

    return result


def compose_full(
    root: Root,
    query: bytes,
    results: dict[
        int,
        dict[str, Any],
    ],
    max_offsets: int | None,
) -> dict[str, Any]:
    if max_offsets is not None:
        u64(
            max_offsets,
            "max_offsets",
        )

    if sorted(results) != list(
        range(len(root.blocks))
    ):
        raise CompositionError(
            "coverage mismatch"
        )

    total = 0
    coordinates: list[
        list[int]
    ] = []

    for block in root.blocks:
        result = results[
            block.ordinal
        ]

        total = checked_add_aggregation(
            total,
            result["match_count"],
            "global_match_count",
        )

        for item in result[
            "coordinates"
        ]:
            local_doc_id, doc_offset = (
                item["coordinate"]
            )

            global_doc_id = checked_add(
                block.start,
                local_doc_id,
                "global_doc_id",
            )

            if not (
                block.start
                <= global_doc_id
                < block.end
            ):
                raise CompositionError(
                    "global doc_id escaped "
                    "block range"
                )

            coordinates.append([
                global_doc_id,
                doc_offset,
            ])

    if coordinates != sorted(
        coordinates
    ):
        raise CompositionError(
            "merged coordinates "
            "not canonical"
        )

    if len(coordinates) != total:
        raise CompositionError(
            "full coordinate count "
            "mismatch"
        )

    returned = (
        coordinates
        if max_offsets is None
        else coordinates[:max_offsets]
    )

    result = make_result(
        root,
        query,
        max_offsets,
        total,
        returned,
    )

    return serialize_and_validate_result(
        root,
        query,
        result,
    )


def compose_two_phase(
    root: Root,
    query: bytes,
    max_offsets: int,
) -> dict[str, Any]:
    u64(
        max_offsets,
        "max_offsets",
    )

    counts: dict[int, int] = {}
    total = 0

    for block in root.blocks:
        count_result = (
            execute_operator_query(
                block.corpus,
                query,
                max_offsets=0,
            )
        )

        if (
            count_result[
                "returned_count"
            ]
            != 0
        ):
            raise CompositionError(
                "count phase returned "
                "coordinates"
            )

        count = u64(
            count_result[
                "match_count"
            ],
            "block_match_count",
        )

        counts[
            block.ordinal
        ] = count

        total = checked_add_aggregation(
            total,
            count,
            "global_match_count",
        )

    remaining = max_offsets
    coordinates: list[
        list[int]
    ] = []

    for block in root.blocks:
        if remaining == 0:
            continue

        locate = (
            execute_operator_query(
                block.corpus,
                query,
                max_offsets=remaining,
            )
        )

        if (
            locate["match_count"]
            != counts[
                block.ordinal
            ]
        ):
            raise CompositionError(
                "count/locate phase "
                "mismatch"
            )

        for item in locate[
            "coordinates"
        ]:
            local_doc_id, doc_offset = (
                item["coordinate"]
            )

            coordinates.append([
                checked_add(
                    block.start,
                    local_doc_id,
                    "global_doc_id",
                ),
                doc_offset,
            ])

        remaining -= locate[
            "returned_count"
        ]

    if coordinates != sorted(
        coordinates
    ):
        raise CompositionError(
            "two-phase coordinates "
            "not canonical"
        )

    if len(coordinates) != min(
        max_offsets,
        total,
    ):
        raise CompositionError(
            "two-phase prefix length "
            "mismatch"
        )

    result = make_result(
        root,
        query,
        max_offsets,
        total,
        coordinates,
    )

    return serialize_and_validate_result(
        root,
        query,
        result,
    )


def validate_oracle(
    root: Root,
    documents: Sequence[Document],
    query: bytes,
    max_offsets: int | None,
    result: dict[str, Any],
) -> None:
    expected = naive_coordinates(
        documents,
        query,
    )

    returned = (
        expected
        if max_offsets is None
        else expected[:max_offsets]
    )

    checks = {
        "runtime_corpus_id":
            root.corpus_id,
        "source_manifest_id":
            root.source_manifest_id,
        "composition_root_id":
            root.composition_root_id,
        "match_count":
            len(expected),
        "returned_count":
            len(returned),
        "coordinates":
            returned,
        "bounded":
            len(returned)
            < len(expected),
        "offsets_complete":
            len(returned)
            == len(expected),
    }

    for field, value in (
        checks.items()
    ):
        if result.get(field) != value:
            raise CompositionError(
                f"oracle mismatch: {field}"
            )


def semantic_view(
    result: dict[str, Any],
) -> dict[str, Any]:
    value = dict(result)

    value.pop(
        "composition_root_id"
    )

    value.pop(
        "composition_result_id"
    )

    return value


def validate_replay_mutations(
    source_root: Root,
    replay_root: Root,
) -> list[dict[str, Any]]:
    query = b"shared"

    if (
        source_root is replay_root
        or source_root.corpus_id
        != replay_root.corpus_id
        or source_root.source_manifest_id
        != replay_root.source_manifest_id
        or source_root.document_count
        != replay_root.document_count
        or source_root.composition_root_id
        == replay_root.composition_root_id
    ):
        raise CompositionError(
            "different-root replay fixture "
            "identity control failed"
        )

    source_result = compose_full(
        source_root,
        query,
        run_full_results(
            source_root,
            query,
            list(
                range(len(source_root.blocks))
            ),
        ),
        None,
    )

    replay_root_result = compose_full(
        replay_root,
        query,
        run_full_results(
            replay_root,
            query,
            list(
                range(len(replay_root.blocks))
            ),
        ),
        None,
    )

    source_artifact = canonical_json_bytes(
        source_result
    )

    replay_root_artifact = (
        canonical_json_bytes(
            replay_root_result
        )
    )

    if (
        source_result.get(
            "composition_root_id"
        )
        != source_root.composition_root_id
        or replay_root_result.get(
            "composition_root_id"
        )
        != replay_root.composition_root_id
        or source_result.get(
            "composition_result_id"
        )
        == replay_root_result.get(
            "composition_result_id"
        )
        or source_artifact
        == replay_root_artifact
        or canonical_json_bytes(
            semantic_view(source_result)
        )
        != canonical_json_bytes(
            semantic_view(
                replay_root_result
            )
        )
    ):
        raise CompositionError(
            "different-root replay fixture "
            "semantic control failed"
        )

    replay_candidate = json.loads(
        source_artifact.decode("utf-8")
    )

    if canonical_json_bytes(
        replay_candidate
    ) != source_artifact:
        raise CompositionError(
            "different-root replay artifact "
            "is not canonical"
        )

    positive = serialize_and_validate_result(
        source_root,
        query,
        replay_candidate,
    )

    if canonical_json_bytes(
        positive
    ) != source_artifact:
        raise CompositionError(
            "same-root replay positive "
            "control failed"
        )

    mutations = [
        expect_result_failure(
            "replay_against_different_root",
            "COMPOSITION_E_IDENTITY",
            lambda: serialize_and_validate_result(
                replay_root,
                query,
                replay_candidate,
            ),
        )
    ]

    if (
        len(mutations) != 1
        or mutations[0]["rejected"]
        is not True
        or canonical_json_bytes(
            replay_candidate
        )
        != source_artifact
    ):
        raise CompositionError(
            "different-root replay mutation "
            "gate failed"
        )

    return mutations


def validate_byte_check_replay_mutations(
    work: Path,
    source_root: Root,
    query: bytes,
    stored_result: dict[str, Any],
) -> list[dict[str, Any]]:
    if not query:
        raise CompositionError(
            "byte-check replay query is empty"
        )

    stored_artifact = canonical_json_bytes(
        stored_result
    )

    replay_candidate = json.loads(
        stored_artifact.decode("utf-8")
    )

    if (
        canonical_json_bytes(replay_candidate)
        != stored_artifact
        or replay_candidate.get("ok")
        is not True
        or replay_candidate.get(
            "verified_blocks"
        ) != list(
            range(len(source_root.blocks))
        )
        or not replay_candidate.get(
            "coordinates"
        )
    ):
        raise CompositionError(
            "stored byte-check replay fixture "
            "control failed"
        )

    positive = serialize_and_validate_result(
        source_root,
        query,
        replay_candidate,
    )

    if canonical_json_bytes(
        positive
    ) != stored_artifact:
        raise CompositionError(
            "stored byte-check positive replay "
            "control failed"
        )

    global_doc_id, doc_offset = (
        replay_candidate["coordinates"][0]
    )

    target_block = next(
        (
            block
            for block in source_root.blocks
            if block.start
            <= global_doc_id
            < block.end
        ),
        None,
    )

    if target_block is None:
        raise CompositionError(
            "stored byte-check coordinate has "
            "no source block"
        )

    cloned_block = clone_block_for_mutation(
        work,
        target_block,
        "stored-byte-check-success",
    )

    source_manifest = load_canonical_json(
        cloned_block.corpus
        / SOURCE_MANIFEST_NAME
    )

    records = source_manifest.get(
        "documents"
    )

    local_doc_id = (
        global_doc_id - cloned_block.start
    )

    if (
        not isinstance(records, list)
        or local_doc_id < 0
        or local_doc_id >= len(records)
        or not isinstance(
            records[local_doc_id],
            dict,
        )
    ):
        raise CompositionError(
            "stored byte-check source record "
            "control failed"
        )

    snapshot_relative = records[
        local_doc_id
    ].get("snapshot_path")

    if not isinstance(
        snapshot_relative,
        str,
    ):
        raise CompositionError(
            "stored byte-check snapshot path "
            "control failed"
        )

    snapshot_path = (
        cloned_block.corpus
        / snapshot_relative
    )

    payload = bytearray(
        snapshot_path.read_bytes()
    )

    span_end = doc_offset + len(query)

    if (
        doc_offset < 0
        or span_end > len(payload)
        or bytes(
            payload[doc_offset:span_end]
        ) != query
    ):
        raise CompositionError(
            "stored byte-check source span "
            "control failed"
        )

    payload[doc_offset] ^= 0x01
    snapshot_path.write_bytes(payload)

    if bytes(
        payload[doc_offset:span_end]
    ) == query:
        raise CompositionError(
            "stored byte-check source mutation "
            "did not invalidate span"
        )

    replay_blocks = tuple(
        cloned_block
        if block.ordinal
        == target_block.ordinal
        else block
        for block in source_root.blocks
    )

    mutated_root = Root(
        name=(
            source_root.name
            + "-stored-byte-check-mutation"
        ),
        blocks=replay_blocks,
        document_count=(
            source_root.document_count
        ),
        corpus_id=source_root.corpus_id,
        source_manifest_id=(
            source_root.source_manifest_id
        ),
        composition_root_id=(
            source_root.composition_root_id
        ),
    )

    if (
        mutated_root.corpus_id
        != source_root.corpus_id
        or mutated_root.source_manifest_id
        != source_root.source_manifest_id
        or mutated_root.composition_root_id
        != source_root.composition_root_id
        or replay_candidate.get(
            "composition_result_id"
        ) != stored_result.get(
            "composition_result_id"
        )
    ):
        raise CompositionError(
            "stored byte-check replay identity "
            "control failed"
        )

    mutations = [
        expect_result_failure(
            "stored_byte_check_success_"
            "without_recomputation",
            "COMPOSITION_E_VERIFY",
            lambda: serialize_and_validate_result(
                mutated_root,
                query,
                replay_candidate,
            ),
        )
    ]

    if (
        len(mutations) != 1
        or mutations[0]["rejected"]
        is not True
        or canonical_json_bytes(
            replay_candidate
        ) != stored_artifact
    ):
        raise CompositionError(
            "stored byte-check replay mutation "
            "gate failed"
        )

    return mutations


def independent_replay_command(
    root: Root,
    result_path: Path,
    block_paths: Sequence[Path],
) -> list[str]:
    command = [
        sys.executable,
        "-I",
        str(INDEPENDENT_REPLAY),
        "--root",
        str(
            result_path.parent
            / (
                f"{root.name}-"
                "composition-root-v1.json"
            )
        ),
        "--result",
        str(result_path),
    ]

    for block_path in block_paths:
        command.extend([
            "--block",
            str(block_path),
        ])

    return command


def run_independent_replay_process(
    command: Sequence[str],
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    cwd.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    return subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )


def require_independent_replay_failure(
    name: str,
    expected_error_class: str,
    command: Sequence[str],
    cwd: Path,
) -> dict[str, Any]:
    replay = run_independent_replay_process(
        command,
        cwd,
    )

    if (
        replay.returncode == 0
        or expected_error_class
        not in replay.stderr
        or INDEPENDENT_REPLAY_MARKER
        in replay.stdout
    ):
        raise CompositionError(
            "independent replay mutation "
            f"did not fail closed: {name}; "
            f"stdout={replay.stdout[-1000:]!r}; "
            f"stderr={replay.stderr[-1000:]!r}"
        )

    return {
        "mutation": name,
        "expected_error_class":
            expected_error_class,
        "rejected": True,
    }


def validate_independent_replay(
    work: Path,
    source_root: Root,
    different_root: Root,
    source_result: dict[str, Any],
) -> dict[str, Any]:
    if not INDEPENDENT_REPLAY.is_file():
        raise CompositionError(
            "independent replay entrypoint "
            "is unavailable"
        )

    result_path = (
        work
        / "composition-reference-"
        "result-v1.json"
    )
    result_payload = canonical_json_bytes(
        source_result
    )
    result_path.write_bytes(result_payload)

    if (
        canonical_json_bytes(
            json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )
        )
        != result_payload
    ):
        raise CompositionError(
            "independent replay result "
            "serialization mismatch"
        )

    source_blocks = [
        block.corpus
        for block in source_root.blocks
    ]
    command = independent_replay_command(
        source_root,
        result_path,
        source_blocks,
    )

    replay_a = run_independent_replay_process(
        command,
        work / "independent-replay-cwd-a",
    )
    replay_b = run_independent_replay_process(
        command,
        work / "independent-replay-cwd-b",
    )

    for label, replay in (
        ("a", replay_a),
        ("b", replay_b),
    ):
        if (
            replay.returncode != 0
            or replay.stderr != ""
            or INDEPENDENT_REPLAY_MARKER
            not in replay.stdout
        ):
            raise CompositionError(
                "independent replay positive "
                f"run failed: {label}; "
                f"stdout={replay.stdout[-1000:]!r}; "
                f"stderr={replay.stderr[-1000:]!r}"
            )

    if replay_a.stdout != replay_b.stdout:
        raise CompositionError(
            "independent replay output depends "
            "on working directory"
        )

    negative_cases = []

    negative_cases.append(
        require_independent_replay_failure(
            "replay_against_different_root",
            "COMPOSITION_E_IDENTITY",
            independent_replay_command(
                different_root,
                result_path,
                [
                    block.corpus
                    for block
                    in different_root.blocks
                ],
            ),
            work / (
                "independent-replay-"
                "different-root"
            ),
        )
    )

    negative_cases.append(
        require_independent_replay_failure(
            "required_runtime_unit_missing",
            "COMPOSITION_E_COVERAGE",
            independent_replay_command(
                source_root,
                result_path,
                source_blocks[:-1],
            ),
            work / (
                "independent-replay-"
                "missing-block"
            ),
        )
    )

    mutated_block = (
        work
        / "independent-replay-"
        "mutated-block"
    )
    shutil.copytree(
        source_root.blocks[0].corpus,
        mutated_block,
    )
    source_manifest = load_canonical_json(
        mutated_block / SOURCE_MANIFEST_NAME
    )
    source_records = source_manifest.get(
        "documents"
    )

    if not isinstance(source_records, list):
        raise CompositionError(
            "independent replay mutation "
            "source manifest is invalid"
        )

    mutable_record = next(
        (
            record
            for record in source_records
            if (
                isinstance(record, dict)
                and record.get("byte_length", 0)
                > 0
            )
        ),
        None,
    )

    if mutable_record is None:
        raise CompositionError(
            "independent replay mutation has "
            "no non-empty source document"
        )

    snapshot_path = (
        mutated_block
        / mutable_record["snapshot_path"]
    )
    snapshot_payload = bytearray(
        snapshot_path.read_bytes()
    )
    snapshot_payload[0] ^= 0x01
    snapshot_path.write_bytes(snapshot_payload)

    mutated_blocks = [
        mutated_block,
        *source_blocks[1:],
    ]

    negative_cases.append(
        require_independent_replay_failure(
            "stored_success_with_changed_source",
            "COMPOSITION_E_VERIFY",
            independent_replay_command(
                source_root,
                result_path,
                mutated_blocks,
            ),
            work / (
                "independent-replay-"
                "changed-source"
            ),
        )
    )

    return {
        "ok": True,
        "format": (
            "GLYPH_COMPOSITION_"
            "INDEPENDENT_REPLAY_GATE_V1"
        ),
        "entrypoint": (
            "tools/"
            "replay_composition_reference_v1.py"
        ),
        "marker": INDEPENDENT_REPLAY_MARKER,
        "different_working_directory_verified":
            True,
        "deterministic_stdout_verified": True,
        "stdout_sha256": hashlib.sha256(
            replay_a.stdout.encode("utf-8")
        ).hexdigest(),
        "complete_block_coverage_verified": True,
        "runtime_query_replay_verified": True,
        "independent_source_oracle_verified": True,
        "returned_byte_check_recomputed": True,
        "negative_case_count":
            len(negative_cases),
        "negative_cases": negative_cases,
    }


def validate_global_manifest(
    work: Path,
    documents: Sequence[Document],
    corpus_id: str,
    manifest_id: str,
) -> None:
    source = work / "global-source"
    corpus = work / "global-corpus"

    write_tree(
        source,
        documents,
    )

    build_snapshot(
        source,
        corpus,
    )

    manifest = load_canonical_json(
        corpus
        / SOURCE_MANIFEST_NAME
    )

    validate_source_manifest(
        manifest,
        documents,
    )

    if manifest.get(
        "corpus_id"
    ) != corpus_id:
        raise CompositionError(
            "global corpus_id mismatch"
        )

    if manifest.get(
        "source_manifest_id"
    ) != manifest_id:
        raise CompositionError(
            "global source_manifest_id "
            "mismatch"
        )


def main() -> int:
    documents = fixture_documents()

    if [
        item.path
        for item in documents
    ] != sorted(
        item.path
        for item in documents
    ):
        raise CompositionError(
            "fixture paths "
            "are not canonical"
        )

    corpus_id = runtime_corpus_id(
        documents
    )

    manifest_id = source_manifest_id(
        documents
    )

    with tempfile.TemporaryDirectory(
        prefix=(
            "glyph-composition-"
            "reference-v1-"
        )
    ) as temporary:
        work = Path(temporary)

        validate_global_manifest(
            work,
            documents,
            corpus_id,
            manifest_id,
        )

        root_a = build_root(
            work,
            "partition-a",
            [
                (0, 3),
                (3, 6),
                (6, 9),
            ],
            documents,
            corpus_id,
            manifest_id,
        )

        root_b = build_root(
            work,
            "partition-b",
            [
                (0, 2),
                (2, 6),
                (6, 9),
            ],
            documents,
            corpus_id,
            manifest_id,
        )

        roots = [
            root_a,
            root_b,
        ]

        if (
            root_a.composition_root_id
            == root_b.composition_root_id
        ):
            raise CompositionError(
                "repartition did not "
                "change root identity"
            )

        validate_straddles(
            documents,
            roots,
        )

        root_mutations = (
            validate_root_mutations(
                root_a
            )
        )

        artifact_mutations = (
            validate_artifact_integrity_mutations(
                work,
                root_a,
                documents,
            )
        )

        queries = [
            b"shared",
            b"dup",
            b"\x00\xff",
            b"alpha LEFT",
            b"a",
            b"not-present",
            b"\x00\xffshared",
            b"dupdup",
            (
                b"this-query-is-longer-"
                b"than-every-document-"
                b"in-the-fixture"
            ),
        ]

        fixtures = []

        for query in queries:
            outputs = []

            for root in roots:
                full = run_full_results(
                    root,
                    query,
                    list(
                        range(
                            len(root.blocks)
                        )
                    ),
                )

                result = compose_full(
                    root,
                    query,
                    full,
                    None,
                )

                validate_oracle(
                    root,
                    documents,
                    query,
                    None,
                    result,
                )

                outputs.append(
                    result
                )

            if (
                canonical_json_bytes(
                    semantic_view(
                        outputs[0]
                    )
                )
                != canonical_json_bytes(
                    semantic_view(
                        outputs[1]
                    )
                )
            ):
                raise CompositionError(
                    "repartition changed "
                    "semantic result"
                )

            fixtures.append({
                "query_hex":
                    query.hex(),
                "match_count":
                    outputs[0][
                        "match_count"
                    ],
                "returned_count":
                    outputs[0][
                        "returned_count"
                    ],
            })

        repeated_query = b"a"

        repeated_count = len(
            naive_coordinates(
                documents,
                repeated_query,
            )
        )

        limits = sorted({
            0,
            1,
            max(
                0,
                repeated_count - 1,
            ),
            repeated_count,
            repeated_count + 1,
        })

        bounded = []

        for limit in limits:
            outputs = []

            for root in roots:
                full = run_full_results(
                    root,
                    repeated_query,
                    list(
                        range(
                            len(root.blocks)
                        )
                    ),
                )

                expected = compose_full(
                    root,
                    repeated_query,
                    full,
                    limit,
                )

                actual = (
                    compose_two_phase(
                        root,
                        repeated_query,
                        limit,
                    )
                )

                if (
                    canonical_json_bytes(
                        actual
                    )
                    != canonical_json_bytes(
                        expected
                    )
                ):
                    raise CompositionError(
                        "two-phase result "
                        "differs from "
                        "full prefix"
                    )

                validate_oracle(
                    root,
                    documents,
                    repeated_query,
                    limit,
                    actual,
                )

                outputs.append(
                    actual
                )

            if (
                canonical_json_bytes(
                    semantic_view(
                        outputs[0]
                    )
                )
                != canonical_json_bytes(
                    semantic_view(
                        outputs[1]
                    )
                )
            ):
                raise CompositionError(
                    "repartition changed "
                    "bounded result"
                )

            bounded.append({
                "max_offsets":
                    limit,
                "match_count":
                    outputs[0][
                        "match_count"
                    ],
                "returned_count":
                    outputs[0][
                        "returned_count"
                    ],
                "bounded":
                    outputs[0][
                        "bounded"
                    ],
            })

        schedule_query = b"shared"

        orders = [
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 2],
        ]

        schedule_results = []

        for order in orders:
            full = run_full_results(
                root_a,
                schedule_query,
                order,
            )

            result = compose_full(
                root_a,
                schedule_query,
                full,
                None,
            )

            validate_oracle(
                root_a,
                documents,
                schedule_query,
                None,
                result,
            )

            schedule_results.append(
                result
            )

        first = canonical_json_bytes(
            schedule_results[0]
        )

        if any(
            canonical_json_bytes(item)
            != first
            for item
            in schedule_results[1:]
        ):
            raise CompositionError(
                "completion order "
                "changed output"
            )

        if (
            naive_coordinates(
                documents,
                b"dup",
            )
            != [
                [5, 0],
                [6, 0],
            ]
        ):
            raise CompositionError(
                "duplicate document "
                "identity lost"
            )

        if (
            naive_coordinates(
                documents,
                b"\x00\xff",
            )
            != [[3, 13]]
        ):
            raise CompositionError(
                "binary fixture "
                "coordinate mismatch"
            )

        mutation_query = b"a"

        mutation_full_results = (
            run_full_results(
                root_a,
                mutation_query,
                [0, 1, 2],
            )
        )

        mutation_positive_result = (
            compose_full(
                root_a,
                mutation_query,
                mutation_full_results,
                3,
            )
        )

        mutation_zero_query = (
            b"not-present"
        )

        mutation_zero_full_results = (
            run_full_results(
                root_a,
                mutation_zero_query,
                [0, 1, 2],
            )
        )

        mutation_zero_result = (
            compose_full(
                root_a,
                mutation_zero_query,
                mutation_zero_full_results,
                None,
            )
        )

        result_mutations = (
            validate_result_mutations(
                root_a,
                mutation_query,
                mutation_positive_result,
                mutation_zero_query,
                mutation_zero_result,
            )
        )

        replay_mutations = (
            validate_replay_mutations(
                root_a,
                root_b,
            )
        )

        byte_check_replay_mutations = (
            validate_byte_check_replay_mutations(
                work,
                root_a,
                mutation_query,
                mutation_positive_result,
            )
        )

        independent_replay = (
            validate_independent_replay(
                work,
                root_a,
                root_b,
                mutation_positive_result,
            )
        )

        document_model_mutations = (
            validate_document_model_mutations(
                root_a,
                documents,
            )
        )

        aggregation_mutations = (
            validate_aggregation_limit_mutations(
                root_a
            )
        )

        mutation_traceability = (
            build_mutation_traceability(
                root_mutations,
                result_mutations,
                artifact_mutations,
                aggregation_mutations,
                document_model_mutations,
                replay_mutations,
                byte_check_replay_mutations,
            )
        )

        summary = {
            "ok": True,
            "format": (
                "GLYPH_COMPOSITION_"
                "REFERENCE_NORMAL_PATH_V1"
            ),
            "runtime_corpus_id":
                corpus_id,
            "source_manifest_id":
                manifest_id,
            "partition_a_root_id":
                root_a.composition_root_id,
            "partition_b_root_id":
                root_b.composition_root_id,
            "global_document_count":
                len(documents),
            "block_count_per_partition":
                3,
            "query_fixture_count":
                len(fixtures),
            "bounded_case_count":
                len(bounded),
            "schedule_order_count":
                len(orders),
            "repartition_identity_stable":
                True,
            "repartition_coordinates_stable":
                True,
            "root_identity_layout_sensitive":
                True,
            "root_manifest_validation_verified":
                True,
            "composition_result_validation_verified":
                True,
            "composition_result_mutations_verified":
                True,
            "result_mutation_count":
                len(result_mutations),
            "different_root_replay_verified":
                True,
            "replay_mutation_count":
                len(replay_mutations),
            "stored_byte_check_recomputation_verified":
                True,
            "independent_composition_replay_verified":
                True,
            "independent_replay_different_cwd_verified":
                independent_replay[
                    "different_working_directory_verified"
                ],
            "independent_replay_deterministic":
                independent_replay[
                    "deterministic_stdout_verified"
                ],
            "independent_replay_negative_case_count":
                independent_replay[
                    "negative_case_count"
                ],
            "byte_check_replay_mutation_count":
                len(
                    byte_check_replay_mutations
                ),
            "artifact_integrity_mutations_verified":
                True,
            "artifact_mutation_count":
                len(artifact_mutations),
            "aggregation_limit_mutations_verified":
                True,
            "aggregation_mutation_count":
                len(aggregation_mutations),
            "document_model_mutations_verified":
                True,
            "document_model_mutation_count":
                len(document_model_mutations),
            "mutation_traceability_verified":
                True,
            "normative_requirement_count":
                mutation_traceability[
                    "normative_requirement_count"
                ],
            "implemented_mutation_test_count":
                mutation_traceability[
                    "implemented_test_count"
                ],
            "normative_exact_count":
                mutation_traceability[
                    "normative_exact_count"
                ],
            "normative_supporting_count":
                mutation_traceability[
                    "normative_supporting_count"
                ],
            "normative_open_count":
                mutation_traceability[
                    "normative_open_count"
                ],
            "normative_not_exact_count":
                mutation_traceability[
                    "normative_not_exact_count"
                ],
            "normative_closure_complete":
                mutation_traceability[
                    "normative_closure_complete"
                ],
            "root_identity_mutations_verified":
                True,
            "root_mutation_count":
                len(root_mutations),
            "global_max_offsets_verified":
                True,
            "completion_order_independent":
                True,
            "binary_domain_verified":
                True,
            "empty_document_preserved":
                True,
            "duplicate_documents_preserved":
                True,
            "cross_document_excluded":
                True,
            "cross_block_excluded":
                True,
            "fixtures":
                fixtures,
            "bounded_results":
                bounded,
            "root_mutations":
                root_mutations,
            "result_mutations":
                result_mutations,
            "replay_mutations":
                replay_mutations,
            "byte_check_replay_mutations":
                byte_check_replay_mutations,
            "independent_replay":
                independent_replay,
            "artifact_mutations":
                artifact_mutations,
            "aggregation_mutations":
                aggregation_mutations,
            "document_model_mutations":
                document_model_mutations,
            "mutation_traceability":
                mutation_traceability,
        }

        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
        )

        print(
            "GLYPH COMPOSITION REFERENCE "
            "NORMAL PATH OK"
        )

        print(
            "GLYPH COMPOSITION REFERENCE OK"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )

    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "COMPOSITION_"
                        "REFERENCE_FAILURE"
                    ),
                    "error_type":
                        type(error).__name__,
                    "message":
                        str(error),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )

        raise SystemExit(1)
