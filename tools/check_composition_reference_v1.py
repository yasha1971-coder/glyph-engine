#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
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


class CompositionError(RuntimeError):
    pass


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

    return Root(
        name=name,
        blocks=tuple(blocks),
        document_count=len(documents),
        corpus_id=corpus_id,
        source_manifest_id=manifest_id,
        composition_root_id=root_id,
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

    return {
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

        total = checked_add(
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

    return make_result(
        root,
        query,
        max_offsets,
        total,
        returned,
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

        total = checked_add(
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

    return make_result(
        root,
        query,
        max_offsets,
        total,
        coordinates,
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

    return value


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
