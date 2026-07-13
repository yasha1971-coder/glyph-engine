#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BUILD = ROOT / "build"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_index_v1 import (  # noqa: E402
    INDEX_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    IndexErrorV1,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    OperatorError,
    canonical_json_bytes,
    load_canonical_json,
    sha256_file,
)

RESULT_VERSION = "GLYPH_OPERATOR_QUERY_RESULT_V1"
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"
INDEX_TOPOLOGY = "one_independent_index_per_document"
BOUNDARY_POLICY = "NO_PHYSICAL_DOCUMENT_CONCATENATION"

QUERY_BINARY_NAMES = [
    "query_fm_binary_v1",
    "query_fm_locate_binary_v1",
]

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


class QueryError(RuntimeError):
    exit_code = 7
    error_code = "INTERNAL_INVARIANT"


class QueryInputError(QueryError):
    exit_code = 2
    error_code = "INVALID_QUERY_INPUT"


class QuerySourceError(QueryError):
    exit_code = 3
    error_code = "QUERY_SOURCE_FAILURE"


class QueryRuntimeError(QueryError):
    exit_code = 5
    error_code = "RUNTIME_QUERY_FAILURE"


class QueryVerificationError(QueryError):
    exit_code = 6
    error_code = "QUERY_VERIFICATION_FAILURE"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def require_uint(
    value: Any,
    field: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise QueryVerificationError(
            f"invalid unsigned integer: {field}"
        )

    return value


def parse_query_hex(value: Any) -> bytes:
    if not isinstance(value, str) or value == "":
        raise QueryInputError("EMPTY_QUERY")

    if value != value.lower():
        raise QueryInputError(
            "query_hex must be lowercase"
        )

    if value.startswith("0x"):
        raise QueryInputError(
            "query_hex must not use 0x prefix"
        )

    if len(value) % 2 != 0:
        raise QueryInputError(
            "query_hex must have even length"
        )

    if any(
        character not in "0123456789abcdef"
        for character in value
    ):
        raise QueryInputError(
            "query_hex contains invalid character"
        )

    query = bytes.fromhex(value)

    if not query:
        raise QueryInputError("EMPTY_QUERY")

    if query.hex() != value:
        raise QueryInputError(
            "query_hex is not canonical"
        )

    return query


def read_all_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []

    while True:
        chunk = os.read(
            descriptor,
            1024 * 1024,
        )

        if not chunk:
            break

        chunks.append(chunk)

    return b"".join(chunks)


def read_stable_query_file(
    path: Path,
    *,
    after_first_read_hook: (
        Callable[[Path], None] | None
    ) = None,
) -> bytes:
    path = path.absolute()
    path_bytes = os.fsencode(os.fspath(path))

    try:
        path_before = os.lstat(path_bytes)
    except OSError as error:
        raise QuerySourceError(
            f"query file unavailable: {error}"
        ) from error

    if (
        stat.S_ISLNK(path_before.st_mode)
        or not stat.S_ISREG(path_before.st_mode)
    ):
        raise QuerySourceError(
            "query file must be a real regular file"
        )

    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(path_bytes, flags)
    except OSError as error:
        raise QuerySourceError(
            f"cannot open query file: {error}"
        ) from error

    try:
        descriptor_before = os.fstat(descriptor)

        if (
            descriptor_before.st_dev
            != path_before.st_dev
            or descriptor_before.st_ino
            != path_before.st_ino
        ):
            raise QuerySourceError(
                "query pathname changed before read"
            )

        first = read_all_descriptor(descriptor)

        if after_first_read_hook is not None:
            after_first_read_hook(path)

        os.lseek(descriptor, 0, os.SEEK_SET)
        second = read_all_descriptor(descriptor)

        descriptor_after = os.fstat(descriptor)
        path_after = os.lstat(path_bytes)

        if first != second:
            raise QuerySourceError(
                "query file changed between reads"
            )

        if (
            file_identity(descriptor_before)
            != file_identity(descriptor_after)
            or file_identity(path_before)
            != file_identity(path_after)
        ):
            raise QuerySourceError(
                "query file metadata changed during read"
            )

        if not first:
            raise QueryInputError("EMPTY_QUERY")

        return first

    finally:
        os.close(descriptor)


def run_process(
    command: list[str],
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
    )

    if result.returncode != 0:
        raise QueryRuntimeError(
            "command failed: "
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout[-4000:]
            + "\nstderr:\n"
            + result.stderr[-4000:]
        )

    return result


def ensure_query_binaries() -> None:
    run_process([
        "cmake",
        "-S",
        ".",
        "-B",
        "build",
    ])

    run_process([
        "cmake",
        "--build",
        "build",
        "--target",
        *QUERY_BINARY_NAMES,
        "-j2",
    ])

    for name in QUERY_BINARY_NAMES:
        path = BUILD / name

        if (
            not path.is_file()
            or not os.access(path, os.X_OK)
        ):
            raise QueryRuntimeError(
                f"query binary unavailable: {name}"
            )


def query_binary_commitments() -> list[dict[str, Any]]:
    records = []

    for name in QUERY_BINARY_NAMES:
        path = BUILD / name

        records.append({
            "name": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return sorted(
        records,
        key=lambda item: item["name"],
    )


def run_json_command(
    command: list[str],
) -> dict[str, Any]:
    result = run_process(command)

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise QueryRuntimeError(
            "runtime returned invalid JSON: "
            + result.stdout[-4000:]
        ) from error

    if not isinstance(value, dict):
        raise QueryRuntimeError(
            "runtime JSON root is not an object"
        )

    return value


def validate_interval(
    value: Any,
) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
    ):
        raise QueryVerificationError(
            "invalid FM interval"
        )

    left = require_uint(value[0], "interval left")
    right = require_uint(value[1], "interval right")

    if left > right:
        raise QueryVerificationError(
            "FM interval is reversed"
        )

    return left, right


def validate_count_result(
    value: dict[str, Any],
    query: bytes,
) -> dict[str, Any]:
    if set(value) != COUNT_KEYS:
        raise QueryVerificationError(
            "count result fields mismatch"
        )

    if value.get("ok") is not True:
        raise QueryVerificationError(
            "count result is not ok"
        )

    if value.get("format") != "GLYPH_QUERY_BINARY_V1":
        raise QueryVerificationError(
            "count format mismatch"
        )

    if value.get("query_hex") != query.hex():
        raise QueryVerificationError(
            "count query_hex mismatch"
        )

    if value.get("query_length_bytes") != len(query):
        raise QueryVerificationError(
            "count query length mismatch"
        )

    left, right = validate_interval(
        value.get("interval")
    )

    count = require_uint(
        value.get("count"),
        "count",
    )

    if count != right - left:
        raise QueryVerificationError(
            "count differs from FM interval"
        )

    if value.get("alphabet_size") != 257:
        raise QueryVerificationError(
            "count alphabet size mismatch"
        )

    if value.get("logical_sentinel") != 256:
        raise QueryVerificationError(
            "count logical sentinel mismatch"
        )

    return {
        "interval": [left, right],
        "count": count,
    }


def validate_locate_result(
    value: dict[str, Any],
    query: bytes,
    expected_count: dict[str, Any],
    local_limit: int | None,
) -> dict[str, Any]:
    expected_keys = set(LOCATE_KEYS)

    if local_limit is not None:
        expected_keys.add("max_offsets")

    if set(value) != expected_keys:
        raise QueryVerificationError(
            "locate result fields mismatch"
        )

    if value.get("ok") is not True:
        raise QueryVerificationError(
            "locate result is not ok"
        )

    constants = {
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

    for key, expected in constants.items():
        if value.get(key) != expected:
            raise QueryVerificationError(
                f"locate constant mismatch: {key}"
            )

    left, right = validate_interval(
        value.get("interval")
    )

    if [left, right] != expected_count["interval"]:
        raise QueryVerificationError(
            "count and locate intervals differ"
        )

    match_count = require_uint(
        value.get("match_count"),
        "locate match_count",
    )

    if match_count != expected_count["count"]:
        raise QueryVerificationError(
            "count and locate match counts differ"
        )

    offsets_raw = value.get("offsets")

    if not isinstance(offsets_raw, list):
        raise QueryVerificationError(
            "locate offsets must be a list"
        )

    offsets = [
        require_uint(offset, "located offset")
        for offset in offsets_raw
    ]

    if offsets != sorted(offsets):
        raise QueryVerificationError(
            "located offsets are not canonical"
        )

    if len(set(offsets)) != len(offsets):
        raise QueryVerificationError(
            "duplicate located offset"
        )

    returned_count = require_uint(
        value.get("returned_count"),
        "returned_count",
    )

    if returned_count != len(offsets):
        raise QueryVerificationError(
            "returned_count differs from offsets"
        )

    expected_returned = (
        match_count
        if local_limit is None
        else min(local_limit, match_count)
    )

    if returned_count != expected_returned:
        raise QueryVerificationError(
            "local bounded result size mismatch"
        )

    expected_bounded = (
        returned_count < match_count
    )

    if value.get("bounded") is not expected_bounded:
        raise QueryVerificationError(
            "local bounded flag mismatch"
        )

    if (
        value.get("offsets_complete")
        is not (not expected_bounded)
    ):
        raise QueryVerificationError(
            "local offsets_complete mismatch"
        )

    if local_limit is not None:
        if value.get("max_offsets") != local_limit:
            raise QueryVerificationError(
                "local max_offsets mismatch"
            )

    coordinates = value.get("coordinates")
    expected_coordinates = [
        [0, offset]
        for offset in offsets
    ]

    if coordinates != expected_coordinates:
        raise QueryVerificationError(
            "local coordinate representation mismatch"
        )

    return {
        "interval": [left, right],
        "match_count": match_count,
        "offsets": offsets,
        "returned_count": returned_count,
        "bounded": expected_bounded,
        "offsets_complete":
            not expected_bounded,
    }


def hash_regular_payload(
    path: Path,
    declared_length: int,
    declared_sha256: str,
) -> tuple[int, ...]:
    path_bytes = os.fsencode(os.fspath(path))

    try:
        before = os.lstat(path_bytes)
    except OSError as error:
        raise QuerySourceError(
            f"committed source unavailable: {error}"
        ) from error

    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
    ):
        raise QuerySourceError(
            "committed source must be regular"
        )

    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(path_bytes, flags)

    try:
        opened = os.fstat(descriptor)

        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise QuerySourceError(
                "committed source pathname changed"
            )

        digest = hashlib.sha256()
        total = 0

        while True:
            chunk = os.read(
                descriptor,
                1024 * 1024,
            )

            if not chunk:
                break

            digest.update(chunk)
            total += len(chunk)

        after_fd = os.fstat(descriptor)
        after_path = os.lstat(path_bytes)

        if (
            file_identity(before)
            != file_identity(after_path)
            or file_identity(opened)
            != file_identity(after_fd)
        ):
            raise QuerySourceError(
                "committed source changed while hashing"
            )

        if total != declared_length:
            raise QuerySourceError(
                "committed source length mismatch"
            )

        if digest.hexdigest() != declared_sha256:
            raise QuerySourceError(
                "committed source SHA256 mismatch"
            )

        return file_identity(after_path)

    finally:
        os.close(descriptor)


def byte_check_offsets(
    path: Path,
    query: bytes,
    offsets: list[int],
    declared_length: int,
    declared_sha256: str,
) -> None:
    path_bytes = os.fsencode(os.fspath(path))
    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(path_bytes, flags)

    try:
        status = os.fstat(descriptor)

        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_size != declared_length
        ):
            raise QueryVerificationError(
                "source changed before byte-check"
            )

        for offset in offsets:
            if (
                offset > declared_length
                or len(query)
                > declared_length - offset
            ):
                raise QueryVerificationError(
                    "located offset outside source"
                )

            if hasattr(os, "pread"):
                candidate = os.pread(
                    descriptor,
                    len(query),
                    offset,
                )
            else:
                os.lseek(
                    descriptor,
                    offset,
                    os.SEEK_SET,
                )
                candidate = os.read(
                    descriptor,
                    len(query),
                )

            if candidate != query:
                raise QueryVerificationError(
                    "independent coordinate "
                    "byte-check failed"
                )

    finally:
        os.close(descriptor)

    hash_regular_payload(
        path,
        declared_length,
        declared_sha256,
    )


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


def execute_operator_query(
    corpus_directory: Path,
    query: bytes,
    *,
    max_offsets: int | None = None,
    after_pre_source_check_hook: (
        Callable[[int, Path], None] | None
    ) = None,
) -> dict[str, Any]:
    if not query:
        raise QueryInputError("EMPTY_QUERY")

    if max_offsets is not None:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
            or max_offsets > (2**64 - 1)
        ):
            raise QueryInputError(
                "invalid max_offsets"
            )

    corpus_directory = corpus_directory.resolve()

    index_verification = verify_runtime_index(
        corpus_directory,
        require_current_binaries=True,
        rebuild=False,
    )

    source_manifest_path = (
        corpus_directory / SOURCE_MANIFEST_NAME
    )
    runtime_manifest_path = (
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )

    source_manifest = load_canonical_json(
        source_manifest_path
    )
    runtime_manifest = load_canonical_json(
        runtime_manifest_path
    )

    ensure_query_binaries()

    commitments_before = (
        query_binary_commitments()
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
            source_manifest["documents"],
            runtime_manifest["documents"],
        )
    ):
        if (
            source_record["doc_id"] != doc_id
            or runtime_record["doc_id"] != doc_id
        ):
            raise QueryVerificationError(
                "document alignment mismatch"
            )

        source_path = (
            corpus_directory
            / source_record["snapshot_path"]
        )

        identity_before = hash_regular_payload(
            source_path,
            source_record["byte_length"],
            source_record["sha256"],
        )

        if after_pre_source_check_hook is not None:
            after_pre_source_check_hook(
                doc_id,
                source_path,
            )

        fm_path = (
            corpus_directory
            / runtime_record["fm"]["path"]
        )
        bwt_path = (
            corpus_directory
            / runtime_record["bwt"]["path"]
        )
        sa_path = (
            corpus_directory
            / runtime_record["sa"]["path"]
        )

        count_raw = run_json_command([
            str(BUILD / "query_fm_binary_v1"),
            str(fm_path),
            str(bwt_path),
            query.hex(),
        ])

        count = validate_count_result(
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

        locate_command = [
            str(
                BUILD
                / "query_fm_locate_binary_v1"
            ),
            str(fm_path),
            str(bwt_path),
            str(sa_path),
            str(source_path),
            query.hex(),
        ]

        if local_limit is not None:
            locate_command.append(
                str(local_limit)
            )

        locate_raw = run_json_command(
            locate_command
        )

        locate = validate_locate_result(
            locate_raw,
            query,
            count,
            local_limit,
        )

        identity_after = hash_regular_payload(
            source_path,
            source_record["byte_length"],
            source_record["sha256"],
        )

        if identity_after != identity_before:
            raise QuerySourceError(
                "committed source changed during query"
            )

        byte_check_offsets(
            source_path,
            query,
            locate["offsets"],
            source_record["byte_length"],
            source_record["sha256"],
        )

        total_match_count += count["count"]

        document_result = {
            "doc_id": doc_id,
            "interval": count["interval"],
            "match_count": count["count"],
            "returned_count":
                locate["returned_count"],
        }

        document_results.append(
            document_result
        )

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

    numeric_coordinates = [
        tuple(item["coordinate"])
        for item in coordinates
    ]

    if numeric_coordinates != sorted(
        numeric_coordinates
    ):
        raise QueryVerificationError(
            "global coordinates are not canonical"
        )

    if len(set(numeric_coordinates)) != len(
        numeric_coordinates
    ):
        raise QueryVerificationError(
            "duplicate global coordinate"
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
        raise QueryVerificationError(
            "global bounded result size mismatch"
        )

    final_index_verification = (
        verify_runtime_index(
            corpus_directory,
            require_current_binaries=True,
            rebuild=False,
        )
    )

    if (
        final_index_verification[
            "runtime_index_id"
        ]
        != index_verification[
            "runtime_index_id"
        ]
    ):
        raise QueryVerificationError(
            "runtime index changed during query"
        )

    commitments_after = (
        query_binary_commitments()
    )

    if commitments_after != commitments_before:
        raise QueryVerificationError(
            "query binaries changed during query"
        )

    result = {
        "ok": True,
        "result_version": RESULT_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "index_topology":
            INDEX_TOPOLOGY,
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "corpus_id":
            index_verification["corpus_id"],
        "source_manifest_id":
            index_verification[
                "source_manifest_id"
            ],
        "runtime_index_id":
            index_verification[
                "runtime_index_id"
            ],
        "source_manifest_sha256":
            sha256_file(
                source_manifest_path
            ),
        "runtime_manifest_sha256":
            sha256_file(
                runtime_manifest_path
            ),
        "query_binary_commitments":
            commitments_before,
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256":
            sha256_bytes(query),
        "max_offsets": max_offsets,
        "document_count":
            source_manifest["document_count"],
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

    return result


def command_query(
    args: argparse.Namespace,
) -> int:
    if args.query_file is not None:
        query = read_stable_query_file(
            Path(args.query_file)
        )
    else:
        query = parse_query_hex(
            args.query_hex
        )

    result = execute_operator_query(
        Path(args.corpus),
        query,
        max_offsets=args.max_offsets,
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

    query = subparsers.add_parser("query")
    query.add_argument(
        "--corpus",
        required=True,
    )

    source = query.add_mutually_exclusive_group(
        required=True
    )
    source.add_argument("--query-file")
    source.add_argument("--query-hex")

    query.add_argument(
        "--max-offsets",
        type=int,
    )

    query.set_defaults(handler=command_query)

    return parser


def main() -> int:
    parser = build_parser()

    try:
        args = parser.parse_args()
        return args.handler(args)

    except (
        QueryError,
        OperatorError,
        IndexErrorV1,
    ) as error:
        failure = {
            "ok": False,
            "error_code": getattr(
                error,
                "error_code",
                "QUERY_FAILURE",
            ),
            "error": str(error),
        }

        print(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
            )
        )

        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        return getattr(
            error,
            "exit_code",
            7,
        )

    except Exception as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code":
                        "INTERNAL_INVARIANT",
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )

        print(
            f"ERROR: {error}",
            file=sys.stderr,
        )

        return 7


if __name__ == "__main__":
    raise SystemExit(main())
