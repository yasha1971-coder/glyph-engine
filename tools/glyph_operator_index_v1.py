#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_binary_runtime_evidence_v1 import (  # noqa: E402
    BUILD,
    ensure_binaries,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    load_canonical_json,
    sha256_file,
    verify_snapshot,
)

INDEX_MANIFEST_VERSION = (
    "GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1"
)
INDEX_COMPLETE_VERSION = (
    "GLYPH_OPERATOR_RUNTIME_INDEX_COMPLETE_V1"
)
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"
INDEX_TOPOLOGY = "one_independent_index_per_document"

INDEX_MANIFEST_NAME = "runtime_manifest_v1.json"
INDEX_COMPLETE_NAME = "RUNTIME_BUILD_COMPLETE_V1.json"
INDEX_DOCUMENTS_DIRECTORY = "documents"

LOGICAL_SENTINEL = 256
ALPHABET_SIZE = 257
CHECKPOINT_STEP = 32

RUNTIME_BINARY_NAMES = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
]

INDEX_FORMATS = {
    "sa": "GLYPH_SA_BINARY_V1",
    "bwt": "GLYPH_BWT_BINARY_V1",
    "fm": "GLYPH_FM_BINARY_V1",
}

MANIFEST_KEYS = {
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

BINARY_RECORD_KEYS = {
    "name",
    "size_bytes",
    "sha256",
}

DOCUMENT_RECORD_KEYS = {
    "doc_id",
    "source_snapshot_path",
    "source_byte_length",
    "source_sha256",
    "index_directory",
    "sa",
    "bwt",
    "fm",
}

ARTIFACT_RECORD_KEYS = {
    "path",
    "format",
    "size_bytes",
    "sha256",
}

COMPLETE_KEYS = {
    "complete_version",
    "runtime_manifest_sha256",
    "runtime_index_id",
    "corpus_id",
    "source_manifest_id",
    "document_count",
    "total_runtime_bytes",
}


class IndexErrorV1(RuntimeError):
    exit_code = 7
    error_code = "INTERNAL_INVARIANT"


class IndexInputError(IndexErrorV1):
    exit_code = 2
    error_code = "INVALID_INPUT"


class IndexSourceError(IndexErrorV1):
    exit_code = 3
    error_code = "SOURCE_FAILURE"


class IndexBuildError(IndexErrorV1):
    exit_code = 4
    error_code = "RUNTIME_CONSTRUCTION_FAILURE"


class IndexVerificationError(IndexErrorV1):
    exit_code = 6
    error_code = "RUNTIME_INDEX_VERIFICATION_FAILURE"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise IndexVerificationError(
            f"invalid SHA256 field: {field}"
        )

    return value


def require_uint(
    value: Any,
    field: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise IndexVerificationError(
            f"invalid unsigned integer: {field}"
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


def run_command(command: list[str]) -> None:
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
        raise IndexBuildError(
            "runtime command failed: "
            + " ".join(command)
            + "\nstdout:\n"
            + result.stdout[-4000:]
            + "\nstderr:\n"
            + result.stderr[-4000:]
        )


def runtime_binary_commitments() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for name in RUNTIME_BINARY_NAMES:
        path = BUILD / name

        if not path.is_file():
            raise IndexBuildError(
                f"runtime builder missing: {name}"
            )

        records.append({
            "name": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    return sorted(
        records,
        key=lambda item: item["name"],
    )


def artifact_record(
    physical_path: Path,
    manifest_path: str,
    role: str,
) -> dict[str, Any]:
    return {
        "path": manifest_path,
        "format": INDEX_FORMATS[role],
        "size_bytes": physical_path.stat().st_size,
        "sha256": sha256_file(physical_path),
    }


def runtime_index_identity(
    manifest: dict[str, Any],
) -> str:
    digest = hashlib.sha256()

    digest.update(
        b"GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1\x00"
    )

    digest.update(
        bytes.fromhex(
            validate_sha256(
                manifest["source_manifest_sha256"],
                "source_manifest_sha256",
            )
        )
    )
    digest.update(
        bytes.fromhex(
            validate_sha256(
                manifest["corpus_id"],
                "corpus_id",
            )
        )
    )
    digest.update(
        bytes.fromhex(
            validate_sha256(
                manifest["source_manifest_id"],
                "source_manifest_id",
            )
        )
    )

    topology = manifest["index_topology"].encode(
        "ascii"
    )

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

    binaries = sorted(
        manifest["runtime_binaries"],
        key=lambda item: item["name"],
    )

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
            document["doc_id"].to_bytes(
                8,
                "big",
            )
        )
        digest.update(
            document[
                "source_byte_length"
            ].to_bytes(
                8,
                "big",
            )
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


def make_complete_marker(
    manifest: dict[str, Any],
    runtime_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "complete_version":
            INDEX_COMPLETE_VERSION,
        "runtime_manifest_sha256":
            runtime_manifest_sha256,
        "runtime_index_id":
            manifest["runtime_index_id"],
        "corpus_id": manifest["corpus_id"],
        "source_manifest_id":
            manifest["source_manifest_id"],
        "document_count":
            manifest["document_count"],
        "total_runtime_bytes":
            manifest["total_runtime_bytes"],
    }


def copy_verified_source(
    source_path: Path,
    destination: Path,
    declared_length: int,
    declared_sha256: str,
) -> None:
    source_bytes = os.fsencode(
        os.fspath(source_path)
    )

    try:
        path_before = os.lstat(source_bytes)
    except OSError as error:
        raise IndexSourceError(
            f"committed source unavailable: {error}"
        ) from error

    if (
        stat.S_ISLNK(path_before.st_mode)
        or not stat.S_ISREG(
            path_before.st_mode
        )
    ):
        raise IndexSourceError(
            "committed source payload must be regular"
        )

    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(
            source_bytes,
            flags,
        )
    except OSError as error:
        raise IndexSourceError(
            f"cannot open committed source: {error}"
        ) from error

    try:
        descriptor_before = os.fstat(
            descriptor
        )

        if not stat.S_ISREG(
            descriptor_before.st_mode
        ):
            raise IndexSourceError(
                "opened committed payload is not regular"
            )

        if (
            descriptor_before.st_dev
            != path_before.st_dev
            or descriptor_before.st_ino
            != path_before.st_ino
        ):
            raise IndexSourceError(
                "committed source pathname changed"
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        digest = hashlib.sha256()
        copied = 0

        with destination.open("xb") as output:
            while True:
                chunk = os.read(
                    descriptor,
                    1024 * 1024,
                )

                if not chunk:
                    break

                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)

            output.flush()
            os.fsync(output.fileno())

        descriptor_after = os.fstat(
            descriptor
        )
        path_after = os.lstat(
            source_bytes
        )

        if (
            file_identity(descriptor_before)
            != file_identity(descriptor_after)
        ):
            raise IndexSourceError(
                "committed source changed during copy"
            )

        if (
            file_identity(path_before)
            != file_identity(path_after)
        ):
            raise IndexSourceError(
                "committed source pathname changed "
                "during copy"
            )

        actual_hash = digest.hexdigest()

        if copied != declared_length:
            raise IndexSourceError(
                "committed source length mismatch"
            )

        if actual_hash != declared_sha256:
            raise IndexSourceError(
                "committed source SHA256 mismatch"
            )

    finally:
        os.close(descriptor)


def build_document_index(
    private_source: Path,
    output_directory: Path,
) -> dict[str, Path]:
    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    paths = {
        "sa": output_directory / "sa.bin",
        "bwt": output_directory / "bwt.bin",
        "fm": output_directory / "fm.bin",
    }

    run_command([
        str(BUILD / "build_sa_binary_v1"),
        str(private_source),
        str(paths["sa"]),
    ])

    run_command([
        str(BUILD / "build_bwt_binary_v1"),
        str(private_source),
        str(paths["sa"]),
        str(paths["bwt"]),
    ])

    run_command([
        str(BUILD / "build_fm_binary_v1"),
        str(paths["bwt"]),
        str(paths["fm"]),
        str(CHECKPOINT_STEP),
    ])

    for role, path in paths.items():
        if (
            not path.is_file()
            or path.is_symlink()
        ):
            raise IndexBuildError(
                f"runtime artifact missing: {role}"
            )

    return paths


def build_runtime_index(
    corpus_directory: Path,
    *,
    after_private_copy_hook: (
        Callable[[int, Path], None] | None
    ) = None,
    test_fail_after_documents: int | None = None,
) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()

    source_verification = verify_snapshot(
        corpus_directory
    )

    source_manifest_path = (
        corpus_directory
        / SOURCE_MANIFEST_NAME
    )
    source_manifest = load_canonical_json(
        source_manifest_path
    )

    final_index_directory = (
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY
    )

    if final_index_directory.exists():
        raise IndexInputError(
            "runtime_index_v1 already exists"
        )

    ensure_binaries()

    binary_commitments = (
        runtime_binary_commitments()
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{corpus_directory.name}."
                "runtime_index_v1.tmp."
            ),
            dir=corpus_directory.parent,
        )
    )

    published = False

    try:
        runtime_documents_directory = (
            temporary_directory
            / INDEX_DOCUMENTS_DIRECTORY
        )
        runtime_documents_directory.mkdir()

        private_inputs = (
            temporary_directory
            / ".private_inputs"
        )
        private_inputs.mkdir()

        records: list[dict[str, Any]] = []

        source_documents = source_manifest[
            "documents"
        ]

        for expected_doc_id, source_record in enumerate(
            source_documents
        ):
            if source_record["doc_id"] != expected_doc_id:
                raise IndexSourceError(
                    "O1 document IDs are not canonical"
                )

            source_snapshot_path = (
                corpus_directory
                / source_record["snapshot_path"]
            )

            private_before = (
                private_inputs
                / f"doc_{expected_doc_id:08d}.before.bin"
            )

            copy_verified_source(
                source_snapshot_path,
                private_before,
                source_record["byte_length"],
                source_record["sha256"],
            )

            if after_private_copy_hook is not None:
                after_private_copy_hook(
                    expected_doc_id,
                    source_snapshot_path,
                )

            physical_index_directory = (
                runtime_documents_directory
                / f"doc_{expected_doc_id:08d}"
            )

            paths = build_document_index(
                private_before,
                physical_index_directory,
            )

            private_after = (
                private_inputs
                / f"doc_{expected_doc_id:08d}.after.bin"
            )

            copy_verified_source(
                source_snapshot_path,
                private_after,
                source_record["byte_length"],
                source_record["sha256"],
            )

            if (
                private_before.read_bytes()
                != private_after.read_bytes()
            ):
                raise IndexSourceError(
                    "committed source changed "
                    "during runtime construction"
                )

            private_before.unlink()
            private_after.unlink()

            manifest_index_directory = (
                f"{RUNTIME_INDEX_DIRECTORY}/"
                f"documents/"
                f"doc_{expected_doc_id:08d}"
            )

            record = {
                "doc_id": expected_doc_id,
                "source_snapshot_path":
                    source_record["snapshot_path"],
                "source_byte_length":
                    source_record["byte_length"],
                "source_sha256":
                    source_record["sha256"],
                "index_directory":
                    manifest_index_directory,
                "sa": artifact_record(
                    paths["sa"],
                    manifest_index_directory
                    + "/sa.bin",
                    "sa",
                ),
                "bwt": artifact_record(
                    paths["bwt"],
                    manifest_index_directory
                    + "/bwt.bin",
                    "bwt",
                ),
                "fm": artifact_record(
                    paths["fm"],
                    manifest_index_directory
                    + "/fm.bin",
                    "fm",
                ),
            }

            records.append(record)

            if (
                test_fail_after_documents
                is not None
                and len(records)
                >= test_fail_after_documents
            ):
                raise IndexBuildError(
                    "simulated interrupted "
                    "runtime-index build"
                )

        private_inputs.rmdir()

        verify_snapshot(corpus_directory)

        total_runtime_bytes = sum(
            document[role]["size_bytes"]
            for document in records
            for role in ("sa", "bwt", "fm")
        )

        manifest = {
            "manifest_version":
                INDEX_MANIFEST_VERSION,
            "runtime_profile":
                RUNTIME_PROFILE,
            "construction_status": "complete",
            "index_topology":
                INDEX_TOPOLOGY,
            "logical_sentinel":
                LOGICAL_SENTINEL,
            "alphabet_size": ALPHABET_SIZE,
            "checkpoint_step":
                CHECKPOINT_STEP,
            "source_manifest_name":
                SOURCE_MANIFEST_NAME,
            "source_manifest_sha256":
                sha256_file(
                    source_manifest_path
                ),
            "corpus_id":
                source_verification["corpus_id"],
            "source_manifest_id":
                source_verification[
                    "source_manifest_id"
                ],
            "document_count":
                source_verification[
                    "document_count"
                ],
            "total_source_bytes":
                source_verification[
                    "total_source_bytes"
                ],
            "total_runtime_bytes":
                total_runtime_bytes,
            "runtime_binaries":
                binary_commitments,
            "documents": records,
            "runtime_index_id": "",
        }

        manifest["runtime_index_id"] = (
            runtime_index_identity(manifest)
        )

        runtime_manifest_path = (
            temporary_directory
            / INDEX_MANIFEST_NAME
        )

        runtime_manifest_path.write_bytes(
            canonical_json_bytes(manifest)
        )
        fsync_file(runtime_manifest_path)

        complete = make_complete_marker(
            manifest,
            sha256_file(runtime_manifest_path),
        )

        complete_path = (
            temporary_directory
            / INDEX_COMPLETE_NAME
        )

        complete_path.write_bytes(
            canonical_json_bytes(complete)
        )
        fsync_file(complete_path)

        for document in records:
            for role in ("sa", "bwt", "fm"):
                fsync_file(
                    corpus_directory.parent
                    / temporary_directory.name
                    / Path(
                        document[role]["path"]
                    ).relative_to(
                        RUNTIME_INDEX_DIRECTORY
                    )
                )

        for entry in runtime_documents_directory.iterdir():
            fsync_directory(entry)

        fsync_directory(
            runtime_documents_directory
        )
        fsync_directory(
            temporary_directory
        )

        verify_runtime_index_directory(
            corpus_directory,
            temporary_directory,
            require_current_binaries=True,
            rebuild=False,
        )

        if final_index_directory.exists():
            raise IndexBuildError(
                "runtime_index_v1 appeared "
                "during construction"
            )

        os.rename(
            temporary_directory,
            final_index_directory,
        )
        published = True

        fsync_directory(corpus_directory)

        result = verify_runtime_index(
            corpus_directory,
            require_current_binaries=True,
            rebuild=False,
        )

        return {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_RUNTIME_INDEX_BUILD_V1",
            "operator_obligation": "O2",
            "corpus_directory":
                str(corpus_directory),
            **{
                key: value
                for key, value in result.items()
                if key != "ok"
            },
        }

    except Exception:
        if not published:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )

        raise


def validate_binary_records(
    value: Any,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise IndexVerificationError(
            "runtime_binaries must be a list"
        )

    if len(value) != len(
        RUNTIME_BINARY_NAMES
    ):
        raise IndexVerificationError(
            "runtime binary count mismatch"
        )

    records: list[dict[str, Any]] = []

    for record in value:
        if (
            not isinstance(record, dict)
            or set(record) != BINARY_RECORD_KEYS
        ):
            raise IndexVerificationError(
                "runtime binary record fields mismatch"
            )

        name = record["name"]

        if name not in RUNTIME_BINARY_NAMES:
            raise IndexVerificationError(
                "unknown runtime binary commitment"
            )

        require_uint(
            record["size_bytes"],
            f"runtime binary {name} size",
        )
        validate_sha256(
            record["sha256"],
            f"runtime binary {name}",
        )

        records.append(record)

    if [
        item["name"]
        for item in records
    ] != sorted(RUNTIME_BINARY_NAMES):
        raise IndexVerificationError(
            "runtime binaries not canonically ordered"
        )

    return records


def verify_regular_payload(
    path: Path,
    declared_size: int,
    declared_sha256: str,
) -> None:
    try:
        status = os.lstat(path)
    except OSError as error:
        raise IndexVerificationError(
            f"runtime payload missing: {path.name}"
        ) from error

    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
    ):
        raise IndexVerificationError(
            "runtime payload must be a regular file"
        )

    if status.st_size != declared_size:
        raise IndexVerificationError(
            "runtime payload size mismatch"
        )

    if sha256_file(path) != declared_sha256:
        raise IndexVerificationError(
            "runtime payload SHA256 mismatch"
        )


def validate_artifact_record(
    value: Any,
    expected_path: str,
    role: str,
    physical_path: Path,
) -> int:
    if (
        not isinstance(value, dict)
        or set(value) != ARTIFACT_RECORD_KEYS
    ):
        raise IndexVerificationError(
            f"{role} record fields mismatch"
        )

    if value["path"] != expected_path:
        raise IndexVerificationError(
            f"{role} path mismatch"
        )

    if value["format"] != INDEX_FORMATS[role]:
        raise IndexVerificationError(
            f"{role} format mismatch"
        )

    size = require_uint(
        value["size_bytes"],
        f"{role} size",
    )
    declared_hash = validate_sha256(
        value["sha256"],
        f"{role} SHA256",
    )

    verify_regular_payload(
        physical_path,
        size,
        declared_hash,
    )

    return size


def verify_runtime_index_directory(
    corpus_directory: Path,
    index_directory: Path,
    *,
    require_current_binaries: bool,
    rebuild: bool,
) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()
    index_directory = index_directory.resolve()

    source_verification = verify_snapshot(
        corpus_directory
    )

    source_manifest_path = (
        corpus_directory
        / SOURCE_MANIFEST_NAME
    )
    source_manifest = load_canonical_json(
        source_manifest_path
    )

    try:
        index_status = os.lstat(
            index_directory
        )
    except OSError as error:
        raise IndexVerificationError(
            "runtime index directory missing"
        ) from error

    if (
        stat.S_ISLNK(index_status.st_mode)
        or not stat.S_ISDIR(
            index_status.st_mode
        )
    ):
        raise IndexVerificationError(
            "runtime index must be a real directory"
        )

    expected_root = {
        INDEX_MANIFEST_NAME,
        INDEX_COMPLETE_NAME,
        INDEX_DOCUMENTS_DIRECTORY,
    }

    actual_root = {
        entry.name
        for entry in os.scandir(
            index_directory
        )
    }

    if actual_root != expected_root:
        raise IndexVerificationError(
            "runtime index root coverage mismatch"
        )

    runtime_manifest_path = (
        index_directory
        / INDEX_MANIFEST_NAME
    )
    complete_path = (
        index_directory
        / INDEX_COMPLETE_NAME
    )
    documents_directory = (
        index_directory
        / INDEX_DOCUMENTS_DIRECTORY
    )

    for path in (
        runtime_manifest_path,
        complete_path,
        documents_directory,
    ):
        status = os.lstat(path)

        if stat.S_ISLNK(status.st_mode):
            raise IndexVerificationError(
                f"runtime symlink forbidden: {path.name}"
            )

    if not stat.S_ISDIR(
        os.lstat(documents_directory).st_mode
    ):
        raise IndexVerificationError(
            "runtime documents payload "
            "must be a directory"
        )

    manifest = load_canonical_json(
        runtime_manifest_path
    )
    complete = load_canonical_json(
        complete_path
    )

    if set(manifest) != MANIFEST_KEYS:
        raise IndexVerificationError(
            "runtime manifest fields mismatch"
        )

    if set(complete) != COMPLETE_KEYS:
        raise IndexVerificationError(
            "runtime complete marker fields mismatch"
        )

    constants = {
        "manifest_version":
            INDEX_MANIFEST_VERSION,
        "runtime_profile":
            RUNTIME_PROFILE,
        "construction_status": "complete",
        "index_topology":
            INDEX_TOPOLOGY,
        "logical_sentinel":
            LOGICAL_SENTINEL,
        "alphabet_size": ALPHABET_SIZE,
        "checkpoint_step":
            CHECKPOINT_STEP,
        "source_manifest_name":
            SOURCE_MANIFEST_NAME,
    }

    for key, expected in constants.items():
        if manifest.get(key) != expected:
            raise IndexVerificationError(
                f"runtime manifest constant "
                f"mismatch: {key}"
            )

    if (
        manifest["source_manifest_sha256"]
        != sha256_file(source_manifest_path)
    ):
        raise IndexVerificationError(
            "source manifest SHA256 binding mismatch"
        )

    if (
        manifest["corpus_id"]
        != source_verification["corpus_id"]
    ):
        raise IndexVerificationError(
            "runtime corpus ID binding mismatch"
        )

    if (
        manifest["source_manifest_id"]
        != source_verification[
            "source_manifest_id"
        ]
    ):
        raise IndexVerificationError(
            "source manifest ID binding mismatch"
        )

    if (
        manifest["document_count"]
        != source_verification[
            "document_count"
        ]
    ):
        raise IndexVerificationError(
            "runtime document_count mismatch"
        )

    if (
        manifest["total_source_bytes"]
        != source_verification[
            "total_source_bytes"
        ]
    ):
        raise IndexVerificationError(
            "runtime total_source_bytes mismatch"
        )

    binary_records = validate_binary_records(
        manifest["runtime_binaries"]
    )

    if require_current_binaries or rebuild:
        ensure_binaries()

        current = runtime_binary_commitments()

        if binary_records != current:
            raise IndexVerificationError(
                "runtime builder commitments mismatch"
            )

    documents = manifest["documents"]

    if (
        not isinstance(documents, list)
        or len(documents)
        != manifest["document_count"]
    ):
        raise IndexVerificationError(
            "runtime documents list mismatch"
        )

    expected_document_directories: set[str] = set()
    total_runtime_bytes = 0

    for expected_doc_id, document in enumerate(
        documents
    ):
        if (
            not isinstance(document, dict)
            or set(document)
            != DOCUMENT_RECORD_KEYS
        ):
            raise IndexVerificationError(
                "runtime document record "
                "fields mismatch"
            )

        if document["doc_id"] != expected_doc_id:
            raise IndexVerificationError(
                "runtime document IDs "
                "are not canonical"
            )

        source_record = source_manifest[
            "documents"
        ][expected_doc_id]

        if (
            document["source_snapshot_path"]
            != source_record["snapshot_path"]
        ):
            raise IndexVerificationError(
                "runtime source snapshot path mismatch"
            )

        if (
            document["source_byte_length"]
            != source_record["byte_length"]
        ):
            raise IndexVerificationError(
                "runtime source byte length mismatch"
            )

        if (
            document["source_sha256"]
            != source_record["sha256"]
        ):
            raise IndexVerificationError(
                "runtime source SHA256 mismatch"
            )

        expected_index_directory = (
            f"{RUNTIME_INDEX_DIRECTORY}/"
            f"documents/"
            f"doc_{expected_doc_id:08d}"
        )

        if (
            document["index_directory"]
            != expected_index_directory
        ):
            raise IndexVerificationError(
                "runtime index directory mismatch"
            )

        directory_name = (
            f"doc_{expected_doc_id:08d}"
        )
        expected_document_directories.add(
            directory_name
        )

        physical_directory = (
            documents_directory
            / directory_name
        )

        try:
            physical_status = os.lstat(
                physical_directory
            )
        except OSError as error:
            raise IndexVerificationError(
                "runtime document index missing"
            ) from error

        if (
            stat.S_ISLNK(physical_status.st_mode)
            or not stat.S_ISDIR(
                physical_status.st_mode
            )
        ):
            raise IndexVerificationError(
                "runtime document index "
                "must be a real directory"
            )

        actual_artifacts = {
            entry.name
            for entry in os.scandir(
                physical_directory
            )
        }

        if actual_artifacts != {
            "sa.bin",
            "bwt.bin",
            "fm.bin",
        }:
            raise IndexVerificationError(
                "runtime document artifact "
                "coverage mismatch"
            )

        for role in ("sa", "bwt", "fm"):
            expected_path = (
                expected_index_directory
                + f"/{role}.bin"
            )

            physical_artifact_path = (
                physical_directory
                / f"{role}.bin"
            )

            total_runtime_bytes += (
                validate_artifact_record(
                    document[role],
                    expected_path,
                    role,
                    physical_artifact_path,
                )
            )

    actual_document_directories: set[str] = set()

    for entry in os.scandir(
        documents_directory
    ):
        status = entry.stat(
            follow_symlinks=False
        )

        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISDIR(
                status.st_mode
            )
        ):
            raise IndexVerificationError(
                "unexpected runtime document payload"
            )

        actual_document_directories.add(
            entry.name
        )

    if (
        actual_document_directories
        != expected_document_directories
    ):
        raise IndexVerificationError(
            "runtime document directory "
            "coverage mismatch"
        )

    if (
        manifest["total_runtime_bytes"]
        != total_runtime_bytes
    ):
        raise IndexVerificationError(
            "total_runtime_bytes mismatch"
        )

    expected_runtime_index_id = (
        runtime_index_identity(manifest)
    )

    if (
        manifest["runtime_index_id"]
        != expected_runtime_index_id
    ):
        raise IndexVerificationError(
            "runtime index identity mismatch"
        )

    expected_complete = make_complete_marker(
        manifest,
        sha256_file(runtime_manifest_path),
    )

    if complete != expected_complete:
        raise IndexVerificationError(
            "runtime complete marker mismatch"
        )

    if rebuild:
        with tempfile.TemporaryDirectory(
            prefix="glyph-operator-o2-rebuild-"
        ) as temporary:
            work = Path(temporary)

            for expected_doc_id, document in enumerate(
                documents
            ):
                source_path = (
                    corpus_directory
                    / document[
                        "source_snapshot_path"
                    ]
                )

                private_source = (
                    work
                    / f"doc_{expected_doc_id:08d}.bin"
                )

                copy_verified_source(
                    source_path,
                    private_source,
                    document[
                        "source_byte_length"
                    ],
                    document[
                        "source_sha256"
                    ],
                )

                rebuilt_directory = (
                    work
                    / f"index_{expected_doc_id:08d}"
                )

                rebuilt_paths = (
                    build_document_index(
                        private_source,
                        rebuilt_directory,
                    )
                )

                for role in ("sa", "bwt", "fm"):
                    rebuilt = {
                        "format":
                            INDEX_FORMATS[role],
                        "size_bytes":
                            rebuilt_paths[
                                role
                            ].stat().st_size,
                        "sha256":
                            sha256_file(
                                rebuilt_paths[
                                    role
                                ]
                            ),
                    }

                    committed = {
                        "format":
                            document[role][
                                "format"
                            ],
                        "size_bytes":
                            document[role][
                                "size_bytes"
                            ],
                        "sha256":
                            document[role][
                                "sha256"
                            ],
                    }

                    if rebuilt != committed:
                        raise IndexVerificationError(
                            "deterministic rebuild "
                            f"mismatch: doc={expected_doc_id} "
                            f"role={role}"
                        )

    return {
        "ok": True,
        "format":
            "GLYPH_OPERATOR_RUNTIME_INDEX_VERIFY_V1",
        "operator_obligation": "O2",
        "runtime_profile": RUNTIME_PROFILE,
        "manifest_version":
            INDEX_MANIFEST_VERSION,
        "index_topology":
            INDEX_TOPOLOGY,
        "logical_sentinel":
            LOGICAL_SENTINEL,
        "alphabet_size": ALPHABET_SIZE,
        "checkpoint_step":
            CHECKPOINT_STEP,
        "document_count":
            manifest["document_count"],
        "total_source_bytes":
            manifest["total_source_bytes"],
        "total_runtime_bytes":
            manifest["total_runtime_bytes"],
        "corpus_id":
            manifest["corpus_id"],
        "source_manifest_id":
            manifest["source_manifest_id"],
        "runtime_index_id":
            manifest["runtime_index_id"],
        "runtime_manifest_sha256":
            sha256_file(runtime_manifest_path),
        "runtime_binary_commitments_verified":
            require_current_binaries or rebuild,
        "deterministic_rebuild_verified":
            rebuild,
        "construction_status": "complete",
        "original_source_directory_required":
            False,
    }


def verify_runtime_index(
    corpus_directory: Path,
    *,
    require_current_binaries: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()

    return verify_runtime_index_directory(
        corpus_directory,
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY,
        require_current_binaries=
            require_current_binaries,
        rebuild=rebuild,
    )


def command_build(
    args: argparse.Namespace,
) -> int:
    result = build_runtime_index(
        Path(args.corpus)
    )

    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def command_verify(
    args: argparse.Namespace,
) -> int:
    result = verify_runtime_index(
        Path(args.corpus),
        require_current_binaries=
            args.require_current_binaries,
        rebuild=args.rebuild,
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

    build = subparsers.add_parser("build")
    build.add_argument(
        "--corpus",
        required=True,
    )
    build.set_defaults(handler=command_build)

    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--corpus",
        required=True,
    )
    verify.add_argument(
        "--require-current-binaries",
        action="store_true",
    )
    verify.add_argument(
        "--rebuild",
        action="store_true",
    )
    verify.set_defaults(handler=command_verify)

    return parser


def main() -> int:
    parser = build_parser()

    try:
        args = parser.parse_args()
        return args.handler(args)

    except IndexErrorV1 as error:
        failure = {
            "ok": False,
            "error_code": error.error_code,
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

        return error.exit_code

    except Exception as error:
        failure = {
            "ok": False,
            "error_code":
                "INTERNAL_INVARIANT",
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

        return 7


if __name__ == "__main__":
    raise SystemExit(main())
