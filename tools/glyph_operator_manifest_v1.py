#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

MANIFEST_VERSION = "GLYPH_OPERATOR_CORPUS_MANIFEST_V1"
CORPUS_IDENTITY_VERSION = (
    "GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1"
)
SOURCE_IDENTITY_VERSION = (
    "GLYPH_OPERATOR_CORPUS_MANIFEST_V1"
)
COMPLETE_VERSION = "GLYPH_OPERATOR_BUILD_COMPLETE_V1"
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"

MANIFEST_NAME = "source_manifest_v1.json"
COMPLETE_NAME = "BUILD_COMPLETE_V1.json"
DOCUMENTS_DIRECTORY = "documents"
RUNTIME_INDEX_DIRECTORY = "runtime_index_v1"

MANIFEST_KEYS = {
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

DOCUMENT_KEYS = {
    "doc_id",
    "relative_path_bytes_hex",
    "display_path",
    "document_type",
    "byte_length",
    "sha256",
    "snapshot_path",
}

COMPLETE_KEYS = {
    "complete_version",
    "manifest_sha256",
    "corpus_id",
    "source_manifest_id",
    "document_count",
}


class OperatorError(RuntimeError):
    exit_code = 7
    error_code = "INTERNAL_INVARIANT"


class OperatorInputError(OperatorError):
    exit_code = 2
    error_code = "INVALID_INPUT"


class SourceError(OperatorError):
    exit_code = 3
    error_code = "SOURCE_FAILURE"


class ConstructionError(OperatorError):
    exit_code = 4
    error_code = "CONSTRUCTION_FAILURE"


class VerificationError(OperatorError):
    exit_code = 6
    error_code = "VERIFICATION_FAILURE"


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


def fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY

    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    descriptor = os.open(path, flags)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_relative_path_bytes(raw: bytes) -> None:
    if not raw:
        raise VerificationError(
            "relative source path is empty"
        )

    if raw.startswith(b"/"):
        raise VerificationError(
            "absolute source path forbidden"
        )

    if b"\x00" in raw:
        raise VerificationError(
            "NUL in source pathname forbidden"
        )

    components = raw.split(b"/")

    if any(
        component in (b"", b".", b"..")
        for component in components
    ):
        raise VerificationError(
            "unsafe source pathname component"
        )


def decode_path_hex(value: Any) -> bytes:
    if not isinstance(value, str) or value == "":
        raise VerificationError(
            "relative_path_bytes_hex must be a string"
        )

    if value != value.lower():
        raise VerificationError(
            "relative_path_bytes_hex must be lowercase"
        )

    if len(value) % 2 != 0:
        raise VerificationError(
            "relative_path_bytes_hex has odd length"
        )

    if any(
        character not in "0123456789abcdef"
        for character in value
    ):
        raise VerificationError(
            "relative_path_bytes_hex is invalid"
        )

    raw = bytes.fromhex(value)

    if raw.hex() != value:
        raise VerificationError(
            "relative_path_bytes_hex is not canonical"
        )

    validate_relative_path_bytes(raw)

    return raw


def display_path(raw: bytes) -> str:
    return raw.decode(
        "utf-8",
        errors="backslashreplace",
    )


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
        doc_id = document["doc_id"]
        raw_path = bytes.fromhex(
            document["relative_path_bytes_hex"]
        )
        byte_length = document["byte_length"]
        document_hash = bytes.fromhex(
            document["sha256"]
        )

        digest.update(doc_id.to_bytes(8, "big"))
        digest.update(
            len(raw_path).to_bytes(8, "big")
        )
        digest.update(raw_path)
        digest.update(
            byte_length.to_bytes(8, "big")
        )
        digest.update(document_hash)

    return digest.hexdigest()


def runtime_corpus_identity(
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


def root_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_mtime_ns,
    )


def file_identity(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        stat.S_IFMT(status.st_mode),
        status.st_size,
        status.st_mtime_ns,
    )


def discover_regular_files(
    root_bytes: bytes,
) -> list[bytes]:
    try:
        root_status = os.lstat(root_bytes)
    except OSError as error:
        raise SourceError(
            f"cannot stat source root: {error}"
        ) from error

    if stat.S_ISLNK(root_status.st_mode):
        raise SourceError(
            "source root must not be a symlink"
        )

    if not stat.S_ISDIR(root_status.st_mode):
        raise SourceError(
            "source root must be a directory"
        )

    directories: list[bytes] = [b""]
    files: list[bytes] = []

    while directories:
        relative_directory = directories.pop()

        absolute_directory = (
            root_bytes
            if relative_directory == b""
            else os.path.join(
                root_bytes,
                relative_directory,
            )
        )

        try:
            with os.scandir(
                absolute_directory
            ) as iterator:
                entries = list(iterator)
        except OSError as error:
            raise SourceError(
                f"cannot scan source directory: {error}"
            ) from error

        entries.sort(key=lambda entry: entry.name)

        child_directories: list[bytes] = []

        for entry in entries:
            name = entry.name

            if not isinstance(name, bytes):
                raise SourceError(
                    "filesystem did not return raw pathname bytes"
                )

            relative = (
                name
                if relative_directory == b""
                else relative_directory
                + b"/"
                + name
            )

            validate_relative_path_bytes(relative)

            try:
                status = entry.stat(
                    follow_symlinks=False
                )
            except OSError as error:
                raise SourceError(
                    f"cannot stat source entry "
                    f"{relative.hex()}: {error}"
                ) from error

            mode = status.st_mode

            if stat.S_ISLNK(mode):
                raise SourceError(
                    "symbolic link rejected: "
                    + relative.hex()
                )

            if stat.S_ISDIR(mode):
                child_directories.append(relative)
                continue

            if stat.S_ISREG(mode):
                files.append(relative)
                continue

            raise SourceError(
                "special filesystem object rejected: "
                + relative.hex()
            )

        for relative in reversed(
            sorted(child_directories)
        ):
            directories.append(relative)

    return sorted(files)


def read_hash_from_fd(
    descriptor: int,
) -> tuple[int, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)

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

    return total, digest.hexdigest()


def copy_stable_file(
    absolute_source: bytes,
    destination: Path,
    *,
    after_initial_stat_hook: (
        Callable[[bytes], None] | None
    ) = None,
) -> tuple[int, str]:
    try:
        path_before = os.lstat(
            absolute_source
        )
    except OSError as error:
        raise SourceError(
            f"cannot lstat source file: {error}"
        ) from error

    if not stat.S_ISREG(path_before.st_mode):
        raise SourceError(
            "source object is no longer a regular file"
        )

    flags = os.O_RDONLY

    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC

    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(
            absolute_source,
            flags,
        )
    except OSError as error:
        raise SourceError(
            f"cannot open source file: {error}"
        ) from error

    try:
        descriptor_before = os.fstat(
            descriptor
        )

        if not stat.S_ISREG(
            descriptor_before.st_mode
        ):
            raise SourceError(
                "opened source object is not regular"
            )

        if (
            path_before.st_dev
            != descriptor_before.st_dev
            or path_before.st_ino
            != descriptor_before.st_ino
        ):
            raise SourceError(
                "source pathname changed before read"
            )

        if after_initial_stat_hook is not None:
            after_initial_stat_hook(
                absolute_source
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        digest = hashlib.sha256()
        copied_bytes = 0

        os.lseek(descriptor, 0, os.SEEK_SET)

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
                copied_bytes += len(chunk)

            output.flush()
            os.fsync(output.fileno())

        second_bytes, second_hash = (
            read_hash_from_fd(descriptor)
        )

        descriptor_after = os.fstat(
            descriptor
        )

        try:
            path_after = os.lstat(
                absolute_source
            )
        except OSError as error:
            raise SourceError(
                "source pathname disappeared "
                "during read"
            ) from error

        first_hash = digest.hexdigest()

        if (
            copied_bytes != second_bytes
            or first_hash != second_hash
        ):
            raise SourceError(
                "source content changed during read"
            )

        if (
            file_identity(descriptor_before)
            != file_identity(descriptor_after)
        ):
            raise SourceError(
                "source metadata changed during read"
            )

        if (
            file_identity(path_before)
            != file_identity(path_after)
        ):
            raise SourceError(
                "source pathname metadata changed "
                "during read"
            )

        if (
            descriptor_after.st_dev
            != path_after.st_dev
            or descriptor_after.st_ino
            != path_after.st_ino
        ):
            raise SourceError(
                "source pathname changed identity "
                "during read"
            )

        if copied_bytes != descriptor_after.st_size:
            raise SourceError(
                "copied byte length differs "
                "from source size"
            )

        return copied_bytes, first_hash

    finally:
        os.close(descriptor)


def load_canonical_json(
    path: Path,
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise VerificationError(
            f"cannot load canonical JSON: {path.name}"
        ) from error

    if not isinstance(value, dict):
        raise VerificationError(
            f"JSON root must be object: {path.name}"
        )

    if canonical_json_bytes(value) != raw:
        raise VerificationError(
            f"JSON is not canonical: {path.name}"
        )

    return value


def verify_snapshot(
    corpus_directory: Path,
) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()

    try:
        root_status = os.lstat(
            corpus_directory
        )
    except OSError as error:
        raise VerificationError(
            f"corpus directory unavailable: {error}"
        ) from error

    if (
        stat.S_ISLNK(root_status.st_mode)
        or not stat.S_ISDIR(root_status.st_mode)
    ):
        raise VerificationError(
            "corpus path must be a real directory"
        )

    required_top_level = {
        MANIFEST_NAME,
        COMPLETE_NAME,
        DOCUMENTS_DIRECTORY,
    }

    allowed_top_level = (
        required_top_level
        | {RUNTIME_INDEX_DIRECTORY}
    )

    actual_top_level = {
        entry.name
        for entry in os.scandir(
            corpus_directory
        )
    }

    missing_top_level = (
        required_top_level - actual_top_level
    )
    unknown_top_level = (
        actual_top_level - allowed_top_level
    )

    if missing_top_level or unknown_top_level:
        raise VerificationError(
            "snapshot top-level coverage mismatch; "
            f"missing={sorted(missing_top_level)}; "
            f"unknown={sorted(unknown_top_level)}"
        )

    runtime_index_directory = (
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY
    )

    if runtime_index_directory.exists():
        runtime_index_status = os.lstat(
            runtime_index_directory
        )

        if (
            stat.S_ISLNK(
                runtime_index_status.st_mode
            )
            or not stat.S_ISDIR(
                runtime_index_status.st_mode
            )
        ):
            raise VerificationError(
                "registered runtime index extension "
                "must be a real directory"
            )

    manifest_path = (
        corpus_directory / MANIFEST_NAME
    )
    complete_path = (
        corpus_directory / COMPLETE_NAME
    )
    documents_directory = (
        corpus_directory / DOCUMENTS_DIRECTORY
    )

    for path in (
        manifest_path,
        complete_path,
        documents_directory,
    ):
        status = os.lstat(path)

        if stat.S_ISLNK(status.st_mode):
            raise VerificationError(
                f"symlink forbidden: {path.name}"
            )

    if not stat.S_ISDIR(
        os.lstat(documents_directory).st_mode
    ):
        raise VerificationError(
            "documents payload is not a directory"
        )

    manifest = load_canonical_json(
        manifest_path
    )
    complete = load_canonical_json(
        complete_path
    )

    if set(manifest) != MANIFEST_KEYS:
        raise VerificationError(
            "manifest fields mismatch"
        )

    if set(complete) != COMPLETE_KEYS:
        raise VerificationError(
            "complete marker fields mismatch"
        )

    constants = {
        "manifest_version": MANIFEST_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "corpus_identity_version":
            CORPUS_IDENTITY_VERSION,
        "source_manifest_identity_version":
            SOURCE_IDENTITY_VERSION,
        "construction_status": "complete",
    }

    for key, expected in constants.items():
        if manifest.get(key) != expected:
            raise VerificationError(
                f"manifest constant mismatch: {key}"
            )

    documents = manifest.get("documents")

    if not isinstance(documents, list):
        raise VerificationError(
            "manifest documents must be a list"
        )

    if manifest.get("document_count") != len(
        documents
    ):
        raise VerificationError(
            "document_count mismatch"
        )

    raw_paths: list[bytes] = []
    total_bytes = 0
    expected_snapshot_names: set[str] = set()

    for expected_doc_id, document in enumerate(
        documents
    ):
        if (
            not isinstance(document, dict)
            or set(document) != DOCUMENT_KEYS
        ):
            raise VerificationError(
                "document record fields mismatch"
            )

        if document.get("doc_id") != expected_doc_id:
            raise VerificationError(
                "document IDs are not canonical"
            )

        if (
            document.get("document_type")
            != "regular_file"
        ):
            raise VerificationError(
                "unsupported document type"
            )

        raw_path = decode_path_hex(
            document.get(
                "relative_path_bytes_hex"
            )
        )

        if (
            document.get("display_path")
            != display_path(raw_path)
        ):
            raise VerificationError(
                "display path mismatch"
            )

        raw_paths.append(raw_path)

        expected_snapshot = (
            f"documents/"
            f"doc_{expected_doc_id:08d}.bin"
        )

        if (
            document.get("snapshot_path")
            != expected_snapshot
        ):
            raise VerificationError(
                "snapshot path mismatch"
            )

        expected_snapshot_names.add(
            Path(expected_snapshot).name
        )

        byte_length = document.get(
            "byte_length"
        )

        if (
            not isinstance(byte_length, int)
            or isinstance(byte_length, bool)
            or byte_length < 0
        ):
            raise VerificationError(
                "invalid document byte length"
            )

        declared_hash = document.get("sha256")

        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(
                character
                not in "0123456789abcdef"
                for character in declared_hash
            )
        ):
            raise VerificationError(
                "invalid document SHA256"
            )

        snapshot_path = (
            corpus_directory / expected_snapshot
        )

        try:
            snapshot_status = os.lstat(
                snapshot_path
            )
        except OSError as error:
            raise VerificationError(
                f"snapshot payload missing: "
                f"{expected_snapshot}"
            ) from error

        if (
            stat.S_ISLNK(snapshot_status.st_mode)
            or not stat.S_ISREG(
                snapshot_status.st_mode
            )
        ):
            raise VerificationError(
                "snapshot payload must be regular"
            )

        if snapshot_status.st_size != byte_length:
            raise VerificationError(
                "snapshot byte length mismatch"
            )

        if sha256_file(
            snapshot_path
        ) != declared_hash:
            raise VerificationError(
                "snapshot SHA256 mismatch"
            )

        total_bytes += byte_length

    if raw_paths != sorted(raw_paths):
        raise VerificationError(
            "documents are not ordered by raw path"
        )

    if len(set(raw_paths)) != len(raw_paths):
        raise VerificationError(
            "duplicate raw source path"
        )

    actual_snapshot_names: set[str] = set()

    for entry in os.scandir(
        documents_directory
    ):
        status = entry.stat(
            follow_symlinks=False
        )

        if (
            stat.S_ISLNK(status.st_mode)
            or not stat.S_ISREG(status.st_mode)
        ):
            raise VerificationError(
                "non-regular document payload"
            )

        actual_snapshot_names.add(entry.name)

    if (
        actual_snapshot_names
        != expected_snapshot_names
    ):
        raise VerificationError(
            "document payload coverage mismatch"
        )

    if (
        manifest.get("total_source_bytes")
        != total_bytes
    ):
        raise VerificationError(
            "total_source_bytes mismatch"
        )

    expected_corpus_id = (
        runtime_corpus_identity(documents)
    )
    expected_source_id = (
        source_manifest_identity(documents)
    )

    if (
        manifest.get("corpus_id")
        != expected_corpus_id
    ):
        raise VerificationError(
            "corpus identity mismatch"
        )

    if (
        manifest.get("source_manifest_id")
        != expected_source_id
    ):
        raise VerificationError(
            "source manifest identity mismatch"
        )

    manifest_sha256 = sha256_file(
        manifest_path
    )

    expected_complete = {
        "complete_version": COMPLETE_VERSION,
        "manifest_sha256": manifest_sha256,
        "corpus_id": expected_corpus_id,
        "source_manifest_id":
            expected_source_id,
        "document_count": len(documents),
    }

    if complete != expected_complete:
        raise VerificationError(
            "complete marker mismatch"
        )

    return {
        "ok": True,
        "format":
            "GLYPH_OPERATOR_MANIFEST_VERIFY_V1",
        "manifest_version": MANIFEST_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "document_count": len(documents),
        "total_source_bytes": total_bytes,
        "corpus_id": expected_corpus_id,
        "source_manifest_id":
            expected_source_id,
        "manifest_sha256": manifest_sha256,
        "construction_status": "complete",
        "source_directory_required": False,
    }


def build_snapshot(
    source_directory: Path,
    output_directory: Path,
    *,
    after_initial_stat_hook: (
        Callable[[bytes, bytes], None] | None
    ) = None,
    test_fail_after_documents: int | None = None,
) -> dict[str, Any]:
    source_directory = source_directory.absolute()
    output_directory = output_directory.absolute()

    source_bytes = os.fsencode(
        os.fspath(source_directory)
    )
    output_bytes = os.fsencode(
        os.fspath(output_directory)
    )

    try:
        source_status_before = os.lstat(
            source_bytes
        )
    except OSError as error:
        raise SourceError(
            f"source root unavailable: {error}"
        ) from error

    if stat.S_ISLNK(
        source_status_before.st_mode
    ):
        raise SourceError(
            "source root symlink rejected"
        )

    if not stat.S_ISDIR(
        source_status_before.st_mode
    ):
        raise SourceError(
            "source root is not a directory"
        )

    source_real = os.path.realpath(
        source_bytes
    )
    output_real = os.path.abspath(
        output_bytes
    )

    try:
        common = os.path.commonpath(
            [source_real, output_real]
        )
    except ValueError as error:
        raise OperatorInputError(
            "source and output path relation invalid"
        ) from error

    if common == source_real:
        raise OperatorInputError(
            "output directory must not be inside "
            "source directory"
        )

    if output_directory.exists():
        raise OperatorInputError(
            "output directory already exists"
        )

    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)

    parent_status = os.lstat(parent)

    if (
        stat.S_ISLNK(parent_status.st_mode)
        or not stat.S_ISDIR(parent_status.st_mode)
    ):
        raise OperatorInputError(
            "output parent must be a real directory"
        )

    initial_files = discover_regular_files(
        source_bytes
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix=(
                f".{output_directory.name}."
                "tmp."
            ),
            dir=parent,
        )
    )

    published = False

    try:
        documents_directory = (
            temporary_directory
            / DOCUMENTS_DIRECTORY
        )
        documents_directory.mkdir()

        records: list[dict[str, Any]] = []

        for doc_id, relative_path in enumerate(
            initial_files
        ):
            absolute_source = os.path.join(
                source_bytes,
                relative_path,
            )

            snapshot_relative = (
                f"documents/"
                f"doc_{doc_id:08d}.bin"
            )
            snapshot_path = (
                temporary_directory
                / snapshot_relative
            )

            hook = None

            if after_initial_stat_hook is not None:
                def hook(
                    absolute: bytes,
                    relative: bytes = relative_path,
                ) -> None:
                    after_initial_stat_hook(
                        relative,
                        absolute,
                    )

            byte_length, document_hash = (
                copy_stable_file(
                    absolute_source,
                    snapshot_path,
                    after_initial_stat_hook=hook,
                )
            )

            records.append({
                "doc_id": doc_id,
                "relative_path_bytes_hex":
                    relative_path.hex(),
                "display_path":
                    display_path(relative_path),
                "document_type": "regular_file",
                "byte_length": byte_length,
                "sha256": document_hash,
                "snapshot_path":
                    snapshot_relative,
            })

            if (
                test_fail_after_documents
                is not None
                and len(records)
                >= test_fail_after_documents
            ):
                raise ConstructionError(
                    "simulated interrupted build"
                )

        second_files = discover_regular_files(
            source_bytes
        )

        if second_files != initial_files:
            raise SourceError(
                "source file set changed during build"
            )

        source_status_after = os.lstat(
            source_bytes
        )

        if (
            root_identity(source_status_before)
            != root_identity(source_status_after)
        ):
            raise SourceError(
                "source root changed during build"
            )

        total_source_bytes = sum(
            record["byte_length"]
            for record in records
        )

        corpus_id = runtime_corpus_identity(
            records
        )
        source_manifest_id = (
            source_manifest_identity(records)
        )

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "runtime_profile":
                RUNTIME_PROFILE,
            "corpus_identity_version":
                CORPUS_IDENTITY_VERSION,
            "source_manifest_identity_version":
                SOURCE_IDENTITY_VERSION,
            "construction_status": "complete",
            "document_count": len(records),
            "total_source_bytes":
                total_source_bytes,
            "corpus_id": corpus_id,
            "source_manifest_id":
                source_manifest_id,
            "documents": records,
        }

        manifest_path = (
            temporary_directory / MANIFEST_NAME
        )
        manifest_path.write_bytes(
            canonical_json_bytes(manifest)
        )
        fsync_file(manifest_path)

        complete = {
            "complete_version":
                COMPLETE_VERSION,
            "manifest_sha256":
                sha256_file(manifest_path),
            "corpus_id": corpus_id,
            "source_manifest_id":
                source_manifest_id,
            "document_count": len(records),
        }

        complete_path = (
            temporary_directory / COMPLETE_NAME
        )
        complete_path.write_bytes(
            canonical_json_bytes(complete)
        )
        fsync_file(complete_path)

        for record in records:
            fsync_file(
                temporary_directory
                / record["snapshot_path"]
            )

        fsync_directory(
            documents_directory
        )
        fsync_directory(
            temporary_directory
        )

        final_files = discover_regular_files(
            source_bytes
        )

        if final_files != initial_files:
            raise SourceError(
                "source file set changed before publication"
            )

        verify_snapshot(
            temporary_directory
        )

        if output_directory.exists():
            raise ConstructionError(
                "output directory appeared "
                "during construction"
            )

        os.rename(
            temporary_directory,
            output_directory,
        )
        published = True

        fsync_directory(parent)

        verification = verify_snapshot(
            output_directory
        )

        return {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_MANIFEST_BUILD_V1",
            "output":
                str(output_directory),
            **{
                key: value
                for key, value
                in verification.items()
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


def command_build(
    args: argparse.Namespace,
) -> int:
    result = build_snapshot(
        Path(args.source),
        Path(args.out),
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
    result = verify_snapshot(
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build = subparsers.add_parser("build")
    build.add_argument(
        "--source",
        required=True,
    )
    build.add_argument(
        "--out",
        required=True,
    )
    build.set_defaults(handler=command_build)

    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--corpus",
        required=True,
    )
    verify.set_defaults(handler=command_verify)

    return parser


def main() -> int:
    parser = build_parser()

    try:
        args = parser.parse_args()
        return args.handler(args)

    except OperatorError as error:
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
