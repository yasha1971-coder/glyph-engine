#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any


FORMAT = (
    "GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_V1"
)
VERIFY_FORMAT = (
    "GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_"
    "VERIFY_V1"
)
SUCCESS_MARKER = (
    "GLYPH WORLD REFERENCE CORPUS REGISTRY "
    "VERIFY OK"
)
EXPECTED_REGISTRY_SHA256 = (
    "0b52a6135937a2218e2f0d0be0ebc764"
    "7cd1f32c48648eda1a60bae0cdc91750"
)
EXPECTED_TOP_KEYS = {
    "format",
    "version",
    "identity_contract",
    "selection_policy",
    "identity_rule",
    "record_count",
    "records",
}
EXPECTED_RECORD_KEYS = {
    "reference_id",
    "class",
    "document_name",
    "bytes",
    "md5",
    "sha256",
    "source",
    "whole_file",
}
REFERENCE_ID_RE = re.compile(
    r"[a-z0-9][a-z0-9-]*"
)
CLASS_RE = re.compile(
    r"[a-z0-9][a-z0-9-]*"
)
MD5_RE = re.compile(r"[0-9a-f]{32}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class VerifyError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
    ) -> None:
        super().__init__(message)
        self.code = code


def require(
    condition: bool,
    code: str,
    message: str,
) -> None:
    if not condition:
        raise VerifyError(code, message)


def reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}

    for key, item in pairs:
        if key in value:
            raise VerifyError(
                "WORLD_REFERENCE_E_JSON",
                f"duplicate JSON key: {key}",
            )
        value[key] = item

    return value


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def read_stable_regular(path: Path) -> bytes:
    require(
        not path.is_symlink(),
        "WORLD_REFERENCE_E_PATH",
        f"symbolic link rejected: {path}",
    )

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerifyError(
            "WORLD_REFERENCE_E_PATH",
            f"cannot open file: {path}",
        ) from error

    try:
        before = os.fstat(descriptor)

        require(
            stat.S_ISREG(before.st_mode),
            "WORLD_REFERENCE_E_PATH",
            f"not a regular file: {path}",
        )

        chunks = []

        while True:
            chunk = os.read(
                descriptor,
                1024 * 1024,
            )
            if not chunk:
                break
            chunks.append(chunk)

        after = os.fstat(descriptor)

    finally:
        os.close(descriptor)

    require(
        (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size
            and before.st_mtime_ns
            == after.st_mtime_ns
        ),
        "WORLD_REFERENCE_E_CHANGED",
        f"file changed during read: {path}",
    )

    return b"".join(chunks)


def validate_document_name(
    value: object,
) -> str:
    require(
        isinstance(value, str),
        "WORLD_REFERENCE_E_SCHEMA",
        "document_name must be a string",
    )
    require(
        value != "",
        "WORLD_REFERENCE_E_SCHEMA",
        "document_name must not be empty",
    )

    pure = PurePosixPath(value)

    require(
        not pure.is_absolute(),
        "WORLD_REFERENCE_E_PATH",
        "absolute document_name rejected",
    )
    require(
        all(
            component not in ("", ".", "..")
            for component in pure.parts
        ),
        "WORLD_REFERENCE_E_PATH",
        "unsafe document_name component",
    )
    require(
        pure.as_posix() == value,
        "WORLD_REFERENCE_E_PATH",
        "document_name is not canonical",
    )

    return value


def load_registry(
    path: Path,
) -> tuple[dict[str, Any], str]:
    raw = read_stable_regular(path)

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate_keys,
        )
    except VerifyError:
        raise
    except Exception as error:
        raise VerifyError(
            "WORLD_REFERENCE_E_JSON",
            "invalid registry JSON",
        ) from error

    require(
        isinstance(value, dict),
        "WORLD_REFERENCE_E_SCHEMA",
        "registry must be an object",
    )
    require(
        raw == canonical_bytes(value),
        "WORLD_REFERENCE_E_CANONICAL",
        "registry JSON is not canonical",
    )
    require(
        set(value) == EXPECTED_TOP_KEYS,
        "WORLD_REFERENCE_E_SCHEMA",
        "registry top-level fields mismatch",
    )
    require(
        value["format"] == FORMAT,
        "WORLD_REFERENCE_E_FORMAT",
        "registry format mismatch",
    )
    require(
        type(value["version"]) is int
        and value["version"] == 1,
        "WORLD_REFERENCE_E_FORMAT",
        "registry version mismatch",
    )
    require(
        value["identity_contract"]
        == "GLYPH_CORPUS_IDENTITY_V1",
        "WORLD_REFERENCE_E_FORMAT",
        "identity contract mismatch",
    )
    require(
        value["selection_policy"]
        == "APPROVED_WHOLE_WORLD_REFERENCE_FILES_ONLY",
        "WORLD_REFERENCE_E_POLICY",
        "selection policy mismatch",
    )
    require(
        value["identity_rule"]
        == "SOURCE_PLUS_MD5_PLUS_SHA256_PLUS_BYTES",
        "WORLD_REFERENCE_E_POLICY",
        "identity rule mismatch",
    )
    require(
        isinstance(value["records"], list),
        "WORLD_REFERENCE_E_SCHEMA",
        "records must be an array",
    )
    require(
        type(value["record_count"]) is int
        and value["record_count"]
        == len(value["records"])
        == 7,
        "WORLD_REFERENCE_E_COVERAGE",
        "registry record count mismatch",
    )

    ids = []
    names = []

    for index, record in enumerate(
        value["records"]
    ):
        require(
            isinstance(record, dict),
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} must be an object",
        )
        require(
            set(record) == EXPECTED_RECORD_KEYS,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} fields mismatch",
        )
        require(
            isinstance(record["reference_id"], str)
            and REFERENCE_ID_RE.fullmatch(
                record["reference_id"]
            ) is not None,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} reference_id invalid",
        )
        require(
            isinstance(record["class"], str)
            and CLASS_RE.fullmatch(
                record["class"]
            ) is not None,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} class invalid",
        )

        document_name = validate_document_name(
            record["document_name"]
        )

        require(
            type(record["bytes"]) is int
            and record["bytes"] > 0,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} byte length invalid",
        )
        require(
            isinstance(record["md5"], str)
            and MD5_RE.fullmatch(
                record["md5"]
            ) is not None,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} MD5 invalid",
        )
        require(
            isinstance(record["sha256"], str)
            and SHA256_RE.fullmatch(
                record["sha256"]
            ) is not None,
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} SHA-256 invalid",
        )
        require(
            isinstance(record["source"], str)
            and record["source"].startswith(
                ("http://", "https://", "ftp://")
            ),
            "WORLD_REFERENCE_E_SCHEMA",
            f"record {index} source invalid",
        )
        require(
            record["whole_file"] is True,
            "WORLD_REFERENCE_E_POLICY",
            f"record {index} is not whole-file",
        )

        ids.append(record["reference_id"])
        names.append(document_name)

    require(
        ids == sorted(ids),
        "WORLD_REFERENCE_E_ORDER",
        "reference IDs are not sorted",
    )
    require(
        len(ids) == len(set(ids)),
        "WORLD_REFERENCE_E_COVERAGE",
        "duplicate reference ID",
    )
    require(
        len(names) == len(set(names)),
        "WORLD_REFERENCE_E_COVERAGE",
        "duplicate document_name",
    )

    registry_sha256 = hashlib.sha256(
        raw
    ).hexdigest()

    require(
        registry_sha256
        == EXPECTED_REGISTRY_SHA256,
        "WORLD_REFERENCE_E_IDENTITY",
        "registry SHA-256 mismatch",
    )

    return value, registry_sha256


def open_reference(
    root: Path,
    document_name: str,
) -> tuple[int, list[int]]:
    require(
        root.exists()
        and root.is_dir()
        and not root.is_symlink(),
        "WORLD_REFERENCE_E_ROOT",
        "golden root must be a real directory",
    )

    flags_directory = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags_directory |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags_directory |= os.O_NOFOLLOW

    flags_file = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags_file |= os.O_NOFOLLOW

    opened: list[int] = []

    try:
        current = os.open(
            root,
            flags_directory,
        )
        opened.append(current)

        parts = PurePosixPath(
            document_name
        ).parts

        for component in parts[:-1]:
            current = os.open(
                component,
                flags_directory,
                dir_fd=current,
            )
            opened.append(current)

        descriptor = os.open(
            parts[-1],
            flags_file,
            dir_fd=current,
        )

        return descriptor, opened

    except OSError as error:
        for opened_descriptor in reversed(opened):
            os.close(opened_descriptor)

        raise VerifyError(
            "WORLD_REFERENCE_E_PATH",
            f"cannot safely open: {document_name}",
        ) from error


def verify_reference(
    root: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    descriptor, opened = open_reference(
        root,
        record["document_name"],
    )

    try:
        before = os.fstat(descriptor)

        require(
            stat.S_ISREG(before.st_mode),
            "WORLD_REFERENCE_E_PATH",
            "reference is not a regular file",
        )
        require(
            before.st_size == record["bytes"],
            "WORLD_REFERENCE_E_BYTES",
            "reference byte length mismatch",
        )

        md5_digest = hashlib.md5(
            usedforsecurity=False
        )
        sha256_digest = hashlib.sha256()

        while True:
            chunk = os.read(
                descriptor,
                16 * 1024 * 1024,
            )
            if not chunk:
                break
            md5_digest.update(chunk)
            sha256_digest.update(chunk)

        after = os.fstat(descriptor)

    finally:
        os.close(descriptor)
        for opened_descriptor in reversed(opened):
            os.close(opened_descriptor)

    require(
        (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size
            and before.st_mtime_ns
            == after.st_mtime_ns
        ),
        "WORLD_REFERENCE_E_CHANGED",
        "reference changed during verification",
    )

    actual_md5 = md5_digest.hexdigest()
    actual_sha256 = sha256_digest.hexdigest()

    require(
        actual_md5 == record["md5"],
        "WORLD_REFERENCE_E_MD5",
        "reference MD5 mismatch",
    )
    require(
        actual_sha256 == record["sha256"],
        "WORLD_REFERENCE_E_SHA256",
        "reference SHA-256 mismatch",
    )

    return {
        "reference_id": record["reference_id"],
        "class": record["class"],
        "document_name":
            record["document_name"],
        "bytes": record["bytes"],
        "md5": actual_md5,
        "sha256": actual_sha256,
        "source": record["source"],
        "whole_file": True,
        "stable_during_verification": True,
    }


def write_atomic(
    path: Path,
    payload: bytes,
) -> None:
    require(
        not path.exists()
        and not path.is_symlink(),
        "WORLD_REFERENCE_E_OUTPUT",
        f"output already exists: {path}",
    )
    require(
        path.parent.is_dir(),
        "WORLD_REFERENCE_E_OUTPUT",
        "output parent is not a directory",
    )

    descriptor = None
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}"
    )

    require(
        not temporary.exists()
        and not temporary.is_symlink(),
        "WORLD_REFERENCE_E_OUTPUT",
        "temporary output already exists",
    )

    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL,
            0o600,
        )

        with os.fdopen(
            descriptor,
            "wb",
            closefd=True,
        ) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise VerifyError(
                "WORLD_REFERENCE_E_OUTPUT",
                f"output already exists: {path}",
            ) from error

        temporary.unlink()

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY

        directory_descriptor = os.open(
            path.parent,
            directory_flags,
        )

        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def choose_records(
    registry: dict[str, Any],
    requested: list[str],
) -> list[dict[str, Any]]:
    by_id = {
        record["reference_id"]: record
        for record in registry["records"]
    }

    require(
        len(requested) == len(set(requested)),
        "WORLD_REFERENCE_E_REQUEST",
        "duplicate requested reference ID",
    )

    if not requested:
        return list(registry["records"])

    unknown = sorted(
        set(requested) - set(by_id)
    )

    require(
        not unknown,
        "WORLD_REFERENCE_E_REQUEST",
        f"unknown reference IDs: {unknown}",
    )

    requested_set = set(requested)

    return [
        record
        for record in registry["records"]
        if record["reference_id"]
        in requested_set
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the exact GLYPH V1 world-reference "
            "registry and optionally stream-verify its "
            "whole files."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--golden-root",
        type=Path,
    )
    parser.add_argument(
        "--reference-id",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    registry, registry_sha256 = load_registry(
        args.registry
    )
    selected = choose_records(
        registry,
        args.reference_id,
    )

    if args.registry_only:
        require(
            args.golden_root is None,
            "WORLD_REFERENCE_E_REQUEST",
            "--golden-root is invalid in registry-only mode",
        )
        verified: list[dict[str, Any]] = []
    else:
        require(
            args.golden_root is not None,
            "WORLD_REFERENCE_E_REQUEST",
            "--golden-root is required",
        )
        verified = [
            verify_reference(
                args.golden_root,
                record,
            )
            for record in selected
        ]

    result = {
        "format": VERIFY_FORMAT,
        "ok": True,
        "registry_format": FORMAT,
        "registry_sha256": registry_sha256,
        "registry_record_count":
            registry["record_count"],
        "registry_only": args.registry_only,
        "files_verified":
            not args.registry_only,
        "selected_reference_ids": [
            record["reference_id"]
            for record in selected
        ],
        "verified_record_count":
            len(verified),
        "verified_total_bytes":
            sum(
                record["bytes"]
                for record in verified
            ),
        "records": verified,
    }

    payload = canonical_bytes(result)

    write_atomic(
        args.output,
        payload,
    )

    print(SUCCESS_MARKER)
    print(
        "registry_sha256="
        f"{registry_sha256}"
    )
    print(
        "selected_reference_count="
        f"{len(selected)}"
    )
    print(
        "verified_record_count="
        f"{len(verified)}"
    )
    print(
        "output_sha256="
        f"{hashlib.sha256(payload).hexdigest()}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerifyError as error:
        print(
            f"ERROR {error.code}: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
