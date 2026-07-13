#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

BUNDLE_VERSION = "GLYPH_BINARY_RUNTIME_BUNDLE_V1"
MANIFEST_VERSION = "GLYPH_BINARY_RUNTIME_BUNDLE_MANIFEST_V1"
RUNTIME_PROFILE = "GLYPH_BINARY_RUNTIME_V1"
ARTIFACT_VERSION = "GLYPH_BINARY_RUNTIME_EVIDENCE_V1"

MANIFEST_NAME = "bundle_manifest_v1.json"

MANIFEST_KEYS = {
    "bundle_version",
    "manifest_version",
    "runtime_profile",
    "artifact_version",
    "evidence_path",
    "evidence_module_path",
    "replay_entrypoint",
    "document_count",
    "documents",
    "runtime_binaries",
    "file_count",
    "files",
    "bundle_root_sha256",
    "external_data_dependencies",
}

FILE_KEYS = {
    "path",
    "role",
    "size_bytes",
    "sha256",
    "executable",
}


class BundleReplayError(RuntimeError):
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


def safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or value == "":
        raise BundleReplayError(
            "bundle path must be a non-empty string"
        )

    if "\\" in value:
        raise BundleReplayError(
            "backslash forbidden in bundle path"
        )

    path = Path(value)

    if path.is_absolute():
        raise BundleReplayError(
            "absolute bundle path forbidden"
        )

    if any(
        part in ("", ".", "..")
        for part in path.parts
    ):
        raise BundleReplayError(
            "noncanonical or traversal bundle path"
        )

    if path.as_posix() != value:
        raise BundleReplayError(
            "bundle path is not canonical"
        )

    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise BundleReplayError(
            f"failed to load JSON: {path.name}"
        ) from error

    if not isinstance(value, dict):
        raise BundleReplayError(
            f"JSON root must be an object: {path.name}"
        )

    return value


def validate_manifest(
    bundle: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    manifest_path = bundle / MANIFEST_NAME

    if not manifest_path.is_file():
        raise BundleReplayError(
            "bundle manifest missing"
        )

    if manifest_path.is_symlink():
        raise BundleReplayError(
            "bundle manifest must not be a symlink"
        )

    manifest = load_json(manifest_path)

    if set(manifest) != MANIFEST_KEYS:
        missing = sorted(
            MANIFEST_KEYS - set(manifest)
        )
        extra = sorted(
            set(manifest) - MANIFEST_KEYS
        )

        raise BundleReplayError(
            f"manifest fields mismatch; "
            f"missing={missing}; extra={extra}"
        )

    constants = {
        "bundle_version": BUNDLE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "artifact_version": ARTIFACT_VERSION,
    }

    for key, expected in constants.items():
        if manifest.get(key) != expected:
            raise BundleReplayError(
                f"manifest constant mismatch: {key}"
            )

    if manifest.get("external_data_dependencies") != []:
        raise BundleReplayError(
            "external data dependencies forbidden"
        )

    files = manifest.get("files")

    if not isinstance(files, list) or not files:
        raise BundleReplayError(
            "manifest files must be a non-empty list"
        )

    if manifest.get("file_count") != len(files):
        raise BundleReplayError(
            "manifest file_count mismatch"
        )

    records: dict[str, dict[str, Any]] = {}

    for record in files:
        if not isinstance(record, dict):
            raise BundleReplayError(
                "manifest file record must be an object"
            )

        if set(record) != FILE_KEYS:
            raise BundleReplayError(
                "manifest file record fields mismatch"
            )

        relative = safe_relative_path(
            record.get("path")
        )
        relative_text = relative.as_posix()

        if relative_text == MANIFEST_NAME:
            raise BundleReplayError(
                "manifest must not list itself"
            )

        if relative_text in records:
            raise BundleReplayError(
                "duplicate manifest path"
            )

        size = record.get("size_bytes")

        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise BundleReplayError(
                "invalid manifest file size"
            )

        sha256 = record.get("sha256")

        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in sha256
            )
        ):
            raise BundleReplayError(
                "invalid manifest SHA256"
            )

        if not isinstance(
            record.get("executable"),
            bool,
        ):
            raise BundleReplayError(
                "invalid executable flag"
            )

        records[relative_text] = record

    expected_paths = set(records)
    expected_paths.add(MANIFEST_NAME)

    actual_paths: set[str] = set()

    for entry in bundle.rglob("*"):
        relative = entry.relative_to(
            bundle
        ).as_posix()

        if entry.is_symlink():
            raise BundleReplayError(
                f"symlink payload forbidden: {relative}"
            )

        if entry.is_file():
            actual_paths.add(relative)

    if actual_paths != expected_paths:
        missing = sorted(
            expected_paths - actual_paths
        )
        extra = sorted(
            actual_paths - expected_paths
        )

        raise BundleReplayError(
            f"manifest coverage mismatch; "
            f"missing={missing}; extra={extra}"
        )

    for relative_text, record in records.items():
        path = bundle / relative_text

        if path.stat().st_size != record["size_bytes"]:
            raise BundleReplayError(
                f"file size mismatch: {relative_text}"
            )

        if sha256_file(path) != record["sha256"]:
            raise BundleReplayError(
                f"file SHA256 mismatch: {relative_text}"
            )

        if (
            record["executable"]
            and not os.access(path, os.X_OK)
        ):
            raise BundleReplayError(
                f"executable payload is not executable: "
                f"{relative_text}"
            )

    sorted_records = sorted(
        files,
        key=lambda item: item["path"],
    )

    actual_root = sha256_bytes(
        canonical_json_bytes(sorted_records)
    )

    if (
        manifest.get("bundle_root_sha256")
        != actual_root
    ):
        raise BundleReplayError(
            "bundle root SHA256 mismatch"
        )

    return manifest, records


def load_evidence_module(
    bundle: Path,
    relative_path: str,
):
    module_path = bundle / safe_relative_path(
        relative_path
    )

    sys.dont_write_bytecode = True

    specification = (
        importlib.util.spec_from_file_location(
            "glyph_binary_runtime_evidence_v1",
            module_path,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise BundleReplayError(
            "failed to load evidence replay module"
        )

    module = importlib.util.module_from_spec(
        specification
    )

    specification.loader.exec_module(module)

    return module


def replay_bundle(bundle: Path) -> dict[str, Any]:
    bundle = bundle.resolve()

    if not bundle.is_dir():
        raise BundleReplayError(
            "bundle directory missing"
        )

    manifest, records = validate_manifest(bundle)

    evidence_path_text = manifest.get(
        "evidence_path"
    )
    module_path_text = manifest.get(
        "evidence_module_path"
    )
    replay_path_text = manifest.get(
        "replay_entrypoint"
    )

    evidence_path = safe_relative_path(
        evidence_path_text
    )
    safe_relative_path(module_path_text)
    safe_relative_path(replay_path_text)

    if records[evidence_path.as_posix()][
        "role"
    ] != "evidence":
        raise BundleReplayError(
            "evidence role mismatch"
        )

    if records[module_path_text][
        "role"
    ] != "evidence_module":
        raise BundleReplayError(
            "evidence module role mismatch"
        )

    if records[replay_path_text][
        "role"
    ] != "replay":
        raise BundleReplayError(
            "replay entrypoint role mismatch"
        )

    documents_metadata = manifest.get(
        "documents"
    )

    if not isinstance(documents_metadata, list):
        raise BundleReplayError(
            "manifest documents must be a list"
        )

    if (
        manifest.get("document_count")
        != len(documents_metadata)
    ):
        raise BundleReplayError(
            "manifest document_count mismatch"
        )

    documents: list[bytes] = []

    for expected_doc_id, metadata in enumerate(
        documents_metadata
    ):
        if not isinstance(metadata, dict):
            raise BundleReplayError(
                "document metadata must be an object"
            )

        if set(metadata) != {
            "doc_id",
            "path",
            "byte_length",
            "sha256",
        }:
            raise BundleReplayError(
                "document metadata fields mismatch"
            )

        if metadata.get("doc_id") != expected_doc_id:
            raise BundleReplayError(
                "document IDs are not canonical"
            )

        relative = safe_relative_path(
            metadata.get("path")
        )
        relative_text = relative.as_posix()

        if records[relative_text][
            "role"
        ] != "document":
            raise BundleReplayError(
                "document payload role mismatch"
            )

        data = (bundle / relative).read_bytes()

        if metadata.get("byte_length") != len(data):
            raise BundleReplayError(
                "document byte length mismatch"
            )

        if metadata.get("sha256") != sha256_bytes(
            data
        ):
            raise BundleReplayError(
                "document SHA256 mismatch"
            )

        documents.append(data)

    expected_runtime_names = {
        "build_sa_binary_v1",
        "build_bwt_binary_v1",
        "build_fm_binary_v1",
        "query_fm_locate_binary_v1",
    }

    runtime_binaries = manifest.get(
        "runtime_binaries"
    )

    if not isinstance(runtime_binaries, list):
        raise BundleReplayError(
            "runtime_binaries must be a list"
        )

    actual_runtime_names: set[str] = set()

    for item in runtime_binaries:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "path"}
        ):
            raise BundleReplayError(
                "runtime binary metadata mismatch"
            )

        name = item.get("name")
        relative = safe_relative_path(
            item.get("path")
        )
        relative_text = relative.as_posix()

        if records[relative_text][
            "role"
        ] != "runtime_binary":
            raise BundleReplayError(
                "runtime binary role mismatch"
            )

        if not records[relative_text][
            "executable"
        ]:
            raise BundleReplayError(
                "runtime binary must be executable"
            )

        actual_runtime_names.add(name)

    if actual_runtime_names != expected_runtime_names:
        raise BundleReplayError(
            "required runtime binaries mismatch"
        )

    evidence = load_json(
        bundle / evidence_path
    )

    module = load_evidence_module(
        bundle,
        module_path_text,
    )

    result = module.replay_artifact(
        evidence,
        documents,
    )

    if result.get("ok") is not True:
        raise BundleReplayError(
            "runtime evidence replay failed"
        )

    return {
        "ok": True,
        "bundle_version": BUNDLE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "artifact_version": ARTIFACT_VERSION,
        "bundle_root_sha256":
            manifest["bundle_root_sha256"],
        "manifest_sha256": sha256_file(
            bundle / MANIFEST_NAME
        ),
        "document_count": len(documents),
        "external_data_dependencies": [],
        "repository_dependency_required": False,
        "network_dependency_required": False,
        "replay_result": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        required=True,
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    try:
        result = replay_bundle(
            Path(args.bundle)
        )

        if args.out is not None:
            output = Path(args.out).resolve()
            output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            output.write_bytes(
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
    except Exception as error:
        failure = {
            "ok": False,
            "bundle_version": BUNDLE_VERSION,
            "error": str(error),
        }

        print(
            json.dumps(
                failure,
                indent=2,
                sort_keys=True,
            )
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
