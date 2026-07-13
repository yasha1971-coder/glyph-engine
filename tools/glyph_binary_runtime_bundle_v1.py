#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

TOOLS = Path(__file__).resolve().parent

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


from glyph_binary_runtime_evidence_v1 import (
    ARTIFACT_VERSION,
    RUNTIME_PROFILE,
    canonical_json_bytes,
    ensure_binaries,
    make_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

BUNDLE_VERSION = "GLYPH_BINARY_RUNTIME_BUNDLE_V1"
MANIFEST_VERSION = (
    "GLYPH_BINARY_RUNTIME_BUNDLE_MANIFEST_V1"
)

RUNTIME_BINARY_NAMES = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
    "query_fm_locate_binary_v1",
]


class BundleBuildError(RuntimeError):
    pass


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


def copy_payload(
    source: Path,
    destination: Path,
    *,
    executable: bool,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(source, destination)

    destination.chmod(
        0o755 if executable else 0o644
    )


def file_record(
    bundle: Path,
    relative_path: str,
    role: str,
    *,
    executable: bool,
) -> dict[str, Any]:
    path = bundle / relative_path

    return {
        "path": relative_path,
        "role": role,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "executable": executable,
    }


def build_bundle(
    document_paths: Sequence[Path],
    query_hex: str,
    max_offsets: int | None,
    output_directory: Path,
) -> dict[str, Any]:
    ensure_binaries()

    output_directory = output_directory.resolve()

    if output_directory.exists():
        if any(output_directory.iterdir()):
            raise BundleBuildError(
                "bundle output directory is not empty"
            )
    else:
        output_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

    documents = [
        path.read_bytes()
        for path in document_paths
    ]

    artifact = make_artifact(
        documents,
        query_hex,
        max_offsets,
    )

    evidence_relative = "evidence.json"
    evidence_path = (
        output_directory / evidence_relative
    )

    evidence_path.write_bytes(
        canonical_json_bytes(artifact)
    )

    document_metadata = []

    for doc_id, data in enumerate(documents):
        relative = (
            f"documents/doc_{doc_id:08d}.bin"
        )
        destination = output_directory / relative

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        destination.write_bytes(data)
        destination.chmod(0o644)

        document_metadata.append({
            "doc_id": doc_id,
            "path": relative,
            "byte_length": len(data),
            "sha256": sha256_bytes(data),
        })

    module_relative = (
        "tools/glyph_binary_runtime_evidence_v1.py"
    )

    copy_payload(
        ROOT
        / "tools/glyph_binary_runtime_evidence_v1.py",
        output_directory / module_relative,
        executable=False,
    )

    replay_relative = "replay.py"

    copy_payload(
        ROOT
        / "tools/"
        "glyph_binary_runtime_bundle_replay_v1.py",
        output_directory / replay_relative,
        executable=True,
    )

    runtime_binaries = []

    for name in RUNTIME_BINARY_NAMES:
        source = BUILD / name

        if not source.is_file():
            raise BundleBuildError(
                f"missing runtime binary: {source}"
            )

        relative = f"build/{name}"

        copy_payload(
            source,
            output_directory / relative,
            executable=True,
        )

        runtime_binaries.append({
            "name": name,
            "path": relative,
        })

    records = [
        file_record(
            output_directory,
            evidence_relative,
            "evidence",
            executable=False,
        ),
        file_record(
            output_directory,
            module_relative,
            "evidence_module",
            executable=False,
        ),
        file_record(
            output_directory,
            replay_relative,
            "replay",
            executable=True,
        ),
    ]

    for metadata in document_metadata:
        records.append(
            file_record(
                output_directory,
                metadata["path"],
                "document",
                executable=False,
            )
        )

    for metadata in runtime_binaries:
        records.append(
            file_record(
                output_directory,
                metadata["path"],
                "runtime_binary",
                executable=True,
            )
        )

    records.sort(
        key=lambda item: item["path"]
    )

    bundle_root_sha256 = sha256_bytes(
        canonical_json_bytes(records)
    )

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "artifact_version": ARTIFACT_VERSION,
        "evidence_path": evidence_relative,
        "evidence_module_path": module_relative,
        "replay_entrypoint": replay_relative,
        "document_count": len(documents),
        "documents": document_metadata,
        "runtime_binaries": runtime_binaries,
        "file_count": len(records),
        "files": records,
        "bundle_root_sha256":
            bundle_root_sha256,
        "external_data_dependencies": [],
    }

    manifest_path = (
        output_directory
        / "bundle_manifest_v1.json"
    )

    manifest_path.write_bytes(
        canonical_json_bytes(manifest)
    )
    manifest_path.chmod(0o644)

    result = {
        "ok": True,
        "bundle": str(output_directory),
        "bundle_version": BUNDLE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "bundle_root_sha256":
            bundle_root_sha256,
        "manifest_sha256":
            sha256_file(manifest_path),
        "file_count": len(records),
        "document_count": len(documents),
        "evidence_sha256":
            sha256_file(evidence_path),
        "self_contained": True,
        "external_data_dependencies": [],
    }

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--query-hex",
        required=True,
    )
    parser.add_argument(
        "--max-offsets",
        type=int,
    )
    parser.add_argument(
        "--out",
        required=True,
    )
    args = parser.parse_args()

    result = build_bundle(
        [
            Path(value).resolve()
            for value in args.document
        ],
        args.query_hex,
        args.max_offsets,
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


if __name__ == "__main__":
    raise SystemExit(main())
