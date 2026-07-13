#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
BUILD = ROOT / "build"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_bundle_replay_v1 import (  # noqa: E402
    ARTIFACT_PATH,
    BUNDLE_COMPLETE_VERSION,
    BUNDLE_MANIFEST_VERSION,
    BUNDLE_VERSION,
    COMPLETE_NAME,
    MANIFEST_NAME,
    QUERY_PATH,
    QUERY_RESULT_VERSION,
    REPLAY_PATH,
    RUNTIME_PROFILE,
    bundle_root_identity,
    canonical_json_bytes,
)
from glyph_operator_index_v1 import (  # noqa: E402
    INDEX_COMPLETE_NAME,
    INDEX_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    COMPLETE_NAME as SOURCE_COMPLETE_NAME,
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    fsync_directory,
    fsync_file,
    load_canonical_json,
    sha256_file,
    verify_snapshot,
)
from glyph_operator_query_v1 import (  # noqa: E402
    execute_operator_query,
    parse_query_hex,
    read_stable_query_file,
)

REPLAY_SOURCE = (
    TOOLS
    / "glyph_operator_bundle_replay_v1.py"
)


class BundleBuildError(RuntimeError):
    pass


def copy_regular(
    source: Path,
    destination: Path,
    *,
    executable: bool,
) -> None:
    source_status = os.lstat(source)

    if (
        stat.S_ISLNK(source_status.st_mode)
        or not stat.S_ISREG(
            source_status.st_mode
        )
    ):
        raise BundleBuildError(
            "source payload must be regular: "
            + str(source)
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        source,
        destination,
    )

    destination.chmod(
        0o755 if executable else 0o644
    )

    if (
        destination.stat().st_size
        != source_status.st_size
        or sha256_file(destination)
        != sha256_file(source)
    ):
        raise BundleBuildError(
            "copied payload differs: "
            + str(source)
        )


def add_entry(
    entries: list[dict[str, Any]],
    bundle: Path,
    relative: str,
    role: str,
    *,
    executable: bool,
) -> None:
    path = bundle / relative
    status = os.lstat(path)

    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISREG(status.st_mode)
    ):
        raise BundleBuildError(
            "bundle payload is not regular: "
            + relative
        )

    entries.append({
        "path": relative,
        "role": role,
        "size_bytes": status.st_size,
        "sha256": sha256_file(path),
        "executable": executable,
    })


def run_standalone_replay(
    bundle: Path,
) -> dict[str, Any]:
    replay = bundle / REPLAY_PATH

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(replay),
            "--bundle",
            str(bundle),
        ],
        cwd=bundle,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=1800,
        check=False,
        env={
            "PATH": os.environ.get(
                "PATH",
                "/usr/bin:/bin",
            ),
            "LANG": "C",
            "LC_ALL": "C",
        },
    )

    if result.returncode != 0:
        raise BundleBuildError(
            "standalone bundle replay failed"
            + "\nstdout:\n"
            + result.stdout[-8000:]
            + "\nstderr:\n"
            + result.stderr[-8000:]
        )

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise BundleBuildError(
            "standalone replay returned "
            "invalid JSON"
        ) from error

    if (
        not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        raise BundleBuildError(
            "standalone replay did not pass"
        )

    return value


def build_bundle(
    corpus_directory: Path,
    output_directory: Path,
    query: bytes,
    *,
    max_offsets: int | None = None,
    test_fail_before_publication: bool = False,
) -> dict[str, Any]:
    corpus_directory = corpus_directory.resolve()
    output_directory = output_directory.absolute()

    if output_directory.exists():
        raise BundleBuildError(
            "bundle output already exists"
        )

    corpus_text = os.fspath(corpus_directory)
    output_text = os.fspath(output_directory)

    try:
        common = os.path.commonpath(
            [corpus_text, output_text]
        )
    except ValueError as error:
        raise BundleBuildError(
            "invalid corpus/output relation"
        ) from error

    if common == corpus_text:
        raise BundleBuildError(
            "bundle output must not be "
            "inside committed corpus"
        )

    source_verification = verify_snapshot(
        corpus_directory
    )
    runtime_verification = verify_runtime_index(
        corpus_directory,
        require_current_binaries=True,
        rebuild=False,
    )

    query_result = execute_operator_query(
        corpus_directory,
        query,
        max_offsets=max_offsets,
    )

    source_manifest_path = (
        corpus_directory / SOURCE_MANIFEST_NAME
    )
    source_complete_path = (
        corpus_directory / SOURCE_COMPLETE_NAME
    )
    runtime_manifest_path = (
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )
    runtime_complete_path = (
        corpus_directory
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_COMPLETE_NAME
    )

    source_manifest = load_canonical_json(
        source_manifest_path
    )
    runtime_manifest = load_canonical_json(
        runtime_manifest_path
    )

    parent = output_directory.parent
    parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(
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
        entries: list[dict[str, Any]] = []

        artifact_output = (
            temporary / ARTIFACT_PATH
        )
        artifact_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        artifact_output.write_bytes(
            canonical_json_bytes(
                query_result
            )
        )
        artifact_output.chmod(0o644)
        add_entry(
            entries,
            temporary,
            ARTIFACT_PATH,
            "query_artifact",
            executable=False,
        )

        query_output = temporary / QUERY_PATH
        query_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        query_output.write_bytes(query)
        query_output.chmod(0o644)
        add_entry(
            entries,
            temporary,
            QUERY_PATH,
            "query_bytes",
            executable=False,
        )

        copy_regular(
            source_manifest_path,
            temporary / (
                "source/"
                + SOURCE_MANIFEST_NAME
            ),
            executable=False,
        )
        add_entry(
            entries,
            temporary,
            "source/" + SOURCE_MANIFEST_NAME,
            "source_manifest",
            executable=False,
        )

        copy_regular(
            source_complete_path,
            temporary / (
                "source/"
                + SOURCE_COMPLETE_NAME
            ),
            executable=False,
        )
        add_entry(
            entries,
            temporary,
            "source/" + SOURCE_COMPLETE_NAME,
            "source_complete",
            executable=False,
        )

        for document in source_manifest[
            "documents"
        ]:
            relative = document[
                "snapshot_path"
            ]
            bundle_relative = (
                "source/" + relative
            )

            copy_regular(
                corpus_directory / relative,
                temporary / bundle_relative,
                executable=False,
            )
            add_entry(
                entries,
                temporary,
                bundle_relative,
                "source_document",
                executable=False,
            )

        copy_regular(
            runtime_manifest_path,
            temporary / (
                "runtime/"
                + INDEX_MANIFEST_NAME
            ),
            executable=False,
        )
        add_entry(
            entries,
            temporary,
            "runtime/" + INDEX_MANIFEST_NAME,
            "runtime_manifest",
            executable=False,
        )

        copy_regular(
            runtime_complete_path,
            temporary / (
                "runtime/"
                + INDEX_COMPLETE_NAME
            ),
            executable=False,
        )
        add_entry(
            entries,
            temporary,
            "runtime/" + INDEX_COMPLETE_NAME,
            "runtime_complete",
            executable=False,
        )

        runtime_prefix = (
            RUNTIME_INDEX_DIRECTORY + "/"
        )

        for document in runtime_manifest[
            "documents"
        ]:
            for role in ("sa", "bwt", "fm"):
                original = document[role][
                    "path"
                ]

                if not original.startswith(
                    runtime_prefix
                ):
                    raise BundleBuildError(
                        "runtime artifact path "
                        "prefix mismatch"
                    )

                relative = original[
                    len(runtime_prefix):
                ]
                bundle_relative = (
                    "runtime/" + relative
                )

                copy_regular(
                    corpus_directory / original,
                    temporary / bundle_relative,
                    executable=False,
                )
                add_entry(
                    entries,
                    temporary,
                    bundle_relative,
                    "runtime_index_artifact",
                    executable=False,
                )

        copied_binary_names: set[str] = set()

        for record in (
            runtime_manifest[
                "runtime_binaries"
            ]
            + query_result[
                "query_binary_commitments"
            ]
        ):
            name = record["name"]

            if name in copied_binary_names:
                continue

            copied_binary_names.add(name)
            bundle_relative = "bin/" + name

            copy_regular(
                BUILD / name,
                temporary / bundle_relative,
                executable=True,
            )

            expected_size = record[
                "size_bytes"
            ]
            expected_hash = record["sha256"]

            if (
                (temporary / bundle_relative)
                .stat().st_size
                != expected_size
                or sha256_file(
                    temporary / bundle_relative
                )
                != expected_hash
            ):
                raise BundleBuildError(
                    "binary commitment mismatch: "
                    + name
                )

            role = (
                "query_binary"
                if name.startswith("query_")
                else "runtime_builder_binary"
            )

            add_entry(
                entries,
                temporary,
                bundle_relative,
                role,
                executable=True,
            )

        copy_regular(
            REPLAY_SOURCE,
            temporary / REPLAY_PATH,
            executable=True,
        )
        add_entry(
            entries,
            temporary,
            REPLAY_PATH,
            "standalone_replay",
            executable=True,
        )

        entries.sort(
            key=lambda item: item["path"]
        )

        bundle_root = bundle_root_identity(
            entries
        )

        manifest = {
            "bundle_manifest_version":
                BUNDLE_MANIFEST_VERSION,
            "bundle_version":
                BUNDLE_VERSION,
            "runtime_profile":
                RUNTIME_PROFILE,
            "construction_status":
                "complete",
            "artifact_version":
                QUERY_RESULT_VERSION,
            "corpus_id":
                query_result["corpus_id"],
            "source_manifest_id":
                query_result[
                    "source_manifest_id"
                ],
            "runtime_index_id":
                query_result[
                    "runtime_index_id"
                ],
            "query_result_id":
                query_result[
                    "query_result_id"
                ],
            "document_count":
                query_result[
                    "document_count"
                ],
            "file_count": len(entries),
            "files": entries,
            "bundle_root_sha256":
                bundle_root,
            "external_dependencies": [],
            "network_dependency_required":
                False,
            "repository_dependency_required":
                False,
        }

        manifest_path = (
            temporary / MANIFEST_NAME
        )
        manifest_path.write_bytes(
            canonical_json_bytes(manifest)
        )
        manifest_path.chmod(0o644)

        complete = {
            "complete_version":
                BUNDLE_COMPLETE_VERSION,
            "bundle_manifest_sha256":
                sha256_file(manifest_path),
            "bundle_root_sha256":
                bundle_root,
            "query_result_id":
                query_result[
                    "query_result_id"
                ],
            "runtime_index_id":
                query_result[
                    "runtime_index_id"
                ],
            "corpus_id":
                query_result["corpus_id"],
            "file_count": len(entries),
        }

        complete_path = (
            temporary / COMPLETE_NAME
        )
        complete_path.write_bytes(
            canonical_json_bytes(complete)
        )
        complete_path.chmod(0o644)

        for entry in entries:
            fsync_file(
                temporary / entry["path"]
            )

        fsync_file(manifest_path)
        fsync_file(complete_path)

        for root, directories, _ in os.walk(
            temporary,
            topdown=False,
        ):
            for directory in directories:
                fsync_directory(
                    Path(root) / directory
                )
            fsync_directory(Path(root))

        verify_snapshot(corpus_directory)
        verify_runtime_index(
            corpus_directory,
            require_current_binaries=True,
            rebuild=False,
        )

        replay_result = (
            run_standalone_replay(
                temporary
            )
        )

        if (
            replay_result[
                "query_result_id"
            ]
            != query_result[
                "query_result_id"
            ]
        ):
            raise BundleBuildError(
                "standalone replay result ID mismatch"
            )

        if test_fail_before_publication:
            raise BundleBuildError(
                "simulated interrupted "
                "bundle build"
            )

        if output_directory.exists():
            raise BundleBuildError(
                "bundle output appeared "
                "during construction"
            )

        os.rename(
            temporary,
            output_directory,
        )
        published = True
        fsync_directory(parent)

        final_replay = run_standalone_replay(
            output_directory
        )

        return {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_EVIDENCE_BUNDLE_BUILD_V1",
            "operator_obligation": "O4",
            "bundle_version":
                BUNDLE_VERSION,
            "bundle_manifest_version":
                BUNDLE_MANIFEST_VERSION,
            "output":
                str(output_directory),
            "file_count": len(entries),
            "document_count":
                source_verification[
                    "document_count"
                ],
            "corpus_id":
                source_verification[
                    "corpus_id"
                ],
            "source_manifest_id":
                source_verification[
                    "source_manifest_id"
                ],
            "runtime_index_id":
                runtime_verification[
                    "runtime_index_id"
                ],
            "query_result_id":
                query_result[
                    "query_result_id"
                ],
            "bundle_root_sha256":
                bundle_root,
            "standalone_replay_verified":
                final_replay["ok"],
            "self_contained": True,
            "repository_dependency_required":
                False,
            "network_dependency_required":
                False,
        }

    except Exception:
        if not published:
            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )
        raise


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
    build.add_argument(
        "--out",
        required=True,
    )

    source = build.add_mutually_exclusive_group(
        required=True
    )
    source.add_argument("--query-file")
    source.add_argument("--query-hex")

    build.add_argument(
        "--max-offsets",
        type=int,
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        query = (
            read_stable_query_file(
                Path(args.query_file)
            )
            if args.query_file is not None
            else parse_query_hex(
                args.query_hex
            )
        )

        result = build_bundle(
            Path(args.corpus),
            Path(args.out),
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
