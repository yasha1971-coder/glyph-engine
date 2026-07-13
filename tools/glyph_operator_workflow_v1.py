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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_bundle_replay_v1 import (  # noqa: E402
    ARTIFACT_PATH,
    QUERY_PATH,
    load_canonical_json as load_bundle_json,
    verify_bundle,
)
from glyph_operator_bundle_v1 import (  # noqa: E402
    build_bundle,
)
from glyph_operator_index_v1 import (  # noqa: E402
    build_runtime_index,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    OperatorError,
    build_snapshot,
    canonical_json_bytes,
    fsync_directory,
    fsync_file,
    load_canonical_json,
    sha256_file,
    verify_snapshot,
)
from glyph_operator_query_v1 import (  # noqa: E402
    QueryError,
    execute_operator_query,
    parse_query_hex,
    read_stable_query_file,
)

WORKFLOW_RESULT_VERSION = (
    "GLYPH_OPERATOR_WORKFLOW_RESULT_V1"
)
WORKFLOW_COMPLETE_VERSION = (
    "GLYPH_OPERATOR_WORKFLOW_COMPLETE_V1"
)
WORKFLOW_PROFILE = "GLYPH_OPERATOR_PATH_V1"

WORKFLOW_RESULT_NAME = "workflow_result_v1.json"
WORKFLOW_COMPLETE_NAME = "WORKFLOW_COMPLETE_V1.json"

CORPUS_DIRECTORY_NAME = "corpus"
BUNDLE_DIRECTORY_NAME = "bundle"

NEXT_OBLIGATION = "O6_OPERATOR_CONFORMANCE_CLOSURE"

CASE_ROOT_ENTRIES = {
    CORPUS_DIRECTORY_NAME,
    BUNDLE_DIRECTORY_NAME,
    WORKFLOW_RESULT_NAME,
    WORKFLOW_COMPLETE_NAME,
}

WORKFLOW_RESULT_KEYS = {
    "ok",
    "workflow_result_version",
    "workflow_profile",
    "operator_obligation",
    "corpus_id",
    "source_manifest_id",
    "runtime_index_id",
    "query_result_id",
    "bundle_root_sha256",
    "bundle_manifest_sha256",
    "query_hex",
    "query_length_bytes",
    "query_sha256",
    "max_offsets",
    "document_count",
    "total_source_bytes",
    "match_count",
    "returned_count",
    "bounded",
    "offsets_complete",
    "case_layout",
    "one_command_workflow",
    "atomic_publication",
    "source_directory_consumed_only_during_snapshot",
    "bundle_self_contained",
    "repository_dependency_required_for_bundle",
    "network_dependency_required",
    "next_operator_obligation",
    "workflow_result_id",
}

WORKFLOW_COMPLETE_KEYS = {
    "complete_version",
    "workflow_result_sha256",
    "workflow_result_id",
    "corpus_id",
    "runtime_index_id",
    "query_result_id",
    "bundle_root_sha256",
}


class WorkflowError(RuntimeError):
    exit_code = 7
    error_code = "WORKFLOW_FAILURE"


class WorkflowInputError(WorkflowError):
    exit_code = 2
    error_code = "INVALID_WORKFLOW_INPUT"


class WorkflowVerificationError(WorkflowError):
    exit_code = 6
    error_code = "WORKFLOW_VERIFICATION_FAILURE"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_sha256(
    value: Any,
    field: str,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise WorkflowVerificationError(
            f"invalid SHA256: {field}"
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
        raise WorkflowVerificationError(
            f"invalid unsigned integer: {field}"
        )

    return value


def workflow_result_identity(
    result: dict[str, Any],
) -> str:
    preimage = {
        "workflow_result_version":
            result["workflow_result_version"],
        "workflow_profile":
            result["workflow_profile"],
        "operator_obligation":
            result["operator_obligation"],
        "corpus_id":
            result["corpus_id"],
        "source_manifest_id":
            result["source_manifest_id"],
        "runtime_index_id":
            result["runtime_index_id"],
        "query_result_id":
            result["query_result_id"],
        "bundle_root_sha256":
            result["bundle_root_sha256"],
        "bundle_manifest_sha256":
            result["bundle_manifest_sha256"],
        "query_hex":
            result["query_hex"],
        "query_length_bytes":
            result["query_length_bytes"],
        "query_sha256":
            result["query_sha256"],
        "max_offsets":
            result["max_offsets"],
        "document_count":
            result["document_count"],
        "total_source_bytes":
            result["total_source_bytes"],
        "match_count":
            result["match_count"],
        "returned_count":
            result["returned_count"],
        "bounded":
            result["bounded"],
        "offsets_complete":
            result["offsets_complete"],
        "case_layout":
            result["case_layout"],
        "one_command_workflow":
            result["one_command_workflow"],
        "atomic_publication":
            result["atomic_publication"],
        "source_directory_consumed_only_during_snapshot":
            result[
                "source_directory_consumed_only_during_snapshot"
            ],
        "bundle_self_contained":
            result["bundle_self_contained"],
        "repository_dependency_required_for_bundle":
            result[
                "repository_dependency_required_for_bundle"
            ],
        "network_dependency_required":
            result["network_dependency_required"],
    }

    return sha256_bytes(
        canonical_json_bytes(preimage)
    )


def require_real_directory(
    path: Path,
    field: str,
) -> Path:
    absolute = path.absolute()

    try:
        status = os.lstat(absolute)
    except OSError as error:
        raise WorkflowInputError(
            f"{field} unavailable: {error}"
        ) from error

    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
    ):
        raise WorkflowInputError(
            f"{field} must be a real directory"
        )

    return absolute.resolve()


def validate_source_output_relation(
    source: Path,
    output: Path,
) -> None:
    source_text = os.fspath(source)
    output_text = os.fspath(output)

    try:
        common = os.path.commonpath([
            source_text,
            output_text,
        ])
    except ValueError as error:
        raise WorkflowInputError(
            "invalid source/output relationship"
        ) from error

    if common == source_text:
        raise WorkflowInputError(
            "output must not be equal to or inside source"
        )

    if common == output_text:
        raise WorkflowInputError(
            "output must not be an ancestor of source"
        )


def verify_case_root(case_directory: Path) -> None:
    actual: set[str] = set()

    for entry in os.scandir(case_directory):
        status = entry.stat(
            follow_symlinks=False
        )

        if stat.S_ISLNK(status.st_mode):
            raise WorkflowVerificationError(
                f"case root symlink forbidden: {entry.name}"
            )

        actual.add(entry.name)

    if actual != CASE_ROOT_ENTRIES:
        raise WorkflowVerificationError(
            "case root coverage mismatch; "
            f"missing={sorted(CASE_ROOT_ENTRIES - actual)}; "
            f"extra={sorted(actual - CASE_ROOT_ENTRIES)}"
        )

    for directory_name in (
        CORPUS_DIRECTORY_NAME,
        BUNDLE_DIRECTORY_NAME,
    ):
        path = case_directory / directory_name
        status = os.lstat(path)

        if not stat.S_ISDIR(status.st_mode):
            raise WorkflowVerificationError(
                f"case payload must be directory: {directory_name}"
            )

    for filename in (
        WORKFLOW_RESULT_NAME,
        WORKFLOW_COMPLETE_NAME,
    ):
        path = case_directory / filename
        status = os.lstat(path)

        if not stat.S_ISREG(status.st_mode):
            raise WorkflowVerificationError(
                f"case payload must be regular: {filename}"
            )


def verify_operator_workflow(
    case_directory: Path,
) -> dict[str, Any]:
    case_directory = require_real_directory(
        case_directory,
        "case directory",
    )

    verify_case_root(case_directory)

    corpus_directory = (
        case_directory / CORPUS_DIRECTORY_NAME
    )
    bundle_directory = (
        case_directory / BUNDLE_DIRECTORY_NAME
    )

    result_path = (
        case_directory / WORKFLOW_RESULT_NAME
    )
    complete_path = (
        case_directory / WORKFLOW_COMPLETE_NAME
    )

    result = load_canonical_json(result_path)
    complete = load_canonical_json(complete_path)

    if set(result) != WORKFLOW_RESULT_KEYS:
        raise WorkflowVerificationError(
            "workflow result fields mismatch"
        )

    constants = {
        "ok": True,
        "workflow_result_version":
            WORKFLOW_RESULT_VERSION,
        "workflow_profile":
            WORKFLOW_PROFILE,
        "operator_obligation": "O5",
        "case_layout": {
            "corpus_directory":
                CORPUS_DIRECTORY_NAME,
            "bundle_directory":
                BUNDLE_DIRECTORY_NAME,
            "workflow_result":
                WORKFLOW_RESULT_NAME,
            "workflow_complete":
                WORKFLOW_COMPLETE_NAME,
        },
        "one_command_workflow": True,
        "atomic_publication": True,
        "source_directory_consumed_only_during_snapshot":
            True,
        "bundle_self_contained": True,
        "repository_dependency_required_for_bundle":
            False,
        "network_dependency_required": False,
        "next_operator_obligation":
            NEXT_OBLIGATION,
    }

    for field, expected in constants.items():
        if result.get(field) != expected:
            raise WorkflowVerificationError(
                f"workflow constant mismatch: {field}"
            )

    source = verify_snapshot(
        corpus_directory
    )

    runtime = verify_runtime_index(
        corpus_directory,
        require_current_binaries=False,
        rebuild=False,
    )

    bundle = verify_bundle(
        bundle_directory
    )

    artifact = load_bundle_json(
        bundle_directory / ARTIFACT_PATH
    )
    query = (
        bundle_directory / QUERY_PATH
    ).read_bytes()

    if not query:
        raise WorkflowVerificationError(
            "workflow query is empty"
        )

    expected_fields = {
        "corpus_id":
            source["corpus_id"],
        "source_manifest_id":
            source["source_manifest_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "query_result_id":
            artifact["query_result_id"],
        "bundle_root_sha256":
            bundle["bundle_root_sha256"],
        "bundle_manifest_sha256":
            bundle["bundle_manifest_sha256"],
        "query_hex":
            query.hex(),
        "query_length_bytes":
            len(query),
        "query_sha256":
            sha256_bytes(query),
        "max_offsets":
            artifact["max_offsets"],
        "document_count":
            source["document_count"],
        "total_source_bytes":
            source["total_source_bytes"],
        "match_count":
            artifact["match_count"],
        "returned_count":
            artifact["returned_count"],
        "bounded":
            artifact["bounded"],
        "offsets_complete":
            artifact["offsets_complete"],
    }

    for field, expected in expected_fields.items():
        if result.get(field) != expected:
            raise WorkflowVerificationError(
                f"workflow binding mismatch: {field}"
            )

    if (
        runtime["corpus_id"]
        != source["corpus_id"]
        or runtime["source_manifest_id"]
        != source["source_manifest_id"]
    ):
        raise WorkflowVerificationError(
            "runtime/source identity mismatch"
        )

    if (
        bundle["corpus_id"]
        != source["corpus_id"]
        or bundle["source_manifest_id"]
        != source["source_manifest_id"]
        or bundle["runtime_index_id"]
        != runtime["runtime_index_id"]
        or bundle["query_result_id"]
        != artifact["query_result_id"]
    ):
        raise WorkflowVerificationError(
            "bundle/case identity mismatch"
        )

    expected_result_id = (
        workflow_result_identity(result)
    )

    if (
        result["workflow_result_id"]
        != expected_result_id
    ):
        raise WorkflowVerificationError(
            "workflow result identity mismatch"
        )

    expected_complete = {
        "complete_version":
            WORKFLOW_COMPLETE_VERSION,
        "workflow_result_sha256":
            sha256_file(result_path),
        "workflow_result_id":
            expected_result_id,
        "corpus_id":
            source["corpus_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "query_result_id":
            artifact["query_result_id"],
        "bundle_root_sha256":
            bundle["bundle_root_sha256"],
    }

    if (
        set(complete) != WORKFLOW_COMPLETE_KEYS
        or complete != expected_complete
    ):
        raise WorkflowVerificationError(
            "workflow complete marker mismatch"
        )

    return {
        "ok": True,
        "workflow_result_version":
            WORKFLOW_RESULT_VERSION,
        "operator_obligation": "O5",
        "workflow_result_id":
            expected_result_id,
        "corpus_id":
            source["corpus_id"],
        "source_manifest_id":
            source["source_manifest_id"],
        "runtime_index_id":
            runtime["runtime_index_id"],
        "query_result_id":
            artifact["query_result_id"],
        "bundle_root_sha256":
            bundle["bundle_root_sha256"],
        "bundle_manifest_sha256":
            bundle["bundle_manifest_sha256"],
        "document_count":
            source["document_count"],
        "total_source_bytes":
            source["total_source_bytes"],
        "match_count":
            artifact["match_count"],
        "returned_count":
            artifact["returned_count"],
        "bounded":
            artifact["bounded"],
        "offsets_complete":
            artifact["offsets_complete"],
        "case_root_coverage_verified": True,
        "source_snapshot_verified": True,
        "runtime_index_verified": True,
        "bundle_replay_verified": True,
        "workflow_complete_verified": True,
        "next_operator_obligation":
            NEXT_OBLIGATION,
    }


def fsync_case_metadata(
    case_directory: Path,
) -> None:
    fsync_file(
        case_directory / WORKFLOW_RESULT_NAME
    )
    fsync_file(
        case_directory / WORKFLOW_COMPLETE_NAME
    )

    for root, directories, _ in os.walk(
        case_directory,
        topdown=False,
    ):
        for directory in directories:
            fsync_directory(
                Path(root) / directory
            )

        fsync_directory(Path(root))


def build_operator_workflow(
    source_directory: Path,
    output_directory: Path,
    query: bytes,
    *,
    max_offsets: int | None = None,
    test_fail_before_publication: bool = False,
) -> dict[str, Any]:
    if not query:
        raise WorkflowInputError("EMPTY_QUERY")

    if max_offsets is not None:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
            or max_offsets > (2**64 - 1)
        ):
            raise WorkflowInputError(
                "invalid max_offsets"
            )

    source_directory = require_real_directory(
        source_directory,
        "source directory",
    )

    output_directory = (
        output_directory.absolute()
    )

    if output_directory.exists():
        raise WorkflowInputError(
            "output path already exists"
        )

    validate_source_output_relation(
        source_directory,
        output_directory,
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
                "workflow-tmp."
            ),
            dir=parent,
        )
    )

    published = False

    try:
        corpus_directory = (
            temporary / CORPUS_DIRECTORY_NAME
        )
        bundle_directory = (
            temporary / BUNDLE_DIRECTORY_NAME
        )

        source = build_snapshot(
            source_directory,
            corpus_directory,
        )

        runtime = build_runtime_index(
            corpus_directory
        )

        query_result = execute_operator_query(
            corpus_directory,
            query,
            max_offsets=max_offsets,
        )

        build_bundle(
            corpus_directory,
            bundle_directory,
            query,
            max_offsets=max_offsets,
        )

        bundle = verify_bundle(
            bundle_directory
        )

        artifact = load_bundle_json(
            bundle_directory / ARTIFACT_PATH
        )

        if artifact != query_result:
            raise WorkflowVerificationError(
                "O3 result differs from O4 artifact"
            )

        result = {
            "ok": True,
            "workflow_result_version":
                WORKFLOW_RESULT_VERSION,
            "workflow_profile":
                WORKFLOW_PROFILE,
            "operator_obligation": "O5",
            "corpus_id":
                source["corpus_id"],
            "source_manifest_id":
                source["source_manifest_id"],
            "runtime_index_id":
                runtime["runtime_index_id"],
            "query_result_id":
                query_result["query_result_id"],
            "bundle_root_sha256":
                bundle["bundle_root_sha256"],
            "bundle_manifest_sha256":
                bundle["bundle_manifest_sha256"],
            "query_hex":
                query.hex(),
            "query_length_bytes":
                len(query),
            "query_sha256":
                sha256_bytes(query),
            "max_offsets":
                max_offsets,
            "document_count":
                source["document_count"],
            "total_source_bytes":
                source["total_source_bytes"],
            "match_count":
                query_result["match_count"],
            "returned_count":
                query_result["returned_count"],
            "bounded":
                query_result["bounded"],
            "offsets_complete":
                query_result["offsets_complete"],
            "case_layout": {
                "corpus_directory":
                    CORPUS_DIRECTORY_NAME,
                "bundle_directory":
                    BUNDLE_DIRECTORY_NAME,
                "workflow_result":
                    WORKFLOW_RESULT_NAME,
                "workflow_complete":
                    WORKFLOW_COMPLETE_NAME,
            },
            "one_command_workflow": True,
            "atomic_publication": True,
            "source_directory_consumed_only_during_snapshot":
                True,
            "bundle_self_contained": True,
            "repository_dependency_required_for_bundle":
                False,
            "network_dependency_required":
                False,
            "next_operator_obligation":
                NEXT_OBLIGATION,
        }

        result["workflow_result_id"] = (
            workflow_result_identity(result)
        )

        result_path = (
            temporary / WORKFLOW_RESULT_NAME
        )
        result_path.write_bytes(
            canonical_json_bytes(result)
        )
        result_path.chmod(0o644)

        complete = {
            "complete_version":
                WORKFLOW_COMPLETE_VERSION,
            "workflow_result_sha256":
                sha256_file(result_path),
            "workflow_result_id":
                result["workflow_result_id"],
            "corpus_id":
                result["corpus_id"],
            "runtime_index_id":
                result["runtime_index_id"],
            "query_result_id":
                result["query_result_id"],
            "bundle_root_sha256":
                result["bundle_root_sha256"],
        }

        complete_path = (
            temporary / WORKFLOW_COMPLETE_NAME
        )
        complete_path.write_bytes(
            canonical_json_bytes(complete)
        )
        complete_path.chmod(0o644)

        fsync_case_metadata(temporary)

        temporary_verification = (
            verify_operator_workflow(
                temporary
            )
        )

        if (
            temporary_verification[
                "workflow_result_id"
            ]
            != result["workflow_result_id"]
        ):
            raise WorkflowVerificationError(
                "temporary workflow verification mismatch"
            )

        if test_fail_before_publication:
            raise WorkflowError(
                "simulated interrupted workflow"
            )

        if output_directory.exists():
            raise WorkflowInputError(
                "output path appeared during workflow"
            )

        os.rename(
            temporary,
            output_directory,
        )
        published = True
        fsync_directory(parent)

        final_verification = (
            verify_operator_workflow(
                output_directory
            )
        )

        return {
            **final_verification,
            "output":
                str(output_directory),
            "atomic_publication_verified":
                True,
        }

    except Exception:
        if published:
            shutil.rmtree(
                output_directory,
                ignore_errors=True,
            )
        else:
            shutil.rmtree(
                temporary,
                ignore_errors=True,
            )

        raise


def command_run(
    args: argparse.Namespace,
) -> int:
    query = (
        read_stable_query_file(
            Path(args.query_file)
        )
        if args.query_file is not None
        else parse_query_hex(
            args.query_hex
        )
    )

    result = build_operator_workflow(
        Path(args.source),
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


def command_verify(
    args: argparse.Namespace,
) -> int:
    result = verify_operator_workflow(
        Path(args.case)
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

    run = subparsers.add_parser("run")
    run.add_argument(
        "--source",
        required=True,
    )
    run.add_argument(
        "--out",
        required=True,
    )

    query = run.add_mutually_exclusive_group(
        required=True
    )
    query.add_argument("--query-file")
    query.add_argument("--query-hex")

    run.add_argument(
        "--max-offsets",
        type=int,
    )
    run.set_defaults(handler=command_run)

    verify = subparsers.add_parser("verify")
    verify.add_argument(
        "--case",
        required=True,
    )
    verify.set_defaults(handler=command_verify)

    return parser


def main() -> int:
    parser = build_parser()

    try:
        args = parser.parse_args()
        return args.handler(args)

    except (
        WorkflowError,
        QueryError,
        OperatorError,
    ) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_code": getattr(
                        error,
                        "error_code",
                        "OPERATOR_WORKFLOW_FAILURE",
                    ),
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
