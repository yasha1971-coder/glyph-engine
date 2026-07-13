#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_manifest_v1 import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)
from glyph_operator_workflow_v1 import (  # noqa: E402
    BUNDLE_DIRECTORY_NAME,
    CORPUS_DIRECTORY_NAME,
    WORKFLOW_COMPLETE_NAME,
    WORKFLOW_COMPLETE_VERSION,
    WORKFLOW_RESULT_NAME,
    build_operator_workflow,
    validate_source_output_relation,
    verify_operator_workflow,
    workflow_result_identity,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_WORKFLOW_V1.json"
)

WORKFLOW_SCRIPT = (
    TOOLS / "glyph_operator_workflow_v1.py"
)


class GateError(RuntimeError):
    pass


def write_bytes(
    root: Path,
    relative: bytes,
    data: bytes,
) -> None:
    root_bytes = os.fsencode(root)
    components = relative.split(b"/")
    current = root_bytes

    for component in components[:-1]:
        current = os.path.join(
            current,
            component,
        )
        os.makedirs(
            current,
            exist_ok=True,
        )

    path = os.path.join(
        current,
        components[-1],
    )

    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_TRUNC,
        0o644,
    )

    try:
        view = memoryview(data)

        while view:
            written = os.write(
                descriptor,
                view,
            )

            if written <= 0:
                raise GateError(
                    "failed to write fixture payload"
                )

            view = view[written:]

    finally:
        os.close(descriptor)


def create_source_tree(root: Path) -> None:
    root.mkdir(parents=True)

    write_bytes(
        root,
        b"00-empty.bin",
        b"",
    )
    write_bytes(
        root,
        b"10-alpha.bin",
        b"prefix-A\x00B-suffix-A\x00B",
    )
    write_bytes(
        root,
        b"20-all-bytes.bin",
        bytes(range(256)),
    )
    write_bytes(
        root,
        b"nested/30-duplicate-a.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/31-duplicate-b.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/invalid-name-\xff.bin",
        b"\x80\x81A\x00B\xfe\xff",
    )


def stable_environment() -> dict[str, str]:
    return {
        "PATH": os.environ.get(
            "PATH",
            "/usr/bin:/bin",
        ),
        "LANG": "C",
        "LC_ALL": "C",
    }


def run_workflow_cli(
    arguments: list[str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(WORKFLOW_SCRIPT),
            *arguments,
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3600,
        check=False,
        env=stable_environment(),
    )

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            "workflow CLI returned invalid JSON"
            f"\nexit_code={result.returncode}"
            f"\nstdout={result.stdout[-4000:]}"
            f"\nstderr={result.stderr[-4000:]}"
        ) from error

    if not isinstance(value, dict):
        raise GateError(
            "workflow CLI JSON root is not object"
        )

    return result, value


def require_cli_success(
    arguments: list[str],
) -> dict[str, Any]:
    process, value = run_workflow_cli(
        arguments
    )

    if (
        process.returncode != 0
        or value.get("ok") is not True
    ):
        raise GateError(
            "workflow CLI unexpectedly failed"
            f"\nexit_code={process.returncode}"
            f"\nstdout={process.stdout[-4000:]}"
            f"\nstderr={process.stderr[-4000:]}"
        )

    return value


def expect_cli_failure(
    name: str,
    arguments: list[str],
) -> dict[str, Any]:
    process, value = run_workflow_cli(
        arguments
    )

    if process.returncode == 0:
        raise GateError(
            f"mutation unexpectedly accepted: {name}"
        )

    return {
        "mutation": name,
        "rejected": True,
        "error_code": value.get(
            "error_code",
            "UNCLASSIFIED_FAILURE",
        ),
    }


def expect_failure(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        function()

    except Exception as error:
        return {
            "mutation": name,
            "rejected": True,
            "error_type":
                type(error).__name__,
        }

    raise GateError(
        f"mutation unexpectedly accepted: {name}"
    )


def tree_snapshot(
    root: Path,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        status = os.lstat(path)

        if stat.S_ISLNK(status.st_mode):
            result[relative] = {
                "kind": "symlink",
                "target": os.readlink(path),
            }
            continue

        if stat.S_ISDIR(status.st_mode):
            result[relative] = {
                "kind": "directory",
            }
            continue

        if not stat.S_ISREG(status.st_mode):
            raise GateError(
                "unexpected case payload type: "
                + relative
            )

        result[relative] = {
            "kind": "file",
            "bytes": path.read_bytes(),
            "executable": bool(
                status.st_mode
                & (
                    stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH
                )
            ),
        }

    return result


def run_bundled_replay(
    bundle: Path,
) -> dict[str, Any]:
    script = bundle / "replay.py"

    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
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
        env=stable_environment(),
    )

    if result.returncode != 0:
        raise GateError(
            "standalone bundle replay failed"
            f"\nstdout={result.stdout[-4000:]}"
            f"\nstderr={result.stderr[-4000:]}"
        )

    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            "standalone replay returned invalid JSON"
        ) from error

    if (
        not isinstance(value, dict)
        or value.get("ok") is not True
    ):
        raise GateError(
            "standalone replay did not return ok"
        )

    return value


def mutation_copy(
    pristine: Path,
    work: Path,
    name: str,
) -> Path:
    destination = work / name

    shutil.copytree(
        pristine,
        destination,
        symlinks=True,
    )

    return destination


def first_nonempty_file(
    directory: Path,
) -> Path:
    for path in sorted(directory.rglob("*")):
        if (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
        ):
            return path

    raise GateError(
        "non-empty mutation payload not found"
    )


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())

    if not data:
        raise GateError(
            "cannot mutate empty payload"
        )

    data[-1] ^= 0x01
    path.write_bytes(data)


def rewrite_workflow_result(
    case: Path,
    result: dict[str, Any],
) -> None:
    result["workflow_result_id"] = (
        workflow_result_identity(result)
    )

    result_path = (
        case / WORKFLOW_RESULT_NAME
    )
    result_path.write_bytes(
        canonical_json_bytes(result)
    )

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

    (
        case / WORKFLOW_COMPLETE_NAME
    ).write_bytes(
        canonical_json_bytes(complete)
    )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-operator-o5-gate-"
    ) as temporary:
        work = Path(temporary)

        source_file = work / "source-file"
        source_hex = work / "source-hex"
        source_guard = work / "source-guard"

        case_file = work / "case-file"
        case_hex = work / "case-hex"

        query_file = work / "query.bin"
        query = b"A\x00B"

        create_source_tree(source_file)
        create_source_tree(source_hex)
        create_source_tree(source_guard)

        query_file.write_bytes(query)

        run_file = require_cli_success([
            "run",
            "--source",
            str(source_file),
            "--query-file",
            str(query_file),
            "--max-offsets",
            "2",
            "--out",
            str(case_file),
        ])

        run_hex = require_cli_success([
            "run",
            "--source",
            str(source_hex),
            "--query-hex",
            query.hex(),
            "--max-offsets",
            "2",
            "--out",
            str(case_hex),
        ])

        if tree_snapshot(
            case_file
        ) != tree_snapshot(case_hex):
            raise GateError(
                "query-file and query-hex "
                "produced different case trees"
            )

        identity_fields = [
            "workflow_result_id",
            "corpus_id",
            "source_manifest_id",
            "runtime_index_id",
            "query_result_id",
            "bundle_root_sha256",
            "bundle_manifest_sha256",
            "document_count",
            "total_source_bytes",
            "match_count",
            "returned_count",
            "bounded",
            "offsets_complete",
        ]

        for field in identity_fields:
            if run_file[field] != run_hex[field]:
                raise GateError(
                    "query transport result mismatch: "
                    + field
                )

        if run_file["match_count"] != 3:
            raise GateError(
                "unexpected complete match count"
            )

        if run_file["returned_count"] != 2:
            raise GateError(
                "unexpected bounded returned count"
            )

        if run_file["bounded"] is not True:
            raise GateError(
                "bounded workflow result expected"
            )

        initial_verify_file = (
            require_cli_success([
                "verify",
                "--case",
                str(case_file),
            ])
        )

        initial_verify_hex = (
            require_cli_success([
                "verify",
                "--case",
                str(case_hex),
            ])
        )

        if (
            initial_verify_file[
                "workflow_result_id"
            ]
            != run_file[
                "workflow_result_id"
            ]
            or initial_verify_hex[
                "workflow_result_id"
            ]
            != run_hex[
                "workflow_result_id"
            ]
        ):
            raise GateError(
                "initial case verification mismatch"
            )

        portable_bundle = (
            work / "portable-bundle"
        )

        shutil.copytree(
            case_file / BUNDLE_DIRECTORY_NAME,
            portable_bundle,
        )

        shutil.rmtree(source_file)
        shutil.rmtree(source_hex)
        query_file.unlink()

        verify_after_removal_file = (
            require_cli_success([
                "verify",
                "--case",
                str(case_file),
            ])
        )

        verify_after_removal_hex = (
            require_cli_success([
                "verify",
                "--case",
                str(case_hex),
            ])
        )

        portable_replay = (
            run_bundled_replay(
                portable_bundle
            )
        )

        if (
            verify_after_removal_file[
                "workflow_result_id"
            ]
            != run_file[
                "workflow_result_id"
            ]
            or verify_after_removal_hex[
                "workflow_result_id"
            ]
            != run_hex[
                "workflow_result_id"
            ]
        ):
            raise GateError(
                "case changed after source removal"
            )

        if (
            portable_replay[
                "query_result_id"
            ]
            != run_file["query_result_id"]
            or portable_replay[
                "bundle_root_sha256"
            ]
            != run_file[
                "bundle_root_sha256"
            ]
        ):
            raise GateError(
                "portable bundle replay identity mismatch"
            )

        mutations: list[dict[str, Any]] = []

        existing_output = work / "existing-output"
        existing_output.mkdir()

        mutations.append(
            expect_failure(
                "existing_output_rejected",
                lambda: build_operator_workflow(
                    source_guard,
                    existing_output,
                    query,
                    max_offsets=2,
                ),
            )
        )

        mutations.append(
            expect_failure(
                "output_inside_source_rejected",
                lambda:
                    validate_source_output_relation(
                        source_guard.resolve(),
                        (
                            source_guard
                            / "nested-output"
                        ).absolute(),
                    ),
            )
        )

        mutations.append(
            expect_failure(
                "output_ancestor_of_source_rejected",
                lambda:
                    validate_source_output_relation(
                        source_guard.resolve(),
                        source_guard.parent.resolve(),
                    ),
            )
        )

        symlink_source = work / "symlink-source"
        symlink_source.symlink_to(
            source_guard,
            target_is_directory=True,
        )

        mutations.append(
            expect_failure(
                "symlink_source_rejected",
                lambda: build_operator_workflow(
                    symlink_source,
                    work / "symlink-source-case",
                    query,
                    max_offsets=2,
                ),
            )
        )

        mutations.append(
            expect_failure(
                "empty_query_rejected",
                lambda: build_operator_workflow(
                    source_guard,
                    work / "empty-query-case",
                    b"",
                    max_offsets=2,
                ),
            )
        )

        mutations.append(
            expect_failure(
                "negative_max_offsets_rejected",
                lambda: build_operator_workflow(
                    source_guard,
                    work / "negative-limit-case",
                    query,
                    max_offsets=-1,
                ),
            )
        )

        mutations.append(
            expect_failure(
                "max_offsets_overflow_rejected",
                lambda: build_operator_workflow(
                    source_guard,
                    work / "overflow-limit-case",
                    query,
                    max_offsets=2**64,
                ),
            )
        )

        interrupted_output = (
            work / "interrupted-case"
        )

        mutations.append(
            expect_failure(
                "interrupted_workflow_rejected",
                lambda: build_operator_workflow(
                    source_guard,
                    interrupted_output,
                    query,
                    max_offsets=2,
                    test_fail_before_publication=True,
                ),
            )
        )

        if interrupted_output.exists():
            raise GateError(
                "interrupted workflow published output"
            )

        extra_root = mutation_copy(
            case_file,
            work,
            "mutation-extra-root",
        )
        (
            extra_root / "undeclared.bin"
        ).write_bytes(b"extra")

        mutations.append(
            expect_failure(
                "extra_case_root_payload_rejected",
                lambda: verify_operator_workflow(
                    extra_root
                ),
            )
        )

        missing_result = mutation_copy(
            case_file,
            work,
            "mutation-missing-result",
        )
        (
            missing_result
            / WORKFLOW_RESULT_NAME
        ).unlink()

        mutations.append(
            expect_failure(
                "missing_workflow_result_rejected",
                lambda: verify_operator_workflow(
                    missing_result
                ),
            )
        )

        missing_bundle = mutation_copy(
            case_file,
            work,
            "mutation-missing-bundle",
        )
        shutil.rmtree(
            missing_bundle
            / BUNDLE_DIRECTORY_NAME
        )

        mutations.append(
            expect_failure(
                "missing_bundle_directory_rejected",
                lambda: verify_operator_workflow(
                    missing_bundle
                ),
            )
        )

        noncanonical_result = mutation_copy(
            case_file,
            work,
            "mutation-noncanonical-result",
        )
        result_path = (
            noncanonical_result
            / WORKFLOW_RESULT_NAME
        )
        result_value = json.loads(
            result_path.read_text()
        )
        result_path.write_text(
            json.dumps(
                result_value,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        mutations.append(
            expect_failure(
                "noncanonical_workflow_result_rejected",
                lambda: verify_operator_workflow(
                    noncanonical_result
                ),
            )
        )

        wrong_complete = mutation_copy(
            case_file,
            work,
            "mutation-wrong-complete",
        )
        complete_path = (
            wrong_complete
            / WORKFLOW_COMPLETE_NAME
        )
        complete_value = json.loads(
            complete_path.read_text()
        )
        complete_value[
            "workflow_result_sha256"
        ] = "0" * 64
        complete_path.write_bytes(
            canonical_json_bytes(
                complete_value
            )
        )

        mutations.append(
            expect_failure(
                "wrong_complete_marker_rejected",
                lambda: verify_operator_workflow(
                    wrong_complete
                ),
            )
        )

        wrong_query_binding = mutation_copy(
            case_file,
            work,
            "mutation-wrong-query-binding",
        )
        result_path = (
            wrong_query_binding
            / WORKFLOW_RESULT_NAME
        )
        result_value = json.loads(
            result_path.read_text()
        )
        result_value[
            "query_result_id"
        ] = "0" * 64
        rewrite_workflow_result(
            wrong_query_binding,
            result_value,
        )

        mutations.append(
            expect_failure(
                "wrong_query_result_binding_rejected",
                lambda: verify_operator_workflow(
                    wrong_query_binding
                ),
            )
        )

        wrong_match_count = mutation_copy(
            case_file,
            work,
            "mutation-wrong-match-count",
        )
        result_path = (
            wrong_match_count
            / WORKFLOW_RESULT_NAME
        )
        result_value = json.loads(
            result_path.read_text()
        )
        result_value["match_count"] += 1
        rewrite_workflow_result(
            wrong_match_count,
            result_value,
        )

        mutations.append(
            expect_failure(
                "wrong_match_count_binding_rejected",
                lambda: verify_operator_workflow(
                    wrong_match_count
                ),
            )
        )

        altered_corpus = mutation_copy(
            case_file,
            work,
            "mutation-altered-corpus",
        )
        corpus_payload = first_nonempty_file(
            altered_corpus
            / CORPUS_DIRECTORY_NAME
            / "documents"
        )
        flip_last_byte(corpus_payload)

        mutations.append(
            expect_failure(
                "altered_corpus_payload_rejected",
                lambda: verify_operator_workflow(
                    altered_corpus
                ),
            )
        )

        altered_runtime = mutation_copy(
            case_file,
            work,
            "mutation-altered-runtime",
        )
        runtime_candidates = sorted(
            (
                altered_runtime
                / CORPUS_DIRECTORY_NAME
                / "runtime_index_v1"
            ).rglob("fm.bin")
        )

        if not runtime_candidates:
            raise GateError(
                "runtime mutation payload missing"
            )

        flip_last_byte(
            runtime_candidates[0]
        )

        mutations.append(
            expect_failure(
                "altered_runtime_index_rejected",
                lambda: verify_operator_workflow(
                    altered_runtime
                ),
            )
        )

        altered_bundle = mutation_copy(
            case_file,
            work,
            "mutation-altered-bundle",
        )
        bundle_query = (
            altered_bundle
            / BUNDLE_DIRECTORY_NAME
            / "query/query.bin"
        )
        flip_last_byte(bundle_query)

        mutations.append(
            expect_failure(
                "altered_bundle_payload_rejected",
                lambda: verify_operator_workflow(
                    altered_bundle
                ),
            )
        )

        symlink_result = mutation_copy(
            case_file,
            work,
            "mutation-symlink-result",
        )
        external_result = (
            work / "external-workflow-result.json"
        )
        external_result.write_bytes(
            (
                symlink_result
                / WORKFLOW_RESULT_NAME
            ).read_bytes()
        )
        (
            symlink_result
            / WORKFLOW_RESULT_NAME
        ).unlink()
        (
            symlink_result
            / WORKFLOW_RESULT_NAME
        ).symlink_to(external_result)

        mutations.append(
            expect_failure(
                "symlink_workflow_result_rejected",
                lambda: verify_operator_workflow(
                    symlink_result
                ),
            )
        )

        if len(mutations) != 19:
            raise GateError(
                "unexpected mutation count: "
                + str(len(mutations))
            )

        if not all(
            mutation["rejected"] is True
            for mutation in mutations
        ):
            raise GateError(
                "O5 mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_WORKFLOW_GATE_V1",
            "operator_obligation": "O5",
            "workflow_result_version":
                "GLYPH_OPERATOR_WORKFLOW_RESULT_V1",
            "workflow_profile":
                "GLYPH_OPERATOR_PATH_V1",
            "one_command_workflow_verified":
                True,
            "o1_snapshot_executed": True,
            "o2_runtime_index_executed": True,
            "o3_binary_query_executed": True,
            "o4_evidence_bundle_executed": True,
            "query_file_cli_verified": True,
            "query_hex_cli_verified": True,
            "query_file_equals_query_hex":
                True,
            "embedded_nul_query_verified": True,
            "bounded_result_verified": True,
            "deterministic_case_tree_verified":
                True,
            "case_verify_verified": True,
            "source_removed_before_verify":
                True,
            "query_file_removed_before_verify":
                True,
            "bundle_replay_outside_repository_verified":
                True,
            "exact_case_root_coverage_verified":
                True,
            "source_snapshot_verified": True,
            "runtime_index_verified": True,
            "bundle_replay_verified": True,
            "workflow_complete_verified": True,
            "atomic_publication_verified": True,
            "interrupted_workflow_rejected":
                True,
            "invalid_utf8_filename_supported":
                True,
            "network_dependency_required":
                False,
            "bundle_repository_dependency_required":
                False,
            "document_count":
                run_file["document_count"],
            "total_source_bytes":
                run_file["total_source_bytes"],
            "match_count":
                run_file["match_count"],
            "returned_count":
                run_file["returned_count"],
            "bounded":
                run_file["bounded"],
            "workflow_result_id":
                run_file["workflow_result_id"],
            "query_result_id":
                run_file["query_result_id"],
            "runtime_index_id":
                run_file["runtime_index_id"],
            "bundle_root_sha256":
                run_file["bundle_root_sha256"],
            "mutation_count":
                len(mutations),
            "mutations": mutations,
            "next_operator_obligation":
                "O6_OPERATOR_CONFORMANCE_CLOSURE",
        }

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUT.write_bytes(
            canonical_json_bytes(output)
        )

        print(
            json.dumps(
                output,
                indent=2,
                sort_keys=True,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
