#!/usr/bin/env python3
from __future__ import annotations

import copy
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

from glyph_operator_bundle_replay_v1 import (  # noqa: E402
    BUNDLE_COMPLETE_VERSION,
    COMPLETE_NAME,
    MANIFEST_NAME,
    bundle_root_identity,
    canonical_json_bytes,
    sha256_file,
)
from glyph_operator_bundle_v1 import (  # noqa: E402
    build_bundle,
)
from glyph_operator_index_v1 import (  # noqa: E402
    build_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    build_snapshot,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1.json"
)

EXTERNAL_REPLAY = (
    TOOLS
    / "glyph_operator_bundle_replay_v1.py"
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
        b"10-first.bin",
        b"A\x00B--A\x00B",
    )
    write_bytes(
        root,
        b"20-all-bytes.bin",
        bytes(range(256)),
    )
    write_bytes(
        root,
        b"30-left.bin",
        b"LEFT-AB",
    )
    write_bytes(
        root,
        b"31-right.bin",
        b"CD-RIGHT",
    )
    write_bytes(
        root,
        b"nested/40-duplicate-a.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/41-duplicate-b.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/invalid-name-\xff.bin",
        b"\x80\x81A\x00B\xfe\xff",
    )


def build_corpus(
    source: Path,
    corpus: Path,
) -> None:
    build_snapshot(source, corpus)
    build_runtime_index(corpus)


def run_replay(
    bundle: Path,
    *,
    bundled: bool,
) -> subprocess.CompletedProcess[str]:
    script = (
        bundle / "replay.py"
        if bundled
        else EXTERNAL_REPLAY
    )

    return subprocess.run(
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
        env={
            "PATH": os.environ.get(
                "PATH",
                "/usr/bin:/bin",
            ),
            "LANG": "C",
            "LC_ALL": "C",
        },
    )


def expect_replay_failure(
    name: str,
    bundle: Path,
) -> dict[str, Any]:
    result = run_replay(
        bundle,
        bundled=False,
    )

    if result.returncode == 0:
        raise GateError(
            "mutation unexpectedly accepted: "
            + name
        )

    message = result.stdout.strip()

    return {
        "mutation": name,
        "rejected": True,
        "exit_code": result.returncode,
        "message": message[-2000:],
    }


def tree_snapshot(
    root: Path,
) -> dict[str, tuple[bytes, bool]]:
    result = {}

    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            status = path.stat()
            executable = bool(
                status.st_mode
                & (
                    stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH
                )
            )

            result[
                path.relative_to(root).as_posix()
            ] = (
                path.read_bytes(),
                executable,
            )

    return result


def refresh_bundle_integrity(
    bundle: Path,
) -> None:
    manifest_path = bundle / MANIFEST_NAME
    manifest = json.loads(
        manifest_path.read_text()
    )

    for entry in manifest["files"]:
        path = bundle / entry["path"]

        if path.exists() and not path.is_symlink():
            status = path.stat()
            entry["size_bytes"] = status.st_size
            entry["sha256"] = sha256_file(path)
            entry["executable"] = bool(
                status.st_mode
                & (
                    stat.S_IXUSR
                    | stat.S_IXGRP
                    | stat.S_IXOTH
                )
            )

    manifest["files"].sort(
        key=lambda item: item["path"]
    )
    manifest["file_count"] = len(
        manifest["files"]
    )
    manifest["bundle_root_sha256"] = (
        bundle_root_identity(
            manifest["files"]
        )
    )

    manifest_path.write_bytes(
        canonical_json_bytes(manifest)
    )

    complete = {
        "complete_version":
            BUNDLE_COMPLETE_VERSION,
        "bundle_manifest_sha256":
            sha256_file(manifest_path),
        "bundle_root_sha256":
            manifest[
                "bundle_root_sha256"
            ],
        "query_result_id":
            manifest["query_result_id"],
        "runtime_index_id":
            manifest["runtime_index_id"],
        "corpus_id":
            manifest["corpus_id"],
        "file_count":
            manifest["file_count"],
    }

    (bundle / COMPLETE_NAME).write_bytes(
        canonical_json_bytes(complete)
    )


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


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-operator-o4-"
    ) as temporary:
        work = Path(temporary)

        source_one = work / "source-one"
        source_two = work / "source-two"
        corpus_one = work / "corpus-one"
        corpus_two = work / "corpus-two"
        bundle_one = work / "bundle-one"
        bundle_two = work / "bundle-two"

        create_source_tree(source_one)
        create_source_tree(source_two)

        build_corpus(source_one, corpus_one)
        build_corpus(source_two, corpus_two)

        query = b"A\x00B"

        first = build_bundle(
            corpus_one,
            bundle_one,
            query,
            max_offsets=2,
        )
        second = build_bundle(
            corpus_two,
            bundle_two,
            query,
            max_offsets=2,
        )

        if tree_snapshot(
            bundle_one
        ) != tree_snapshot(bundle_two):
            raise GateError(
                "equivalent inputs produced "
                "different bundles"
            )

        first_replay = run_replay(
            bundle_one,
            bundled=True,
        )

        if first_replay.returncode != 0:
            raise GateError(
                "bundled replay failed:\n"
                + first_replay.stdout
                + first_replay.stderr
            )

        portable = work / "portable-copy"
        shutil.copytree(
            bundle_one,
            portable,
        )

        shutil.rmtree(source_one)
        shutil.rmtree(source_two)
        shutil.rmtree(corpus_one)
        shutil.rmtree(corpus_two)
        shutil.rmtree(bundle_one)
        shutil.rmtree(bundle_two)

        portable_replay = run_replay(
            portable,
            bundled=True,
        )

        if portable_replay.returncode != 0:
            raise GateError(
                "portable copied replay failed:\n"
                + portable_replay.stdout
                + portable_replay.stderr
            )

        portable_result = json.loads(
            portable_replay.stdout
        )

        if portable_result.get("ok") is not True:
            raise GateError(
                "portable replay did not return ok"
            )

        mutations: list[dict[str, Any]] = []

        missing_artifact = mutation_copy(
            portable,
            work,
            "mutation-missing-artifact",
        )
        (
            missing_artifact
            / "artifact/query_result_v1.json"
        ).unlink()
        mutations.append(
            expect_replay_failure(
                "missing_query_artifact",
                missing_artifact,
            )
        )

        missing_replay = mutation_copy(
            portable,
            work,
            "mutation-missing-replay",
        )
        (missing_replay / "replay.py").unlink()
        mutations.append(
            expect_replay_failure(
                "missing_replay_code",
                missing_replay,
            )
        )

        extra_file = mutation_copy(
            portable,
            work,
            "mutation-extra-file",
        )
        (extra_file / "undeclared.bin").write_bytes(
            b"extra"
        )
        mutations.append(
            expect_replay_failure(
                "undeclared_extra_file",
                extra_file,
            )
        )

        altered_query = mutation_copy(
            portable,
            work,
            "mutation-altered-query",
        )
        query_path = altered_query / "query/query.bin"
        query_data = bytearray(
            query_path.read_bytes()
        )
        query_data[-1] ^= 0x01
        query_path.write_bytes(query_data)
        mutations.append(
            expect_replay_failure(
                "altered_query_bytes",
                altered_query,
            )
        )

        altered_source = mutation_copy(
            portable,
            work,
            "mutation-altered-source",
        )
        source_path = (
            altered_source
            / "source/documents/"
            "doc_00000001.bin"
        )
        source_data = bytearray(
            source_path.read_bytes()
        )
        source_data[0] ^= 0x01
        source_path.write_bytes(source_data)
        mutations.append(
            expect_replay_failure(
                "altered_source_document",
                altered_source,
            )
        )

        altered_runtime = mutation_copy(
            portable,
            work,
            "mutation-altered-runtime",
        )
        fm_path = (
            altered_runtime
            / "runtime/documents/"
            "doc_00000001/fm.bin"
        )
        fm_data = bytearray(
            fm_path.read_bytes()
        )
        fm_data[-1] ^= 0x01
        fm_path.write_bytes(fm_data)
        mutations.append(
            expect_replay_failure(
                "altered_runtime_index",
                altered_runtime,
            )
        )

        altered_query_binary = mutation_copy(
            portable,
            work,
            "mutation-query-binary",
        )
        binary_path = (
            altered_query_binary
            / "bin/query_fm_binary_v1"
        )
        binary_data = bytearray(
            binary_path.read_bytes()
        )
        binary_data[-1] ^= 0x01
        binary_path.write_bytes(binary_data)
        mutations.append(
            expect_replay_failure(
                "altered_query_binary",
                altered_query_binary,
            )
        )

        altered_builder = mutation_copy(
            portable,
            work,
            "mutation-builder-binary",
        )
        builder_path = (
            altered_builder
            / "bin/build_fm_binary_v1"
        )
        builder_data = bytearray(
            builder_path.read_bytes()
        )
        builder_data[-1] ^= 0x01
        builder_path.write_bytes(builder_data)
        mutations.append(
            expect_replay_failure(
                "altered_builder_binary",
                altered_builder,
            )
        )

        non_executable = mutation_copy(
            portable,
            work,
            "mutation-non-executable",
        )
        executable_path = (
            non_executable
            / "bin/query_fm_binary_v1"
        )
        executable_path.chmod(0o644)
        mutations.append(
            expect_replay_failure(
                "non_executable_query_binary",
                non_executable,
            )
        )

        symlink_payload = mutation_copy(
            portable,
            work,
            "mutation-symlink",
        )
        target = (
            work / "external-source.bin"
        )
        original = (
            symlink_payload
            / "source/documents/"
            "doc_00000001.bin"
        )
        target.write_bytes(
            original.read_bytes()
        )
        original.unlink()
        original.symlink_to(target)
        mutations.append(
            expect_replay_failure(
                "symlink_source_payload",
                symlink_payload,
            )
        )

        wrong_result_id = mutation_copy(
            portable,
            work,
            "mutation-result-id",
        )
        artifact_path = (
            wrong_result_id
            / "artifact/query_result_v1.json"
        )
        artifact = json.loads(
            artifact_path.read_text()
        )
        artifact["query_result_id"] = "0" * 64
        artifact_path.write_bytes(
            canonical_json_bytes(artifact)
        )
        refresh_bundle_integrity(
            wrong_result_id
        )
        mutations.append(
            expect_replay_failure(
                "wrong_query_result_id",
                wrong_result_id,
            )
        )

        wrong_mapping = mutation_copy(
            portable,
            work,
            "mutation-source-mapping",
        )
        artifact_path = (
            wrong_mapping
            / "artifact/query_result_v1.json"
        )
        artifact = json.loads(
            artifact_path.read_text()
        )

        if artifact["coordinates"]:
            artifact["coordinates"][0][
                "relative_path_bytes_hex"
            ] = b"wrong/path.bin".hex()

        artifact_path.write_bytes(
            canonical_json_bytes(artifact)
        )
        refresh_bundle_integrity(
            wrong_mapping
        )
        mutations.append(
            expect_replay_failure(
                "wrong_source_path_mapping",
                wrong_mapping,
            )
        )

        noncanonical_artifact = mutation_copy(
            portable,
            work,
            "mutation-noncanonical-artifact",
        )
        artifact_path = (
            noncanonical_artifact
            / "artifact/query_result_v1.json"
        )
        artifact = json.loads(
            artifact_path.read_text()
        )
        artifact_path.write_text(
            json.dumps(
                artifact,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        refresh_bundle_integrity(
            noncanonical_artifact
        )
        mutations.append(
            expect_replay_failure(
                "noncanonical_query_artifact",
                noncanonical_artifact,
            )
        )

        wrong_complete = mutation_copy(
            portable,
            work,
            "mutation-complete-marker",
        )
        complete_path = (
            wrong_complete / COMPLETE_NAME
        )
        complete = json.loads(
            complete_path.read_text()
        )
        complete[
            "bundle_root_sha256"
        ] = "0" * 64
        complete_path.write_bytes(
            canonical_json_bytes(complete)
        )
        mutations.append(
            expect_replay_failure(
                "wrong_complete_marker",
                wrong_complete,
            )
        )

        unsafe_manifest = mutation_copy(
            portable,
            work,
            "mutation-unsafe-path",
        )
        manifest_path = (
            unsafe_manifest / MANIFEST_NAME
        )
        manifest = json.loads(
            manifest_path.read_text()
        )
        manifest["files"][0]["path"] = (
            "../outside.bin"
        )
        manifest_path.write_bytes(
            canonical_json_bytes(manifest)
        )
        mutations.append(
            expect_replay_failure(
                "unsafe_manifest_path",
                unsafe_manifest,
            )
        )

        interrupted_source = (
            work / "interrupted-source"
        )
        interrupted_corpus = (
            work / "interrupted-corpus"
        )
        interrupted_output = (
            work / "interrupted-bundle"
        )

        create_source_tree(
            interrupted_source
        )
        build_corpus(
            interrupted_source,
            interrupted_corpus,
        )

        interrupted_rejected = False

        try:
            build_bundle(
                interrupted_corpus,
                interrupted_output,
                query,
                max_offsets=2,
                test_fail_before_publication=True,
            )
        except Exception:
            interrupted_rejected = True

        if not interrupted_rejected:
            raise GateError(
                "interrupted bundle build accepted"
            )

        if interrupted_output.exists():
            raise GateError(
                "interrupted bundle was published"
            )

        mutations.append({
            "mutation":
                "interrupted_bundle_not_published",
            "rejected": True,
            "exit_code": 1,
            "message":
                "simulated interrupted build rejected",
        })

        if not all(
            item["rejected"] is True
            for item in mutations
        ):
            raise GateError(
                "O4 mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_EVIDENCE_BUNDLE_GATE_V1",
            "operator_obligation": "O4",
            "bundle_version":
                "GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1",
            "deterministic_bundle_verified":
                True,
            "self_contained": True,
            "source_documents_bundled": True,
            "runtime_indexes_bundled": True,
            "runtime_builder_binaries_bundled":
                True,
            "query_binaries_bundled": True,
            "query_bytes_bundled": True,
            "query_artifact_bundled": True,
            "replay_code_bundled": True,
            "exact_manifest_coverage_verified":
                True,
            "payload_hashes_verified": True,
            "bundle_root_verified": True,
            "source_manifest_verified": True,
            "runtime_manifest_verified": True,
            "compiled_query_replay_verified":
                True,
            "copied_bundle_replay_verified":
                True,
            "replay_outside_repository_verified":
                True,
            "original_source_paths_removed_before_replay":
                True,
            "atomic_publication_verified": True,
            "interrupted_build_rejected": True,
            "repository_dependency_required":
                False,
            "network_dependency_required":
                False,
            "external_data_dependencies": [],
            "document_count":
                portable_result[
                    "document_count"
                ],
            "match_count":
                portable_result["match_count"],
            "returned_count":
                portable_result[
                    "returned_count"
                ],
            "file_count": first["file_count"],
            "bundle_root_sha256":
                first["bundle_root_sha256"],
            "query_result_id":
                first["query_result_id"],
            "runtime_index_id":
                first["runtime_index_id"],
            "mutation_count": len(mutations),
            "mutations": mutations,
            "next_operator_obligation":
                "O5_ONE_COMMAND_OPERATOR_WORKFLOW",
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
