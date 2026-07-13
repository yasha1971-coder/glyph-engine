#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parent

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


from glyph_binary_runtime_bundle_v1 import (
    build_bundle,
)

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_BINARY_RUNTIME_BUNDLE_V1.json"
)


class GateError(RuntimeError):
    pass


def run_replay(
    bundle: Path,
    *,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [
            "python3",
            "-I",
            str(bundle / "replay.py"),
            "--bundle",
            str(bundle),
        ],
        cwd=bundle.parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=300,
        check=False,
    )

    if expect_success and result.returncode != 0:
        raise GateError(
            f"bundle replay failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if (
        not expect_success
        and result.returncode == 0
    ):
        raise GateError(
            "mutated bundle unexpectedly replayed"
        )

    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())

    if not isinstance(value, dict):
        raise GateError(
            "JSON root must be an object"
        )

    return value


def save_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    )


def copy_mutation(
    original: Path,
    root: Path,
    name: str,
) -> Path:
    destination = root / name
    shutil.copytree(original, destination)
    return destination


def expect_mutation_failure(
    original: Path,
    root: Path,
    name: str,
    mutate,
) -> dict[str, Any]:
    bundle = copy_mutation(
        original,
        root,
        f"mutation_{name}",
    )

    mutate(bundle)

    result = run_replay(
        bundle,
        expect_success=False,
    )

    return {
        "mutation": name,
        "rejected": True,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def flip_last_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())

    if not data:
        raise GateError(
            "cannot mutate empty payload"
        )

    data[-1] ^= 0x01
    path.write_bytes(data)


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-runtime-bundle-v1-"
    ) as temporary:
        work = Path(temporary)

        source_one = work / "source_one"
        source_two = work / "source_two"
        source_one.mkdir()
        source_two.mkdir()

        documents = [
            b"A\x00B\xffA",
            b"\xff\x00",
            b"",
            bytes(range(256)),
        ]

        first_paths = []
        second_paths = []

        for doc_id, data in enumerate(documents):
            first = (
                source_one
                / f"document_{doc_id}.bin"
            )
            second = (
                source_two
                / f"document_{doc_id}.bin"
            )

            first.write_bytes(data)
            second.write_bytes(data)

            first_paths.append(first)
            second_paths.append(second)

        first_bundle = work / "bundle_first"
        second_bundle = work / "bundle_second"

        first_result = build_bundle(
            first_paths,
            b"\xff\x00".hex(),
            3,
            first_bundle,
        )

        second_result = build_bundle(
            second_paths,
            b"\xff\x00".hex(),
            3,
            second_bundle,
        )

        first_manifest_bytes = (
            first_bundle
            / "bundle_manifest_v1.json"
        ).read_bytes()

        second_manifest_bytes = (
            second_bundle
            / "bundle_manifest_v1.json"
        ).read_bytes()

        if (
            first_manifest_bytes
            != second_manifest_bytes
        ):
            raise GateError(
                "identical bundle builds differ"
            )

        if (
            (first_bundle / "evidence.json")
            .read_bytes()
            !=
            (second_bundle / "evidence.json")
            .read_bytes()
        ):
            raise GateError(
                "identical evidence payloads differ"
            )

        shutil.rmtree(source_one)
        shutil.rmtree(source_two)

        copied_bundle = (
            work / "outside_repository_bundle"
        )
        shutil.copytree(
            first_bundle,
            copied_bundle,
        )

        replay_process = run_replay(
            copied_bundle,
            expect_success=True,
        )

        replay_result = json.loads(
            replay_process.stdout
        )

        if replay_result.get("ok") is not True:
            raise GateError(
                "copied bundle replay result not ok"
            )

        if (
            replay_result.get(
                "repository_dependency_required"
            )
            is not False
        ):
            raise GateError(
                "repository dependency not eliminated"
            )

        manifest = load_json(
            copied_bundle
            / "bundle_manifest_v1.json"
        )

        if (
            manifest.get(
                "external_data_dependencies"
            )
            != []
        ):
            raise GateError(
                "external data dependency declared"
            )

        if any(
            Path(record["path"]).is_absolute()
            or ".." in Path(
                record["path"]
            ).parts
            for record in manifest["files"]
        ):
            raise GateError(
                "unsafe path in generated manifest"
            )

        mutations = []

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "missing_evidence",
                lambda bundle: (
                    bundle / "evidence.json"
                ).unlink(),
            )
        )

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "altered_document",
                lambda bundle: flip_last_byte(
                    bundle
                    / "documents/doc_00000000.bin"
                ),
            )
        )

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "altered_runtime_binary",
                lambda bundle: flip_last_byte(
                    bundle
                    / "build/build_fm_binary_v1"
                ),
            )
        )

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "missing_evidence_module",
                lambda bundle: (
                    bundle
                    / "tools/"
                    "glyph_binary_runtime_evidence_v1.py"
                ).unlink(),
            )
        )

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "undeclared_extra_file",
                lambda bundle: (
                    bundle / "extra.bin"
                ).write_bytes(b"extra"),
            )
        )

        def wrong_root(bundle: Path) -> None:
            path = (
                bundle
                / "bundle_manifest_v1.json"
            )
            value = load_json(path)
            value["bundle_root_sha256"] = (
                "0" * 64
            )
            save_json(path, value)

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "wrong_bundle_root",
                wrong_root,
            )
        )

        def traversal_path(bundle: Path) -> None:
            path = (
                bundle
                / "bundle_manifest_v1.json"
            )
            value = load_json(path)
            value["files"][0]["path"] = (
                "../outside.bin"
            )
            save_json(path, value)

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "parent_traversal_path",
                traversal_path,
            )
        )

        def wrong_file_count(bundle: Path) -> None:
            path = (
                bundle
                / "bundle_manifest_v1.json"
            )
            value = load_json(path)
            value["file_count"] += 1
            save_json(path, value)

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "wrong_file_count",
                wrong_file_count,
            )
        )

        def symlink_document(bundle: Path) -> None:
            document = (
                bundle
                / "documents/doc_00000000.bin"
            )
            external = (
                work / "external_document.bin"
            )
            external.write_bytes(
                b"A\x00B\xffA"
            )
            document.unlink()
            document.symlink_to(external)

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "symlink_document",
                symlink_document,
            )
        )

        def non_executable_binary(
            bundle: Path,
        ) -> None:
            binary = (
                bundle
                / "build/build_sa_binary_v1"
            )
            binary.chmod(0o644)

        mutations.append(
            expect_mutation_failure(
                copied_bundle,
                work,
                "non_executable_runtime_binary",
                non_executable_binary,
            )
        )

        if not all(
            item["rejected"] is True
            for item in mutations
        ):
            raise GateError(
                "bundle mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_BINARY_RUNTIME_BUNDLE_GATE_V1",
            "bundle_version":
                "GLYPH_BINARY_RUNTIME_BUNDLE_V1",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "count_path_conformant": True,
            "locate_path_conformant": True,
            "multidoc_path_conformant": True,
            "runtime_evidence_conformant": True,
            "runtime_bundle_conformant": True,
            "runtime_conformant": False,
            "self_contained": True,
            "source_documents_bundled": True,
            "runtime_binaries_bundled": True,
            "replay_code_bundled": True,
            "exact_manifest_coverage_verified":
                True,
            "payload_hashes_verified": True,
            "bundle_root_verified": True,
            "deterministic_bundle_verified": True,
            "copied_bundle_replay_verified": True,
            "replay_outside_repository_verified":
                True,
            "original_source_paths_removed_before_replay":
                True,
            "repository_dependency_required": False,
            "network_dependency_required": False,
            "external_data_dependencies": [],
            "document_count":
                first_result["document_count"],
            "file_count":
                first_result["file_count"],
            "bundle_root_sha256":
                first_result[
                    "bundle_root_sha256"
                ],
            "mutation_count": len(mutations),
            "mutations": mutations,
            "remaining_runtime_work": [
                "integrate runtime gates into executable proof graph",
                "make final VERIFY OK depend on runtime conformance",
                "full runtime conformance closure",
            ],
            "non_claims": [
                "The bundle assumes a compatible Linux userspace.",
                "The bundle is not a hermetic VM or container image.",
                "Full runtime conformance awaits proof-graph integration.",
            ],
        }

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUT.write_text(
            json.dumps(
                output,
                indent=2,
                sort_keys=True,
            )
            + "\n"
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
