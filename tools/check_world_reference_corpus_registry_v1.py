#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


FORMAT = (
    "GLYPH_WORLD_REFERENCE_CORPUS_REGISTRY_"
    "HOSTILE_GATE_V1"
)
SUCCESS = (
    "GLYPH WORLD REFERENCE CORPUS REGISTRY "
    "HOSTILE GATE OK"
)


class CheckError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invoke(verifier, arguments, cwd):
    return subprocess.run(
        [sys.executable, str(verifier), *arguments],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def expect_success(verifier, arguments, cwd):
    completed = invoke(verifier, arguments, cwd)

    require(
        completed.returncode == 0,
        "expected verifier success",
    )
    require(
        SUCCESS.replace(" HOSTILE GATE", " VERIFY")
        in completed.stdout,
        "verifier success marker missing",
    )
    require(
        completed.stderr == "",
        "successful verifier wrote stderr",
    )

    return completed


def expect_failure(
    mutations,
    name,
    verifier,
    arguments,
    cwd,
    needle=None,
):
    completed = invoke(verifier, arguments, cwd)
    combined = completed.stdout + completed.stderr

    require(
        completed.returncode != 0,
        f"mutation accepted: {name}",
    )
    require(
        "GLYPH WORLD REFERENCE CORPUS "
        "REGISTRY VERIFY OK"
        not in combined,
        f"mutation emitted success marker: {name}",
    )

    if needle is not None:
        require(
            needle in combined,
            f"wrong rejection for {name}",
        )

    mutations.append({
        "mutation": name,
        "rejected": True,
    })


def write_raw(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def verifier_arguments(
    registry,
    output,
    *,
    registry_only=False,
    golden_root=None,
    reference_ids=(),
):
    arguments = [
        "--registry",
        str(registry),
        "--output",
        str(output),
    ]

    if registry_only:
        arguments.append("--registry-only")

    if golden_root is not None:
        arguments.extend([
            "--golden-root",
            str(golden_root),
        ])

    for reference_id in reference_ids:
        arguments.extend([
            "--reference-id",
            reference_id,
        ])

    return arguments


def publish_output(path, payload):
    require(
        not path.exists() and not path.is_symlink(),
        f"output already exists: {path}",
    )
    require(
        path.parent.is_dir(),
        "output parent is not a directory",
    )

    temporary = path.parent / (
        f".{path.name}.tmp-{os.getpid()}"
    )

    descriptor = None

    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
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
            raise CheckError(
                f"output already exists: {path}"
            ) from error

        temporary.unlink()

        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY

        directory_descriptor = os.open(
            path.parent,
            flags,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("verifier", type=Path)
    parser.add_argument("registry", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    verifier = arguments.verifier.resolve()
    registry_path = arguments.registry.resolve()
    output_path = arguments.output.resolve()

    require(verifier.is_file(), "verifier missing")
    require(registry_path.is_file(), "registry missing")

    registry_raw = registry_path.read_bytes()
    registry = json.loads(registry_raw)

    require(
        registry_raw == canonical_bytes(registry),
        "input registry is not canonical",
    )

    records = registry["records"]
    reference_ids = sorted(
        item["reference_id"]
        for item in records
    )

    require(len(reference_ids) == 7, "bad reference count")

    probe_id = "matt-mahoney-enwik8"
    record_by_id = {
        item["reference_id"]: item
        for item in records
    }
    require(probe_id in record_by_id, "enwik8 absent")

    mutations = []

    with tempfile.TemporaryDirectory(
        prefix="glyph-world-reference-check-"
    ) as temporary_name:
        work = Path(temporary_name)
        cwd_a = work / "cwd-a"
        cwd_b = work / "cwd-b"
        cwd_a.mkdir()
        cwd_b.mkdir()

        positive_a = work / "positive-a.json"
        positive_b = work / "positive-b.json"

        expect_success(
            verifier,
            verifier_arguments(
                registry_path,
                positive_a,
                registry_only=True,
            ),
            cwd_a,
        )

        expect_success(
            verifier,
            verifier_arguments(
                registry_path,
                positive_b,
                registry_only=True,
                reference_ids=reversed(reference_ids),
            ),
            cwd_b,
        )

        require(
            positive_a.read_bytes()
            == positive_b.read_bytes(),
            "selection order changed registry-only result",
        )

        mutation_directory = work / "registries"
        mutation_directory.mkdir()

        invalid_json = mutation_directory / "invalid.json"
        write_raw(invalid_json, b"{")

        expect_failure(
            mutations,
            "invalid_json",
            verifier,
            verifier_arguments(
                invalid_json,
                work / "invalid-result.json",
                registry_only=True,
            ),
            cwd_a,
            "invalid registry JSON",
        )

        duplicate_key = (
            mutation_directory / "duplicate-key.json"
        )
        write_raw(
            duplicate_key,
            b'{"format":"duplicate",'
            + registry_raw[1:],
        )

        expect_failure(
            mutations,
            "duplicate_json_key",
            verifier,
            verifier_arguments(
                duplicate_key,
                work / "duplicate-key-result.json",
                registry_only=True,
            ),
            cwd_a,
            "duplicate JSON key",
        )

        noncanonical = (
            mutation_directory / "noncanonical.json"
        )
        write_raw(noncanonical, b" " + registry_raw)

        expect_failure(
            mutations,
            "noncanonical_json",
            verifier,
            verifier_arguments(
                noncanonical,
                work / "noncanonical-result.json",
                registry_only=True,
            ),
            cwd_a,
            "registry JSON is not canonical",
        )

        canonical_mutations = []

        changed = copy.deepcopy(registry)
        changed["records"][0]["source"] += "#changed"
        canonical_mutations.append(
            ("source_changed", changed)
        )

        changed = copy.deepcopy(registry)
        changed["records"][0]["bytes"] += 1
        canonical_mutations.append(
            ("byte_length_changed", changed)
        )

        changed = copy.deepcopy(registry)
        changed["records"][0]["md5"] = "0" * 32
        canonical_mutations.append(
            ("md5_changed", changed)
        )

        changed = copy.deepcopy(registry)
        changed["records"][0]["sha256"] = "0" * 64
        canonical_mutations.append(
            ("sha256_changed", changed)
        )

        changed = copy.deepcopy(registry)
        changed["records"] = changed["records"][:-1]
        canonical_mutations.append(
            ("record_removed", changed)
        )

        changed = copy.deepcopy(registry)
        changed["records"] = list(
            reversed(changed["records"])
        )
        canonical_mutations.append(
            ("record_order_changed", changed)
        )

        for name, value in canonical_mutations:
            mutation_path = (
                mutation_directory / f"{name}.json"
            )
            write_raw(
                mutation_path,
                canonical_bytes(value),
            )

            expect_failure(
                mutations,
                name,
                verifier,
                verifier_arguments(
                    mutation_path,
                    work / f"{name}-result.json",
                    registry_only=True,
                ),
                cwd_a,
                (
                    "registry record count mismatch"
                    if name == "record_removed"
                    else (
                        "reference IDs are not sorted"
                        if name == "record_order_changed"
                        else "registry SHA-256 mismatch"
                    )
                ),
            )

        expect_failure(
            mutations,
            "golden_root_in_registry_only_mode",
            verifier,
            verifier_arguments(
                registry_path,
                work / "invalid-mode-result.json",
                registry_only=True,
                golden_root=work,
            ),
            cwd_a,
            "--golden-root is invalid",
        )

        expect_failure(
            mutations,
            "golden_root_missing",
            verifier,
            verifier_arguments(
                registry_path,
                work / "missing-root-result.json",
            ),
            cwd_a,
            "--golden-root is required",
        )

        expect_failure(
            mutations,
            "unknown_reference_id",
            verifier,
            verifier_arguments(
                registry_path,
                work / "unknown-result.json",
                registry_only=True,
                reference_ids=("not-a-reference",),
            ),
            cwd_a,
            "unknown reference IDs",
        )

        expect_failure(
            mutations,
            "duplicate_requested_reference_id",
            verifier,
            verifier_arguments(
                registry_path,
                work / "duplicate-id-result.json",
                registry_only=True,
                reference_ids=(probe_id, probe_id),
            ),
            cwd_a,
            "duplicate requested reference ID",
        )

        existing_output = work / "existing-output.json"
        existing_output.write_bytes(b"preserve")

        expect_failure(
            mutations,
            "existing_output",
            verifier,
            verifier_arguments(
                registry_path,
                existing_output,
                registry_only=True,
            ),
            cwd_a,
            "output already exists",
        )

        expected_record = record_by_id[probe_id]
        document_name = expected_record["document_name"]
        expected_bytes = expected_record["bytes"]

        roots = work / "roots"
        roots.mkdir()

        missing_root = roots / "missing"
        missing_root.mkdir()

        expect_failure(
            mutations,
            "reference_file_missing",
            verifier,
            verifier_arguments(
                registry_path,
                work / "missing-file-result.json",
                golden_root=missing_root,
                reference_ids=(probe_id,),
            ),
            cwd_a,
        )

        symlink_root = roots / "symlink"
        symlink_path = symlink_root / document_name
        symlink_path.parent.mkdir(parents=True)
        external = roots / "external.bin"
        external.write_bytes(b"")
        symlink_path.symlink_to(external)

        expect_failure(
            mutations,
            "reference_symlink",
            verifier,
            verifier_arguments(
                registry_path,
                work / "symlink-result.json",
                golden_root=symlink_root,
                reference_ids=(probe_id,),
            ),
            cwd_a,
        )

        directory_root = roots / "directory"
        directory_path = directory_root / document_name
        directory_path.mkdir(parents=True)

        expect_failure(
            mutations,
            "reference_not_regular_file",
            verifier,
            verifier_arguments(
                registry_path,
                work / "directory-result.json",
                golden_root=directory_root,
                reference_ids=(probe_id,),
            ),
            cwd_a,
        )

        short_root = roots / "short"
        short_path = short_root / document_name
        short_path.parent.mkdir(parents=True)
        short_path.write_bytes(b"")

        expect_failure(
            mutations,
            "reference_byte_length_mismatch",
            verifier,
            verifier_arguments(
                registry_path,
                work / "short-result.json",
                golden_root=short_root,
                reference_ids=(probe_id,),
            ),
            cwd_a,
            "reference byte length mismatch",
        )

        sparse_root = roots / "sparse"
        sparse_path = sparse_root / document_name
        sparse_path.parent.mkdir(parents=True)

        with sparse_path.open("wb") as stream:
            stream.truncate(expected_bytes)

        expect_failure(
            mutations,
            "reference_digest_mismatch",
            verifier,
            verifier_arguments(
                registry_path,
                work / "sparse-result.json",
                golden_root=sparse_root,
                reference_ids=(probe_id,),
            ),
            cwd_a,
            "reference MD5 mismatch",
        )

    require(len(mutations) == 19, "mutation count mismatch")
    require(
        all(item["rejected"] for item in mutations),
        "not all mutations rejected",
    )

    result = {
        "format": FORMAT,
        "ok": True,
        "registry_sha256":
            sha256_file(registry_path),
        "verifier_sha256":
            sha256_file(verifier),
        "positive_case_count": 2,
        "mutation_count": len(mutations),
        "all_mutations_rejected": True,
        "different_cwd_verified": True,
        "different_request_order_verified": True,
        "golden_data_required": False,
        "mutations": mutations,
    }

    payload = canonical_bytes(result)
    publish_output(output_path, payload)

    print(SUCCESS)
    print("positive_case_count=2")
    print(f"mutation_count={len(mutations)}")
    print("all_mutations_rejected=true")
    print(
        "output_sha256="
        + hashlib.sha256(payload).hexdigest()
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckError as error:
        print(
            f"GLYPH WORLD REFERENCE CHECK ERROR: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
