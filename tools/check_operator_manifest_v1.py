#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_manifest_v1 import (  # noqa: E402
    COMPLETE_NAME,
    MANIFEST_NAME,
    OperatorError,
    SourceError,
    VerificationError,
    build_snapshot,
    canonical_json_bytes,
    load_canonical_json,
    verify_snapshot,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_MANIFEST_V1.json"
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
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


def create_reference_tree(root: Path) -> None:
    root.mkdir(parents=True)

    write_bytes(
        root,
        b"00-empty.bin",
        b"",
    )
    write_bytes(
        root,
        b"10-ascii.txt",
        b"banana\n",
    )
    write_bytes(
        root,
        b"20-binary.bin",
        b"A\x00B\xffA",
    )
    write_bytes(
        root,
        b"30-invalid-content.bin",
        b"\x80\x81\xfe\xff",
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
        b"invalid-name-content",
    )


def copy_snapshot(
    source: Path,
    destination: Path,
) -> Path:
    shutil.copytree(
        source,
        destination,
    )
    return destination


def expect_failure(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        function()
    except (
        OperatorError,
        GateError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise GateError(
        f"mutation unexpectedly accepted: {name}"
    )


def save_canonical(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.write_bytes(
        canonical_json_bytes(value)
    )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-operator-o1-"
    ) as temporary:
        work = Path(temporary)

        source_one = work / "source-one"
        source_two = work / "source-two"

        create_reference_tree(source_one)
        create_reference_tree(source_two)

        corpus_one = work / "corpus-one"
        corpus_two = work / "corpus-two"

        first = build_snapshot(
            source_one,
            corpus_one,
        )
        second = build_snapshot(
            source_two,
            corpus_two,
        )

        first_manifest_bytes = (
            corpus_one / MANIFEST_NAME
        ).read_bytes()
        second_manifest_bytes = (
            corpus_two / MANIFEST_NAME
        ).read_bytes()

        if (
            first_manifest_bytes
            != second_manifest_bytes
        ):
            raise GateError(
                "equivalent source trees produced "
                "different manifests"
            )

        first_manifest = load_canonical_json(
            corpus_one / MANIFEST_NAME
        )

        second_manifest = load_canonical_json(
            corpus_two / MANIFEST_NAME
        )

        if first_manifest != second_manifest:
            raise GateError(
                "equivalent source manifests differ"
            )

        raw_paths = [
            bytes.fromhex(
                document[
                    "relative_path_bytes_hex"
                ]
            )
            for document
            in first_manifest["documents"]
        ]

        if raw_paths != sorted(raw_paths):
            raise GateError(
                "raw path ordering is not deterministic"
            )

        if [
            document["doc_id"]
            for document
            in first_manifest["documents"]
        ] != list(
            range(
                first_manifest["document_count"]
            )
        ):
            raise GateError(
                "doc_id assignment is not canonical"
            )

        invalid_name = (
            b"nested/invalid-name-\xff.bin"
        )

        if invalid_name not in raw_paths:
            raise GateError(
                "invalid UTF-8 filename missing"
            )

        if not any(
            document["byte_length"] == 0
            for document
            in first_manifest["documents"]
        ):
            raise GateError(
                "empty file was not preserved"
            )

        duplicate_hashes = [
            document["sha256"]
            for document
            in first_manifest["documents"]
            if document["display_path"].startswith(
                "nested/4"
            )
        ]

        if len(duplicate_hashes) != 2:
            raise GateError(
                "duplicate documents missing"
            )

        if len(set(duplicate_hashes)) != 1:
            raise GateError(
                "duplicate document content differs"
            )

        verified = verify_snapshot(
            corpus_one
        )

        if verified.get("ok") is not True:
            raise GateError(
                "independent snapshot verification failed"
            )

        rename_source_a = work / "rename-a"
        rename_source_b = work / "rename-b"
        rename_source_a.mkdir()
        rename_source_b.mkdir()

        write_bytes(
            rename_source_a,
            b"a.bin",
            b"same-content",
        )
        write_bytes(
            rename_source_b,
            b"b.bin",
            b"same-content",
        )

        rename_corpus_a = work / "rename-corpus-a"
        rename_corpus_b = work / "rename-corpus-b"

        build_snapshot(
            rename_source_a,
            rename_corpus_a,
        )
        build_snapshot(
            rename_source_b,
            rename_corpus_b,
        )

        rename_manifest_a = load_canonical_json(
            rename_corpus_a / MANIFEST_NAME
        )
        rename_manifest_b = load_canonical_json(
            rename_corpus_b / MANIFEST_NAME
        )

        if (
            rename_manifest_a["corpus_id"]
            != rename_manifest_b["corpus_id"]
        ):
            raise GateError(
                "rename unexpectedly changed corpus_id"
            )

        if (
            rename_manifest_a[
                "source_manifest_id"
            ]
            ==
            rename_manifest_b[
                "source_manifest_id"
            ]
        ):
            raise GateError(
                "rename did not change source_manifest_id"
            )

        changed_source = work / "changed-source"
        changed_source.mkdir()

        write_bytes(
            changed_source,
            b"a.bin",
            b"different-content",
        )

        changed_corpus = work / "changed-corpus"

        build_snapshot(
            changed_source,
            changed_corpus,
        )

        changed_manifest = load_canonical_json(
            changed_corpus / MANIFEST_NAME
        )

        if (
            changed_manifest["corpus_id"]
            == rename_manifest_a["corpus_id"]
        ):
            raise GateError(
                "content change did not change corpus_id"
            )

        if (
            changed_manifest["source_manifest_id"]
            == rename_manifest_a[
                "source_manifest_id"
            ]
        ):
            raise GateError(
                "content change did not change "
                "source_manifest_id"
            )

        mutations: list[dict[str, Any]] = []

        symlink_source = work / "symlink-source"
        symlink_source.mkdir()
        write_bytes(
            symlink_source,
            b"real.bin",
            b"real",
        )
        (
            symlink_source / "link.bin"
        ).symlink_to("real.bin")

        mutations.append(
            expect_failure(
                "symlink_rejected",
                lambda: build_snapshot(
                    symlink_source,
                    work / "symlink-corpus",
                ),
            )
        )

        fifo_source = work / "fifo-source"
        fifo_source.mkdir()
        fifo_path = fifo_source / "named-pipe"
        os.mkfifo(fifo_path)

        mutations.append(
            expect_failure(
                "special_file_rejected",
                lambda: build_snapshot(
                    fifo_source,
                    work / "fifo-corpus",
                ),
            )
        )

        mutation_source = work / "mutation-source"
        mutation_source.mkdir()
        write_bytes(
            mutation_source,
            b"changing.bin",
            b"before",
        )

        mutated_once = False

        def mutate_during_read(
            relative: bytes,
            absolute: bytes,
        ) -> None:
            nonlocal mutated_once

            if (
                relative == b"changing.bin"
                and not mutated_once
            ):
                mutated_once = True

                descriptor = os.open(
                    absolute,
                    os.O_WRONLY
                    | os.O_TRUNC,
                )

                try:
                    os.write(
                        descriptor,
                        b"after-change",
                    )
                finally:
                    os.close(descriptor)

        mutation_output = (
            work / "mutation-corpus"
        )

        mutations.append(
            expect_failure(
                "source_mutation_during_read",
                lambda: build_snapshot(
                    mutation_source,
                    mutation_output,
                    after_initial_stat_hook=
                        mutate_during_read,
                ),
            )
        )

        if mutation_output.exists():
            raise GateError(
                "failed source mutation build "
                "published final output"
            )

        interrupted_source = (
            work / "interrupted-source"
        )
        interrupted_source.mkdir()

        write_bytes(
            interrupted_source,
            b"a.bin",
            b"a",
        )
        write_bytes(
            interrupted_source,
            b"b.bin",
            b"b",
        )

        interrupted_output = (
            work / "interrupted-corpus"
        )

        mutations.append(
            expect_failure(
                "interrupted_build_not_published",
                lambda: build_snapshot(
                    interrupted_source,
                    interrupted_output,
                    test_fail_after_documents=1,
                ),
            )
        )

        if interrupted_output.exists():
            raise GateError(
                "interrupted build published output"
            )

        incomplete = work / "incomplete-corpus"
        incomplete.mkdir()
        (
            incomplete / "documents"
        ).mkdir()

        mutations.append(
            expect_failure(
                "incomplete_snapshot_rejected",
                lambda: verify_snapshot(
                    incomplete
                ),
            )
        )

        tampered_payload = copy_snapshot(
            corpus_one,
            work / "tampered-payload",
        )

        first_payload = sorted(
            (
                tampered_payload
                / "documents"
            ).iterdir()
        )[0]

        payload_data = bytearray(
            first_payload.read_bytes()
        )

        if payload_data:
            payload_data[0] ^= 0x01
        else:
            payload_data.extend(b"x")

        first_payload.write_bytes(
            payload_data
        )

        mutations.append(
            expect_failure(
                "document_payload_tamper",
                lambda: verify_snapshot(
                    tampered_payload
                ),
            )
        )

        extra_payload = copy_snapshot(
            corpus_one,
            work / "extra-payload",
        )
        (
            extra_payload
            / "documents/undeclared.bin"
        ).write_bytes(b"extra")

        mutations.append(
            expect_failure(
                "undeclared_document_payload",
                lambda: verify_snapshot(
                    extra_payload
                ),
            )
        )

        unknown_extension = copy_snapshot(
            corpus_one,
            work / "unknown-extension",
        )

        (
            unknown_extension
            / "unregistered_extension_v1"
        ).mkdir()

        mutations.append(
            expect_failure(
                "unknown_top_level_extension",
                lambda: verify_snapshot(
                    unknown_extension
                ),
            )
        )

        traversal = copy_snapshot(
            corpus_one,
            work / "traversal-manifest",
        )

        traversal_manifest_path = (
            traversal / MANIFEST_NAME
        )
        traversal_manifest = (
            load_canonical_json(
                traversal_manifest_path
            )
        )

        traversal_manifest[
            "documents"
        ][0][
            "relative_path_bytes_hex"
        ] = b"../outside.bin".hex()

        save_canonical(
            traversal_manifest_path,
            traversal_manifest,
        )

        mutations.append(
            expect_failure(
                "parent_traversal_manifest_path",
                lambda: verify_snapshot(
                    traversal
                ),
            )
        )

        noncanonical = copy_snapshot(
            corpus_one,
            work / "noncanonical-json",
        )

        manifest_value = load_canonical_json(
            noncanonical / MANIFEST_NAME
        )

        (
            noncanonical / MANIFEST_NAME
        ).write_text(
            json.dumps(
                manifest_value,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        mutations.append(
            expect_failure(
                "noncanonical_manifest_json",
                lambda: verify_snapshot(
                    noncanonical
                ),
            )
        )

        output_exists_source = (
            work / "output-exists-source"
        )
        output_exists_source.mkdir()
        write_bytes(
            output_exists_source,
            b"a.bin",
            b"a",
        )

        output_exists = (
            work / "output-already-exists"
        )
        output_exists.mkdir()

        mutations.append(
            expect_failure(
                "existing_output_rejected",
                lambda: build_snapshot(
                    output_exists_source,
                    output_exists,
                ),
            )
        )

        inside_source = work / "inside-source"
        inside_source.mkdir()
        write_bytes(
            inside_source,
            b"a.bin",
            b"a",
        )

        mutations.append(
            expect_failure(
                "output_inside_source_rejected",
                lambda: build_snapshot(
                    inside_source,
                    inside_source / "corpus",
                ),
            )
        )

        if not all(
            mutation["rejected"] is True
            for mutation in mutations
        ):
            raise GateError(
                "O1 mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_MANIFEST_GATE_V1",
            "operator_obligation": "O1",
            "manifest_version":
                "GLYPH_OPERATOR_CORPUS_MANIFEST_V1",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "deterministic_source_discovery":
                True,
            "raw_path_byte_ordering_verified":
                True,
            "stable_doc_id_assignment":
                True,
            "invalid_utf8_filename_supported":
                True,
            "embedded_nul_content_supported":
                True,
            "byte_ff_content_supported":
                True,
            "empty_files_preserved": True,
            "duplicate_documents_preserved":
                True,
            "source_stability_verified": True,
            "source_mutation_rejected": True,
            "symlink_rejected": True,
            "special_files_rejected": True,
            "atomic_publication_verified": True,
            "incomplete_snapshot_rejected": True,
            "independent_snapshot_verification":
                True,
            "rename_preserves_corpus_id":
                True,
            "rename_changes_source_manifest_id":
                True,
            "content_change_changes_identities":
                True,
            "source_directory_required_for_verify":
                False,
            "document_count":
                first["document_count"],
            "total_source_bytes":
                first["total_source_bytes"],
            "corpus_id":
                first["corpus_id"],
            "source_manifest_id":
                first["source_manifest_id"],
            "manifest_sha256":
                verified["manifest_sha256"],
            "mutation_count": len(mutations),
            "mutations": mutations,
            "next_operator_obligation":
                "O2_RUNTIME_INDEX_BUILD",
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
