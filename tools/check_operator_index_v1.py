#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from glyph_operator_index_v1 import (  # noqa: E402
    INDEX_COMPLETE_NAME,
    INDEX_MANIFEST_NAME,
    RUNTIME_INDEX_DIRECTORY,
    IndexErrorV1,
    build_runtime_index,
    canonical_json_bytes,
    make_complete_marker,
    runtime_index_identity,
    verify_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME as SOURCE_MANIFEST_NAME,
    OperatorError,
    build_snapshot,
    load_canonical_json,
    sha256_file,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_RUNTIME_INDEX_V1.json"
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
        b"10-ascii.txt",
        b"banana\n",
    )
    write_bytes(
        root,
        b"20-nul.bin",
        b"A\x00B\x00A",
    )
    write_bytes(
        root,
        b"30-ff.bin",
        b"\xff\x00\xff",
    )
    write_bytes(
        root,
        b"40-all-bytes.bin",
        bytes(range(256)),
    )
    write_bytes(
        root,
        b"nested/50-duplicate-a.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/51-duplicate-b.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/invalid-name-\xff.bin",
        b"\x80\x81\xfe\xff",
    )


def copy_corpus(
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
        IndexErrorV1,
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


def runtime_manifest_path(
    corpus: Path,
) -> Path:
    return (
        corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_MANIFEST_NAME
    )


def runtime_complete_path(
    corpus: Path,
) -> Path:
    return (
        corpus
        / RUNTIME_INDEX_DIRECTORY
        / INDEX_COMPLETE_NAME
    )


def rewrite_runtime_manifest(
    corpus: Path,
    manifest: dict[str, Any],
    *,
    recompute_runtime_index_id: bool,
) -> None:
    if recompute_runtime_index_id:
        manifest["runtime_index_id"] = (
            runtime_index_identity(manifest)
        )

    manifest_path = runtime_manifest_path(
        corpus
    )

    manifest_path.write_bytes(
        canonical_json_bytes(manifest)
    )

    complete = make_complete_marker(
        manifest,
        sha256_file(manifest_path),
    )

    runtime_complete_path(
        corpus
    ).write_bytes(
        canonical_json_bytes(complete)
    )


def tree_payloads(
    root: Path,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}

    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[
                path.relative_to(root).as_posix()
            ] = path.read_bytes()

    return result


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-operator-o2-"
    ) as temporary:
        work = Path(temporary)

        source_one = work / "source-one"
        source_two = work / "source-two"

        create_source_tree(source_one)
        create_source_tree(source_two)

        corpus_one = work / "corpus-one"
        corpus_two = work / "corpus-two"

        build_snapshot(
            source_one,
            corpus_one,
        )
        build_snapshot(
            source_two,
            corpus_two,
        )

        first = build_runtime_index(
            corpus_one
        )
        second = build_runtime_index(
            corpus_two
        )

        first_manifest = (
            runtime_manifest_path(
                corpus_one
            ).read_bytes()
        )
        second_manifest = (
            runtime_manifest_path(
                corpus_two
            ).read_bytes()
        )

        if first_manifest != second_manifest:
            raise GateError(
                "equivalent snapshots produced "
                "different runtime manifests"
            )

        first_payloads = tree_payloads(
            corpus_one
            / RUNTIME_INDEX_DIRECTORY
        )
        second_payloads = tree_payloads(
            corpus_two
            / RUNTIME_INDEX_DIRECTORY
        )

        if first_payloads != second_payloads:
            raise GateError(
                "equivalent snapshots produced "
                "different runtime payloads"
            )

        verified = verify_runtime_index(
            corpus_one,
            require_current_binaries=True,
            rebuild=True,
        )

        if verified["ok"] is not True:
            raise GateError(
                "O2 rebuild verification failed"
            )

        source_manifest = load_canonical_json(
            corpus_one
            / SOURCE_MANIFEST_NAME
        )

        if (
            verified["document_count"]
            != source_manifest[
                "document_count"
            ]
        ):
            raise GateError(
                "O2 document count differs from O1"
            )

        empty_doc_id = next(
            document["doc_id"]
            for document
            in source_manifest["documents"]
            if document["byte_length"] == 0
        )

        runtime_manifest = load_canonical_json(
            runtime_manifest_path(
                corpus_one
            )
        )

        empty_runtime = runtime_manifest[
            "documents"
        ][empty_doc_id]

        for role in ("sa", "bwt", "fm"):
            if empty_runtime[role]["size_bytes"] <= 0:
                raise GateError(
                    "empty document runtime "
                    f"artifact missing: {role}"
                )

        mutations: list[dict[str, Any]] = []

        mutations.append(
            expect_failure(
                "existing_runtime_index_rejected",
                lambda: build_runtime_index(
                    corpus_one
                ),
            )
        )

        missing_sa = copy_corpus(
            corpus_one,
            work / "missing-sa",
        )

        (
            missing_sa
            / RUNTIME_INDEX_DIRECTORY
            / "documents/doc_00000000/sa.bin"
        ).unlink()

        mutations.append(
            expect_failure(
                "missing_sa_payload",
                lambda: verify_runtime_index(
                    missing_sa
                ),
            )
        )

        altered_bwt = copy_corpus(
            corpus_one,
            work / "altered-bwt",
        )

        bwt_path = (
            altered_bwt
            / RUNTIME_INDEX_DIRECTORY
            / "documents/doc_00000001/bwt.bin"
        )

        bwt_data = bytearray(
            bwt_path.read_bytes()
        )

        if not bwt_data:
            raise GateError(
                "BWT fixture unexpectedly empty"
            )

        bwt_data[-1] ^= 0x01
        bwt_path.write_bytes(bwt_data)

        mutations.append(
            expect_failure(
                "altered_bwt_payload",
                lambda: verify_runtime_index(
                    altered_bwt
                ),
            )
        )

        extra_payload = copy_corpus(
            corpus_one,
            work / "extra-runtime-payload",
        )

        (
            extra_payload
            / RUNTIME_INDEX_DIRECTORY
            / "documents/doc_00000000/extra.bin"
        ).write_bytes(b"extra")

        mutations.append(
            expect_failure(
                "undeclared_runtime_payload",
                lambda: verify_runtime_index(
                    extra_payload
                ),
            )
        )

        symlink_payload = copy_corpus(
            corpus_one,
            work / "symlink-runtime-payload",
        )

        symlink_sa = (
            symlink_payload
            / RUNTIME_INDEX_DIRECTORY
            / "documents/doc_00000000/sa.bin"
        )

        original_sa = symlink_sa.read_bytes()
        external_sa = (
            work / "external-sa.bin"
        )
        external_sa.write_bytes(original_sa)

        symlink_sa.unlink()
        symlink_sa.symlink_to(external_sa)

        mutations.append(
            expect_failure(
                "symlink_runtime_payload",
                lambda: verify_runtime_index(
                    symlink_payload
                ),
            )
        )

        unknown_root = copy_corpus(
            corpus_one,
            work / "unknown-runtime-root",
        )

        (
            unknown_root
            / RUNTIME_INDEX_DIRECTORY
            / "unknown.bin"
        ).write_bytes(b"unknown")

        mutations.append(
            expect_failure(
                "unknown_runtime_root_payload",
                lambda: verify_runtime_index(
                    unknown_root
                ),
            )
        )

        noncanonical = copy_corpus(
            corpus_one,
            work / "noncanonical-runtime-json",
        )

        noncanonical_manifest = (
            load_canonical_json(
                runtime_manifest_path(
                    noncanonical
                )
            )
        )

        runtime_manifest_path(
            noncanonical
        ).write_text(
            json.dumps(
                noncanonical_manifest,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        mutations.append(
            expect_failure(
                "noncanonical_runtime_manifest",
                lambda: verify_runtime_index(
                    noncanonical
                ),
            )
        )

        wrong_source_binding = copy_corpus(
            corpus_one,
            work / "wrong-source-binding",
        )

        wrong_source_manifest = (
            load_canonical_json(
                runtime_manifest_path(
                    wrong_source_binding
                )
            )
        )

        wrong_source_manifest[
            "source_manifest_sha256"
        ] = "0" * 64

        rewrite_runtime_manifest(
            wrong_source_binding,
            wrong_source_manifest,
            recompute_runtime_index_id=True,
        )

        mutations.append(
            expect_failure(
                "wrong_source_manifest_binding",
                lambda: verify_runtime_index(
                    wrong_source_binding
                ),
            )
        )

        wrong_doc_id = copy_corpus(
            corpus_one,
            work / "wrong-runtime-doc-id",
        )

        wrong_doc_manifest = (
            load_canonical_json(
                runtime_manifest_path(
                    wrong_doc_id
                )
            )
        )

        wrong_doc_manifest[
            "documents"
        ][0]["doc_id"] = 99

        rewrite_runtime_manifest(
            wrong_doc_id,
            wrong_doc_manifest,
            recompute_runtime_index_id=True,
        )

        mutations.append(
            expect_failure(
                "noncanonical_runtime_doc_id",
                lambda: verify_runtime_index(
                    wrong_doc_id
                ),
            )
        )

        wrong_binary = copy_corpus(
            corpus_one,
            work / "wrong-runtime-binary",
        )

        wrong_binary_manifest = (
            load_canonical_json(
                runtime_manifest_path(
                    wrong_binary
                )
            )
        )

        wrong_binary_manifest[
            "runtime_binaries"
        ][0]["sha256"] = "0" * 64

        rewrite_runtime_manifest(
            wrong_binary,
            wrong_binary_manifest,
            recompute_runtime_index_id=True,
        )

        mutations.append(
            expect_failure(
                "runtime_binary_commitment_changed",
                lambda: verify_runtime_index(
                    wrong_binary,
                    require_current_binaries=True,
                ),
            )
        )

        tampered_source = copy_corpus(
            corpus_one,
            work / "tampered-o1-source",
        )

        source_payload = (
            tampered_source
            / "documents/doc_00000001.bin"
        )

        source_data = bytearray(
            source_payload.read_bytes()
        )

        source_data[0] ^= 0x01
        source_payload.write_bytes(
            source_data
        )

        mutations.append(
            expect_failure(
                "committed_source_payload_changed",
                lambda: verify_runtime_index(
                    tampered_source
                ),
            )
        )

        interrupted_source = (
            work / "interrupted-source"
        )
        create_source_tree(
            interrupted_source
        )

        interrupted_corpus = (
            work / "interrupted-corpus"
        )

        build_snapshot(
            interrupted_source,
            interrupted_corpus,
        )

        mutations.append(
            expect_failure(
                "interrupted_index_not_published",
                lambda: build_runtime_index(
                    interrupted_corpus,
                    test_fail_after_documents=1,
                ),
            )
        )

        if (
            interrupted_corpus
            / RUNTIME_INDEX_DIRECTORY
        ).exists():
            raise GateError(
                "interrupted O2 build published "
                "runtime_index_v1"
            )

        mutation_source = (
            work / "mutation-source"
        )
        create_source_tree(
            mutation_source
        )

        mutation_corpus = (
            work / "mutation-corpus"
        )

        build_snapshot(
            mutation_source,
            mutation_corpus,
        )

        mutated_once = False

        def mutate_snapshot_during_build(
            doc_id: int,
            source_path: Path,
        ) -> None:
            nonlocal mutated_once

            if doc_id == 0 and not mutated_once:
                mutated_once = True
                source_path.write_bytes(
                    b"changed-after-private-copy"
                )

        mutations.append(
            expect_failure(
                "snapshot_mutation_during_index_build",
                lambda: build_runtime_index(
                    mutation_corpus,
                    after_private_copy_hook=
                        mutate_snapshot_during_build,
                ),
            )
        )

        if (
            mutation_corpus
            / RUNTIME_INDEX_DIRECTORY
        ).exists():
            raise GateError(
                "mutated O1 snapshot produced "
                "published runtime index"
            )

        if not all(
            mutation["rejected"] is True
            for mutation in mutations
        ):
            raise GateError(
                "O2 mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_RUNTIME_INDEX_GATE_V1",
            "operator_obligation": "O2",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "manifest_version":
                "GLYPH_OPERATOR_RUNTIME_INDEX_MANIFEST_V1",
            "built_from_committed_snapshot":
                True,
            "original_source_directory_used":
                False,
            "private_verified_inputs_used":
                True,
            "binary_safe_arbitrary_bytes":
                True,
            "logical_sentinel": 256,
            "alphabet_size": 257,
            "one_index_per_document": True,
            "empty_document_indexed": True,
            "duplicate_documents_preserved":
                True,
            "runtime_binary_commitments_bound":
                True,
            "source_manifest_bound": True,
            "corpus_identity_bound": True,
            "source_manifest_identity_bound":
                True,
            "per_document_index_commitments":
                True,
            "deterministic_manifest_verified":
                True,
            "deterministic_payloads_verified":
                True,
            "deterministic_rebuild_verified":
                True,
            "atomic_publication_verified": True,
            "interrupted_build_rejected": True,
            "snapshot_mutation_rejected": True,
            "structural_verification_passed":
                True,
            "document_count":
                first["document_count"],
            "total_source_bytes":
                first["total_source_bytes"],
            "total_runtime_bytes":
                first["total_runtime_bytes"],
            "corpus_id": first["corpus_id"],
            "source_manifest_id":
                first["source_manifest_id"],
            "runtime_index_id":
                first["runtime_index_id"],
            "mutation_count": len(mutations),
            "mutations": mutations,
            "next_operator_obligation":
                "O3_BINARY_QUERY_AND_SOURCE_MAPPING",
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
