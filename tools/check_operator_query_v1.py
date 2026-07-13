#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import os
import re
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
    RUNTIME_INDEX_DIRECTORY,
    IndexErrorV1,
    build_runtime_index,
)
from glyph_operator_manifest_v1 import (  # noqa: E402
    MANIFEST_NAME,
    OperatorError,
    build_snapshot,
    canonical_json_bytes,
    load_canonical_json,
)
from glyph_operator_query_v1 import (  # noqa: E402
    QueryError,
    execute_operator_query,
    parse_query_hex,
    read_stable_query_file,
    validate_count_result,
    validate_locate_result,
)

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_OPERATOR_QUERY_V1.json"
)


class GateError(RuntimeError):
    pass


def stable_error_message(
    error: BaseException,
) -> str:
    message = str(error)

    # Result artifacts must not bind the checkout location.
    message = message.replace(
        str(ROOT),
        "<GLYPH_ROOT>",
    )

    # tempfile creates a random suffix on every gate run.
    message = re.sub(
        r"/(?:tmp|var/tmp)/"
        r"glyph-operator-o3-[^/\s]+",
        "<GLYPH_O3_TMP>",
        message,
    )

    return message


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
        b"10-banana.bin",
        b"banana\x00\xff",
    )
    write_bytes(
        root,
        b"11-repeated.bin",
        b"anaana",
    )
    write_bytes(
        root,
        b"20-left.bin",
        b"LEFT-AB",
    )
    write_bytes(
        root,
        b"21-right.bin",
        b"CD-RIGHT",
    )
    write_bytes(
        root,
        b"30-all-bytes.bin",
        bytes(range(256)),
    )
    write_bytes(
        root,
        b"nested/40-same-a.bin",
        b"same",
    )
    write_bytes(
        root,
        b"nested/41-same-b.bin",
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


def naive_coordinates(
    corpus: Path,
    query: bytes,
) -> list[list[int]]:
    manifest = load_canonical_json(
        corpus / MANIFEST_NAME
    )

    result: list[list[int]] = []

    for document in manifest["documents"]:
        data = (
            corpus
            / document["snapshot_path"]
        ).read_bytes()

        if len(query) > len(data):
            continue

        for offset in range(
            len(data) - len(query) + 1
        ):
            if (
                data[
                    offset:offset + len(query)
                ]
                == query
            ):
                result.append([
                    document["doc_id"],
                    offset,
                ])

    return sorted(result)


def validate_against_oracle(
    corpus: Path,
    query: bytes,
    result: dict[str, Any],
    max_offsets: int | None,
) -> None:
    expected = naive_coordinates(
        corpus,
        query,
    )

    expected_returned = (
        expected
        if max_offsets is None
        else expected[:max_offsets]
    )

    actual = [
        item["coordinate"]
        for item in result["coordinates"]
    ]

    if result["match_count"] != len(expected):
        raise GateError(
            "operator count differs from oracle"
        )

    if actual != expected_returned:
        raise GateError(
            "operator coordinates differ from oracle"
        )

    if (
        result["returned_count"]
        != len(expected_returned)
    ):
        raise GateError(
            "operator returned_count mismatch"
        )

    bounded = (
        len(expected_returned) < len(expected)
    )

    if result["bounded"] is not bounded:
        raise GateError(
            "operator bounded mismatch"
        )

    if (
        result["offsets_complete"]
        is not (not bounded)
    ):
        raise GateError(
            "operator offsets_complete mismatch"
        )

    if result["byte_check"] is not True:
        raise GateError(
            "operator byte_check is not true"
        )


def expect_failure(
    name: str,
    function: Callable[[], Any],
) -> dict[str, Any]:
    try:
        function()
    except (
        QueryError,
        OperatorError,
        IndexErrorV1,
        GateError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": stable_error_message(
                error
            ),
        }

    raise GateError(
        f"mutation unexpectedly accepted: {name}"
    )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph-operator-o3-"
    ) as temporary:
        work = Path(temporary)

        source_one = work / "source-one"
        source_two = work / "source-two"
        corpus_one = work / "corpus-one"
        corpus_two = work / "corpus-two"

        create_source_tree(source_one)
        create_source_tree(source_two)

        build_corpus(source_one, corpus_one)
        build_corpus(source_two, corpus_two)

        cross_document_query = b"ABCD-R"

        queries = [
            b"ana",
            b"A\x00B",
            b"\xff",
            b"\x80\x81",
            bytes(range(256)),
            b"same",
            b"not-present",
            cross_document_query,
        ]

        fixtures = []

        for query_index, query in enumerate(queries):
            first = execute_operator_query(
                corpus_one,
                query,
            )
            second = execute_operator_query(
                corpus_two,
                query,
            )

            if first != second:
                raise GateError(
                    "equivalent corpora produced "
                    "different query results"
                )

            validate_against_oracle(
                corpus_one,
                query,
                first,
                None,
            )

            query_file = (
                work
                / (
                    f"query-{query_index:03d}-"
                    f"{len(query):08d}.bin"
                )
            )
            query_file.write_bytes(query)

            from_file = execute_operator_query(
                corpus_one,
                read_stable_query_file(
                    query_file
                ),
            )

            if from_file != first:
                raise GateError(
                    "query-file and query-hex "
                    "semantic results differ"
                )

            fixtures.append({
                "query_hex": query.hex(),
                "match_count":
                    first["match_count"],
                "returned_count":
                    first["returned_count"],
                "query_result_id":
                    first["query_result_id"],
            })

        repeated_query = b"ana"
        repeated_count = len(
            naive_coordinates(
                corpus_one,
                repeated_query,
            )
        )

        bounded_results = []

        for limit in sorted({
            0,
            1,
            max(0, repeated_count - 1),
            repeated_count,
            repeated_count + 1,
        }):
            result = execute_operator_query(
                corpus_one,
                repeated_query,
                max_offsets=limit,
            )

            validate_against_oracle(
                corpus_one,
                repeated_query,
                result,
                limit,
            )

            bounded_results.append({
                "max_offsets": limit,
                "match_count":
                    result["match_count"],
                "returned_count":
                    result["returned_count"],
                "bounded":
                    result["bounded"],
                "query_result_id":
                    result["query_result_id"],
            })

        invalid_name_result = (
            execute_operator_query(
                corpus_one,
                b"\x80\x81",
            )
        )

        invalid_path_hex = (
            b"nested/invalid-name-\xff.bin"
        ).hex()

        if not any(
            item[
                "relative_path_bytes_hex"
            ] == invalid_path_hex
            for item
            in invalid_name_result[
                "coordinates"
            ]
        ):
            raise GateError(
                "invalid UTF-8 path mapping missing"
            )

        source_manifest = load_canonical_json(
            corpus_one / MANIFEST_NAME
        )

        source_payloads = [
            (
                corpus_one
                / document["snapshot_path"]
            ).read_bytes()
            for document
            in source_manifest["documents"]
        ]

        physical_concatenation = b"".join(
            source_payloads
        )

        if (
            cross_document_query
            not in physical_concatenation
        ):
            raise GateError(
                "cross-document control is absent "
                "from physical concatenation"
            )

        if any(
            cross_document_query in payload
            for payload in source_payloads
        ):
            raise GateError(
                "cross-document control also exists "
                "inside an individual document"
            )

        cross_only = execute_operator_query(
            corpus_one,
            cross_document_query,
        )

        if cross_only["match_count"] != 0:
            raise GateError(
                "cross-document-only query matched"
            )

        mutations: list[dict[str, Any]] = []

        mutations.append(
            expect_failure(
                "empty_query_hex",
                lambda: parse_query_hex(""),
            )
        )

        mutations.append(
            expect_failure(
                "uppercase_query_hex",
                lambda: parse_query_hex("00FF"),
            )
        )

        mutations.append(
            expect_failure(
                "odd_query_hex",
                lambda: parse_query_hex("0"),
            )
        )

        mutations.append(
            expect_failure(
                "prefixed_query_hex",
                lambda: parse_query_hex("0x00"),
            )
        )

        empty_query_file = work / "empty-query.bin"
        empty_query_file.write_bytes(b"")

        mutations.append(
            expect_failure(
                "empty_query_file",
                lambda: read_stable_query_file(
                    empty_query_file
                ),
            )
        )

        real_query_file = work / "real-query.bin"
        real_query_file.write_bytes(b"ana")

        symlink_query_file = (
            work / "symlink-query.bin"
        )
        symlink_query_file.symlink_to(
            real_query_file
        )

        mutations.append(
            expect_failure(
                "symlink_query_file",
                lambda: read_stable_query_file(
                    symlink_query_file
                ),
            )
        )

        changing_query = work / "changing-query.bin"
        changing_query.write_bytes(b"before")

        mutations.append(
            expect_failure(
                "query_file_mutation",
                lambda: read_stable_query_file(
                    changing_query,
                    after_first_read_hook=
                        lambda path:
                            path.write_bytes(
                                b"after-change"
                            ),
                ),
            )
        )

        mutations.append(
            expect_failure(
                "negative_max_offsets",
                lambda: execute_operator_query(
                    corpus_one,
                    b"ana",
                    max_offsets=-1,
                ),
            )
        )

        mutations.append(
            expect_failure(
                "max_offsets_overflow",
                lambda: execute_operator_query(
                    corpus_one,
                    b"ana",
                    max_offsets=2**64,
                ),
            )
        )

        tampered_source_corpus = (
            work / "tampered-source-corpus"
        )
        shutil.copytree(
            corpus_one,
            tampered_source_corpus,
        )

        source_payload = (
            tampered_source_corpus
            / "documents/doc_00000001.bin"
        )
        source_data = bytearray(
            source_payload.read_bytes()
        )
        source_data[0] ^= 0x01
        source_payload.write_bytes(source_data)

        mutations.append(
            expect_failure(
                "committed_source_tamper",
                lambda: execute_operator_query(
                    tampered_source_corpus,
                    b"ana",
                ),
            )
        )

        tampered_runtime_corpus = (
            work / "tampered-runtime-corpus"
        )
        shutil.copytree(
            corpus_one,
            tampered_runtime_corpus,
        )

        fm_path = (
            tampered_runtime_corpus
            / RUNTIME_INDEX_DIRECTORY
            / "documents/doc_00000001/fm.bin"
        )
        fm_data = bytearray(
            fm_path.read_bytes()
        )
        fm_data[-1] ^= 0x01
        fm_path.write_bytes(fm_data)

        mutations.append(
            expect_failure(
                "runtime_index_tamper",
                lambda: execute_operator_query(
                    tampered_runtime_corpus,
                    b"ana",
                ),
            )
        )

        mutation_source = (
            work / "during-query-source"
        )
        mutation_corpus = (
            work / "during-query-corpus"
        )

        create_source_tree(mutation_source)
        build_corpus(
            mutation_source,
            mutation_corpus,
        )

        mutated = False

        def mutate_during_query(
            doc_id: int,
            path: Path,
        ) -> None:
            nonlocal mutated

            if doc_id == 1 and not mutated:
                mutated = True
                path.write_bytes(
                    b"changed-during-query"
                )

        mutations.append(
            expect_failure(
                "source_mutation_during_query",
                lambda: execute_operator_query(
                    mutation_corpus,
                    b"ana",
                    after_pre_source_check_hook=
                        mutate_during_query,
                ),
            )
        )

        valid_count = {
            "ok": True,
            "format":
                "GLYPH_QUERY_BINARY_V1",
            "query_hex": "616e61",
            "query_length_bytes": 3,
            "interval": [5, 7],
            "count": 2,
            "alphabet_size": 257,
            "logical_sentinel": 256,
        }

        invalid_count = copy.deepcopy(
            valid_count
        )
        invalid_count["count"] = 3

        mutations.append(
            expect_failure(
                "count_interval_mismatch",
                lambda: validate_count_result(
                    invalid_count,
                    b"ana",
                ),
            )
        )

        valid_locate = {
            "ok": True,
            "format":
                "GLYPH_QUERY_LOCATE_BINARY_V1",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "document_count": 1,
            "query_hex": "616e61",
            "query_length_bytes": 3,
            "interval": [5, 7],
            "match_count": 2,
            "returned_count": 2,
            "bounded": False,
            "offsets_complete": True,
            "byte_check": True,
            "offsets": [1, 3],
            "coordinates": [
                [0, 1],
                [0, 3],
            ],
            "alphabet_size": 257,
            "logical_sentinel": 256,
        }

        duplicate_locate = copy.deepcopy(
            valid_locate
        )
        duplicate_locate["offsets"] = [1, 1]
        duplicate_locate["coordinates"] = [
            [0, 1],
            [0, 1],
        ]

        mutations.append(
            expect_failure(
                "duplicate_locate_offset",
                lambda: validate_locate_result(
                    duplicate_locate,
                    b"ana",
                    {
                        "interval": [5, 7],
                        "count": 2,
                    },
                    None,
                ),
            )
        )

        false_byte_check = copy.deepcopy(
            valid_locate
        )
        false_byte_check["byte_check"] = False

        mutations.append(
            expect_failure(
                "runtime_false_byte_check",
                lambda: validate_locate_result(
                    false_byte_check,
                    b"ana",
                    {
                        "interval": [5, 7],
                        "count": 2,
                    },
                    None,
                ),
            )
        )

        if not all(
            item["rejected"] is True
            for item in mutations
        ):
            raise GateError(
                "O3 mutation gate failed"
            )

        output = {
            "ok": True,
            "format":
                "GLYPH_OPERATOR_QUERY_GATE_V1",
            "operator_obligation": "O3",
            "result_version":
                "GLYPH_OPERATOR_QUERY_RESULT_V1",
            "runtime_profile":
                "GLYPH_BINARY_RUNTIME_V1",
            "compiled_count_used": True,
            "compiled_locate_used": True,
            "count_locate_agreement_verified":
                True,
            "query_file_transport_verified":
                True,
            "query_hex_transport_verified":
                True,
            "query_file_equals_hex": True,
            "embedded_nul_query_verified": True,
            "byte_ff_query_verified": True,
            "all_256_byte_query_verified": True,
            "invalid_utf8_path_mapping_verified":
                True,
            "zero_match_verified": True,
            "cross_document_match_rejected":
                True,
            "bounded_multidoc_locate_verified":
                True,
            "global_coordinate_order_verified":
                True,
            "independent_byte_check_verified":
                True,
            "query_binary_commitments_bound":
                True,
            "source_mutation_rejected": True,
            "runtime_mutation_rejected": True,
            "query_file_mutation_rejected": True,
            "deterministic_results_verified":
                True,
            "fixture_count": len(fixtures),
            "bounded_fixture_count":
                len(bounded_results),
            "mutation_count": len(mutations),
            "fixtures": fixtures,
            "bounded_results":
                bounded_results,
            "mutations": mutations,
            "next_operator_obligation":
                "O4_OPERATOR_EVIDENCE_BUNDLE",
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
