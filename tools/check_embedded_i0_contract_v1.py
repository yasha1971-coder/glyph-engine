#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

HEADER = ROOT / "include/glyph/glyph.h"

RESULT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_EMBEDDED_I0_CONTRACT_V1.json"
)

FORMAT = "GLYPH_EMBEDDED_I0_CONTRACT_V1"

EXPECTED_FUNCTIONS = {
    "glyph_abi_version_v1",
    "glyph_index_open_v1",
    "glyph_index_get_info_v1",
    "glyph_document_get_info_v1",
    "glyph_document_path_v1",
    "glyph_query_count_v1",
    "glyph_query_locate_v1",
    "glyph_index_close_v1",
}

EXPECTED_STATUS = {
    "GLYPH_OK": 0,
    "GLYPH_E_ARG": 1,
    "GLYPH_E_FORMAT": 2,
    "GLYPH_E_VERIFY": 3,
    "GLYPH_E_VERSION": 4,
    "GLYPH_E_IO": 5,
    "GLYPH_E_NOMEM": 6,
    "GLYPH_E_LIMIT": 7,
    "GLYPH_E_TIMEOUT": 8,
    "GLYPH_E_BUSY": 9,
    "GLYPH_E_CLOSED": 10,
    "GLYPH_E_INTERNAL": 11,
    "GLYPH_E_UNSUPPORTED": 12,
}

EXPECTED_LAYOUT = {
    "glyph_open_options_v1": 72,
    "glyph_query_options_v1": 56,
    "glyph_coordinate_v1": 16,
    "glyph_locate_result_v1": 56,
    "glyph_index_info_v1": 120,
    "glyph_document_info_v1": 96,
}


class ContractError(RuntimeError):
    pass


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json_bytes(
    value: Any,
) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def audit_header_text(
    text: str,
) -> dict[str, Any]:
    for forbidden in (
        "std::",
        "size_t",
        "template<",
        "throw ",
        "new ",
        "delete ",
    ):
        require(
            forbidden not in text,
            f"forbidden public-header token: {forbidden}",
        )

    typedef_names = set(
        re.findall(
            r"}\s*(glyph_[a-z0-9_]+)\s*;",
            text,
        )
    )

    typedef_names.update(
        re.findall(
            r"typedef\s+struct\s+"
            r"[a-z0-9_]+\s+"
            r"(glyph_[a-z0-9_]+)\s*;",
            text,
        )
    )

    function_names = set(
        re.findall(
            r"\b(glyph_[a-z0-9_]+_v1)\s*\(",
            text,
        )
    )

    collisions = sorted(
        typedef_names & function_names
    )

    require(
        not collisions,
        "ordinary C identifier collision: "
        + ",".join(collisions),
    )

    require(
        function_names == EXPECTED_FUNCTIONS,
        "exported function set mismatch",
    )

    status_pairs = re.findall(
        r"^\s*(GLYPH_(?:OK|E_[A-Z_]+))"
        r"\s*=\s*([0-9]+)\s*,?\s*$",
        text,
        flags=re.MULTILINE,
    )

    status_map = {
        name: int(number)
        for name, number in status_pairs
    }

    require(
        status_map == EXPECTED_STATUS,
        "status-code map mismatch",
    )

    for structure in (
        "glyph_open_options_v1",
        "glyph_query_options_v1",
        "glyph_locate_result_v1",
        "glyph_index_info_v1",
        "glyph_document_info_v1",
    ):
        pattern = (
            rf"typedef struct {structure}\s*\{{"
            rf"\s*uint32_t struct_size;"
        )

        require(
            re.search(pattern, text) is not None,
            f"struct_size is not first: {structure}",
        )

    require(
        "GLYPH_SUPPORTED_POINTER_BITS_MIN_V1 "
        "UINT32_C(64)"
        in text,
        "64-bit minimum policy missing",
    )

    return {
        "typedef_count": len(typedef_names),
        "function_count": len(function_names),
        "status_count": len(status_map),
        "identifier_namespace_ok": True,
        "fixed_width_public_types": True,
    }


def expect_header_rejection(
    name: str,
    text: str,
) -> dict[str, Any]:
    try:
        audit_header_text(text)
    except ContractError as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise ContractError(
        f"header mutation unexpectedly accepted: {name}"
    )


def compiler_path(name: str) -> str:
    path = shutil.which(name)

    if path is None:
        raise ContractError(
            f"required compiler missing: {name}"
        )

    return path


def run_compile(
    command: list[str],
) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise ContractError(
            "compile check failed:\n"
            + completed.stdout
            + completed.stderr
        )


def consumer_source(
    *,
    cpp: bool,
    layout_asserts: bool,
) -> str:
    assert_keyword = (
        "static_assert"
        if cpp
        else "_Static_assert"
    )

    assertions = ""

    if layout_asserts:
        assertions = f'''
{assert_keyword}(
    sizeof(void *) == 8,
    "V1 supported profile requires 64-bit pointers"
);

{assert_keyword}(
    sizeof(glyph_open_options_v1) == 72,
    "glyph_open_options_v1 size"
);
{assert_keyword}(
    offsetof(glyph_open_options_v1, struct_size) == 0,
    "glyph_open_options_v1.struct_size"
);
{assert_keyword}(
    offsetof(glyph_open_options_v1, max_mapped_bytes) == 8,
    "glyph_open_options_v1.max_mapped_bytes"
);
{assert_keyword}(
    offsetof(glyph_open_options_v1, reserved) == 32,
    "glyph_open_options_v1.reserved"
);

{assert_keyword}(
    sizeof(glyph_query_options_v1) == 56,
    "glyph_query_options_v1 size"
);
{assert_keyword}(
    offsetof(glyph_query_options_v1, timeout_ns) == 8,
    "glyph_query_options_v1.timeout_ns"
);
{assert_keyword}(
    offsetof(glyph_query_options_v1, reserved) == 16,
    "glyph_query_options_v1.reserved"
);

{assert_keyword}(
    sizeof(glyph_coordinate_v1) == 16,
    "glyph_coordinate_v1 size"
);
{assert_keyword}(
    offsetof(glyph_coordinate_v1, doc_id) == 0,
    "glyph_coordinate_v1.doc_id"
);
{assert_keyword}(
    offsetof(glyph_coordinate_v1, doc_offset) == 8,
    "glyph_coordinate_v1.doc_offset"
);

{assert_keyword}(
    sizeof(glyph_locate_result_v1) == 56,
    "glyph_locate_result_v1 size"
);
{assert_keyword}(
    offsetof(glyph_locate_result_v1, complete) == 4,
    "glyph_locate_result_v1.complete"
);
{assert_keyword}(
    offsetof(glyph_locate_result_v1, total_matches) == 8,
    "glyph_locate_result_v1.total_matches"
);
{assert_keyword}(
    offsetof(glyph_locate_result_v1, returned_matches) == 16,
    "glyph_locate_result_v1.returned_matches"
);
{assert_keyword}(
    offsetof(glyph_locate_result_v1, reserved) == 24,
    "glyph_locate_result_v1.reserved"
);

{assert_keyword}(
    sizeof(glyph_index_info_v1) == 120,
    "glyph_index_info_v1 size"
);
{assert_keyword}(
    offsetof(glyph_index_info_v1, document_count) == 8,
    "glyph_index_info_v1.document_count"
);
{assert_keyword}(
    offsetof(glyph_index_info_v1, corpus_id_sha256) == 24,
    "glyph_index_info_v1.corpus_id_sha256"
);
{assert_keyword}(
    offsetof(glyph_index_info_v1, runtime_index_id_sha256) == 56,
    "glyph_index_info_v1.runtime_index_id_sha256"
);
{assert_keyword}(
    offsetof(glyph_index_info_v1, reserved) == 88,
    "glyph_index_info_v1.reserved"
);

{assert_keyword}(
    sizeof(glyph_document_info_v1) == 96,
    "glyph_document_info_v1 size"
);
{assert_keyword}(
    offsetof(glyph_document_info_v1, doc_id) == 8,
    "glyph_document_info_v1.doc_id"
);
{assert_keyword}(
    offsetof(glyph_document_info_v1, source_sha256) == 24,
    "glyph_document_info_v1.source_sha256"
);
{assert_keyword}(
    offsetof(glyph_document_info_v1, path_length_bytes) == 56,
    "glyph_document_info_v1.path_length_bytes"
);
{assert_keyword}(
    offsetof(glyph_document_info_v1, reserved) == 64,
    "glyph_document_info_v1.reserved"
);
'''

    return f'''
#include <stddef.h>
#include <stdint.h>
#include "glyph/glyph.h"

{assertions}

static uint32_t (*p_abi)(void) =
    glyph_abi_version_v1;

static glyph_status_v1 (*p_open)(
    const char *,
    const glyph_open_options_v1 *,
    glyph_index_v1 **
) = glyph_index_open_v1;

static glyph_status_v1 (*p_index_info)(
    glyph_index_v1 *,
    glyph_index_info_v1 *
) = glyph_index_get_info_v1;

static glyph_status_v1 (*p_document_info)(
    glyph_index_v1 *,
    uint64_t,
    glyph_document_info_v1 *
) = glyph_document_get_info_v1;

static glyph_status_v1 (*p_document_path)(
    glyph_index_v1 *,
    uint64_t,
    uint8_t *,
    uint64_t,
    uint64_t *
) = glyph_document_path_v1;

static glyph_status_v1 (*p_count)(
    glyph_index_v1 *,
    const uint8_t *,
    uint64_t,
    const glyph_query_options_v1 *,
    uint64_t *
) = glyph_query_count_v1;

static glyph_status_v1 (*p_locate)(
    glyph_index_v1 *,
    const uint8_t *,
    uint64_t,
    const glyph_query_options_v1 *,
    uint64_t,
    glyph_coordinate_v1 *,
    uint64_t,
    glyph_locate_result_v1 *
) = glyph_query_locate_v1;

static glyph_status_v1 (*p_close)(
    glyph_index_v1 **
) = glyph_index_close_v1;

int main(void) {{
    glyph_open_options_v1 open_options =
        GLYPH_OPEN_OPTIONS_V1_INIT;

    glyph_query_options_v1 query_options =
        GLYPH_QUERY_OPTIONS_V1_INIT;

    glyph_locate_result_v1 locate_result =
        GLYPH_LOCATE_RESULT_V1_INIT;

    glyph_index_info_v1 index_info =
        GLYPH_INDEX_INFO_V1_INIT;

    glyph_document_info_v1 document_info =
        GLYPH_DOCUMENT_INFO_V1_INIT;

    (void)open_options;
    (void)query_options;
    (void)locate_result;
    (void)index_info;
    (void)document_info;

    (void)p_abi;
    (void)p_open;
    (void)p_index_info;
    (void)p_document_info;
    (void)p_document_path;
    (void)p_count;
    (void)p_locate;
    (void)p_close;

    return 0;
}}
'''


def compile_contract() -> dict[str, Any]:
    cc = compiler_path(
        os.environ.get("CC", "cc")
    )
    cxx = compiler_path(
        os.environ.get("CXX", "c++")
    )

    with tempfile.TemporaryDirectory(
        prefix="glyph-i0-contract-"
    ) as raw:
        directory = Path(raw)

        c99 = directory / "consumer-c99.c"
        c11 = directory / "consumer-c11.c"
        cpp = directory / "consumer-cpp17.cpp"

        c99.write_text(
            consumer_source(
                cpp=False,
                layout_asserts=False,
            )
        )
        c11.write_text(
            consumer_source(
                cpp=False,
                layout_asserts=True,
            )
        )
        cpp.write_text(
            consumer_source(
                cpp=True,
                layout_asserts=True,
            )
        )

        common = [
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-Iinclude",
            "-fsyntax-only",
        ]

        run_compile(
            [
                cc,
                "-std=c99",
                *common,
                str(c99),
            ]
        )

        run_compile(
            [
                cc,
                "-std=c11",
                *common,
                str(c11),
            ]
        )

        run_compile(
            [
                cxx,
                "-std=c++17",
                *common,
                str(cpp),
            ]
        )

    return {
        "strict_c99_header": True,
        "strict_c11_header": True,
        "cpp17_consumer_header": True,
        "layout_static_assertions": True,
        "function_pointer_signatures": True,
    }


def audit_specs() -> dict[str, Any]:
    abi = (
        ROOT
        / "docs/specs/GLYPH_C_ABI_V1.md"
    ).read_text()

    threat = (
        ROOT
        / "docs/specs/"
        / "GLYPH_EMBEDDED_THREAT_MODEL_V1.md"
    ).read_text()

    mmap_spec = (
        ROOT
        / "docs/specs/"
        / "GLYPH_MMAP_TRUST_MODEL_V1.md"
    ).read_text()

    resource = (
        ROOT
        / "docs/specs/"
        / "GLYPH_RESOURCE_FAILURE_MODEL_V1.md"
    ).read_text()

    signature = (
        ROOT
        / "docs/specs/"
        / "GLYPH_SIGNED_STATEMENT_V1.md"
    ).read_text()

    required_abi_phrases = (
        "64-bit little-endian Linux",
        "options == NULL",
        "*out_index = NULL",
        "*out_count = 0",
        "Document-path size probe",
        "Repeated close",
        "Close concurrency barrier",
        "GLYPH_E_CLOSED",
        "Frozen V1 structure layout",
    )

    for phrase in required_abi_phrases:
        require(
            phrase in abi,
            f"C ABI specification phrase missing: {phrase}",
        )

    require(
        "no new operation may begin"
        in threat,
        "close barrier missing from threat model",
    )

    for phrase in (
        "opened directory descriptor",
        "descriptor-relative lookup",
        "same opened file description",
    ):
        require(
            phrase in mmap_spec,
            f"mmap trust phrase missing: {phrase}",
        )

    require(
        "No hidden unbounded coordinate collection"
        in resource,
        "bounded-resource rule missing",
    )

    require(
        "REQUIRE_TRUSTED_SIGNATURE"
        in signature,
        "mandatory signature policy missing",
    )

    return {
        "nullable_options_defined": True,
        "failure_outputs_defined": True,
        "path_probe_defined": True,
        "repeated_close_defined": True,
        "close_barrier_defined": True,
        "host_profile_defined": True,
        "root_directory_anchored": True,
        "signature_policy_defined": True,
    }


def main() -> int:
    header_text = HEADER.read_text()

    header_summary = audit_header_text(
        header_text
    )

    mutations = [
        expect_header_rejection(
            "ordinary_identifier_collision",
            header_text.replace(
                "glyph_index_get_info_v1",
                "glyph_index_info_v1",
                1,
            ),
        ),
        expect_header_rejection(
            "missing_exported_function",
            header_text.replace(
                "glyph_query_count_v1",
                "glyph_query_count_removed_v1",
                1,
            ),
        ),
        expect_header_rejection(
            "forbidden_platform_width_type",
            header_text + "\nsize_t forbidden;\n",
        ),
        expect_header_rejection(
            "renumbered_status_code",
            header_text.replace(
                "GLYPH_E_BUSY = 9",
                "GLYPH_E_BUSY = 99",
                1,
            ),
        ),
    ]

    compile_summary = compile_contract()
    specification_summary = audit_specs()

    implementation_paths = [
        ROOT / "src/glyph_c_api.cpp",
        ROOT / "src/glyph_runtime.cpp",
    ]

    runtime_implementation_present = any(
        path.exists()
        for path in implementation_paths
    )

    require(
        runtime_implementation_present is False,
        "runtime implementation exists before I0 freeze",
    )

    output = {
        "ok": True,
        "format": FORMAT,
        "phase": "I0_CONTRACT_FREEZE_PREPARATION",
        "status": "DRAFT_NOT_FROZEN",
        "abi_version": 1,
        "supported_host_profile":
            "64-bit little-endian Linux",
        "runtime_implementation_present": False,
        "public_header_present": True,
        "contract_spec_count": 5,
        "function_count":
            header_summary["function_count"],
        "status_code_count":
            header_summary["status_count"],
        "identifier_namespace_verified":
            header_summary[
                "identifier_namespace_ok"
            ],
        "fixed_width_public_types":
            header_summary[
                "fixed_width_public_types"
            ],
        "strict_c99_header":
            compile_summary["strict_c99_header"],
        "strict_c11_header":
            compile_summary["strict_c11_header"],
        "cpp17_consumer_header":
            compile_summary[
                "cpp17_consumer_header"
            ],
        "layout_static_assertions":
            compile_summary[
                "layout_static_assertions"
            ],
        "function_pointer_signatures":
            compile_summary[
                "function_pointer_signatures"
            ],
        "layout": EXPECTED_LAYOUT,
        **specification_summary,
        "mutation_count": len(mutations),
        "mutations": mutations,
        "next":
            "EXTERNAL_LINE_BY_LINE_PREFREEZE_REVIEW",
    }

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_bytes(
        canonical_json_bytes(output)
    )

    print(
        canonical_json_bytes(output).decode(
            "utf-8"
        ),
        end="",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
