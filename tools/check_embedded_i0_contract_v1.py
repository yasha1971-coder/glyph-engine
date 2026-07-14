#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
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


EXPECTED_PUBLIC_STRUCTS = {
    "glyph_open_options_v1",
    "glyph_query_options_v1",
    "glyph_coordinate_v1",
    "glyph_locate_result_v1",
    "glyph_index_info_v1",
    "glyph_document_info_v1",
}

ALLOWED_PUBLIC_FIELD_TYPES = {
    "uint8_t",
    "uint32_t",
    "uint64_t",
}

ALLOWED_PUBLIC_RETURN_TYPES = {
    "uint32_t",
    "glyph_status_v1",
}

ALLOWED_PUBLIC_PARAMETER_TYPES = {
    "const char*",
    "const glyph_open_options_v1*",
    "const glyph_query_options_v1*",
    "const uint8_t*",
    "glyph_coordinate_v1*",
    "glyph_document_info_v1*",
    "glyph_index_info_v1*",
    "glyph_index_v1*",
    "glyph_index_v1**",
    "glyph_locate_result_v1*",
    "uint64_t",
    "uint64_t*",
    "uint8_t*",
}

LAYOUT_ASSERTIONS = [
    (
        "pointer_bits",
        "sizeof(void *) == 8",
        "V1 supported profile requires 64-bit pointers",
    ),
    (
        "open_options_size",
        "sizeof(glyph_open_options_v1) == 72",
        "glyph_open_options_v1 size",
    ),
    (
        "open_options_struct_size",
        "offsetof(glyph_open_options_v1, struct_size) == 0",
        "glyph_open_options_v1.struct_size",
    ),
    (
        "open_options_max_mapped_bytes",
        "offsetof(glyph_open_options_v1, max_mapped_bytes) == 8",
        "glyph_open_options_v1.max_mapped_bytes",
    ),
    (
        "open_options_reserved",
        "offsetof(glyph_open_options_v1, reserved) == 32",
        "glyph_open_options_v1.reserved",
    ),
    (
        "query_options_size",
        "sizeof(glyph_query_options_v1) == 56",
        "glyph_query_options_v1 size",
    ),
    (
        "query_options_timeout",
        "offsetof(glyph_query_options_v1, timeout_ns) == 8",
        "glyph_query_options_v1.timeout_ns",
    ),
    (
        "query_options_reserved",
        "offsetof(glyph_query_options_v1, reserved) == 16",
        "glyph_query_options_v1.reserved",
    ),
    (
        "coordinate_size",
        "sizeof(glyph_coordinate_v1) == 16",
        "glyph_coordinate_v1 size",
    ),
    (
        "coordinate_doc_id",
        "offsetof(glyph_coordinate_v1, doc_id) == 0",
        "glyph_coordinate_v1.doc_id",
    ),
    (
        "coordinate_doc_offset",
        "offsetof(glyph_coordinate_v1, doc_offset) == 8",
        "glyph_coordinate_v1.doc_offset",
    ),
    (
        "locate_result_size",
        "sizeof(glyph_locate_result_v1) == 56",
        "glyph_locate_result_v1 size",
    ),
    (
        "locate_result_complete",
        "offsetof(glyph_locate_result_v1, complete) == 4",
        "glyph_locate_result_v1.complete",
    ),
    (
        "locate_result_total",
        "offsetof(glyph_locate_result_v1, total_matches) == 8",
        "glyph_locate_result_v1.total_matches",
    ),
    (
        "locate_result_returned",
        "offsetof(glyph_locate_result_v1, returned_matches) == 16",
        "glyph_locate_result_v1.returned_matches",
    ),
    (
        "locate_result_reserved",
        "offsetof(glyph_locate_result_v1, reserved) == 24",
        "glyph_locate_result_v1.reserved",
    ),
    (
        "index_info_size",
        "sizeof(glyph_index_info_v1) == 120",
        "glyph_index_info_v1 size",
    ),
    (
        "index_info_document_count",
        "offsetof(glyph_index_info_v1, document_count) == 8",
        "glyph_index_info_v1.document_count",
    ),
    (
        "index_info_corpus_id",
        "offsetof(glyph_index_info_v1, corpus_id_sha256) == 24",
        "glyph_index_info_v1.corpus_id_sha256",
    ),
    (
        "index_info_runtime_id",
        "offsetof(glyph_index_info_v1, runtime_index_id_sha256) == 56",
        "glyph_index_info_v1.runtime_index_id_sha256",
    ),
    (
        "index_info_format_version",
        "offsetof(glyph_index_info_v1, index_format_version) == 88",
        "glyph_index_info_v1.index_format_version",
    ),
    (
        "index_info_runtime_profile",
        "offsetof(glyph_index_info_v1, runtime_profile_id) == 92",
        "glyph_index_info_v1.runtime_profile_id",
    ),
    (
        "index_info_reserved",
        "offsetof(glyph_index_info_v1, reserved) == 96",
        "glyph_index_info_v1.reserved",
    ),
    (
        "document_info_size",
        "sizeof(glyph_document_info_v1) == 96",
        "glyph_document_info_v1 size",
    ),
    (
        "document_info_doc_id",
        "offsetof(glyph_document_info_v1, doc_id) == 8",
        "glyph_document_info_v1.doc_id",
    ),
    (
        "document_info_source_hash",
        "offsetof(glyph_document_info_v1, source_sha256) == 24",
        "glyph_document_info_v1.source_sha256",
    ),
    (
        "document_info_path_length",
        "offsetof(glyph_document_info_v1, path_length_bytes) == 56",
        "glyph_document_info_v1.path_length_bytes",
    ),
    (
        "document_info_reserved",
        "offsetof(glyph_document_info_v1, reserved) == 64",
        "glyph_document_info_v1.reserved",
    ),
]


SPEC_PATHS = (
    "docs/specs/GLYPH_C_ABI_V1.md",
    "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md",
    "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md",
    "docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md",
    "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md",
)

CONTRACT_SOURCE_PATHS = (
    "include/glyph/glyph.h",
    *SPEC_PATHS,
    "tools/check_embedded_i0_contract_v1.py",
)

REQUIRED_SPEC_HEADINGS = {
    "docs/specs/GLYPH_C_ABI_V1.md": {
        "## ABI version",
        "## Close behavior",
        "## Locate result semantics",
        "## Document identity",
        "## Document path semantics",
        "## Structure versioning",
        "## Status codes",
        "## Open options",
        "## Thread-safety contract",
        "### Initial supported host profile",
        "### Nullable option structures",
        "### Exact output initialization",
        "### Exact argument rules",
        "### Document-path size probe",
        "### Repeated close",
        "### Close concurrency barrier",
        "### Frozen V1 structure layout",
    },
    "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md": {
        "## Hostile inputs",
        "## Trusted assumptions",
        "## Filesystem attacker boundary",
        "## Concurrency boundary",
        "## Resource boundary",
    },
    "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md": {
        "## Required publication model",
        "## Root directory anchoring",
        "## Required open sequence",
        "## Required structural checks",
        "## Permitted integrity claim",
        "## Forbidden integrity claim",
        "## Post-open mutation",
        "## File change during open",
    },
    "docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md": {
        "## Query-plane limits",
        "## Checked arithmetic",
        "## Bounded locate",
        "## Open-time work boundary",
        "## Timeout contract",
        "## Partial-result rule",
        "## Build-plane isolation",
        "## Interrupted build",
    },
    "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md": {
        "## Verifier-side policy",
        "## Signed statement fields",
        "## Signing preimage",
        "## Canonical statement encoding",
        "## Target algorithm",
        "## Required rejection cases",
    },
}

CROSS_FILE_REQUIREMENTS = {
    "document_identity_dense": (
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Document identity",
            "0 <= doc_id < document_count",
        ),
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Document identity",
            "committed canonical source-manifest order",
        ),
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Document identity",
            "out-of-range `doc_id` returns `GLYPH_E_ARG`",
        ),
    ),
    "close_model_consistent": (
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "### Close concurrency barrier",
            "no persistent library-side close barrier or closing latch",
        ),
        (
            "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md",
            "## Concurrency boundary",
            "atomic active-operation count",
        ),
        (
            "docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md",
            "## Partial-result rule",
            "close/query race is not a recoverable query result class",
        ),
    ),
    "mapped_bytes_validated": (
        (
            "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md",
            "## Required open sequence",
            "create the final read-only mapping",
        ),
        (
            "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md",
            "## Required open sequence",
            "compute full SHA-256 through the final mapped region",
        ),
        (
            "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md",
            "## File change during open",
            "size-preserving rewrite",
        ),
        (
            "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md",
            "## Filesystem attacker boundary",
            "hostile local writer",
        ),
    ),
    "path_validation_defined": (
        (
            "docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md",
            "## Required structural checks",
            "free of `.` and `..` components",
        ),
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Document path semantics",
            "raw relative path bytes",
        ),
    ),
    "signature_policy_defined": (
        (
            "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md",
            "## Verifier-side policy",
            "ALLOW_UNSIGNED",
        ),
        (
            "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md",
            "## Verifier-side policy",
            "REQUIRE_TRUSTED_SIGNATURE",
        ),
        (
            "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md",
            "## Verifier-side policy",
            "signature stripping",
        ),
        (
            "docs/specs/GLYPH_SIGNED_STATEMENT_V1.md",
            "## Canonical statement encoding",
            "mandatory pre-implementation gate",
        ),
    ),
    "pointer_memory_directionality": (
        (
            "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md",
            "## Trusted assumptions",
            "valid readable caller memory for every input pointer",
        ),
        (
            "docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md",
            "## Trusted assumptions",
            "valid writable caller memory for every output pointer",
        ),
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Query input lifetime",
            "The runtime must not: - retain the query pointer; - write to query memory",
        ),
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Locate output model",
            "caller-owned storage",
        ),
    ),
    "open_deadline_nonclaim": (
        (
            "docs/specs/GLYPH_C_ABI_V1.md",
            "## Open options",
            "no open deadline",
        ),
        (
            "docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md",
            "## Open-time work boundary",
            "no open deadline",
        ),
    ),
}

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
}

SOURCE_SCAN_EXCLUDED_ROOTS = {
    ".git",
    "benchmarks",
    "build",
    "docs",
    "include",
    "tools",
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

    struct_pattern = re.compile(
        r"typedef\s+struct\s+"
        r"(?P<tag>glyph_[a-z0-9_]+)"
        r"\s*\{"
        r"(?P<body>.*?)"
        r"\}\s*"
        r"(?P<alias>glyph_[a-z0-9_]+)"
        r"\s*;",
        flags=re.DOTALL,
    )

    struct_matches = list(
        struct_pattern.finditer(text)
    )

    public_structs = {
        match.group("alias")
        for match in struct_matches
    }

    require(
        public_structs
        == EXPECTED_PUBLIC_STRUCTS,
        "public structure set mismatch",
    )

    public_field_count = 0

    for match in struct_matches:
        tag = match.group("tag")
        alias = match.group("alias")

        require(
            tag == alias,
            f"struct tag/alias mismatch: {tag}/{alias}",
        )

        body = re.sub(
            r"/\*.*?\*/",
            "",
            match.group("body"),
            flags=re.DOTALL,
        )

        body = re.sub(
            r"//[^\n]*",
            "",
            body,
        )

        declarations = [
            item.strip()
            for item in body.split(";")
            if item.strip()
        ]

        for declaration in declarations:
            field = re.fullmatch(
                r"(?P<base>[A-Za-z_][A-Za-z0-9_]*)"
                r"\s+"
                r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
                r"(?:"
                r"\s*\[\s*"
                r"(?P<extent>[A-Za-z0-9_]+)"
                r"\s*\]"
                r")?",
                declaration,
            )

            require(
                field is not None,
                "unparseable public structure field: "
                + declaration,
            )

            base = field.group("base")

            require(
                base in ALLOWED_PUBLIC_FIELD_TYPES,
                "forbidden public field type: "
                + base,
            )

            public_field_count += 1

    prototype_pattern = re.compile(
        r"GLYPH_API"
        r"\s+"
        r"(?P<return_type>"
        r"[A-Za-z_][A-Za-z0-9_]*"
        r")"
        r"\s+"
        r"GLYPH_CALL"
        r"\s+"
        r"(?P<name>"
        r"glyph_[a-z0-9_]+_v1"
        r")"
        r"\s*\("
        r"(?P<parameters>.*?)"
        r"\)"
        r"\s*;",
        flags=re.DOTALL,
    )

    prototypes = list(
        prototype_pattern.finditer(text)
    )

    annotated_names = [
        match.group("name")
        for match in prototypes
    ]

    require(
        len(annotated_names)
        == len(set(annotated_names)),
        "duplicate annotated exported function",
    )

    require(
        set(annotated_names)
        == EXPECTED_FUNCTIONS,
        "GLYPH_API/GLYPH_CALL export annotation mismatch",
    )

    public_parameter_count = 0

    for match in prototypes:
        return_type = match.group(
            "return_type"
        )

        require(
            return_type
            in ALLOWED_PUBLIC_RETURN_TYPES,
            "forbidden public return type: "
            + return_type,
        )

        raw_parameters = (
            match.group("parameters").strip()
        )

        if raw_parameters == "void":
            continue

        require(
            raw_parameters != "",
            "empty C parameter list must use void",
        )

        for raw_parameter in raw_parameters.split(","):
            parameter = " ".join(
                raw_parameter.split()
            )

            name_match = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)$",
                parameter,
            )

            require(
                name_match is not None,
                "unparseable public parameter: "
                + parameter,
            )

            type_expression = (
                parameter[
                    :name_match.start()
                ].strip()
            )

            type_expression = re.sub(
                r"\s*\*\s*",
                "*",
                type_expression,
            )

            type_expression = " ".join(
                type_expression.split()
            )

            require(
                type_expression
                in ALLOWED_PUBLIC_PARAMETER_TYPES,
                "forbidden public parameter type: "
                + type_expression,
            )

            public_parameter_count += 1

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
        "public_struct_count":
            len(public_structs),
        "public_field_count":
            public_field_count,
        "public_parameter_count":
            public_parameter_count,
        "identifier_namespace_ok": True,
        "fixed_width_public_types": True,
        "export_annotations_verified": True,
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


def layout_assertion_source(
    style: str,
) -> str:
    lines: list[str] = []

    for identifier, expression, message in (
        LAYOUT_ASSERTIONS
    ):
        if style == "c99":
            lines.append(
                "typedef char "
                f"glyph_i0_assert_{identifier}"
                f"[({expression}) ? 1 : -1];"
            )
        elif style == "c11":
            lines.append(
                "_Static_assert("
                f"{expression}, "
                f"\"{message}\""
                ");"
            )
        elif style == "cpp17":
            lines.append(
                "static_assert("
                f"{expression}, "
                f"\"{message}\""
                ");"
            )
        else:
            raise ContractError(
                "unknown layout assertion style: "
                + style
            )

    return "\n".join(lines)


def consumer_source(
    *,
    cpp: bool,
    layout_style: str,
) -> str:
    assertions = layout_assertion_source(
        layout_style
    )

    return f"""
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
"""


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
                layout_style="c99",
            )
        )

        c11.write_text(
            consumer_source(
                cpp=False,
                layout_style="c11",
            )
        )

        cpp.write_text(
            consumer_source(
                cpp=True,
                layout_style="cpp17",
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
        "c99_layout_assertions": True,
        "c11_layout_assertions": True,
        "cpp17_layout_assertions": True,
        "layout_static_assertions": True,
        "function_pointer_signatures": True,
    }


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_markdown(
    text: str,
) -> str:
    return " ".join(text.split())


def load_spec_texts() -> dict[str, str]:
    result: dict[str, str] = {}

    for relative in SPEC_PATHS:
        path = ROOT / relative

        require(
            path.is_file(),
            f"specification missing: {relative}",
        )

        raw = path.read_bytes()

        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError(
                f"specification is not UTF-8: {relative}"
            ) from error

        result[relative] = value

    return result


def extract_spec_sections(
    text: str,
    relative: str,
) -> dict[str, str]:
    lines = text.splitlines(
        keepends=True
    )

    headings: list[
        tuple[int, str, int]
    ] = []

    for index, line in enumerate(lines):
        candidate = line.rstrip("\r\n")

        match = re.match(
            r"^(#{2,3})\s+.+$",
            candidate,
        )

        if match is None:
            continue

        headings.append(
            (
                index,
                candidate,
                len(match.group(1)),
            )
        )

    require(
        headings,
        f"no normative headings found: {relative}",
    )

    names = [
        heading
        for _, heading, _ in headings
    ]

    require(
        len(names) == len(set(names)),
        f"duplicate normative heading: {relative}",
    )

    sections: dict[str, str] = {}

    for position, (
        start,
        heading,
        level,
    ) in enumerate(headings):
        end = len(lines)

        for next_start, _, next_level in (
            headings[position + 1:]
        ):
            if next_level <= level:
                end = next_start
                break

        section = "".join(
            lines[start:end]
        )

        if not section.endswith("\n"):
            section += "\n"

        sections[heading] = section

    return sections


def parse_layout_table(
    section: str,
) -> dict[str, int]:
    pairs = re.findall(
        r"\|\s*`(?P<name>glyph_[a-z0-9_]+)`"
        r"\s*\|\s*(?P<size>[0-9]+)\s+bytes\s*\|",
        section,
    )

    return {
        name: int(size)
        for name, size in pairs
    }


def expect_spec_rejection(
    name: str,
    spec_texts: dict[str, str],
) -> dict[str, Any]:
    try:
        audit_specs(spec_texts)
    except ContractError as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise ContractError(
        f"spec mutation unexpectedly accepted: {name}"
    )


def audit_specs(
    supplied_texts: dict[str, str] | None = None,
) -> dict[str, Any]:
    texts = (
        load_spec_texts()
        if supplied_texts is None
        else dict(supplied_texts)
    )

    require(
        set(texts) == set(SPEC_PATHS),
        "specification file set mismatch",
    )

    section_texts: dict[
        str,
        dict[str, str],
    ] = {}

    section_records: dict[
        str,
        dict[str, dict[str, Any]],
    ] = {}

    file_hashes: dict[str, str] = {}

    for relative in SPEC_PATHS:
        text = texts[relative]

        sections = extract_spec_sections(
            text,
            relative,
        )

        section_texts[relative] = sections

        required = REQUIRED_SPEC_HEADINGS[
            relative
        ]

        missing = sorted(
            required - set(sections)
        )

        require(
            not missing,
            "required normative heading missing: "
            + relative
            + ": "
            + ", ".join(missing),
        )

        file_hashes[relative] = sha256_bytes(
            text.encode("utf-8")
        )

        section_records[relative] = {}

        for heading, section in (
            sections.items()
        ):
            encoded = section.encode("utf-8")

            section_records[relative][
                heading
            ] = {
                "sha256": sha256_bytes(
                    encoded
                ),
                "size_bytes": len(encoded),
            }

    cross_file_invariants: dict[
        str,
        bool,
    ] = {}

    for invariant, requirements in (
        CROSS_FILE_REQUIREMENTS.items()
    ):
        for relative, heading, marker in (
            requirements
        ):
            section = section_texts[
                relative
            ][heading]

            normalized = normalize_markdown(
                section
            )

            require(
                marker in normalized,
                f"{invariant}: normative marker "
                f"missing: {relative}: "
                f"{heading}: {marker}",
            )

        cross_file_invariants[
            invariant
        ] = True

    all_text = "\n".join(
        texts[relative]
        for relative in SPEC_PATHS
    )

    referenced_statuses = set(
        re.findall(
            r"\bGLYPH_(?:OK|E_[A-Z_]+)\b",
            all_text,
        )
    )

    expected_statuses = set(
        EXPECTED_STATUS
    )

    require(
        referenced_statuses
        == expected_statuses,
        "specification status registry mismatch",
    )

    status_section = section_texts[
        "docs/specs/GLYPH_C_ABI_V1.md"
    ]["## Status codes"]

    status_section_names = set(
        re.findall(
            r"`(GLYPH_(?:OK|E_[A-Z_]+))`",
            status_section,
        )
    )

    require(
        status_section_names
        == expected_statuses,
        "C ABI status section mismatch",
    )

    layout_section = section_texts[
        "docs/specs/GLYPH_C_ABI_V1.md"
    ]["### Frozen V1 structure layout"]

    specification_layout = (
        parse_layout_table(
            layout_section
        )
    )

    require(
        specification_layout
        == EXPECTED_LAYOUT,
        "C ABI structure-layout table mismatch",
    )

    section_count = sum(
        len(records)
        for records in section_records.values()
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
        "normative_section_hashes_verified":
            True,
        "normative_section_count":
            section_count,
        "normative_sections":
            section_records,
        "specification_file_sha256":
            file_hashes,
        "cross_file_invariants_verified":
            True,
        "cross_file_invariant_count":
            len(cross_file_invariants),
        "cross_file_invariants":
            cross_file_invariants,
        "spec_status_registry_verified":
            True,
        "spec_layout_table_verified":
            True,
        "referenced_statuses":
            sorted(referenced_statuses),
    }


def source_has_embedded_definition(
    text: str,
) -> tuple[str, str] | None:
    for name in sorted(
        EXPECTED_FUNCTIONS
    ):
        pattern = (
            r"\b"
            + re.escape(name)
            + r"\s*\("
            + r"[^;{}]*"
            + r"\)"
            + r"\s*(?:noexcept\s*)?"
            + r"\{"
        )

        if re.search(
            pattern,
            text,
            flags=re.DOTALL,
        ):
            return (
                "public_function_definition",
                name,
            )

    if re.search(
        r"#\s*include\s*"
        r"[<\"]glyph/glyph\.h[>\"]",
        text,
    ):
        return (
            "public_header_include",
            "glyph/glyph.h",
        )

    return None

def tracked_source_files() -> list[Path]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    files: list[Path] = []

    for line in completed.stdout.splitlines():
        if not line:
            continue

        path = ROOT / line

        if (
            not path.is_file()
            or path.suffix
            not in SOURCE_SUFFIXES
        ):
            continue

        relative = path.relative_to(ROOT)

        if (
            relative.parts
            and relative.parts[0]
            in SOURCE_SCAN_EXCLUDED_ROOTS
        ):
            continue

        files.append(path)

    return sorted(files)

def scan_runtime_implementation() -> dict[str, Any]:
    hits: list[dict[str, str]] = []
    scanned = 0

    tracked = tracked_source_files()

    scanned = len(tracked)

    for path in tracked:

        if (
            not path.is_file()
            or path.suffix
            not in SOURCE_SUFFIXES
        ):
            continue

        relative = path.relative_to(ROOT)

        if (
            relative.parts
            and relative.parts[0]
            in SOURCE_SCAN_EXCLUDED_ROOTS
        ):
            continue


        text = path.read_text(
            errors="replace"
        )

        reason = (
            source_has_embedded_definition(
                text
            )
        )

        if reason is None:
            continue

        kind, symbol = reason

        hits.append(
            {
                "path": relative.as_posix(),
                "reason": kind,
                "symbol": symbol,
            }
        )

    require(
        not hits,
        "runtime implementation exists "
        "before I0 freeze: "
        + json.dumps(
            hits,
            sort_keys=True,
        ),
    )

    return {
        "runtime_implementation_present":
            False,
        "implementation_source_files_scanned":
            scanned,
        "implementation_hits": hits,
        "implementation_tree_scan_verified":
            True,
    }


def expect_implementation_detection() -> dict[str, Any]:
    synthetic = """
#include "glyph/glyph.h"

glyph_status_v1 glyph_query_count_v1(
    glyph_index_v1 *index,
    const uint8_t *query,
    uint64_t query_size,
    const glyph_query_options_v1 *options,
    uint64_t *out_count
) {
    return GLYPH_E_UNSUPPORTED;
}
"""

    detected = (
        source_has_embedded_definition(
            synthetic
        )
    )

    require(
        detected is not None,
        "synthetic embedded implementation "
        "was not detected",
    )

    return {
        "mutation":
            "premature_runtime_implementation",
        "rejected": True,
        "message":
            "synthetic ABI implementation detected",
    }


def contract_source_commit() -> str:
    command = [
        "git",
        "rev-list",
        "-1",
        "HEAD",
        "--",
        *CONTRACT_SOURCE_PATHS,
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    require(
        completed.returncode == 0,
        "failed to resolve reviewed commit: "
        + completed.stderr,
    )

    commit = completed.stdout.strip()

    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            commit,
        )
        is not None,
        "invalid reviewed commit",
    )

    return commit


def ensure_contract_sources_clean() -> None:
    completed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *CONTRACT_SOURCE_PATHS,
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    require(
        completed.returncode == 0,
        "failed to inspect contract-source state",
    )

    require(
        completed.stdout.strip() == "",
        "contract sources are not committed:\n"
        + completed.stdout,
    )


def contract_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}

    for relative in CONTRACT_SOURCE_PATHS:
        path = ROOT / relative

        require(
            path.is_file(),
            f"contract source missing: {relative}",
        )

        result[relative] = sha256_bytes(
            path.read_bytes()
        )

    return result


def build_output(
    reviewed_commit: str,
) -> dict[str, Any]:
    require(
        re.fullmatch(
            r"[0-9a-f]{40}",
            reviewed_commit,
        )
        is not None,
        "reviewed_commit must be 40 lowercase hex",
    )

    header_text = HEADER.read_text()

    header_summary = audit_header_text(
        header_text
    )

    field_anchor = (
        "uint32_t struct_size;\n"
        "    uint32_t flags;"
    )

    export_anchor = (
        "GLYPH_API uint32_t GLYPH_CALL\n"
        "glyph_abi_version_v1"
    )

    require(
        header_text.count(field_anchor) >= 1,
        "field mutation anchor missing",
    )

    require(
        header_text.count(export_anchor) == 1,
        "export mutation anchor missing",
    )

    spec_texts = load_spec_texts()

    abi_path = (
        "docs/specs/GLYPH_C_ABI_V1.md"
    )

    threat_path = (
        "docs/specs/"
        "GLYPH_EMBEDDED_THREAT_MODEL_V1.md"
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
            "forbidden_platform_width_long",
            header_text.replace(
                field_anchor,
                (
                    "uint32_t struct_size;\n"
                    "    long forbidden_platform_width;\n"
                    "    uint32_t flags;"
                ),
                1,
            ),
        ),
        expect_header_rejection(
            "forbidden_bool_type",
            header_text.replace(
                field_anchor,
                (
                    "uint32_t struct_size;\n"
                    "    bool forbidden_boolean;\n"
                    "    uint32_t flags;"
                ),
                1,
            ),
        ),
        expect_header_rejection(
            "missing_glyph_api_annotation",
            header_text.replace(
                export_anchor,
                (
                    "uint32_t GLYPH_CALL\n"
                    "glyph_abi_version_v1"
                ),
                1,
            ),
        ),
        expect_header_rejection(
            "missing_glyph_call_annotation",
            header_text.replace(
                export_anchor,
                (
                    "GLYPH_API uint32_t\n"
                    "glyph_abi_version_v1"
                ),
                1,
            ),
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

    missing_heading = dict(spec_texts)
    missing_heading[abi_path] = (
        missing_heading[abi_path].replace(
            "## Document identity",
            "## Removed document identity",
            1,
        )
    )

    mutations.append(
        expect_spec_rejection(
            "missing_normative_heading",
            missing_heading,
        )
    )

    unknown_status = dict(spec_texts)
    unknown_status[abi_path] += (
        "\nUnknown status `GLYPH_E_FAKE`.\n"
    )

    mutations.append(
        expect_spec_rejection(
            "unknown_status_reference",
            unknown_status,
        )
    )

    bad_layout = dict(spec_texts)

    layout_anchor = (
        "| `glyph_index_info_v1` | "
        "120 bytes |"
    )

    require(
        bad_layout[abi_path].count(
            layout_anchor
        )
        == 1,
        "layout mutation anchor missing",
    )

    bad_layout[abi_path] = (
        bad_layout[abi_path].replace(
            layout_anchor,
            (
                "| `glyph_index_info_v1` | "
                "121 bytes |"
            ),
            1,
        )
    )

    mutations.append(
        expect_spec_rejection(
            "layout_table_mismatch",
            bad_layout,
        )
    )

    close_contradiction = dict(spec_texts)

    close_anchor = (
        "no persistent library-side "
        "close barrier or\n"
        "  closing latch"
    )

    require(
        close_contradiction[
            abi_path
        ].count(close_anchor)
        == 1,
        "close-model mutation anchor missing",
    )

    close_contradiction[abi_path] = (
        close_contradiction[
            abi_path
        ].replace(
            close_anchor,
            "persistent library-side closing latch",
            1,
        )
    )

    mutations.append(
        expect_spec_rejection(
            "close_model_contradiction",
            close_contradiction,
        )
    )

    pointer_assumption = dict(spec_texts)

    pointer_anchor = (
        "valid readable caller memory"
    )

    require(
        pointer_assumption[
            threat_path
        ].count(pointer_anchor)
        >= 1,
        "pointer-model mutation anchor missing",
    )

    pointer_assumption[threat_path] = (
        pointer_assumption[
            threat_path
        ].replace(
            pointer_anchor,
            "valid caller memory",
            1,
        )
    )

    mutations.append(
        expect_spec_rejection(
            "pointer_model_regression",
            pointer_assumption,
        )
    )

    mutations.append(
        expect_implementation_detection()
    )

    compile_summary = compile_contract()
    specification_summary = audit_specs(
        spec_texts
    )
    implementation_summary = (
        scan_runtime_implementation()
    )

    output = {
        "ok": True,
        "format": FORMAT,
        "phase":
            "I0_CONTRACT_FREEZE_PREPARATION",
        "status": "DRAFT_NOT_FROZEN",
        "reviewed_commit": reviewed_commit,
        "abi_version": 1,
        "supported_host_profile":
            "64-bit little-endian Linux",
        "public_header_present": True,
        "contract_spec_count":
            len(SPEC_PATHS),
        "contract_source_sha256":
            contract_source_hashes(),
        "function_count":
            header_summary["function_count"],
        "status_code_count":
            header_summary["status_count"],
        "public_struct_count":
            header_summary[
                "public_struct_count"
            ],
        "public_field_count":
            header_summary[
                "public_field_count"
            ],
        "public_parameter_count":
            header_summary[
                "public_parameter_count"
            ],
        "identifier_namespace_verified":
            header_summary[
                "identifier_namespace_ok"
            ],
        "fixed_width_public_types":
            header_summary[
                "fixed_width_public_types"
            ],
        "export_annotations_verified":
            header_summary[
                "export_annotations_verified"
            ],
        "strict_c99_header":
            compile_summary[
                "strict_c99_header"
            ],
        "strict_c11_header":
            compile_summary[
                "strict_c11_header"
            ],
        "cpp17_consumer_header":
            compile_summary[
                "cpp17_consumer_header"
            ],
        "c99_layout_assertions":
            compile_summary[
                "c99_layout_assertions"
            ],
        "c11_layout_assertions":
            compile_summary[
                "c11_layout_assertions"
            ],
        "cpp17_layout_assertions":
            compile_summary[
                "cpp17_layout_assertions"
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
        "write_mode_supported": True,
        "verify_mode_supported": True,
        "committed_result_byte_comparison":
            True,
        **implementation_summary,
        **specification_summary,
        "mutation_count": len(mutations),
        "mutations": mutations,
        "next":
            "SECOND_EXTERNAL_PREFREEZE_REVIEW",
    }

    return output


def atomic_write(
    path: Path,
    payload: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_args(
    argv: list[str] | None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build or verify the GLYPH "
            "Embedded I0 contract artifact."
        )
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--write",
        action="store_true",
        help="write the canonical result artifact",
    )

    mode.add_argument(
        "--verify",
        action="store_true",
        help=(
            "byte-compare the committed result "
            "with a fresh checker run"
        ),
    )

    parser.add_argument(
        "--result",
        type=Path,
        default=RESULT,
        help="result artifact path",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(argv)

    result_path = args.result

    if not result_path.is_absolute():
        result_path = ROOT / result_path

    ensure_contract_sources_clean()

    reviewed_commit = (
        contract_source_commit()
    )

    output = build_output(
        reviewed_commit
    )

    payload = canonical_json_bytes(
        output
    )

    if args.write:
        atomic_write(
            result_path,
            payload,
        )

        print(
            "GLYPH EMBEDDED I0 CONTRACT WRITE PASS"
        )
    else:
        require(
            result_path.is_file(),
            "committed result missing: "
            + str(result_path),
        )

        committed = (
            result_path.read_bytes()
        )

        require(
            committed == payload,
            "committed I0 result mismatch: "
            f"expected_sha256="
            f"{sha256_bytes(payload)} "
            f"actual_sha256="
            f"{sha256_bytes(committed)}",
        )

        print(
            "GLYPH EMBEDDED I0 CONTRACT VERIFY PASS"
        )

    print(
        "reviewed_commit =",
        reviewed_commit,
    )
    print(
        "normative_section_count =",
        output["normative_section_count"],
    )
    print(
        "cross_file_invariant_count =",
        output[
            "cross_file_invariant_count"
        ],
    )
    print(
        "mutation_count =",
        output["mutation_count"],
    )
    print(
        "result_sha256 =",
        sha256_bytes(payload),
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as error:
        print(
            "ERROR:",
            error,
            file=sys.stderr,
        )
        raise SystemExit(1)
