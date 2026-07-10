#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


FORMAT = "GLYPH_CORPUS_IDENTITY_V1"
PROOF_OBLIGATION = "P4"
DOMAIN = FORMAT.encode("ascii") + b"\x00"


class CorpusIdentityError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Document:
    name: str
    data: bytes


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_u64(value: int) -> bytes:
    if not isinstance(value, int):
        raise CorpusIdentityError(
            "invalid_integer",
            f"value must be integer: {value!r}",
        )

    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise CorpusIdentityError(
            "integer_out_of_range",
            f"value outside u64 range: {value}",
        )

    return struct.pack(">Q", value)


def validate_document_name(name: str) -> bytes:
    if not isinstance(name, str):
        raise CorpusIdentityError(
            "invalid_document_name_type",
            f"document name must be str: {type(name).__name__}",
        )

    if "\x00" in name:
        raise CorpusIdentityError(
            "document_name_contains_nul",
            f"document name contains NUL: {name!r}",
        )

    try:
        encoded = name.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise CorpusIdentityError(
            "invalid_utf8_document_name",
            f"document name is not valid UTF-8: {name!r}",
        ) from error

    return encoded


def canonical_preimage(
    documents: Sequence[Document],
    *,
    format_name: str = FORMAT,
) -> bytes:
    if format_name != FORMAT:
        raise CorpusIdentityError(
            "unsupported_format",
            f"unsupported corpus identity format: {format_name}",
        )

    names: set[str] = set()
    parts = [DOMAIN, encode_u64(len(documents))]

    for doc_id, document in enumerate(documents):
        if document.name in names:
            raise CorpusIdentityError(
                "duplicate_document_name",
                f"duplicate document name: {document.name!r}",
            )

        names.add(document.name)

        name_bytes = validate_document_name(document.name)

        if not isinstance(document.data, bytes):
            raise CorpusIdentityError(
                "invalid_document_bytes",
                (
                    f"document {doc_id} data must be bytes, "
                    f"got {type(document.data).__name__}"
                ),
            )

        parts.extend(
            [
                encode_u64(doc_id),
                encode_u64(len(name_bytes)),
                name_bytes,
                encode_u64(len(document.data)),
                sha256_bytes(document.data),
            ]
        )

    return b"".join(parts)


def build_manifest(
    documents: Sequence[Document],
) -> dict[str, Any]:
    preimage = canonical_preimage(documents)
    corpus_id = sha256_hex(preimage)

    manifest_documents = []

    for doc_id, document in enumerate(documents):
        manifest_documents.append(
            {
                "doc_id": doc_id,
                "doc_name": document.name,
                "byte_length": len(document.data),
                "sha256": sha256_hex(document.data),
            }
        )

    return {
        "format": FORMAT,
        "document_count": len(documents),
        "total_bytes": sum(
            len(document.data)
            for document in documents
        ),
        "documents": manifest_documents,
        "corpus_id_preimage_sha256": corpus_id,
        "corpus_id": corpus_id,
    }


def is_lower_sha256_hex(value: object) -> bool:
    if not isinstance(value, str):
        return False

    if len(value) != 64:
        return False

    return all(
        character in "0123456789abcdef"
        for character in value
    )


def validate_manifest(
    documents: Sequence[Document],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise CorpusIdentityError(
            "manifest_not_object",
            "manifest must be an object",
        )

    if manifest.get("format") != FORMAT:
        raise CorpusIdentityError(
            "incorrect_format",
            (
                f"manifest format {manifest.get('format')!r} "
                f"!= {FORMAT!r}"
            ),
        )

    manifest_documents = manifest.get("documents")

    if not isinstance(manifest_documents, list):
        raise CorpusIdentityError(
            "documents_not_array",
            "manifest documents must be an array",
        )

    if manifest.get("document_count") != len(documents):
        raise CorpusIdentityError(
            "incorrect_document_count",
            (
                f"document_count {manifest.get('document_count')!r} "
                f"!= {len(documents)}"
            ),
        )

    if len(manifest_documents) != len(documents):
        raise CorpusIdentityError(
            "incorrect_document_count",
            (
                f"manifest document entries "
                f"{len(manifest_documents)} != {len(documents)}"
            ),
        )

    expected_total = sum(
        len(document.data)
        for document in documents
    )

    if manifest.get("total_bytes") != expected_total:
        raise CorpusIdentityError(
            "incorrect_total_bytes",
            (
                f"total_bytes {manifest.get('total_bytes')!r} "
                f"!= {expected_total}"
            ),
        )

    seen_names: set[str] = set()

    for expected_doc_id, (
        source_document,
        manifest_document,
    ) in enumerate(zip(documents, manifest_documents)):
        if not isinstance(manifest_document, dict):
            raise CorpusIdentityError(
                "document_entry_not_object",
                f"document entry {expected_doc_id} is not object",
            )

        actual_doc_id = manifest_document.get("doc_id")

        if actual_doc_id != expected_doc_id:
            raise CorpusIdentityError(
                "non_consecutive_document_ids",
                (
                    f"document entry {expected_doc_id} "
                    f"has doc_id {actual_doc_id!r}"
                ),
            )

        actual_name = manifest_document.get("doc_name")

        if actual_name != source_document.name:
            raise CorpusIdentityError(
                "incorrect_document_name",
                (
                    f"document {expected_doc_id} name "
                    f"{actual_name!r} != {source_document.name!r}"
                ),
            )

        validate_document_name(actual_name)

        if actual_name in seen_names:
            raise CorpusIdentityError(
                "duplicate_document_name",
                f"duplicate document name: {actual_name!r}",
            )

        seen_names.add(actual_name)

        actual_length = manifest_document.get("byte_length")
        expected_length = len(source_document.data)

        if actual_length != expected_length:
            raise CorpusIdentityError(
                "incorrect_document_length",
                (
                    f"document {expected_doc_id} length "
                    f"{actual_length!r} != {expected_length}"
                ),
            )

        actual_sha256 = manifest_document.get("sha256")

        if not is_lower_sha256_hex(actual_sha256):
            raise CorpusIdentityError(
                "malformed_sha256",
                (
                    f"document {expected_doc_id} SHA256 "
                    f"is malformed: {actual_sha256!r}"
                ),
            )

        expected_sha256 = sha256_hex(source_document.data)

        if actual_sha256 != expected_sha256:
            raise CorpusIdentityError(
                "incorrect_document_sha256",
                (
                    f"document {expected_doc_id} SHA256 "
                    f"{actual_sha256} != {expected_sha256}"
                ),
            )

    expected_manifest = build_manifest(documents)

    for field in (
        "corpus_id_preimage_sha256",
        "corpus_id",
    ):
        actual_value = manifest.get(field)

        if not is_lower_sha256_hex(actual_value):
            raise CorpusIdentityError(
                "malformed_sha256",
                f"{field} is malformed: {actual_value!r}",
            )

        expected_value = expected_manifest[field]

        if actual_value != expected_value:
            raise CorpusIdentityError(
                "incorrect_corpus_id",
                (
                    f"{field} {actual_value} "
                    f"!= {expected_value}"
                ),
            )

    return {
        "ok": True,
        "format": FORMAT,
        "document_count": len(documents),
        "total_bytes": expected_total,
        "corpus_id": expected_manifest["corpus_id"],
    }


def clone_manifest(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return json.loads(json.dumps(manifest))


def expect_rejection(
    name: str,
    documents: Sequence[Document],
    manifest: dict[str, Any],
    expected_codes: set[str],
) -> dict[str, Any]:
    try:
        validate_manifest(documents, manifest)
    except CorpusIdentityError as error:
        if error.code not in expected_codes:
            raise AssertionError(
                {
                    "mutation": name,
                    "expected_codes": sorted(expected_codes),
                    "actual_code": error.code,
                    "message": error.message,
                }
            ) from error

        return {
            "mutation": name,
            "rejected": True,
            "error_code": error.code,
            "message": error.message,
        }

    raise AssertionError(
        f"mutation unexpectedly accepted: {name}"
    )


def fixture_result(
    name: str,
    documents: Sequence[Document],
) -> dict[str, Any]:
    first = build_manifest(documents)
    second = build_manifest(documents)

    if first != second:
        raise AssertionError(
            f"non-deterministic manifest: {name}"
        )

    validation = validate_manifest(documents, first)

    return {
        "fixture": name,
        **validation,
        "manifest": first,
    }


def run_positive_fixtures() -> list[dict[str, Any]]:
    fixtures = [
        (
            "empty_corpus",
            [],
        ),
        (
            "one_empty_document",
            [Document("empty", b"")],
        ),
        (
            "multiple_empty_documents",
            [
                Document("empty-a", b""),
                Document("empty-b", b""),
            ],
        ),
        (
            "single_zero",
            [Document("zero", b"\x00")],
        ),
        (
            "single_ff",
            [Document("ff", b"\xff")],
        ),
        (
            "zero_ff",
            [Document("zero-ff", b"\x00\xff")],
        ),
        (
            "duplicate_content_distinct_names",
            [
                Document("copy-a", b"same"),
                Document("copy-b", b"same"),
            ],
        ),
        (
            "multiple_documents",
            [
                Document("alpha", b"abc"),
                Document("beta", b"\x00\xff"),
                Document("gamma", b""),
            ],
        ),
        (
            "full_alphabet",
            [Document("alphabet", bytes(range(256)))],
        ),
        (
            "unicode_name",
            [Document("данные-猫", b"payload")],
        ),
    ]

    return [
        fixture_result(name, documents)
        for name, documents in fixtures
    ]


def assert_identity_difference(
    name: str,
    left: Sequence[Document],
    right: Sequence[Document],
) -> dict[str, Any]:
    left_manifest = build_manifest(left)
    right_manifest = build_manifest(right)

    left_id = left_manifest["corpus_id"]
    right_id = right_manifest["corpus_id"]

    if left_id == right_id:
        raise AssertionError(
            f"identity mutation did not change corpus ID: {name}"
        )

    return {
        "mutation": name,
        "different": True,
        "left_corpus_id": left_id,
        "right_corpus_id": right_id,
    }


def run_identity_mutations() -> list[dict[str, Any]]:
    base = [
        Document("alpha", b"ab"),
        Document("beta", b"cd"),
    ]

    return [
        assert_identity_difference(
            "one_byte_mutation",
            base,
            [
                Document("alpha", b"ab"),
                Document("beta", b"ce"),
            ],
        ),
        assert_identity_difference(
            "document_reordering",
            base,
            [
                Document("beta", b"cd"),
                Document("alpha", b"ab"),
            ],
        ),
        assert_identity_difference(
            "document_rename",
            base,
            [
                Document("alpha-renamed", b"ab"),
                Document("beta", b"cd"),
            ],
        ),
        assert_identity_difference(
            "boundary_repartition_equal_concatenation",
            base,
            [
                Document("alpha", b"abc"),
                Document("beta", b"d"),
            ],
        ),
        assert_identity_difference(
            "empty_document_insertion",
            base,
            [
                Document("alpha", b"ab"),
                Document("empty", b""),
                Document("beta", b"cd"),
            ],
        ),
        assert_identity_difference(
            "document_deletion",
            base,
            [Document("alpha", b"ab")],
        ),
        assert_identity_difference(
            "zero_byte_mutation",
            [Document("binary", b"\x00\xff")],
            [Document("binary", b"\x01\xff")],
        ),
        assert_identity_difference(
            "ff_byte_mutation",
            [Document("binary", b"\x00\xff")],
            [Document("binary", b"\x00\xfe")],
        ),
    ]


def run_negative_manifest_mutations() -> list[dict[str, Any]]:
    documents = [
        Document("alpha", b"abc"),
        Document("beta", b"\x00\xff"),
    ]

    canonical = build_manifest(documents)
    results: list[dict[str, Any]] = []

    mutated = clone_manifest(canonical)
    mutated["format"] = "GLYPH_CORPUS_IDENTITY_V2"

    results.append(
        expect_rejection(
            "incorrect_format",
            documents,
            mutated,
            {"incorrect_format"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["document_count"] = 3

    results.append(
        expect_rejection(
            "incorrect_document_count",
            documents,
            mutated,
            {"incorrect_document_count"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["total_bytes"] += 1

    results.append(
        expect_rejection(
            "incorrect_total_bytes",
            documents,
            mutated,
            {"incorrect_total_bytes"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["documents"][1]["doc_id"] = 7

    results.append(
        expect_rejection(
            "non_consecutive_document_ids",
            documents,
            mutated,
            {"non_consecutive_document_ids"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["documents"][1]["doc_name"] = "alpha"

    results.append(
        expect_rejection(
            "duplicate_document_name",
            documents,
            mutated,
            {
                "incorrect_document_name",
                "duplicate_document_name",
            },
        )
    )

    mutated = clone_manifest(canonical)
    mutated["documents"][0]["byte_length"] += 1

    results.append(
        expect_rejection(
            "incorrect_document_length",
            documents,
            mutated,
            {"incorrect_document_length"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["documents"][0]["sha256"] = "0" * 64

    results.append(
        expect_rejection(
            "incorrect_document_sha256",
            documents,
            mutated,
            {"incorrect_document_sha256"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["documents"][0]["sha256"] = "XYZ"

    results.append(
        expect_rejection(
            "malformed_document_sha256",
            documents,
            mutated,
            {"malformed_sha256"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["corpus_id"] = "0" * 64

    results.append(
        expect_rejection(
            "incorrect_corpus_id",
            documents,
            mutated,
            {"incorrect_corpus_id"},
        )
    )

    mutated = clone_manifest(canonical)
    mutated["corpus_id_preimage_sha256"] = "not-a-hash"

    results.append(
        expect_rejection(
            "malformed_preimage_sha256",
            documents,
            mutated,
            {"malformed_sha256"},
        )
    )

    nul_name_documents = [
        Document("bad\x00name", b"x"),
    ]

    try:
        build_manifest(nul_name_documents)
    except CorpusIdentityError as error:
        if error.code != "document_name_contains_nul":
            raise

        results.append(
            {
                "mutation": "document_name_contains_nul",
                "rejected": True,
                "error_code": error.code,
                "message": error.message,
            }
        )
    else:
        raise AssertionError(
            "NUL document name unexpectedly accepted"
        )

    duplicate_name_documents = [
        Document("same", b"a"),
        Document("same", b"b"),
    ]

    try:
        build_manifest(duplicate_name_documents)
    except CorpusIdentityError as error:
        if error.code != "duplicate_document_name":
            raise

        results.append(
            {
                "mutation": "duplicate_document_names_at_build",
                "rejected": True,
                "error_code": error.code,
                "message": error.message,
            }
        )
    else:
        raise AssertionError(
            "duplicate document names unexpectedly accepted"
        )

    return results


def main() -> int:
    positive = run_positive_fixtures()
    identity_mutations = run_identity_mutations()
    negative = run_negative_manifest_mutations()

    boundary_left = [
        Document("alpha", b"ab"),
        Document("beta", b"cd"),
    ]

    boundary_right = [
        Document("alpha", b"abc"),
        Document("beta", b"d"),
    ]

    if (
        b"".join(document.data for document in boundary_left)
        !=
        b"".join(document.data for document in boundary_right)
    ):
        raise AssertionError(
            "boundary fixture concatenations must be equal"
        )

    output = {
        "ok": True,
        "proof_obligation": PROOF_OBLIGATION,
        "format": FORMAT,
        "positive_fixture_count": len(positive),
        "identity_mutation_count": len(identity_mutations),
        "negative_manifest_mutation_count": len(negative),
        "positive_fixtures": positive,
        "identity_mutations": identity_mutations,
        "negative_manifest_mutations": negative,
    }

    print(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
