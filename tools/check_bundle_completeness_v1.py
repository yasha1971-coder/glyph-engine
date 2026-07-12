#!/usr/bin/env python3
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

PROOF = "P11"
FORMAT = "GLYPH_BUNDLE_COMPLETENESS_V1"
MANIFEST_VERSION = "GLYPH_BUNDLE_MANIFEST_V1"
ARTIFACT_VERSION = "GLYPH_PORTABLE_EVIDENCE_V1"
SCHEMA_VERSION = "GLYPH_PORTABLE_EVIDENCE_SCHEMA_V1"
BOUNDARY_POLICY = "DOCUMENT_LOCAL_MATCHES_ONLY_V1"

MANIFEST_NAME = "bundle_manifest_v1.json"

REQUIRED_ROLES = {
    "artifact",
    "corpus",
    "schema",
    "replay",
}

ALLOWED_REPLAY_IMPORTS = {
    "__future__",
    "hashlib",
    "json",
    "pathlib",
    "sys",
    "typing",
}

FORBIDDEN_AUTHORITATIVE_KEYS = {
    "absolute_path",
    "source_path",
    "external_path",
    "repository_path",
    "temporary_path",
    "url",
    "network_url",
    "hostname",
    "cwd",
    "created_at",
    "timestamp",
    "pid",
    "random_nonce",
}


class BundleError(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise BundleError(
            "value is not canonical JSON"
        ) from error

    return encoded.encode("utf-8")


def write_canonical_json(
    path: Path,
    value: Any,
) -> None:
    path.write_bytes(
        canonical_json_bytes(value) + b"\n"
    )


def is_sha256(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 64
    ):
        return False

    return all(
        character in "0123456789abcdef"
        for character in value
    )


def validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BundleError("path must be non-empty string")

    if "\x00" in value:
        raise BundleError("path contains NUL")

    if "\\" in value:
        raise BundleError("path must use slash separators")

    path = Path(value)

    if path.is_absolute():
        raise BundleError("absolute path forbidden")

    parts = value.split("/")

    if any(part in {"", ".", ".."} for part in parts):
        raise BundleError("noncanonical or traversal path")

    if path.as_posix() != value:
        raise BundleError("path is not canonical")

    return value


def list_payload_files(bundle: Path) -> set[str]:
    files: set[str] = set()

    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise BundleError(
                f"symlink forbidden: {path}"
            )

        if path.is_file():
            relative = path.relative_to(bundle).as_posix()

            if relative != MANIFEST_NAME:
                files.add(relative)

    return files


def check_forbidden_metadata(
    value: Any,
    *,
    path: str = "$",
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_AUTHORITATIVE_KEYS:
                raise BundleError(
                    f"forbidden authoritative key: {path}.{key}"
                )

            check_forbidden_metadata(
                child,
                path=f"{path}.{key}",
            )

    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_forbidden_metadata(
                child,
                path=f"{path}[{index}]",
            )

    elif isinstance(value, str):
        lowered = value.lower()

        if (
            lowered.startswith("http://")
            or lowered.startswith("https://")
            or lowered.startswith("file://")
        ):
            raise BundleError(
                f"external URL forbidden: {path}"
            )


def naive_offsets(
    corpus: bytes,
    query: bytes,
) -> list[int]:
    if not query:
        raise BundleError("EMPTY_QUERY")

    if len(query) > len(corpus):
        return []

    return [
        offset
        for offset in range(
            len(corpus) - len(query) + 1
        )
        if corpus[offset:offset + len(query)] == query
    ]


def make_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "type": "object",
        "required": [
            "artifact_version",
            "proof_dependencies",
            "corpus",
            "query_hex",
            "query_length_bytes",
            "query_sha256",
            "document_boundary_policy",
            "match_count",
            "coordinates",
            "returned_count",
            "bounded",
            "offsets_complete",
            "byte_check",
        ],
        "properties": {
            "artifact_version": {
                "type": "string",
                "const": ARTIFACT_VERSION,
            },
            "proof_dependencies": {
                "type": "array",
            },
            "corpus": {
                "type": "object",
            },
            "query_hex": {
                "type": "string",
            },
            "query_length_bytes": {
                "type": "integer",
            },
            "query_sha256": {
                "type": "string",
            },
            "document_boundary_policy": {
                "type": "string",
                "const": BOUNDARY_POLICY,
            },
            "match_count": {
                "type": "integer",
            },
            "coordinates": {
                "type": "array",
            },
            "returned_count": {
                "type": "integer",
            },
            "bounded": {
                "type": "boolean",
            },
            "offsets_complete": {
                "type": "boolean",
            },
            "byte_check": {
                "type": "boolean",
            },
        },
        "additionalProperties": False,
    }


def make_artifact(
    corpus: bytes,
    query: bytes,
    *,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    if not query:
        raise BundleError("EMPTY_QUERY")

    if max_offsets is not None and (
        not isinstance(max_offsets, int)
        or isinstance(max_offsets, bool)
        or max_offsets < 0
    ):
        raise BundleError("invalid max_offsets")

    offsets = naive_offsets(corpus, query)

    returned = (
        offsets
        if max_offsets is None
        else offsets[:max_offsets]
    )

    bounded = len(returned) < len(offsets)

    artifact: dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "proof_dependencies": [
            "P4",
            "P7",
            "P8",
            "P9",
            "P10",
        ],
        "corpus": {
            "path": "corpus.bin",
            "sha256": sha256_bytes(corpus),
            "size_bytes": len(corpus),
            "document_count": 1,
            "document_lengths": [len(corpus)],
        },
        "query_hex": query.hex(),
        "query_length_bytes": len(query),
        "query_sha256": sha256_bytes(query),
        "document_boundary_policy":
            BOUNDARY_POLICY,
        "match_count": len(offsets),
        "coordinates": [
            [0, offset]
            for offset in returned
        ],
        "returned_count": len(returned),
        "bounded": bounded,
        "offsets_complete": not bounded,
        "byte_check": True,
    }

    if max_offsets is not None:
        artifact["max_offsets"] = max_offsets

    return artifact


def schema_type_matches(
    expected: str,
    value: Any,
) -> bool:
    if expected == "string":
        return isinstance(value, str)

    if expected == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    if expected == "boolean":
        return isinstance(value, bool)

    if expected == "array":
        return isinstance(value, list)

    if expected == "object":
        return isinstance(value, dict)

    return False


def validate_schema_and_artifact(
    schema: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("schema version mismatch")

    if schema.get("artifact_version") != ARTIFACT_VERSION:
        raise BundleError(
            "schema artifact version mismatch"
        )

    if schema.get("type") != "object":
        raise BundleError("schema root type mismatch")

    required = schema.get("required")

    if not isinstance(required, list):
        raise BundleError("schema required must be list")

    for field in required:
        if field not in artifact:
            raise BundleError(
                f"missing required artifact field: {field}"
            )

    properties = schema.get("properties")

    if not isinstance(properties, dict):
        raise BundleError(
            "schema properties must be object"
        )

    if schema.get("additionalProperties") is False:
        allowed = set(properties)

        unexpected = set(artifact) - allowed - {
            "max_offsets"
        }

        if unexpected:
            raise BundleError(
                "unexpected artifact fields: "
                + ",".join(sorted(unexpected))
            )

    for field, definition in properties.items():
        if field not in artifact:
            continue

        if not isinstance(definition, dict):
            raise BundleError(
                f"invalid schema definition: {field}"
            )

        expected_type = definition.get("type")

        if (
            isinstance(expected_type, str)
            and not schema_type_matches(
                expected_type,
                artifact[field],
            )
        ):
            raise BundleError(
                f"artifact type mismatch: {field}"
            )

        if (
            "const" in definition
            and artifact[field] != definition["const"]
        ):
            raise BundleError(
                f"artifact constant mismatch: {field}"
            )


def validate_artifact_semantics(
    artifact: dict[str, Any],
    corpus: bytes,
) -> dict[str, Any]:
    check_forbidden_metadata(artifact)

    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise BundleError("artifact version mismatch")

    corpus_meta = artifact.get("corpus")

    if not isinstance(corpus_meta, dict):
        raise BundleError("corpus metadata missing")

    corpus_path = validate_relative_path(
        corpus_meta.get("path")
    )

    if corpus_path != "corpus.bin":
        raise BundleError(
            "artifact corpus path mismatch"
        )

    if corpus_meta.get("sha256") != sha256_bytes(corpus):
        raise BundleError(
            "artifact corpus SHA256 mismatch"
        )

    if corpus_meta.get("size_bytes") != len(corpus):
        raise BundleError(
            "artifact corpus size mismatch"
        )

    if corpus_meta.get("document_count") != 1:
        raise BundleError(
            "artifact document_count mismatch"
        )

    if corpus_meta.get("document_lengths") != [
        len(corpus)
    ]:
        raise BundleError(
            "artifact document_lengths mismatch"
        )

    query_hex = artifact.get("query_hex")

    if (
        not isinstance(query_hex, str)
        or not query_hex
        or query_hex != query_hex.lower()
        or len(query_hex) % 2
    ):
        raise BundleError(
            "invalid canonical query_hex"
        )

    try:
        query = bytes.fromhex(query_hex)
    except ValueError as error:
        raise BundleError("invalid query_hex") from error

    if query.hex() != query_hex:
        raise BundleError(
            "query_hex is not canonical"
        )

    if artifact.get("query_length_bytes") != len(query):
        raise BundleError("query length mismatch")

    if artifact.get("query_sha256") != sha256_bytes(query):
        raise BundleError("query SHA256 mismatch")

    if (
        artifact.get("document_boundary_policy")
        != BOUNDARY_POLICY
    ):
        raise BundleError(
            "document boundary policy mismatch"
        )

    expected_offsets = naive_offsets(corpus, query)

    raw_coordinates = artifact.get("coordinates")

    if not isinstance(raw_coordinates, list):
        raise BundleError(
            "coordinates must be list"
        )

    coordinates: list[tuple[int, int]] = []

    for value in raw_coordinates:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(
                isinstance(item, int)
                and not isinstance(item, bool)
                for item in value
            )
        ):
            raise BundleError("invalid coordinate")

        coordinate = (value[0], value[1])

        if coordinate[0] != 0:
            raise BundleError(
                "coordinate document mismatch"
            )

        coordinates.append(coordinate)

    if coordinates != sorted(coordinates):
        raise BundleError(
            "coordinates are not canonical"
        )

    if len(coordinates) != len(set(coordinates)):
        raise BundleError("duplicate coordinates")

    max_offsets = artifact.get("max_offsets")

    if max_offsets is None:
        expected_returned = expected_offsets
    else:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
        ):
            raise BundleError(
                "invalid max_offsets"
            )

        expected_returned = expected_offsets[:max_offsets]

    expected_coordinates = [
        (0, offset)
        for offset in expected_returned
    ]

    if coordinates != expected_coordinates:
        raise BundleError(
            "coordinates differ from byte oracle"
        )

    if artifact.get("match_count") != len(expected_offsets):
        raise BundleError("match_count mismatch")

    if artifact.get("returned_count") != len(
        expected_coordinates
    ):
        raise BundleError(
            "returned_count mismatch"
        )

    bounded = (
        len(expected_coordinates)
        < len(expected_offsets)
    )

    if artifact.get("bounded") is not bounded:
        raise BundleError("bounded mismatch")

    if (
        artifact.get("offsets_complete")
        is not (not bounded)
    ):
        raise BundleError(
            "offsets_complete mismatch"
        )

    if artifact.get("byte_check") is not True:
        raise BundleError(
            "byte_check must be true"
        )

    return {
        "query": query,
        "match_count": len(expected_offsets),
        "coordinates": expected_coordinates,
        "bounded": bounded,
    }


def replay_source() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAME = "bundle_manifest_v1.json"
MANIFEST_VERSION = "GLYPH_BUNDLE_MANIFEST_V1"
ARTIFACT_VERSION = "GLYPH_PORTABLE_EVIDENCE_V1"
SCHEMA_VERSION = "GLYPH_PORTABLE_EVIDENCE_SCHEMA_V1"
BOUNDARY_POLICY = "DOCUMENT_LOCAL_MATCHES_ONLY_V1"

def fail(message):
    raise SystemExit("REPLAY FAIL: " + message)

def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def safe_relative(value):
    if not isinstance(value, str) or not value:
        fail("invalid path")
    if "\x00" in value or "\\" in value:
        fail("unsafe path")
    path = Path(value)
    if path.is_absolute():
        fail("absolute path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        fail("unsafe path component")
    if path.as_posix() != value:
        fail("noncanonical path")
    return value

def payload_files(root):
    result = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            fail("symlink forbidden")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative != MANIFEST_NAME:
                result.add(relative)
    return result

def naive_offsets(corpus, query):
    if not query:
        fail("empty query")
    if len(query) > len(corpus):
        return []
    return [
        offset
        for offset in range(len(corpus) - len(query) + 1)
        if corpus[offset:offset + len(query)] == query
    ]

def main():
    if len(sys.argv) != 2:
        fail("usage: replay.py BUNDLE_DIR")

    root = Path(sys.argv[1]).resolve()
    manifest_path = root / MANIFEST_NAME

    if not manifest_path.is_file():
        fail("manifest missing")

    manifest = json.loads(manifest_path.read_text())

    if manifest.get("bundle_version") != MANIFEST_VERSION:
        fail("bundle version mismatch")

    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        fail("artifact version mismatch")

    if manifest.get("required_runtime") != "python3-stdlib":
        fail("runtime mismatch")

    if manifest.get("external_dependencies") != []:
        fail("external dependencies declared")

    entrypoint = manifest.get("replay_entrypoint")

    if entrypoint != ["python3", "-I", "replay.py", "."]:
        fail("replay entrypoint mismatch")

    entries = manifest.get("files")

    if not isinstance(entries, list):
        fail("manifest files missing")

    normalized = []
    seen_paths = set()
    seen_roles = set()

    for entry in entries:
        if not isinstance(entry, dict):
            fail("invalid manifest entry")

        path_value = safe_relative(entry.get("path"))
        role = entry.get("role")

        if path_value in seen_paths:
            fail("duplicate manifest path")

        if role in seen_roles:
            fail("duplicate manifest role")

        seen_paths.add(path_value)
        seen_roles.add(role)

        path = root / path_value

        try:
            path.resolve().relative_to(root)
        except ValueError:
            fail("path escapes bundle")

        if path.is_symlink():
            fail("symlink forbidden")

        if not path.is_file():
            fail("declared file missing")

        size = path.stat().st_size
        digest = sha256_file(path)

        if entry.get("size_bytes") != size:
            fail("size mismatch")

        if entry.get("sha256") != digest:
            fail("sha256 mismatch")

        normalized.append({
            "path": path_value,
            "role": role,
            "size_bytes": size,
            "sha256": digest,
        })

    normalized.sort(key=lambda item: item["path"])

    if seen_roles != {"artifact", "corpus", "schema", "replay"}:
        fail("required roles mismatch")

    if seen_paths != payload_files(root):
        fail("manifest coverage mismatch")

    root_hash = sha256_bytes(canonical(normalized))

    if manifest.get("bundle_root_sha256") != root_hash:
        fail("bundle root mismatch")

    role_paths = {
        entry["role"]: root / entry["path"]
        for entry in normalized
    }

    artifact = json.loads(role_paths["artifact"].read_text())
    schema = json.loads(role_paths["schema"].read_text())
    corpus = role_paths["corpus"].read_bytes()

    if schema.get("schema_version") != SCHEMA_VERSION:
        fail("schema version mismatch")

    if schema.get("artifact_version") != ARTIFACT_VERSION:
        fail("schema artifact mismatch")

    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        fail("artifact version mismatch")

    required = schema.get("required")

    if not isinstance(required, list):
        fail("schema required invalid")

    for field in required:
        if field not in artifact:
            fail("required artifact field missing")

    corpus_meta = artifact.get("corpus")

    if not isinstance(corpus_meta, dict):
        fail("corpus metadata missing")

    if corpus_meta.get("path") != "corpus.bin":
        fail("corpus path mismatch")

    if corpus_meta.get("sha256") != sha256_bytes(corpus):
        fail("corpus sha mismatch")

    if corpus_meta.get("size_bytes") != len(corpus):
        fail("corpus size mismatch")

    query_hex = artifact.get("query_hex")

    if (
        not isinstance(query_hex, str)
        or not query_hex
        or query_hex != query_hex.lower()
        or len(query_hex) % 2
    ):
        fail("query hex invalid")

    try:
        query = bytes.fromhex(query_hex)
    except ValueError:
        fail("query hex invalid")

    if query.hex() != query_hex:
        fail("query hex noncanonical")

    if artifact.get("query_length_bytes") != len(query):
        fail("query length mismatch")

    if artifact.get("query_sha256") != sha256_bytes(query):
        fail("query sha mismatch")

    if artifact.get("document_boundary_policy") != BOUNDARY_POLICY:
        fail("boundary policy mismatch")

    offsets = naive_offsets(corpus, query)
    max_offsets = artifact.get("max_offsets")

    if max_offsets is None:
        returned = offsets
    else:
        if (
            not isinstance(max_offsets, int)
            or isinstance(max_offsets, bool)
            or max_offsets < 0
        ):
            fail("max offsets invalid")
        returned = offsets[:max_offsets]

    expected_coordinates = [[0, offset] for offset in returned]

    if artifact.get("coordinates") != expected_coordinates:
        fail("coordinates mismatch")

    if artifact.get("match_count") != len(offsets):
        fail("match count mismatch")

    if artifact.get("returned_count") != len(returned):
        fail("returned count mismatch")

    bounded = len(returned) < len(offsets)

    if artifact.get("bounded") is not bounded:
        fail("bounded mismatch")

    if artifact.get("offsets_complete") is not (not bounded):
        fail("offsets complete mismatch")

    if artifact.get("byte_check") is not True:
        fail("byte check mismatch")

    result = {
        "ok": True,
        "bundle_version": MANIFEST_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "bundle_root_sha256": root_hash,
        "match_count": len(offsets),
        "coordinates": expected_coordinates,
        "returned_count": len(returned),
        "bounded": bounded,
        "self_contained": True,
        "external_dependencies": [],
    }

    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def validate_replay_imports(path: Path) -> None:
    tree = ast.parse(
        path.read_text(),
        filename=str(path),
    )

    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(
                    alias.name.split(".", 1)[0]
                )

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(
                    node.module.split(".", 1)[0]
                )

    unexpected = imported - ALLOWED_REPLAY_IMPORTS

    if unexpected:
        raise BundleError(
            "replay imports undeclared modules: "
            + ",".join(sorted(unexpected))
        )

    source = path.read_text()

    forbidden_fragments = [
        "GLYPH_CPP_BACKEND",
        "sys.path.append",
        "sys.path.insert",
        "site.addsitedir",
        "urllib",
        "requests",
        "socket",
        "subprocess",
        "http://",
        "https://",
    ]

    for fragment in forbidden_fragments:
        if fragment in source:
            raise BundleError(
                f"forbidden replay dependency: {fragment}"
            )


def make_manifest(
    bundle: Path,
) -> dict[str, Any]:
    roles = {
        "artifact.json": "artifact",
        "corpus.bin": "corpus",
        "schema.json": "schema",
        "replay.py": "replay",
    }

    files = []

    for relative, role in roles.items():
        path = bundle / relative

        files.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    files.sort(key=lambda item: item["path"])

    return {
        "bundle_version": MANIFEST_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "required_runtime": "python3-stdlib",
        "replay_entrypoint": [
            "python3",
            "-I",
            "replay.py",
            ".",
        ],
        "external_dependencies": [],
        "files": files,
        "bundle_root_sha256": sha256_bytes(
            canonical_json_bytes(files)
        ),
    }


def build_bundle(
    bundle: Path,
    corpus: bytes,
    query: bytes,
    *,
    max_offsets: int | None = None,
) -> None:
    bundle.mkdir(parents=True, exist_ok=False)

    (bundle / "corpus.bin").write_bytes(corpus)

    write_canonical_json(
        bundle / "schema.json",
        make_schema(),
    )

    write_canonical_json(
        bundle / "artifact.json",
        make_artifact(
            corpus,
            query,
            max_offsets=max_offsets,
        ),
    )

    replay_path = bundle / "replay.py"
    replay_path.write_text(replay_source())
    replay_path.chmod(0o755)

    write_canonical_json(
        bundle / MANIFEST_NAME,
        make_manifest(bundle),
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())

    if not isinstance(value, dict):
        raise BundleError(
            f"expected JSON object: {path}"
        )

    return value


def validate_manifest(
    bundle: Path,
) -> dict[str, Any]:
    manifest_path = bundle / MANIFEST_NAME

    if manifest_path.is_symlink():
        raise BundleError("manifest symlink forbidden")

    if not manifest_path.is_file():
        raise BundleError("manifest missing")

    manifest = load_json(manifest_path)

    check_forbidden_metadata(manifest)

    if manifest.get("bundle_version") != MANIFEST_VERSION:
        raise BundleError("bundle version mismatch")

    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        raise BundleError(
            "manifest artifact version mismatch"
        )

    if manifest.get("required_runtime") != "python3-stdlib":
        raise BundleError(
            "required runtime mismatch"
        )

    if manifest.get("external_dependencies") != []:
        raise BundleError(
            "external dependencies forbidden"
        )

    if manifest.get("replay_entrypoint") != [
        "python3",
        "-I",
        "replay.py",
        ".",
    ]:
        raise BundleError(
            "replay entrypoint mismatch"
        )

    entries = manifest.get("files")

    if not isinstance(entries, list):
        raise BundleError(
            "manifest files must be list"
        )

    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            raise BundleError(
                "manifest entry must be object"
            )

        relative = validate_relative_path(
            entry.get("path")
        )
        role = entry.get("role")

        if role not in REQUIRED_ROLES:
            raise BundleError(
                f"invalid file role: {role}"
            )

        if relative in seen_paths:
            raise BundleError(
                "duplicate manifest path"
            )

        if role in seen_roles:
            raise BundleError(
                "duplicate singleton role"
            )

        seen_paths.add(relative)
        seen_roles.add(role)

        path = bundle / relative

        try:
            path.resolve().relative_to(
                bundle.resolve()
            )
        except ValueError as error:
            raise BundleError(
                "manifest path escapes bundle"
            ) from error

        if path.is_symlink():
            raise BundleError(
                f"symlink payload forbidden: {relative}"
            )

        if not path.is_file():
            raise BundleError(
                f"declared file missing: {relative}"
            )

        size = path.stat().st_size
        digest = sha256_file(path)

        if entry.get("size_bytes") != size:
            raise BundleError(
                f"file size mismatch: {relative}"
            )

        if not is_sha256(entry.get("sha256")):
            raise BundleError(
                f"invalid manifest SHA256: {relative}"
            )

        if entry.get("sha256") != digest:
            raise BundleError(
                f"file SHA256 mismatch: {relative}"
            )

        normalized.append(
            {
                "path": relative,
                "role": role,
                "size_bytes": size,
                "sha256": digest,
            }
        )

    if seen_roles != REQUIRED_ROLES:
        raise BundleError(
            "required bundle roles incomplete"
        )

    actual_files = list_payload_files(bundle)

    if seen_paths != actual_files:
        missing = sorted(seen_paths - actual_files)
        extra = sorted(actual_files - seen_paths)

        raise BundleError(
            "manifest coverage mismatch"
            f"; missing={missing}; extra={extra}"
        )

    normalized.sort(key=lambda item: item["path"])

    expected_root = sha256_bytes(
        canonical_json_bytes(normalized)
    )

    if manifest.get("bundle_root_sha256") != expected_root:
        raise BundleError(
            "bundle root SHA256 mismatch"
        )

    return {
        "manifest": manifest,
        "normalized_files": normalized,
        "bundle_root_sha256": expected_root,
        "role_paths": {
            entry["role"]: bundle / entry["path"]
            for entry in normalized
        },
    }


def validate_bundle(
    bundle: Path,
) -> dict[str, Any]:
    state = validate_manifest(bundle)
    role_paths = state["role_paths"]

    validate_replay_imports(
        role_paths["replay"]
    )

    schema = load_json(role_paths["schema"])
    artifact = load_json(role_paths["artifact"])
    corpus = role_paths["corpus"].read_bytes()

    validate_schema_and_artifact(
        schema,
        artifact,
    )

    semantic = validate_artifact_semantics(
        artifact,
        corpus,
    )

    return {
        "ok": True,
        "bundle_root_sha256":
            state["bundle_root_sha256"],
        "file_count":
            len(state["normalized_files"]),
        "exact_file_coverage": True,
        "all_hashes_match": True,
        "schema_matches_artifact": True,
        "corpus_identity_matches": True,
        "query_identity_matches": True,
        "match_count": semantic["match_count"],
        "coordinates": [
            [doc_id, offset]
            for doc_id, offset
            in semantic["coordinates"]
        ],
        "bounded": semantic["bounded"],
        "hidden_data_dependencies": [],
    }


def run_independent_replay(
    bundle: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        str(bundle / "replay.py"),
        str(bundle),
    ]

    environment = {
        "PATH": os.environ.get(
            "PATH",
            "/usr/bin:/bin",
        ),
        "LANG": "C",
        "LC_ALL": "C",
        "HOME": str(bundle / "_isolated_home"),
        "PYTHONNOUSERSITE": "1",
    }

    outside_cwd = bundle.parent / "outside"
    outside_cwd.mkdir(exist_ok=True)

    completed = subprocess.run(
        command,
        cwd=outside_cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    if completed.returncode != 0:
        raise BundleError(
            "independent replay failed: "
            + completed.stderr.strip()
            + completed.stdout.strip()
        )

    try:
        result = json.loads(
            completed.stdout.strip()
        )
    except json.JSONDecodeError as error:
        raise BundleError(
            "independent replay returned invalid JSON"
        ) from error

    if result.get("ok") is not True:
        raise BundleError(
            "independent replay did not pass"
        )

    if result.get("self_contained") is not True:
        raise BundleError(
            "replay did not declare self-contained result"
        )

    if result.get("external_dependencies") != []:
        raise BundleError(
            "replay exposed external dependencies"
        )

    return result


def copy_and_replay(
    source_bundle: Path,
    destination: Path,
) -> dict[str, Any]:
    shutil.copytree(
        source_bundle,
        destination,
    )

    validate_bundle(destination)

    return run_independent_replay(
        destination
    )


def mutate_manifest(
    bundle: Path,
    operation: Callable[
        [dict[str, Any]],
        None,
    ],
) -> None:
    path = bundle / MANIFEST_NAME
    manifest = load_json(path)
    operation(manifest)
    write_canonical_json(path, manifest)


def refresh_manifest(
    bundle: Path,
) -> None:
    write_canonical_json(
        bundle / MANIFEST_NAME,
        make_manifest(bundle),
    )


def clone_bundle(
    source: Path,
    destination: Path,
) -> Path:
    shutil.copytree(source, destination)
    return destination


def expect_failure(
    name: str,
    operation: Callable[[], Any],
) -> dict[str, Any]:
    try:
        operation()
    except (
        BundleError,
        AssertionError,
        ValueError,
        TypeError,
        OSError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
        SyntaxError,
    ) as error:
        return {
            "mutation": name,
            "rejected": True,
            "message": str(error),
        }

    raise AssertionError(
        f"mutation unexpectedly accepted: {name}"
    )


def validate_and_replay(bundle: Path) -> None:
    validate_bundle(bundle)
    run_independent_replay(bundle)


def fixture(
    root: Path,
    name: str,
    corpus: bytes,
    query: bytes,
    *,
    max_offsets: int | None = None,
) -> dict[str, Any]:
    bundle = root / name

    build_bundle(
        bundle,
        corpus,
        query,
        max_offsets=max_offsets,
    )

    validation = validate_bundle(bundle)
    replay = run_independent_replay(bundle)

    copied = root / f"{name}_copied"
    copied_replay = copy_and_replay(
        bundle,
        copied,
    )

    if (
        replay["bundle_root_sha256"]
        != copied_replay["bundle_root_sha256"]
    ):
        raise AssertionError(
            "copied bundle root changed"
        )

    return {
        "fixture": name,
        "corpus_bytes": len(corpus),
        "query_hex": query.hex(),
        "file_count": validation["file_count"],
        "bundle_root_sha256":
            validation["bundle_root_sha256"],
        "exact_file_coverage": True,
        "all_hashes_match": True,
        "schema_matches_artifact": True,
        "independent_replay_ok": True,
        "copied_bundle_replay_ok": True,
        "self_contained": True,
        "external_dependencies": [],
    }


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="glyph_p11_"
    ) as temporary:
        root = Path(temporary)

        alphabet = bytes(range(256))

        fixtures = [
            fixture(
                root,
                "ascii",
                b"banana",
                b"ana",
            ),
            fixture(
                root,
                "embedded_nul",
                b"A\x00B\x00A\x00B",
                b"A\x00B",
            ),
            fixture(
                root,
                "byte_ff",
                b"\xff\x00\xff",
                b"\xff",
            ),
            fixture(
                root,
                "invalid_utf8",
                b"\x80\x81\xfe\xff\x81\xfe",
                b"\x81\xfe",
            ),
            fixture(
                root,
                "zero_match",
                b"abc",
                b"xyz",
            ),
            fixture(
                root,
                "repeated_matches",
                b"aaaaaa",
                b"aa",
            ),
            fixture(
                root,
                "bounded_locate",
                b"aaaaaa",
                b"aa",
                max_offsets=2,
            ),
            fixture(
                root,
                "all_256_bytes",
                alphabet,
                alphabet,
            ),
        ]

        base = root / "mutation_base"

        build_bundle(
            base,
            b"banana",
            b"ana",
            max_offsets=1,
        )

        mutations: list[dict[str, Any]] = []

        def case(name: str) -> Path:
            return clone_bundle(
                base,
                root / f"mutation_{name}",
            )

        bundle = case("missing_artifact")
        (bundle / "artifact.json").unlink()
        mutations.append(
            expect_failure(
                "missing_artifact",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("missing_corpus")
        (bundle / "corpus.bin").unlink()
        mutations.append(
            expect_failure(
                "missing_corpus",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("missing_schema")
        (bundle / "schema.json").unlink()
        mutations.append(
            expect_failure(
                "missing_schema",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("missing_replay")
        (bundle / "replay.py").unlink()
        mutations.append(
            expect_failure(
                "missing_replay",
                lambda: validate_and_replay(bundle),
            )
        )

        for relative in [
            "artifact.json",
            "corpus.bin",
            "schema.json",
            "replay.py",
        ]:
            bundle = case(
                "altered_" + relative.replace(".", "_")
            )
            with (bundle / relative).open("ab") as stream:
                stream.write(b"\nMUTATION")

            mutations.append(
                expect_failure(
                    f"altered_{relative}",
                    lambda bundle=bundle:
                        validate_and_replay(bundle),
                )
            )

        bundle = case("incorrect_size")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest["files"][0].update(
                    {
                        "size_bytes":
                            manifest["files"][0][
                                "size_bytes"
                            ]
                            + 1
                    }
                ),
        )
        mutations.append(
            expect_failure(
                "incorrect_size",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("incorrect_sha")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest["files"][0].update(
                    {"sha256": "0" * 64}
                ),
        )
        mutations.append(
            expect_failure(
                "incorrect_sha256",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("incorrect_root")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest.update(
                    {
                        "bundle_root_sha256":
                            "0" * 64
                    }
                ),
        )
        mutations.append(
            expect_failure(
                "incorrect_bundle_root",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("extra_file")
        (bundle / "undeclared.bin").write_bytes(
            b"extra"
        )
        mutations.append(
            expect_failure(
                "unlisted_extra_file",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("duplicate_path")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest["files"].append(
                    copy.deepcopy(
                        manifest["files"][0]
                    )
                ),
        )
        mutations.append(
            expect_failure(
                "duplicate_manifest_path",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("duplicate_role")

        def duplicate_role(manifest: dict[str, Any]) -> None:
            manifest["files"][1]["role"] = (
                manifest["files"][0]["role"]
            )

        mutate_manifest(bundle, duplicate_role)
        mutations.append(
            expect_failure(
                "duplicate_singleton_role",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("absolute_path")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest["files"][0].update(
                    {"path": "/tmp/artifact.json"}
                ),
        )
        mutations.append(
            expect_failure(
                "absolute_manifest_path",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("parent_path")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest["files"][0].update(
                    {"path": "../artifact.json"}
                ),
        )
        mutations.append(
            expect_failure(
                "parent_traversal_path",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("symlink")
        original = bundle / "corpus.bin"
        target = bundle / "corpus.real"
        original.rename(target)
        original.symlink_to(target.name)
        mutations.append(
            expect_failure(
                "symlink_payload",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("schema_version")
        schema = load_json(bundle / "schema.json")
        schema["artifact_version"] = "WRONG"
        write_canonical_json(
            bundle / "schema.json",
            schema,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "schema_artifact_version_mismatch",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("missing_artifact_field")
        artifact = load_json(
            bundle / "artifact.json"
        )
        del artifact["query_sha256"]
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "missing_required_artifact_field",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("corpus_hash")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["corpus"]["sha256"] = "0" * 64
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "artifact_corpus_sha256_mismatch",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("corpus_size")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["corpus"]["size_bytes"] += 1
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "artifact_corpus_size_mismatch",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("query_hex")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["query_hex"] = "0"
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "malformed_query_hex",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("query_sha")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["query_sha256"] = "0" * 64
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "query_sha256_mismatch",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("match_count")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["match_count"] = 999
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "incorrect_match_count",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("coordinate")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["coordinates"] = [[0, 0]]
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "incorrect_coordinate",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("unsorted")
        artifact = make_artifact(
            b"banana",
            b"ana",
        )
        artifact["coordinates"] = list(
            reversed(
                artifact["coordinates"]
            )
        )
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "unsorted_coordinates",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("duplicate_coordinates")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["coordinates"] = [
            [0, 1],
            [0, 1],
        ]
        artifact["returned_count"] = 2
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "duplicate_coordinates",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("false_byte_check")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["byte_check"] = False
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "false_byte_check",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("wrong_boundary")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact[
            "document_boundary_policy"
        ] = "PHYSICAL_CONCATENATION"
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "wrong_boundary_policy",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("external_corpus_path")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["corpus"]["path"] = (
            "/tmp/external.bin"
        )
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "external_corpus_path",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("url_dependency")
        artifact = load_json(
            bundle / "artifact.json"
        )
        artifact["url"] = (
            "https://example.invalid/corpus"
        )
        write_canonical_json(
            bundle / "artifact.json",
            artifact,
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "url_dependency",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("external_dependency")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest.update(
                    {
                        "external_dependencies": [
                            "/tmp/source.bin"
                        ]
                    }
                ),
        )
        mutations.append(
            expect_failure(
                "undeclared_external_dependency",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("outside_entrypoint")
        mutate_manifest(
            bundle,
            lambda manifest:
                manifest.update(
                    {
                        "replay_entrypoint": [
                            "python3",
                            "-I",
                            "../replay.py",
                            ".",
                        ]
                    }
                ),
        )
        mutations.append(
            expect_failure(
                "replay_entrypoint_outside_bundle",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("repo_import")
        replay = bundle / "replay.py"
        replay.write_text(
            replay.read_text()
            + "\nimport glyph_engine\n"
        )
        refresh_manifest(bundle)
        mutations.append(
            expect_failure(
                "replay_imports_repository_code",
                lambda: validate_and_replay(bundle),
            )
        )

        bundle = case("copied_missing_file")
        copied = root / "copied_incomplete"
        shutil.copytree(bundle, copied)
        (copied / "schema.json").unlink()
        mutations.append(
            expect_failure(
                "copied_bundle_missing_required_file",
                lambda: validate_and_replay(copied),
            )
        )

        bundle = case("role_mismatch")

        def role_mismatch(
            manifest: dict[str, Any],
        ) -> None:
            for entry in manifest["files"]:
                if entry["role"] == "artifact":
                    entry["role"] = "corpus"
                    break

        mutate_manifest(
            bundle,
            role_mismatch,
        )
        mutations.append(
            expect_failure(
                "manifest_role_mismatch",
                lambda: validate_and_replay(bundle),
            )
        )

        if len(mutations) < 36:
            raise AssertionError(
                "insufficient mutation coverage"
            )

        for mutation in mutations:
            if mutation["rejected"] is not True:
                raise AssertionError(
                    "mutation was not rejected"
                )

        output = {
            "ok": True,
            "proof_obligation": PROOF,
            "format": FORMAT,
            "manifest_version": MANIFEST_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "fixture_count": len(fixtures),
            "mutation_count": len(mutations),
            "required_roles": sorted(
                REQUIRED_ROLES
            ),
            "manifest_coverage_exact": True,
            "all_payload_hashes_verified": True,
            "bundle_root_verified": True,
            "schema_matches_artifact": True,
            "corpus_identity_verified": True,
            "query_identity_verified": True,
            "independent_replay_verified": True,
            "copied_bundle_replay_verified": True,
            "replay_outside_repository_verified": True,
            "isolated_python_mode_verified": True,
            "hidden_data_dependencies": [],
            "network_dependency_required": False,
            "self_contained": True,
            "p12_ready": True,
            "p12_handoff": {
                "node":
                    "portable_bundle_replay",
                "requires": [
                    "P4",
                    "P7",
                    "P8",
                    "P9",
                    "P10",
                    "P11",
                ],
                "establishes": [
                    "exact_manifest_coverage",
                    "payload_integrity",
                    "schema_artifact_binding",
                    "corpus_binding",
                    "query_binding",
                    "portable_independent_replay",
                    "no_hidden_data_dependency",
                ],
            },
            "fixtures": fixtures,
            "mutations": mutations,
        }

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
