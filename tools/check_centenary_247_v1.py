#!/usr/bin/env python3
"""Fail-closed structural check for GLYPH Centenary 247 V1."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/governance/GLYPH_CENTENARY_247_V1.md"
LAW = ROOT / "docs/governance/GLYPH_PRESERVATION_LAW_V1.md"
LEDGER = ROOT / "docs/governance/GLYPH_MOVEMENT_LEDGER.md"
IDENTITY_MODEL = (
    ROOT
    / "docs/governance/GLYPH_OBJECT_IDENTITY_AND_DEPENDENCY_V1.md"
)
OBJECT_GRAPH = ROOT / "docs/governance/GLYPH_OBJECT_GRAPH_V1.json"

EXPECTED_SECTIONS = {
    "A": (1, 15),
    "B": (16, 35),
    "C": (36, 55),
    "D": (56, 75),
    "E": (76, 99),
    "F": (100, 123),
    "G": (124, 145),
    "H": (146, 169),
    "I": (170, 191),
    "J": (192, 209),
    "K": (210, 223),
    "L": (224, 237),
    "M": (238, 247),
}

FACET_RE = re.compile(r"^\| F(\d{3}) \|")
SECTION_RE = re.compile(r"^### ([A-M])\.")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_OBJECT_IDS = {
    "glyph-object:governance:preservation-law:v1",
    "glyph-object:governance:movement-ledger:v1",
    "glyph-object:governance:centenary-247:v1",
    "glyph-object:governance:object-identity-dependency:v1",
    "glyph-object:governance:object-graph:v1",
    "glyph-object:tool:check-centenary-247:v1",
}

RELATION_TYPES = {
    "DEPENDS_ON",
    "IMPLEMENTS",
    "VERIFIES",
    "EVIDENCES",
    "PRODUCES",
    "REFINES",
    "SUPERSEDES",
    "MOVED_FROM",
    "COMPATIBLE_WITH",
    "INVALIDATES",
}


def fail(message: str) -> None:
    raise SystemExit(f"CENTENARY_247_FAIL: {message}")


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_object_graph() -> tuple[int, int]:
    try:
        graph = json.loads(
            OBJECT_GRAPH.read_text(encoding="utf-8"),
            object_pairs_hook=strict_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        fail(f"invalid object graph JSON: {exc}")

    if graph.get("format") != "GLYPH_OBJECT_GRAPH_V1":
        fail("object graph format mismatch")
    if graph.get("completeness") != "PARTIAL_EXPLICIT_UNKNOWN_OUTSIDE_LIST":
        fail("object graph must fail closed on unknown coverage")

    objects = graph.get("objects")
    relations = graph.get("relations")
    if not isinstance(objects, list) or not isinstance(relations, list):
        fail("object graph objects/relations must be arrays")

    object_ids: list[str] = []
    for item in objects:
        if not isinstance(item, dict):
            fail("object entry must be an object")
        object_id = item.get("object_id")
        if not isinstance(object_id, str) or not object_id:
            fail("object entry missing object_id")
        object_ids.append(object_id)

        revisions = item.get("revisions", [])
        if not isinstance(revisions, list):
            fail(f"{object_id} revisions must be an array")
        for revision in revisions:
            if not isinstance(revision, dict):
                fail(f"{object_id} revision must be an object")
            content_sha256 = revision.get("content_sha256")
            path_value = revision.get("path")
            if content_sha256 is None and path_value is None:
                continue
            if not isinstance(content_sha256, str) or not SHA256_RE.fullmatch(
                content_sha256
            ):
                fail(f"{object_id} invalid content_sha256")
            if not isinstance(path_value, str) or not path_value:
                fail(f"{object_id} revision missing path")
            local_path = ROOT / path_value
            if not local_path.is_file():
                fail(f"registered path missing: {path_value}")
            if sha256_file(local_path) != content_sha256:
                fail(f"registered digest mismatch: {path_value}")

    if len(object_ids) != len(set(object_ids)):
        fail("duplicate object_id")
    if not REQUIRED_OBJECT_IDS.issubset(set(object_ids)):
        fail("required Centenary governance identity missing")

    relation_ids: list[str] = []
    relation_tuples: set[tuple[str, str, str]] = set()
    for relation in relations:
        if not isinstance(relation, dict):
            fail("relation entry must be an object")
        relation_id = relation.get("relation_id")
        source = relation.get("source")
        target = relation.get("target")
        relation_type = relation.get("type")
        binds = relation.get("binds")
        if not isinstance(relation_id, str) or not relation_id:
            fail("relation missing relation_id")
        relation_ids.append(relation_id)
        if source not in object_ids or target not in object_ids:
            fail(f"dangling relation endpoint: {relation_id}")
        if relation_type not in RELATION_TYPES:
            fail(f"unknown relation type: {relation_id}")
        if binds not in {"logical_object", "exact_revision"}:
            fail(f"invalid relation binding: {relation_id}")
        relation_tuples.add((source, relation_type, target))

    if len(relation_ids) != len(set(relation_ids)):
        fail("duplicate relation_id")

    required_relations = {
        (
            "glyph-object:governance:centenary-247:v1",
            "DEPENDS_ON",
            "glyph-object:governance:preservation-law:v1",
        ),
        (
            "glyph-object:governance:preservation-law:v1",
            "DEPENDS_ON",
            "glyph-object:governance:object-identity-dependency:v1",
        ),
        (
            "glyph-object:tool:check-centenary-247:v1",
            "VERIFIES",
            "glyph-object:governance:object-graph:v1",
        ),
    }
    if not required_relations.issubset(relation_tuples):
        fail("required identity/dependency relation missing")

    return len(objects), len(relations)


def main() -> None:
    for path in (SPEC, LAW, LEDGER, IDENTITY_MODEL, OBJECT_GRAPH):
        if not path.is_file():
            fail(f"missing required file: {path.relative_to(ROOT)}")

    lines = SPEC.read_text(encoding="utf-8").splitlines()
    current_section: str | None = None
    by_section: dict[str, list[int]] = {
        section: [] for section in EXPECTED_SECTIONS
    }
    observed: list[int] = []

    for line in lines:
        section_match = SECTION_RE.match(line)
        if section_match:
            current_section = section_match.group(1)
            continue

        facet_match = FACET_RE.match(line)
        if not facet_match:
            continue
        if current_section is None:
            fail("facet appears before a section")

        number = int(facet_match.group(1))
        observed.append(number)
        by_section[current_section].append(number)

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or any(not cell for cell in cells):
            fail(f"F{number:03d} does not have four non-empty cells")

    expected = list(range(1, 248))
    if observed != expected:
        fail("facet IDs are not exactly F001 through F247 in order")
    if len(set(observed)) != 247:
        fail("duplicate facet ID")

    for section, (first, last) in EXPECTED_SECTIONS.items():
        wanted = list(range(first, last + 1))
        if by_section[section] != wanted:
            fail(f"section {section} range mismatch")

    law_text = LAW.read_text(encoding="utf-8")
    required_law_phrases = (
        "Effective date: **2026-08-14**",
        "This authority belongs only to the Owner.",
        "GLYPH_MOVEMENT_LEDGER.md",
    )
    for phrase in required_law_phrases:
        if phrase not in law_text:
            fail(f"preservation law missing: {phrase}")

    ledger_text = LEDGER.read_text(encoding="utf-8")
    if "2026-08-14" not in ledger_text:
        fail("movement ledger has no establishment date")

    object_count, relation_count = validate_object_graph()

    print("facet_count = 247")
    print("facet_ids = F001..F247")
    print("section_count = 13")
    print("preservation_law = PRESENT")
    print("movement_ledger = PRESENT")
    print(f"registered_object_count = {object_count}")
    print(f"registered_relation_count = {relation_count}")
    print("dependency_unknown_policy = FAIL_CLOSED")
    print("GLYPH CENTENARY 247 STRUCTURE OK")


if __name__ == "__main__":
    main()
