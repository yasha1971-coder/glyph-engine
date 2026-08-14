#!/usr/bin/env python3
"""Fail-closed structural check for GLYPH Centenary 247 V1."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/governance/GLYPH_CENTENARY_247_V1.md"
LAW = ROOT / "docs/governance/GLYPH_PRESERVATION_LAW_V1.md"
LEDGER = ROOT / "docs/governance/GLYPH_MOVEMENT_LEDGER.md"

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


def fail(message: str) -> None:
    raise SystemExit(f"CENTENARY_247_FAIL: {message}")


def main() -> None:
    for path in (SPEC, LAW, LEDGER):
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

    print("facet_count = 247")
    print("facet_ids = F001..F247")
    print("section_count = 13")
    print("preservation_law = PRESENT")
    print("movement_ledger = PRESENT")
    print("GLYPH CENTENARY 247 STRUCTURE OK")


if __name__ == "__main__":
    main()
