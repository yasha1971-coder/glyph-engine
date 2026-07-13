#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
OUT = ROOT / "benchmarks/results/GLYPH_RUNTIME_CONFORMANCE_BASELINE_V1.json"

FORMAT = "GLYPH_RUNTIME_CONFORMANCE_BASELINE_V1"

TARGETS = [
    "build_sa_sentinel_v1",
    "build_bwt_sentinel_v1",
    "build_fm",
    "query_fm_v1",
]


class BaselineError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=180,
    )

    if expect_success and completed.returncode != 0:
        raise BaselineError(
            f"command failed: {command}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    return completed


def ensure_binaries() -> dict[str, str]:
    if not (BUILD / "CMakeCache.txt").is_file():
        run(["cmake", "-S", ".", "-B", "build"])

    run([
        "cmake",
        "--build",
        "build",
        "--target",
        *TARGETS,
        "-j2",
    ])

    binaries: dict[str, str] = {}

    for target in TARGETS:
        path = BUILD / target

        if not path.is_file():
            raise BaselineError(f"missing binary: {path}")

        binaries[target] = str(path)

    return binaries


def build_index(
    corpus: bytes,
    directory: Path,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)

    corpus_path = directory / "corpus.bin"
    sa_path = directory / "sa.bin"
    bwt_path = directory / "bwt.bin"
    fm_path = directory / "fm.bin"

    corpus_path.write_bytes(corpus)

    run([
        str(BUILD / "build_sa_sentinel_v1"),
        str(corpus_path),
        str(sa_path),
    ])

    run([
        str(BUILD / "build_bwt_sentinel_v1"),
        str(corpus_path),
        str(sa_path),
        str(bwt_path),
    ])

    run([
        str(BUILD / "build_fm"),
        str(bwt_path),
        str(fm_path),
        "128",
    ])

    return fm_path, bwt_path


def query(
    fm_path: Path,
    bwt_path: Path,
    pattern: bytes,
) -> dict[str, Any]:
    completed = run([
        str(BUILD / "query_fm_v1"),
        str(fm_path),
        str(bwt_path),
        pattern.hex(),
        "--json",
    ])

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BaselineError(
            f"query output is not JSON: {completed.stdout}"
        ) from error

    return result


def main() -> int:
    binaries = ensure_binaries()

    with tempfile.TemporaryDirectory(
        prefix="glyph-runtime-baseline-"
    ) as temp:
        work = Path(temp)

        ascii_fm, ascii_bwt = build_index(
            b"banana",
            work / "ascii",
        )

        ascii_ana = query(
            ascii_fm,
            ascii_bwt,
            b"ana",
        )

        if ascii_ana.get("count") != 2:
            raise BaselineError(
                f"ASCII count mismatch: {ascii_ana}"
            )

        ff_fm, ff_bwt = build_index(
            b"\xffA\xff",
            work / "byte_ff",
        )

        ff_query = query(
            ff_fm,
            ff_bwt,
            b"\xff",
        )

        if ff_query.get("count") != 2:
            raise BaselineError(
                f"0xFF count mismatch: {ff_query}"
            )

        nul_corpus_path = work / "embedded_nul.bin"
        nul_sa_path = work / "embedded_nul.sa.bin"
        nul_corpus_path.write_bytes(b"A\x00B")

        nul_build = run(
            [
                str(BUILD / "build_sa_sentinel_v1"),
                str(nul_corpus_path),
                str(nul_sa_path),
            ],
            expect_success=False,
        )

        nul_rejected = (
            nul_build.returncode != 0
            and "0x00" in (
                nul_build.stdout + nul_build.stderr
            )
        )

        if not nul_rejected:
            raise BaselineError(
                "current runtime did not produce the expected "
                "explicit embedded-NUL rejection"
            )

        sentinel_byte_query = query(
            ascii_fm,
            ascii_bwt,
            b"\x00",
        )

        sentinel_exposed_as_byte = (
            sentinel_byte_query.get("count") == 1
        )

        source_checks = {
            "sa_rejects_nul": (
                "input contains 0x00"
                in (
                    ROOT
                    / "src/build_sa_sentinel_v1.cpp"
                ).read_text()
            ),
            "sa_appends_physical_zero": (
                "text.push_back(0)"
                in (
                    ROOT
                    / "src/build_sa_sentinel_v1.cpp"
                ).read_text()
            ),
            "bwt_stored_as_uint8": (
                "std::vector<uint8_t> bwt"
                in (
                    ROOT
                    / "src/build_bwt_sentinel_v1.cpp"
                ).read_text()
            ),
            "fm_uses_256_symbols": (
                "std::array<uint64_t, 256> C"
                in (
                    ROOT
                    / "src/build_fm.cpp"
                ).read_text()
            ),
            "query_pattern_uses_uint8": (
                "std::vector<uint8_t> pattern"
                in (
                    ROOT
                    / "src/query_fm_v1.cpp"
                ).read_text()
            ),
        }

        if not all(source_checks.values()):
            raise BaselineError(
                f"unexpected source baseline: {source_checks}"
            )

        gaps = [
            {
                "id": "RUNTIME-GAP-01",
                "obligation": "R1",
                "status": "OPEN",
                "reason": "source corpus containing 0x00 is rejected",
            },
            {
                "id": "RUNTIME-GAP-02",
                "obligation": "R2",
                "status": "OPEN",
                "reason": "physical byte 0x00 is used as sentinel",
            },
            {
                "id": "RUNTIME-GAP-03",
                "obligation": "R2",
                "status": "OPEN",
                "reason": (
                    "query 00 can address the physical sentinel "
                    "instead of a source byte"
                ),
            },
            {
                "id": "RUNTIME-GAP-04",
                "obligation": "R4",
                "status": "OPEN",
                "reason": "BWT symbol storage is uint8",
            },
            {
                "id": "RUNTIME-GAP-05",
                "obligation": "R5",
                "status": "OPEN",
                "reason": "FM alphabet contains 256 entries, not 257",
            },
        ]

        output = {
            "ok": True,
            "audit_ok": True,
            "format": FORMAT,
            "runtime_profile": "GLYPH_SENTINEL_SAFE_V0X",
            "runtime_conformant": False,
            "binary_safe_arbitrary_bytes": False,
            "virtual_sentinel_256": False,
            "proof_graph_target": "GLYPH_PROOF_GRAPH_V1",
            "binaries": binaries,
            "tests": {
                "ascii_ana_count": ascii_ana["count"],
                "byte_ff_count": ff_query["count"],
                "embedded_nul_rejected": nul_rejected,
                "query_00_result": sentinel_byte_query,
                "sentinel_exposed_as_byte_query":
                    sentinel_exposed_as_byte,
            },
            "source_checks": source_checks,
            "open_gap_count": len(gaps),
            "gaps": gaps,
            "next_required_runtime": {
                "name": "GLYPH_BINARY_RUNTIME_V1",
                "source_symbols": "0..255",
                "virtual_sentinel": 256,
                "minimum_storage_width_bits": 9,
            },
            "non_claims": [
                "This baseline is not runtime conformance.",
                "P1-P12 currently verify reference semantics.",
                "The current C++ runtime remains sentinel-safe v0.x.",
            ],
        }

        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            json.dumps(output, indent=2, sort_keys=True)
            + "\n"
        )

        print(json.dumps(output, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
