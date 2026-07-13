#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_BINARY_RUNTIME_COUNT_V1.json"
)

TARGETS = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
    "query_fm_binary_v1",
]


class GateError(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    expect_success: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=240,
        check=False,
    )

    if expect_success and result.returncode != 0:
        raise GateError(
            f"command failed: {command}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return result


def configure_and_build() -> None:
    run(["cmake", "-S", ".", "-B", "build"])
    run([
        "cmake",
        "--build",
        "build",
        "--target",
        *TARGETS,
        "-j2",
    ])


def naive_count(corpus: bytes, query: bytes) -> int:
    if not query:
        raise GateError("empty query")

    if len(query) > len(corpus):
        return 0

    return sum(
        corpus[offset:offset + len(query)] == query
        for offset in range(
            len(corpus) - len(query) + 1
        )
    )


def build_index(
    directory: Path,
    corpus: bytes,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)

    corpus_path = directory / "corpus.bin"
    sa_path = directory / "sa.binary_v1"
    bwt_path = directory / "bwt.binary_v1"
    fm_path = directory / "fm.binary_v1"

    corpus_path.write_bytes(corpus)

    run([
        str(BUILD / "build_sa_binary_v1"),
        str(corpus_path),
        str(sa_path),
    ])

    run([
        str(BUILD / "build_bwt_binary_v1"),
        str(corpus_path),
        str(sa_path),
        str(bwt_path),
    ])

    run([
        str(BUILD / "build_fm_binary_v1"),
        str(bwt_path),
        str(fm_path),
        "32",
    ])

    return {
        "corpus": corpus_path,
        "sa": sa_path,
        "bwt": bwt_path,
        "fm": fm_path,
    }


def query_index(
    paths: dict[str, Path],
    query: bytes,
) -> dict:
    result = run([
        str(BUILD / "query_fm_binary_v1"),
        str(paths["fm"]),
        str(paths["bwt"]),
        query.hex(),
    ])

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            f"query returned invalid JSON: {result.stdout}"
        ) from error

    return parsed


def parse_sa(path: Path) -> dict:
    data = path.read_bytes()

    header_format = "<8sIQQIIIIQQ"
    header_size = struct.calcsize(header_format)

    if len(data) < header_size:
        raise GateError("SA file too small")

    (
        magic,
        version,
        corpus_bytes,
        row_count,
        alphabet_size,
        logical_sentinel,
        internal_sentinel,
        byte_shift,
        payload_bytes,
        payload_checksum,
    ) = struct.unpack_from(header_format, data, 0)

    if magic != b"GLYSAB1\x00":
        raise GateError("bad SA magic")

    rows = list(struct.unpack_from(
        f"<{row_count}I",
        data,
        header_size,
    ))

    return {
        "version": version,
        "corpus_bytes": corpus_bytes,
        "row_count": row_count,
        "alphabet_size": alphabet_size,
        "logical_sentinel": logical_sentinel,
        "internal_sentinel": internal_sentinel,
        "byte_shift": byte_shift,
        "payload_bytes": payload_bytes,
        "payload_checksum": payload_checksum,
        "rows": rows,
    }


def parse_bwt(path: Path) -> dict:
    data = path.read_bytes()

    header_format = "<8sIQQIIIQQ"
    header_size = struct.calcsize(header_format)

    if len(data) < header_size:
        raise GateError("BWT file too small")

    (
        magic,
        version,
        corpus_bytes,
        row_count,
        alphabet_size,
        logical_sentinel,
        width_bits,
        payload_bytes,
        payload_checksum,
    ) = struct.unpack_from(header_format, data, 0)

    if magic != b"GLYBWT1\x00":
        raise GateError("bad BWT magic")

    symbols = list(struct.unpack_from(
        f"<{row_count}H",
        data,
        header_size,
    ))

    return {
        "version": version,
        "corpus_bytes": corpus_bytes,
        "row_count": row_count,
        "alphabet_size": alphabet_size,
        "logical_sentinel": logical_sentinel,
        "width_bits": width_bits,
        "payload_bytes": payload_bytes,
        "payload_checksum": payload_checksum,
        "symbols": symbols,
    }


def validate_fixture(
    work: Path,
    name: str,
    corpus: bytes,
    queries: list[bytes],
) -> dict:
    paths = build_index(work / name, corpus)

    sa = parse_sa(paths["sa"])
    bwt = parse_bwt(paths["bwt"])

    if sa["row_count"] != len(corpus) + 1:
        raise GateError(f"{name}: SA cardinality mismatch")

    if sa["rows"][0] != len(corpus):
        raise GateError(f"{name}: terminal suffix not first")

    if sorted(sa["rows"]) != list(range(len(corpus) + 1)):
        raise GateError(f"{name}: SA is not a permutation")

    if bwt["row_count"] != len(corpus) + 1:
        raise GateError(f"{name}: BWT cardinality mismatch")

    if bwt["symbols"].count(256) != 1:
        raise GateError(f"{name}: sentinel multiplicity mismatch")

    results = []

    for query in queries:
        actual = query_index(paths, query)
        expected_count = naive_count(corpus, query)

        if actual.get("count") != expected_count:
            raise GateError({
                "fixture": name,
                "query_hex": query.hex(),
                "expected_count": expected_count,
                "actual": actual,
            })

        results.append({
            "query_hex": query.hex(),
            "query_length_bytes": len(query),
            "expected_count": expected_count,
            "actual_count": actual["count"],
            "interval": actual["interval"],
        })

    return {
        "fixture": name,
        "corpus_bytes": len(corpus),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "sa_rows": sa["row_count"],
        "bwt_rows": bwt["row_count"],
        "sentinel_count": bwt["symbols"].count(256),
        "source_zero_count": corpus.count(b"\x00"),
        "bwt_zero_symbol_count": bwt["symbols"].count(0),
        "query_count": len(results),
        "queries": results,
    }


def main() -> int:
    configure_and_build()

    with tempfile.TemporaryDirectory(
        prefix="glyph-binary-count-v1-"
    ) as temp:
        work = Path(temp)

        all_bytes = bytes(range(256))

        fixtures = [
            validate_fixture(
                work,
                "ascii",
                b"banana",
                [
                    b"ana",
                    b"a",
                    b"banana",
                    b"not-present",
                ],
            ),
            validate_fixture(
                work,
                "embedded_nul",
                b"A\x00B\x00A",
                [
                    b"\x00",
                    b"A\x00B",
                    b"\x00A",
                    b"B\x00",
                    b"\x00\x00",
                ],
            ),
            validate_fixture(
                work,
                "repeated_nul",
                b"\x00\x00\x00\xff\x00",
                [
                    b"\x00",
                    b"\x00\x00",
                    b"\x00\x00\x00",
                    b"\xff\x00",
                    b"\x00\xff",
                ],
            ),
            validate_fixture(
                work,
                "all_256_bytes",
                all_bytes,
                [
                    *[
                        bytes([value])
                        for value in range(256)
                    ],
                    all_bytes,
                    b"\x00\x01",
                    b"\xfe\xff",
                    b"\xff\x00",
                ],
            ),
            validate_fixture(
                work,
                "empty_corpus",
                b"",
                [
                    b"\x00",
                    b"\xff",
                    b"A",
                ],
            ),
        ]

        malformed = []

        malformed_cases = [
            "",
            "0",
            "00FF",
            "0x00",
            "00 ff",
            "gg",
        ]

        reference = build_index(
            work / "malformed_reference",
            b"A\x00B",
        )

        for value in malformed_cases:
            result = run(
                [
                    str(BUILD / "query_fm_binary_v1"),
                    str(reference["fm"]),
                    str(reference["bwt"]),
                    value,
                ],
                expect_success=False,
            )

            if result.returncode == 0:
                raise GateError(
                    f"malformed query accepted: {value!r}"
                )

            malformed.append({
                "query_text": value,
                "rejected": True,
                "exit_code": result.returncode,
            })

        all_256_fixture = next(
            item
            for item in fixtures
            if item["fixture"] == "all_256_bytes"
        )

        if all_256_fixture["source_zero_count"] != 1:
            raise GateError("all-256 source lost byte 0x00")

        if all_256_fixture["bwt_zero_symbol_count"] != 1:
            raise GateError(
                "BWT source byte 0x00 collides or disappeared"
            )

        output = {
            "ok": True,
            "format": "GLYPH_BINARY_RUNTIME_COUNT_V1",
            "runtime_profile": "GLYPH_BINARY_RUNTIME_V1",
            "count_path_conformant": True,
            "runtime_conformant": False,
            "source_byte_domain": "0x00..0xFF",
            "logical_sentinel": 256,
            "sa_internal_mapping": {
                "sentinel": 0,
                "source_byte_formula": "byte + 1",
                "alphabet_size": 257,
            },
            "bwt_storage": {
                "symbol_width_bits": 16,
                "source_bytes": "0..255",
                "sentinel": 256,
            },
            "fixture_count": len(fixtures),
            "query_count": sum(
                item["query_count"]
                for item in fixtures
            ),
            "malformed_query_count": len(malformed),
            "fixtures": fixtures,
            "malformed_queries": malformed,
            "closed_baseline_gaps": [
                "RUNTIME-GAP-01",
                "RUNTIME-GAP-02",
                "RUNTIME-GAP-03",
                "RUNTIME-GAP-04",
                "RUNTIME-GAP-05",
            ],
            "remaining_runtime_work": [
                "C++ locate path",
                "multi-document boundary path",
                "runtime artifact replay",
                "self-contained binary-runtime bundle",
                "proof-graph integration",
            ],
            "non_claims": [
                "Count-path conformance is not full runtime conformance.",
                "The existing sentinel-safe v0.x format remains unchanged.",
                "Locate and bundle conformance are not yet established.",
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
