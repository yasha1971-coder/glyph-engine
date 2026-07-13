#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

OUT = (
    ROOT
    / "benchmarks/results/"
    / "GLYPH_BINARY_RUNTIME_LOCATE_V1.json"
)

TARGETS = [
    "build_sa_binary_v1",
    "build_bwt_binary_v1",
    "build_fm_binary_v1",
    "query_fm_binary_v1",
    "query_fm_locate_binary_v1",
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


def build_index(
    directory: Path,
    corpus: bytes,
) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "corpus": directory / "corpus.bin",
        "sa": directory / "sa.binary_v1",
        "bwt": directory / "bwt.binary_v1",
        "fm": directory / "fm.binary_v1",
    }

    paths["corpus"].write_bytes(corpus)

    run([
        str(BUILD / "build_sa_binary_v1"),
        str(paths["corpus"]),
        str(paths["sa"]),
    ])

    run([
        str(BUILD / "build_bwt_binary_v1"),
        str(paths["corpus"]),
        str(paths["sa"]),
        str(paths["bwt"]),
    ])

    run([
        str(BUILD / "build_fm_binary_v1"),
        str(paths["bwt"]),
        str(paths["fm"]),
        "32",
    ])

    return paths


def naive_offsets(
    corpus: bytes,
    query: bytes,
) -> list[int]:
    if not query:
        raise GateError("empty query")

    if len(query) > len(corpus):
        return []

    return [
        offset
        for offset in range(
            len(corpus) - len(query) + 1
        )
        if corpus[
            offset:offset + len(query)
        ] == query
    ]


def query_locate(
    paths: dict[str, Path],
    query: bytes,
    max_offsets: int | None = None,
) -> dict:
    command = [
        str(BUILD / "query_fm_locate_binary_v1"),
        str(paths["fm"]),
        str(paths["bwt"]),
        str(paths["sa"]),
        str(paths["corpus"]),
        query.hex(),
    ]

    if max_offsets is not None:
        command.append(str(max_offsets))

    result = run(command)

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise GateError(
            f"invalid locate JSON: {result.stdout}"
        ) from error


def validate_result(
    corpus: bytes,
    query: bytes,
    result: dict,
    max_offsets: int | None,
) -> None:
    expected = naive_offsets(corpus, query)
    expected_count = len(expected)

    if max_offsets is None:
        expected_returned = expected
    else:
        expected_returned = expected[:max_offsets]

    expected_bounded = (
        len(expected_returned) < expected_count
    )

    if result.get("ok") is not True:
        raise GateError("locate result not ok")

    if result.get("match_count") != expected_count:
        raise GateError({
            "query_hex": query.hex(),
            "expected_count": expected_count,
            "result": result,
        })

    if result.get("offsets") != expected_returned:
        raise GateError({
            "query_hex": query.hex(),
            "expected_offsets": expected_returned,
            "result": result,
        })

    expected_coordinates = [
        [0, offset]
        for offset in expected_returned
    ]

    if result.get("coordinates") != expected_coordinates:
        raise GateError(
            "canonical coordinate mismatch"
        )

    if result.get("returned_count") != len(
        expected_returned
    ):
        raise GateError("returned_count mismatch")

    if result.get("bounded") is not expected_bounded:
        raise GateError("bounded flag mismatch")

    if (
        result.get("offsets_complete")
        is not (not expected_bounded)
    ):
        raise GateError("offsets_complete mismatch")

    if result.get("byte_check") is not True:
        raise GateError("byte_check not true")

    if result["interval"][1] - result["interval"][0] != expected_count:
        raise GateError("FM interval/count mismatch")

    if len(corpus) in result["offsets"]:
        raise GateError(
            "terminal suffix leaked into locate output"
        )


def validate_fixture(
    work: Path,
    name: str,
    corpus: bytes,
    queries: list[bytes],
    *,
    bounds: bool = True,
) -> tuple[dict, dict[str, Path]]:
    paths = build_index(work / name, corpus)
    query_results = []

    for query in queries:
        full = query_locate(paths, query)
        validate_result(corpus, query, full, None)

        bounded_results = []

        if bounds:
            count = len(naive_offsets(corpus, query))

            candidates = sorted({
                0,
                1,
                max(0, count - 1),
                count,
                count + 1,
            })

            for max_offsets in candidates:
                bounded = query_locate(
                    paths,
                    query,
                    max_offsets,
                )

                validate_result(
                    corpus,
                    query,
                    bounded,
                    max_offsets,
                )

                bounded_results.append({
                    "max_offsets": max_offsets,
                    "returned_count":
                        bounded["returned_count"],
                    "bounded": bounded["bounded"],
                })

        query_results.append({
            "query_hex": query.hex(),
            "match_count": full["match_count"],
            "offsets": full["offsets"],
            "bounded_results": bounded_results,
        })

    return (
        {
            "fixture": name,
            "corpus_bytes": len(corpus),
            "corpus_sha256":
                hashlib.sha256(corpus).hexdigest(),
            "query_count": len(queries),
            "queries": query_results,
        },
        paths,
    )


def corrupt_copy(
    source: Path,
    destination: Path,
) -> Path:
    data = bytearray(source.read_bytes())

    if not data:
        raise GateError("cannot corrupt empty file")

    data[-1] ^= 0x01
    destination.write_bytes(data)
    return destination


def main() -> int:
    configure_and_build()

    with tempfile.TemporaryDirectory(
        prefix="glyph-binary-locate-v1-"
    ) as temp:
        work = Path(temp)

        all_bytes = bytes(range(256))

        fixtures_with_paths = [
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
                "repeated_matches",
                b"aaaaaa",
                [
                    b"a",
                    b"aa",
                    b"aaa",
                    b"aaaa",
                    b"aaaaaa",
                    b"aaaaaaa",
                ],
            ),
            validate_fixture(
                work,
                "invalid_utf8",
                b"\x80\x81\xfe\xff\x00\x80",
                [
                    b"\x80",
                    b"\x81\xfe",
                    b"\xff\x00",
                    b"\x00\x80",
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
                bounds=False,
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

        fixtures = [
            fixture
            for fixture, _ in fixtures_with_paths
        ]

        path_map = {
            fixture["fixture"]: paths
            for fixture, paths in fixtures_with_paths
        }

        mutations = []

        malformed_cases = [
            "",
            "0",
            "00FF",
            "0x00",
            "00 ff",
            "gg",
        ]

        reference = path_map["embedded_nul"]

        for value in malformed_cases:
            result = run(
                [
                    str(BUILD / "query_fm_locate_binary_v1"),
                    str(reference["fm"]),
                    str(reference["bwt"]),
                    str(reference["sa"]),
                    str(reference["corpus"]),
                    value,
                ],
                expect_success=False,
            )

            if result.returncode == 0:
                raise GateError(
                    f"malformed query accepted: {value!r}"
                )

            mutations.append({
                "mutation":
                    f"malformed_query_{value!r}",
                "rejected": True,
            })

        for bad_max in ["-1", "x", "1x"]:
            result = run(
                [
                    str(BUILD / "query_fm_locate_binary_v1"),
                    str(reference["fm"]),
                    str(reference["bwt"]),
                    str(reference["sa"]),
                    str(reference["corpus"]),
                    "00",
                    bad_max,
                ],
                expect_success=False,
            )

            if result.returncode == 0:
                raise GateError(
                    f"invalid max_offsets accepted: {bad_max}"
                )

            mutations.append({
                "mutation":
                    f"invalid_max_offsets_{bad_max}",
                "rejected": True,
            })

        ascii_paths = path_map["ascii"]
        nul_paths = path_map["embedded_nul"]

        mismatch_result = run(
            [
                str(BUILD / "query_fm_locate_binary_v1"),
                str(nul_paths["fm"]),
                str(nul_paths["bwt"]),
                str(ascii_paths["sa"]),
                str(nul_paths["corpus"]),
                "00",
            ],
            expect_success=False,
        )

        if mismatch_result.returncode == 0:
            raise GateError("mismatched SA accepted")

        mutations.append({
            "mutation": "mismatched_sa_identity",
            "rejected": True,
        })

        corrupt_targets = [
            ("corrupt_sa", reference["sa"]),
            ("corrupt_bwt", reference["bwt"]),
            ("corrupt_fm", reference["fm"]),
        ]

        for name, source in corrupt_targets:
            destination = work / f"{name}.bin"
            corrupt_copy(source, destination)

            paths = dict(reference)

            if name == "corrupt_sa":
                paths["sa"] = destination
            elif name == "corrupt_bwt":
                paths["bwt"] = destination
            else:
                paths["fm"] = destination

            result = run(
                [
                    str(BUILD / "query_fm_locate_binary_v1"),
                    str(paths["fm"]),
                    str(paths["bwt"]),
                    str(paths["sa"]),
                    str(paths["corpus"]),
                    "00",
                ],
                expect_success=False,
            )

            if result.returncode == 0:
                raise GateError(
                    f"corrupted runtime file accepted: {name}"
                )

            mutations.append({
                "mutation": name,
                "rejected": True,
            })

        output = {
            "ok": True,
            "format": "GLYPH_BINARY_RUNTIME_LOCATE_V1",
            "runtime_profile": "GLYPH_BINARY_RUNTIME_V1",
            "count_path_conformant": True,
            "locate_path_conformant": True,
            "runtime_conformant": False,
            "source_byte_domain": "0x00..0xFF",
            "logical_sentinel": 256,
            "coordinate_model":
                "(document_id, document_offset)",
            "document_count_supported": 1,
            "fixture_count": len(fixtures),
            "query_count": sum(
                fixture["query_count"]
                for fixture in fixtures
            ),
            "mutation_count": len(mutations),
            "all_offsets_byte_checked": True,
            "terminal_suffix_never_returned": True,
            "bounded_locate_verified": True,
            "fixtures": fixtures,
            "mutations": mutations,
            "remaining_runtime_work": [
                "multi-document boundary runtime",
                "runtime evidence artifact",
                "deterministic runtime replay",
                "self-contained runtime bundle",
                "proof-graph integration",
            ],
            "non_claims": [
                "Single-document locate is not full runtime conformance.",
                "Multi-document boundaries are not yet implemented.",
                "Runtime artifact and bundle replay are not yet established.",
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
