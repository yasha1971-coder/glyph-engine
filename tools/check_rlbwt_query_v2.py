#!/usr/bin/env python3

from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import tempfile


SUCCESS = "GLYPH RLBWT QUERY V2 HOSTILE GATE OK"
QUERY_SUCCESS = "GLYPH RLBWT BINARY SAFE QUERY V2 OK"
EXPECTED_MUTATIONS = 40

RUNTIME_FORMAT = "GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2"
RLB2_FORMAT = "GLYPH_RLB2_EXPERIMENTAL_V2"
RLR2_FORMAT = "GLYPH_RLR2_V2"

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1


class CheckError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def fnv1a64(payload):
    value = FNV_OFFSET

    for byte in payload:
        value ^= byte
        value = (value * FNV_PRIME) & U64_MASK

    return value


def execute(arguments, cwd=None):
    return subprocess.run(
        [str(item) for item in arguments],
        cwd=None if cwd is None else str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )


def load_module(path):
    specification = importlib.util.spec_from_file_location(
        "glyph_query_hostile_rlb2",
        path,
    )
    require(specification is not None, "module spec unavailable")
    require(specification.loader is not None, "module loader unavailable")

    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def parse_query(process):
    require(
        process.returncode == 0,
        "query failed: " + process.stderr.strip(),
    )

    lines = [
        line
        for line in process.stdout.splitlines()
        if line.strip()
    ]

    require(len(lines) == 2, "unexpected query output")
    require(lines[1] == QUERY_SUCCESS, "query marker missing")

    result = json.loads(lines[0])
    require(result.get("ok") is True, "query result not OK")
    require(
        result.get("format")
        == "GLYPH_RLBWT_BINARY_SAFE_QUERY_V2",
        "query result format mismatch",
    )
    return result


def direct_offsets(source, pattern):
    return [
        offset
        for offset in range(len(source) - len(pattern) + 1)
        if source[offset:offset + len(pattern)] == pattern
    ]


def build_fixture(directory, rlb2_tool, rank_tool):
    source = (
        b"\x00"
        b"banana_bandana"
        b"\xff\x00"
        b"banana!"
    )

    mapped = tuple(byte + 1 for byte in source) + (0,)
    suffix_array = sorted(
        range(len(mapped)),
        key=lambda offset: mapped[offset:],
    )

    bwt_symbols = [
        256 if suffix_offset == 0
        else source[suffix_offset - 1]
        for suffix_offset in suffix_array
    ]
    require(
        bwt_symbols.count(256) == 1,
        "sentinel cardinality mismatch",
    )

    module = load_module(rlb2_tool)
    require(module.BWT_HEADER.size == 56, "BWT header mismatch")

    payload = b"".join(
        struct.pack("<H", symbol)
        for symbol in bwt_symbols
    )

    bwt = directory / "fixture.bwt.binary_v1"
    rlb2 = directory / "fixture.rlb2"
    rlr2 = directory / "fixture.rlr2"
    locate = directory / "locate_core_s2.bin"
    manifest_path = directory / "runtime_manifest_v2.json"

    header = module.BWT_HEADER.pack(
        module.BWT_MAGIC,
        module.BWT_VERSION,
        len(source),
        len(mapped),
        module.ALPHABET_SIZE,
        module.LOGICAL_SENTINEL,
        module.SYMBOL_WIDTH_BITS,
        len(payload),
        fnv1a64(payload),
    )
    bwt.write_bytes(header + payload)

    encoded = execute([
        sys.executable,
        rlb2_tool,
        "encode",
        bwt,
        rlb2,
    ])
    require(
        encoded.returncode == 0,
        "RLB2 encode failed: " + encoded.stderr.strip(),
    )

    ranked = execute([
        sys.executable,
        rank_tool,
        "build",
        rlb2,
        rlr2,
        "--rank-step",
        "4",
    ])
    require(
        ranked.returncode == 0,
        "RLR2 build failed: " + ranked.stderr.strip(),
    )

    sample_step = 2
    samples = [
        (row, suffix_array[row])
        for row in range(len(mapped))
        if row % sample_step == 0
    ]

    locate_payload = bytearray(
        struct.pack(
            "<4sQIQ",
            b"LOC1",
            len(mapped),
            sample_step,
            len(samples),
        )
    )

    for row, suffix_offset in samples:
        locate_payload.extend(
            struct.pack("<QQ", row, suffix_offset)
        )

    locate.write_bytes(locate_payload)

    runtime_bytes = sum(
        path.stat().st_size
        for path in (rlb2, rlr2, locate)
    )

    manifest = {
        "corpus_identity": {
            "bytes": len(source),
            "md5": hashlib.md5(source).hexdigest(),
            "reference_id":
                "portable-query-hostile-fixture-v1",
            "sha256": hashlib.sha256(source).hexdigest(),
        },
        "files": {
            "locate": {
                "bytes": locate.stat().st_size,
                "format": "LOC1",
                "name": locate.name,
                "sha256": sha256_file(locate),
            },
            "rlb2": {
                "bytes": rlb2.stat().st_size,
                "format": RLB2_FORMAT,
                "name": rlb2.name,
                "sha256": sha256_file(rlb2),
            },
            "rlr2": {
                "bytes": rlr2.stat().st_size,
                "format": RLR2_FORMAT,
                "name": rlr2.name,
                "sha256": sha256_file(rlr2),
            },
        },
        "format": RUNTIME_FORMAT,
        "rank_step": 4,
        "row_count": len(mapped),
        "runtime_data_bytes": runtime_bytes,
        "sample_step": sample_step,
        "version": 1,
    }
    manifest_path.write_bytes(canonical_bytes(manifest))

    return {
        "locate": locate,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "rlb2": rlb2,
        "rlr2": rlr2,
        "sample_step": sample_step,
        "source": source,
        "suffix_array": suffix_array,
    }


def manifest_query(query_tool, manifest, pattern, cwd=None):
    return parse_query(
        execute(
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                manifest,
                "--pattern-hex",
                pattern.hex(),
                "--max-offsets",
                "-1",
            ],
            cwd=cwd,
        )
    )


def explicit_query(query_tool, fixture, pattern):
    return parse_query(
        execute([
            sys.executable,
            query_tool,
            "--rlb2",
            fixture["rlb2"],
            "--rank-index",
            fixture["rlr2"],
            "--locate-core",
            fixture["locate"],
            "--pattern-hex",
            pattern.hex(),
            "--max-offsets",
            "-1",
        ])
    )


def stable_result(value):
    result = dict(value)
    result.pop("query_elapsed_ns", None)
    return result


def expect_failure(name, arguments):
    process = execute(arguments)
    rejected = (
        process.returncode != 0
        and QUERY_SUCCESS not in process.stdout.splitlines()
    )
    require(rejected, "mutation accepted: " + name)
    return name


def run_checker(query_tool, rank_tool, rlb2_tool, output):
    query_tool = query_tool.resolve()
    rank_tool = rank_tool.resolve()
    rlb2_tool = rlb2_tool.resolve()
    output = output.resolve()

    for path in (query_tool, rank_tool, rlb2_tool):
        require(path.is_file(), "missing tool: " + path.name)
        require(not path.is_symlink(), "symlink tool: " + path.name)

    require(not output.exists(), "output already exists")

    with tempfile.TemporaryDirectory(
        prefix="glyph-query-v2-hostile-"
    ) as raw_directory:
        directory = Path(raw_directory)
        fixture = build_fixture(
            directory,
            rlb2_tool,
            rank_tool,
        )

        source = fixture["source"]
        row_by_offset = {
            suffix_offset: row
            for row, suffix_offset
            in enumerate(fixture["suffix_array"])
        }
        suffix_owner = {
            source[offset:]: offset
            for offset in range(len(source))
        }

        patterns = [
            b"\x00",
            b"\xff",
            b"ana",
            b"banana",
            b"bandana",
            b"_",
            b"!",
            b"\x00banana",
            b"\xfe\xfe",
            *suffix_owner.keys(),
        ]
        patterns = list(dict.fromkeys(patterns))

        verified_suffixes = set()
        forced_non_sampled = 0
        total_lf_steps = 0
        total_offsets = 0

        for pattern in patterns:
            expected = direct_offsets(source, pattern)
            result = manifest_query(
                query_tool,
                fixture["manifest_path"],
                pattern,
            )

            require(
                result["count"] == len(expected),
                "direct count mismatch",
            )
            require(
                result["locate_offsets"] == expected,
                "direct locate mismatch",
            )
            require(
                result["locate_offsets_complete"] is True,
                "incomplete locate result",
            )

            total_offsets += len(expected)
            total_lf_steps += result["total_lf_steps"]

            for offset in expected:
                if (
                    row_by_offset[offset]
                    % fixture["sample_step"]
                    != 0
                ):
                    forced_non_sampled += 1

            if pattern in suffix_owner:
                owner = suffix_owner[pattern]
                require(owner in expected, "suffix row absent")
                verified_suffixes.add(owner)

        require(
            verified_suffixes == set(range(len(source))),
            "not all ordinary suffix rows verified",
        )
        require(forced_non_sampled > 0, "LF locate not exercised")
        require(total_lf_steps > 0, "LF step count is zero")

        explicit = explicit_query(
            query_tool,
            fixture,
            b"banana",
        )
        expected = direct_offsets(source, b"banana")
        require(explicit["count"] == len(expected), "explicit count")
        require(explicit["locate_offsets"] == expected, "explicit locate")

        alternate = directory / "alternate-cwd"
        alternate.mkdir()

        first = manifest_query(
            query_tool,
            fixture["manifest_path"],
            b"bandana",
            cwd=directory,
        )
        second = manifest_query(
            query_tool,
            fixture["manifest_path"],
            b"bandana",
            cwd=alternate,
        )
        require(
            stable_result(first) == stable_result(second),
            "different-CWD mismatch",
        )

        baseline = fixture["manifest"]
        baseline_bytes = canonical_bytes(baseline)
        mutations = []

        def arguments(path):
            return [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                path,
                "--pattern-hex",
                b"banana".hex(),
                "--max-offsets",
                "10",
            ]

        duplicate = baseline_bytes.decode("utf-8").replace(
            '"format":"GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2"',
            '"format":"GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2",'
            '"format":"GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2"',
            1,
        )
        path = directory / "duplicate.json"
        path.write_text(duplicate, encoding="utf-8")
        mutations.append(expect_failure(
            "duplicate_json_key",
            arguments(path),
        ))

        path = directory / "noncanonical.json"
        path.write_text(
            json.dumps(baseline, indent=2) + "\n",
            encoding="utf-8",
        )
        mutations.append(expect_failure(
            "noncanonical_json",
            arguments(path),
        ))

        path = directory / "trailing.json"
        path.write_bytes(baseline_bytes + b"\n")
        mutations.append(expect_failure(
            "trailing_json_whitespace",
            arguments(path),
        ))

        def mutate(name, callback):
            value = copy.deepcopy(baseline)
            callback(value)
            path = directory / (name + ".json")
            path.write_bytes(canonical_bytes(value))
            mutations.append(expect_failure(name, arguments(path)))

        mutate(
            "manifest_format",
            lambda value: value.__setitem__("format", "BAD"),
        )
        mutate(
            "manifest_version",
            lambda value: value.__setitem__("version", 2),
        )
        mutate(
            "empty_reference_id",
            lambda value: value["corpus_identity"].__setitem__(
                "reference_id", ""
            ),
        )
        mutate(
            "unsafe_reference_id",
            lambda value: value["corpus_identity"].__setitem__(
                "reference_id", "../unsafe"
            ),
        )
        mutate(
            "negative_corpus_bytes",
            lambda value: value["corpus_identity"].__setitem__(
                "bytes", -1
            ),
        )
        mutate(
            "bad_md5",
            lambda value: value["corpus_identity"].__setitem__(
                "md5", "0" * 31
            ),
        )
        mutate(
            "bad_corpus_sha256",
            lambda value: value["corpus_identity"].__setitem__(
                "sha256", "0" * 63
            ),
        )
        mutate(
            "bad_row_count",
            lambda value: value.__setitem__(
                "row_count", value["row_count"] + 1
            ),
        )
        mutate(
            "zero_rank_step",
            lambda value: value.__setitem__("rank_step", 0),
        )
        mutate(
            "zero_sample_step",
            lambda value: value.__setitem__("sample_step", 0),
        )
        mutate(
            "bad_runtime_data_bytes",
            lambda value: value.__setitem__(
                "runtime_data_bytes",
                value["runtime_data_bytes"] + 1,
            ),
        )
        mutate(
            "missing_locate_role",
            lambda value: value["files"].pop("locate"),
        )
        mutate(
            "extra_role",
            lambda value: value["files"].__setitem__(
                "extra",
                copy.deepcopy(value["files"]["locate"]),
            ),
        )
        mutate(
            "bad_rlb2_format",
            lambda value: value["files"]["rlb2"].__setitem__(
                "format", "BAD"
            ),
        )
        mutate(
            "bad_rlr2_format",
            lambda value: value["files"]["rlr2"].__setitem__(
                "format", "BAD"
            ),
        )
        mutate(
            "bad_locate_format",
            lambda value: value["files"]["locate"].__setitem__(
                "format", "BAD"
            ),
        )
        mutate(
            "unsafe_rlb2_name",
            lambda value: value["files"]["rlb2"].__setitem__(
                "name", "../fixture.rlb2"
            ),
        )
        mutate(
            "bad_rlb2_bytes",
            lambda value: value["files"]["rlb2"].__setitem__(
                "bytes",
                value["files"]["rlb2"]["bytes"] + 1,
            ),
        )
        mutate(
            "bad_rlb2_sha256",
            lambda value: value["files"]["rlb2"].__setitem__(
                "sha256", "0" * 64
            ),
        )
        mutate(
            "bad_rlr2_sha256",
            lambda value: value["files"]["rlr2"].__setitem__(
                "sha256", "0" * 64
            ),
        )
        mutate(
            "bad_locate_sha256",
            lambda value: value["files"]["locate"].__setitem__(
                "sha256", "0" * 64
            ),
        )
        mutate(
            "top_level_extra",
            lambda value: value.__setitem__("extra", 1),
        )
        mutate(
            "files_not_object",
            lambda value: value.__setitem__("files", []),
        )
        mutate(
            "corpus_identity_extra",
            lambda value: value["corpus_identity"].__setitem__(
                "extra", 1
            ),
        )

        for role in ("rlb2", "rlr2", "locate"):
            target_directory = directory / ("mutated-" + role)
            target_directory.mkdir()

            for source_path in (
                fixture["rlb2"],
                fixture["rlr2"],
                fixture["locate"],
            ):
                shutil.copy2(
                    source_path,
                    target_directory / source_path.name,
                )

            target = (
                target_directory
                / baseline["files"][role]["name"]
            )
            changed = bytearray(target.read_bytes())
            changed[-1] ^= 1
            target.write_bytes(changed)

            manifest = target_directory / "runtime_manifest_v2.json"
            manifest.write_bytes(baseline_bytes)

            mutations.append(expect_failure(
                "mutated_" + role + "_artifact",
                arguments(manifest),
            ))

        mutations.append(expect_failure(
            "mixed_binding_modes",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
                "--rlb2",
                fixture["rlb2"],
                "--rank-index",
                fixture["rlr2"],
                "--locate-core",
                fixture["locate"],
                "--pattern-hex",
                b"banana".hex(),
            ],
        ))
        mutations.append(expect_failure(
            "partial_explicit_binding",
            [
                sys.executable,
                query_tool,
                "--rlb2",
                fixture["rlb2"],
                "--pattern-hex",
                b"banana".hex(),
            ],
        ))
        mutations.append(expect_failure(
            "invalid_pattern_hex",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
                "--pattern-hex",
                "xyz",
            ],
        ))
        mutations.append(expect_failure(
            "empty_pattern",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
                "--pattern-hex",
                "",
            ],
        ))
        mutations.append(expect_failure(
            "invalid_max_offsets",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
                "--pattern-hex",
                b"banana".hex(),
                "--max-offsets",
                "-2",
            ],
        ))
        mutations.append(expect_failure(
            "missing_manifest",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                directory / "absent.json",
                "--pattern-hex",
                b"banana".hex(),
            ],
        ))
        mutations.append(expect_failure(
            "missing_pattern_argument",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
            ],
        ))
        mutations.append(expect_failure(
            "missing_binding",
            [
                sys.executable,
                query_tool,
                "--pattern-hex",
                b"banana".hex(),
            ],
        ))
        mutations.append(expect_failure(
            "noninteger_max_offsets",
            [
                sys.executable,
                query_tool,
                "--runtime-manifest",
                fixture["manifest_path"],
                "--pattern-hex",
                b"banana".hex(),
                "--max-offsets",
                "invalid",
            ],
        ))

        manifest_directory = directory / "manifest-directory"
        manifest_directory.mkdir()
        mutations.append(expect_failure(
            "manifest_is_directory",
            arguments(manifest_directory),
        ))

        require(
            len(mutations) == EXPECTED_MUTATIONS,
            "mutation count mismatch: " + str(len(mutations)),
        )

        result = {
            "all_mutations_rejected": True,
            "all_ordinary_suffix_rows_verified": True,
            "count_and_locate_equal_direct_source": True,
            "different_cwd_verified": True,
            "explicit_binding_verified": True,
            "fixture_locate_sha256":
                sha256_file(fixture["locate"]),
            "fixture_rlb2_sha256":
                sha256_file(fixture["rlb2"]),
            "fixture_rlr2_sha256":
                sha256_file(fixture["rlr2"]),
            "fixture_source_bytes": len(source),
            "fixture_source_sha256":
                hashlib.sha256(source).hexdigest(),
            "forced_non_sampled_locate_count":
                forced_non_sampled,
            "format": "GLYPH_RLBWT_QUERY_V2_HOSTILE_GATE_V1",
            "manifest_binding_verified": True,
            "mutation_count": len(mutations),
            "mutations": mutations,
            "ok": True,
            "positive_case_count": len(patterns) + 3,
            "query_tool_sha256": sha256_file(query_tool),
            "rank_tool_sha256": sha256_file(rank_tool),
            "rlb2_tool_sha256": sha256_file(rlb2_tool),
            "total_lf_steps": total_lf_steps,
            "total_located_offsets": total_offsets,
        }

        payload = canonical_bytes(result)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)

    print(SUCCESS)
    print(
        "positive_case_count="
        + str(result["positive_case_count"])
    )
    print("mutation_count=" + str(result["mutation_count"]))
    print("all_mutations_rejected=true")
    print("all_ordinary_suffix_rows_verified=true")
    print(
        "output_sha256="
        + hashlib.sha256(payload).hexdigest()
    )


def main():
    if len(sys.argv) != 5:
        print(
            "usage: checker QUERY RANK RLB2 OUTPUT",
            file=sys.stderr,
        )
        return 2

    try:
        run_checker(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
        )
        return 0
    except Exception as error:
        print(
            "QUERY V2 CHECK ERROR: " + str(error),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
