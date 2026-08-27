#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


MAGIC = b"GLYBWT1\x00"
VERSION = 1
ALPHABET_SIZE = 257
SENTINEL = 256
WIDTH_BITS = 16
HEADER = struct.Struct("<8sIQQIIIQQ")

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1
SUCCESS_MARKER = (
    "GLYPH RLBWT BINARY SAFE V2 PROBE OK"
)


def fnv_u16(values):
    value = FNV_OFFSET

    for symbol in values:
        value ^= symbol & 0xff
        value = (value * FNV_PRIME) & U64_MASK
        value ^= (symbol >> 8) & 0xff
        value = (value * FNV_PRIME) & U64_MASK

    return value


def make_bwt(values, **overrides):
    payload = b"".join(
        struct.pack("<H", value)
        for value in values
    )

    fields = {
        "magic": MAGIC,
        "version": VERSION,
        "corpus_bytes": len(values) - 1,
        "row_count": len(values),
        "alphabet_size": ALPHABET_SIZE,
        "sentinel": SENTINEL,
        "width_bits": WIDTH_BITS,
        "payload_bytes": len(payload),
        "checksum": fnv_u16(values),
    }
    fields.update(overrides)

    return HEADER.pack(
        fields["magic"],
        fields["version"],
        fields["corpus_bytes"],
        fields["row_count"],
        fields["alphabet_size"],
        fields["sentinel"],
        fields["width_bits"],
        fields["payload_bytes"],
        fields["checksum"],
    ) + payload


def run_probe(tool, items, output, cwd):
    command = [
        sys.executable,
        "-I",
        str(tool),
    ]

    for label, path in items:
        command.extend([
            "--item",
            f"{label}:{path}",
        ])

    command.extend([
        "--out",
        str(output),
    ])

    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
        check=False,
    )


def require_failure(
    mutations,
    name,
    expected,
    process,
):
    if (
        process.returncode == 0
        or expected not in process.stderr
        or SUCCESS_MARKER in process.stdout
    ):
        raise RuntimeError(
            f"mutation not rejected: {name}; "
            f"rc={process.returncode}; "
            f"stdout={process.stdout!r}; "
            f"stderr={process.stderr!r}"
        )

    mutations.append({
        "expected_error": expected,
        "mutation": name,
        "rejected": True,
    })


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: checker TOOL RESULT RECORD_DIR"
        )

    tool = Path(sys.argv[1]).resolve()
    result_path = Path(sys.argv[2])
    record_dir = Path(sys.argv[3])

    if result_path.exists():
        raise RuntimeError("result already exists")

    with tempfile.TemporaryDirectory(
        prefix="rlbwt-probe-hostile-",
        dir=record_dir,
    ) as temporary_name:
        temporary = Path(temporary_name)

        fixtures = {
            "all256_bound": list(range(257)),
            "embedded_nul": [
                0, 0, 255, 255, 0, 256
            ],
            "escape_absent": [
                1, 1, 2, 2, 256
            ],
            "repetitive": [
                *([97] * 1024),
                256,
            ],
        }

        fixture_paths = {}

        for label, values in fixtures.items():
            path = temporary / f"{label}.bwt"
            path.write_bytes(make_bwt(values))
            fixture_paths[label] = path

        cwd_a = temporary / "cwd-a"
        cwd_b = temporary / "cwd-b"
        cwd_a.mkdir()
        cwd_b.mkdir()

        output_a = temporary / "positive-a.json"
        output_b = temporary / "positive-b.json"

        ordered = sorted(fixture_paths.items())
        reversed_order = list(reversed(ordered))

        positive_a = run_probe(
            tool,
            ordered,
            output_a,
            cwd_a,
        )
        positive_b = run_probe(
            tool,
            reversed_order,
            output_b,
            cwd_b,
        )

        for label, process in (
            ("a", positive_a),
            ("b", positive_b),
        ):
            if (
                process.returncode != 0
                or process.stderr != ""
                or SUCCESS_MARKER
                not in process.stdout
            ):
                raise RuntimeError(
                    f"positive run failed: {label}; "
                    f"stdout={process.stdout!r}; "
                    f"stderr={process.stderr!r}"
                )

        if output_a.read_bytes() != output_b.read_bytes():
            raise RuntimeError(
                "positive output depends on cwd/order"
            )

        if positive_a.stdout != positive_b.stdout:
            raise RuntimeError(
                "positive stdout depends on cwd/order"
            )

        positive = json.loads(
            output_a.read_text(encoding="utf-8")
        )

        by_label = {
            item["label"]: item
            for item in positive["items"]
        }

        bound_item = by_label["all256_bound"]

        if not (
            bound_item["all_256_byte_values_present"]
            and bound_item["run_count"] == 257
            and bound_item["escape_choice"]["symbol"] == 0
            and bound_item["escape_choice"][
                "symbol_runs"
            ] == 1
            and bound_item["theorem_candidate"][
                "actual_escape_overhead_bytes"
            ] == 2
            and bound_item["theorem_candidate"][
                "upper_bound_bytes"
            ] == 2
        ):
            raise RuntimeError(
                "all-256 bound fixture mismatch"
            )

        base_values = [0, 1, 2, 256]
        base_bytes = make_bwt(base_values)

        mutation_specs = [
            (
                "bad_magic",
                make_bwt(
                    base_values,
                    magic=b"BADBWT1\x00",
                ),
                "BWT magic mismatch",
            ),
            (
                "bad_version",
                make_bwt(
                    base_values,
                    version=2,
                ),
                "BWT version mismatch",
            ),
            (
                "row_count_relation",
                make_bwt(
                    base_values,
                    corpus_bytes=4,
                ),
                "BWT row count mismatch",
            ),
            (
                "bad_alphabet",
                make_bwt(
                    base_values,
                    alphabet_size=256,
                ),
                "BWT alphabet size mismatch",
            ),
            (
                "bad_header_sentinel",
                make_bwt(
                    base_values,
                    sentinel=255,
                ),
                "BWT sentinel mismatch",
            ),
            (
                "bad_symbol_width",
                make_bwt(
                    base_values,
                    width_bits=9,
                ),
                "BWT symbol width mismatch",
            ),
            (
                "bad_payload_size_field",
                make_bwt(
                    base_values,
                    payload_bytes=10,
                ),
                "BWT payload size field mismatch",
            ),
            (
                "bad_checksum",
                make_bwt(
                    base_values,
                    checksum=fnv_u16(base_values) ^ 1,
                ),
                "BWT payload checksum mismatch",
            ),
            (
                "truncated_payload",
                base_bytes[:-2],
                "BWT file size mismatch",
            ),
            (
                "trailing_payload",
                base_bytes + b"\x00\x00",
                "BWT file size mismatch",
            ),
            (
                "symbol_257",
                make_bwt([0, 257, 256]),
                "BWT symbol outside 0..256",
            ),
            (
                "missing_sentinel",
                make_bwt([0, 1, 2]),
                "logical sentinel occurrence "
                "count mismatch",
            ),
            (
                "duplicate_sentinel",
                make_bwt([0, 256, 256]),
                "logical sentinel occurrence "
                "count mismatch",
            ),
        ]

        mutations = []

        for name, payload, expected in mutation_specs:
            path = temporary / f"{name}.bwt"
            path.write_bytes(payload)

            process = run_probe(
                tool,
                [(name, path)],
                temporary / f"{name}.json",
                cwd_a,
            )

            require_failure(
                mutations,
                name,
                expected,
                process,
            )

        symlink_path = temporary / "symlink.bwt"
        symlink_path.symlink_to(
            fixture_paths["escape_absent"]
        )

        require_failure(
            mutations,
            "symlink_input",
            "symbolic-link input rejected",
            run_probe(
                tool,
                [("symlink", symlink_path)],
                temporary / "symlink.json",
                cwd_a,
            ),
        )

        require_failure(
            mutations,
            "duplicate_label",
            "duplicate item label",
            run_probe(
                tool,
                [
                    (
                        "duplicate",
                        fixture_paths[
                            "escape_absent"
                        ],
                    ),
                    (
                        "duplicate",
                        fixture_paths[
                            "repetitive"
                        ],
                    ),
                ],
                temporary / "duplicate.json",
                cwd_a,
            ),
        )

        existing_output = temporary / "existing.json"
        existing_output.write_text(
            "{}\n",
            encoding="utf-8",
        )

        require_failure(
            mutations,
            "existing_output",
            "output already exists",
            run_probe(
                tool,
                [
                    (
                        "existing",
                        fixture_paths[
                            "escape_absent"
                        ],
                    )
                ],
                existing_output,
                cwd_a,
            ),
        )

        result = {
            "all_256_bound_equality_verified":
                True,
            "different_cwd_verified": True,
            "different_item_order_verified": True,
            "format":
                "GLYPH_RLBWT_BINARY_SAFE_V2_"
                "PROBE_HOSTILE_GATE_V1",
            "mutation_count": len(mutations),
            "mutations": mutations,
            "ok": True,
            "positive_fixture_count":
                len(fixtures),
            "probe_sha256":
                hashlib.sha256(
                    tool.read_bytes()
                ).hexdigest(),
            "success_marker": SUCCESS_MARKER,
        }

        payload = canonical_bytes(result)
        result_path.write_bytes(payload)

        print(
            "GLYPH RLBWT BINARY SAFE V2 "
            "PROBE HOSTILE GATE OK"
        )
        print(
            f"positive_fixture_count="
            f"{len(fixtures)}"
        )
        print(
            f"mutation_count={len(mutations)}"
        )
        print(
            "all_256_bound_equality_verified=true"
        )
        print(
            "result_sha256="
            f"{hashlib.sha256(payload).hexdigest()}"
        )


if __name__ == "__main__":
    main()
