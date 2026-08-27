#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


BWT_HEADER = struct.Struct("<8sIQQIIIQQ")
RLB2_HEADER = struct.Struct(
    "<8sIIQQQIIIIIIQQQQ32s32s"
)

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1

SUCCESS = "GLYPH RLB2 HOSTILE GATE OK"


class CheckError(RuntimeError):
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


def fnv1a64(payload):
    value = FNV_OFFSET

    for byte in payload:
        value ^= byte
        value = (value * FNV_PRIME) & U64_MASK

    return value


def make_bwt(path):
    symbols = [
        256,
        0, 0, 0,
        1, 1,
        255, 255, 255, 255,
        0,
    ]

    payload = b"".join(
        struct.pack("<H", symbol)
        for symbol in symbols
    )

    header = BWT_HEADER.pack(
        b"GLYBWT1\x00",
        1,
        len(symbols) - 1,
        len(symbols),
        257,
        256,
        16,
        len(payload),
        fnv1a64(payload),
    )

    path.write_bytes(header + payload)


def invoke(tool, arguments, cwd):
    return subprocess.run(
        [sys.executable, str(tool), *map(str, arguments)],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def expect_success(tool, arguments, cwd):
    completed = invoke(tool, arguments, cwd)

    require(
        completed.returncode == 0,
        "expected command success",
    )
    require(
        completed.stderr == b"",
        "successful command wrote stderr",
    )

    return json.loads(completed.stdout)


def expect_failure(
    mutations,
    name,
    tool,
    arguments,
    cwd,
):
    completed = invoke(tool, arguments, cwd)

    require(
        completed.returncode != 0,
        f"mutation accepted: {name}",
    )
    require(
        b'"ok":true' not in completed.stdout,
        f"mutation emitted success: {name}",
    )

    mutations.append({
        "mutation": name,
        "rejected": True,
    })


def patch_bytes(payload, offset, encoded):
    result = bytearray(payload)
    result[offset:offset + len(encoded)] = encoded
    return bytes(result)


def mutate_header(baseline, index, value):
    values = list(RLB2_HEADER.unpack(
        baseline[:RLB2_HEADER.size]
    ))
    values[index] = value

    return (
        RLB2_HEADER.pack(*values)
        + baseline[RLB2_HEADER.size:]
    )


def replace_payload(baseline, payload):
    values = list(RLB2_HEADER.unpack(
        baseline[:RLB2_HEADER.size]
    ))

    values[12] = len(payload)
    values[15] = fnv1a64(payload)
    values[17] = hashlib.sha256(payload).digest()

    return RLB2_HEADER.pack(*values) + payload


def write_mutation(directory, name, payload):
    path = directory / f"{name}.rlb2"
    path.write_bytes(payload)
    return path


def run_checker(tool, output):
    require(tool.is_file(), "tool missing")
    require(
        not output.exists() and not output.is_symlink(),
        "output already exists",
    )

    mutations = []

    with tempfile.TemporaryDirectory(
        prefix="glyph-rlb2-hostile-"
    ) as temporary_name:
        work = Path(temporary_name)
        cwd_a = work / "cwd-a"
        cwd_b = work / "cwd-b"
        cwd_a.mkdir()
        cwd_b.mkdir()

        bwt = work / "fixture.bwt"
        make_bwt(bwt)

        rlb2_a = work / "fixture-a.rlb2"
        rlb2_b = work / "fixture-b.rlb2"

        encode_a = expect_success(
            tool,
            ("encode", bwt, rlb2_a),
            cwd_a,
        )
        encode_b = expect_success(
            tool,
            ("encode", bwt, rlb2_b),
            cwd_b,
        )

        require(
            rlb2_a.read_bytes() == rlb2_b.read_bytes(),
            "RLB2 encoding is not deterministic",
        )
        require(
            encode_a == encode_b,
            "encode result is not deterministic",
        )

        inspect = expect_success(
            tool,
            ("inspect", rlb2_a),
            cwd_b,
        )

        decoded = work / "decoded.bwt"
        decode = expect_success(
            tool,
            ("decode", rlb2_a, decoded),
            cwd_a,
        )

        require(
            decoded.read_bytes() == bwt.read_bytes(),
            "tiny BWT roundtrip mismatch",
        )
        require(encode_a["run_count"] == 5, "bad run count")
        require(inspect["payload_bytes"] == 11, "bad payload")
        require(decode["decoded_bwt_bytes"] == 78, "bad decode size")

        baseline_bwt = bwt.read_bytes()
        bwt_mutations = [
            (
                "bwt_magic",
                patch_bytes(
                    baseline_bwt,
                    0,
                    b"BADBWT!\x00",
                ),
            ),
            (
                "bwt_version",
                patch_bytes(
                    baseline_bwt,
                    8,
                    struct.pack("<I", 2),
                ),
            ),
            (
                "bwt_alphabet",
                patch_bytes(
                    baseline_bwt,
                    28,
                    struct.pack("<I", 256),
                ),
            ),
            (
                "bwt_sentinel",
                patch_bytes(
                    baseline_bwt,
                    32,
                    struct.pack("<I", 255),
                ),
            ),
            (
                "bwt_symbol_width",
                patch_bytes(
                    baseline_bwt,
                    36,
                    struct.pack("<I", 9),
                ),
            ),
            (
                "bwt_checksum",
                patch_bytes(
                    baseline_bwt,
                    48,
                    struct.pack("<Q", 0),
                ),
            ),
            (
                "bwt_payload",
                baseline_bwt[:-1]
                + bytes((baseline_bwt[-1] ^ 1,)),
            ),
        ]

        for name, payload in bwt_mutations:
            candidate = work / f"{name}.bwt"
            candidate.write_bytes(payload)

            expect_failure(
                mutations,
                name,
                tool,
                (
                    "encode",
                    candidate,
                    work / f"{name}.out.rlb2",
                ),
                cwd_a,
            )

        baseline = rlb2_a.read_bytes()
        header = list(RLB2_HEADER.unpack(
            baseline[:RLB2_HEADER.size]
        ))

        header_mutations = [
            ("rlb2_magic", 0, b"BADRLB2\x00"),
            ("rlb2_version", 1, 3),
            ("rlb2_header_size", 2, 159),
            ("rlb2_row_count", 4, 12),
            ("rlb2_run_count_zero", 5, 0),
            ("rlb2_alphabet", 6, 256),
            ("rlb2_sentinel", 7, 255),
            ("rlb2_escape", 8, 300),
            ("rlb2_head_encoding", 9, 2),
            ("rlb2_length_encoding", 10, 2),
            ("rlb2_reserved", 11, 1),
            (
                "rlb2_declared_payload_size",
                12,
                header[12] + 1,
            ),
        ]

        for name, index, value in header_mutations:
            candidate = write_mutation(
                work,
                name,
                mutate_header(baseline, index, value),
            )

            expect_failure(
                mutations,
                name,
                tool,
                ("inspect", candidate),
                cwd_a,
            )

        truncated = write_mutation(
            work,
            "rlb2_truncated",
            baseline[:-1],
        )
        expect_failure(
            mutations,
            "rlb2_truncated",
            tool,
            ("inspect", truncated),
            cwd_a,
        )

        trailing = write_mutation(
            work,
            "rlb2_trailing",
            baseline + b"\x00",
        )
        expect_failure(
            mutations,
            "rlb2_trailing",
            tool,
            ("inspect", trailing),
            cwd_a,
        )

        payload_fnv = write_mutation(
            work,
            "payload_fnv",
            mutate_header(
                baseline,
                15,
                header[15] ^ 1,
            ),
        )
        expect_failure(
            mutations,
            "payload_fnv",
            tool,
            ("inspect", payload_fnv),
            cwd_a,
        )

        altered_payload_sha = bytearray(header[17])
        altered_payload_sha[0] ^= 1
        payload_sha = write_mutation(
            work,
            "payload_sha256",
            mutate_header(
                baseline,
                17,
                bytes(altered_payload_sha),
            ),
        )
        expect_failure(
            mutations,
            "payload_sha256",
            tool,
            ("inspect", payload_sha),
            cwd_a,
        )

        source_fnv = write_mutation(
            work,
            "source_fnv",
            mutate_header(
                baseline,
                14,
                header[14] ^ 1,
            ),
        )
        expect_failure(
            mutations,
            "source_fnv",
            tool,
            (
                "decode",
                source_fnv,
                work / "source-fnv.bwt",
            ),
            cwd_a,
        )

        altered_source_sha = bytearray(header[16])
        altered_source_sha[0] ^= 1
        source_sha = write_mutation(
            work,
            "source_sha256",
            mutate_header(
                baseline,
                16,
                bytes(altered_source_sha),
            ),
        )
        expect_failure(
            mutations,
            "source_sha256",
            tool,
            (
                "decode",
                source_sha,
                work / "source-sha.bwt",
            ),
            cwd_a,
        )

        payload = baseline[RLB2_HEADER.size:]

        semantic_payloads = []

        changed = bytearray(payload)
        changed[1] = 2
        semantic_payloads.append(
            ("invalid_escape_tag", bytes(changed))
        )

        changed = bytearray(payload)
        changed[2] = 0
        semantic_payloads.append(
            ("zero_run_length", bytes(changed))
        )

        changed = bytearray(payload)
        changed[5] = 0
        semantic_payloads.append(
            ("adjacent_equal_runs", bytes(changed))
        )

        changed = bytearray(payload)
        changed[2] = 12
        semantic_payloads.append(
            ("rows_exceed_declared", bytes(changed))
        )

        changed = bytearray(payload)
        changed[1] = 0
        semantic_payloads.append(
            ("sentinel_missing", bytes(changed))
        )

        noncanonical = (
            payload[:2]
            + b"\x81\x00"
            + payload[3:]
        )
        semantic_payloads.append(
            ("noncanonical_uleb128", noncanonical)
        )

        for name, changed_payload in semantic_payloads:
            candidate = write_mutation(
                work,
                name,
                replace_payload(
                    baseline,
                    changed_payload,
                ),
            )

            expect_failure(
                mutations,
                name,
                tool,
                (
                    "decode",
                    candidate,
                    work / f"{name}.bwt",
                ),
                cwd_b,
            )

        existing_rlb2 = work / "existing.rlb2"
        existing_rlb2.write_bytes(b"preserve")

        expect_failure(
            mutations,
            "encode_existing_output",
            tool,
            ("encode", bwt, existing_rlb2),
            cwd_a,
        )
        require(
            existing_rlb2.read_bytes() == b"preserve",
            "encode overwrote existing output",
        )

        existing_bwt = work / "existing.bwt"
        existing_bwt.write_bytes(b"preserve")

        expect_failure(
            mutations,
            "decode_existing_output",
            tool,
            ("decode", rlb2_a, existing_bwt),
            cwd_a,
        )
        require(
            existing_bwt.read_bytes() == b"preserve",
            "decode overwrote existing output",
        )

    require(len(mutations) == 33, "mutation count mismatch")
    require(
        all(item["rejected"] for item in mutations),
        "not all mutations rejected",
    )

    result = {
        "format": "GLYPH_RLB2_HOSTILE_GATE_V1",
        "ok": True,
        "tool_sha256":
            hashlib.sha256(
                tool.read_bytes()
            ).hexdigest(),
        "positive_case_count": 4,
        "mutation_count": len(mutations),
        "all_mutations_rejected": True,
        "deterministic_encode_verified": True,
        "different_cwd_verified": True,
        "tiny_roundtrip_verified": True,
        "golden_data_required": False,
        "mutations": mutations,
    }

    payload = canonical_bytes(result)

    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    print(SUCCESS)
    print("positive_case_count=4")
    print(f"mutation_count={len(mutations)}")
    print("all_mutations_rejected=true")
    print(
        "output_sha256="
        + hashlib.sha256(payload).hexdigest()
    )


def main():
    if len(sys.argv) != 3:
        print("usage: checker TOOL OUTPUT", file=sys.stderr)
        return 2

    try:
        run_checker(
            Path(sys.argv[1]).resolve(),
            Path(sys.argv[2]).resolve(),
        )
        return 0
    except CheckError as error:
        print(f"RLB2 CHECK ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
