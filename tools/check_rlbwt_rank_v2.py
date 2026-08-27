#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


HEADER_BYTES = 160
RECORD_BYTES = 1052
ALPHABET_SIZE = 257

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1

SUCCESS = "GLYPH RLR2 HOSTILE GATE OK"


class CheckError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckError(message)


def fnv1a64(payload):
    value = FNV_OFFSET

    for byte in payload:
        value ^= byte
        value = (value * FNV_PRIME) & U64_MASK

    return value


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def run_command(arguments, cwd):
    return subprocess.run(
        [str(item) for item in arguments],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def require_success(arguments, cwd, marker):
    completed = run_command(arguments, cwd)

    require(
        completed.returncode == 0,
        "expected command success",
    )
    require(
        marker in completed.stdout,
        "success marker missing",
    )
    require(
        completed.stderr == "",
        "successful command emitted stderr",
    )

    return completed


def expect_failure(
    name,
    arguments,
    cwd,
    success_marker,
):
    completed = run_command(arguments, cwd)
    combined = completed.stdout + completed.stderr

    require(
        completed.returncode != 0,
        f"mutation accepted: {name}",
    )
    require(
        success_marker not in combined,
        f"mutation emitted success marker: {name}",
    )

    return {
        "name": name,
        "rejected": True,
        "returncode": completed.returncode,
    }


def refresh_header_fnv(data):
    struct.pack_into(
        "<Q",
        data,
        152,
        fnv1a64(data[:152]),
    )


def refresh_payload_identity(data):
    payload = bytes(data[HEADER_BYTES:])

    struct.pack_into(
        "<Q",
        data,
        112,
        fnv1a64(payload),
    )

    data[120:152] = hashlib.sha256(
        payload
    ).digest()

    refresh_header_fnv(data)


def write_candidate(directory, name, payload):
    path = directory / f"{name}.rlr2"
    path.write_bytes(payload)
    return path


def parse_json_output(completed):
    lines = completed.stdout.splitlines()

    require(lines, "missing JSON output")

    try:
        return json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise CheckError(
            "invalid JSON output"
        ) from error


def inspect_success(tool, candidate, cwd):
    completed = require_success(
        [
            sys.executable,
            tool,
            "inspect",
            candidate,
        ],
        cwd,
        "GLYPH RLR2 INSPECT OK",
    )

    value = parse_json_output(completed)
    require(value.get("ok") is True, "inspect not OK")
    return value


def run_checker(tool, source_rlb2, output):
    tool = tool.resolve()
    source_rlb2 = source_rlb2.resolve()
    output = output.resolve()

    require(tool.is_file(), "tool not found")
    require(source_rlb2.is_file(), "source RLB2 not found")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")

    mutations = []

    with tempfile.TemporaryDirectory(
        prefix="glyph-rlr2-hostile-"
    ) as temporary_name:
        work = Path(temporary_name)
        other_cwd = work / "other-cwd"
        other_cwd.mkdir()

        baseline = work / "baseline.rlr2"
        deterministic = work / "deterministic.rlr2"

        build_a = require_success(
            [
                sys.executable,
                tool,
                "build",
                source_rlb2,
                baseline,
                "--rank-step",
                "4",
            ],
            work,
            "GLYPH RLR2 BUILD OK",
        )

        build_b = require_success(
            [
                sys.executable,
                tool,
                "build",
                source_rlb2,
                deterministic,
                "--rank-step",
                "4",
            ],
            other_cwd,
            "GLYPH RLR2 BUILD OK",
        )

        require(
            baseline.read_bytes()
            == deterministic.read_bytes(),
            "deterministic build mismatch",
        )
        require(
            build_a.stdout == build_b.stdout,
            "build output changed with CWD",
        )

        baseline_value = inspect_success(
            tool,
            baseline,
            work,
        )

        verify = require_success(
            [
                sys.executable,
                tool,
                "verify",
                source_rlb2,
                baseline,
            ],
            work,
            "GLYPH RLR2 VERIFY OK",
        )

        verify_value = parse_json_output(verify)

        require(
            verify_value.get(
                "deterministic_rebuild_verified"
            ) is True,
            "deterministic verify missing",
        )

        raw = baseline.read_bytes()

        require(len(raw) == 4368, "baseline size mismatch")
        require(
            baseline_value["checkpoint_count"] == 4,
            "baseline checkpoint mismatch",
        )

        def header_mutation(
            name,
            offset,
            fmt,
            value,
            mode="inspect",
        ):
            changed = bytearray(raw)
            struct.pack_into(fmt, changed, offset, value)
            refresh_header_fnv(changed)
            candidate = write_candidate(
                work,
                name,
                changed,
            )

            if mode == "inspect":
                arguments = [
                    sys.executable,
                    tool,
                    "inspect",
                    candidate,
                ]
                marker = "GLYPH RLR2 INSPECT OK"
            else:
                arguments = [
                    sys.executable,
                    tool,
                    "verify",
                    source_rlb2,
                    candidate,
                ]
                marker = "GLYPH RLR2 VERIFY OK"

            mutations.append(
                expect_failure(
                    name,
                    arguments,
                    work,
                    marker,
                )
            )

        changed = bytearray(raw)
        changed[0:8] = b"BADRLR2\x00"
        refresh_header_fnv(changed)
        mutations.append(
            expect_failure(
                "magic",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "magic",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        header_mutation("version", 8, "<I", 3)
        header_mutation("header_size", 12, "<I", 159)
        header_mutation("raw_length", 16, "<Q", 12)
        header_mutation(
            "run_count",
            24,
            "<Q",
            6,
            mode="verify",
        )
        header_mutation("rank_step_zero", 32, "<I", 0)
        header_mutation("rank_step_changed", 32, "<I", 5)
        header_mutation("alphabet", 36, "<I", 256)
        header_mutation("sentinel", 40, "<I", 255)
        header_mutation("counter_width", 44, "<I", 64)
        header_mutation("checkpoint_count", 48, "<Q", 5)
        header_mutation("record_bytes", 56, "<I", 1053)
        header_mutation("reserved", 60, "<I", 1)
        header_mutation("payload_bytes", 104, "<Q", 4209)

        changed = bytearray(raw)
        struct.pack_into(
            "<Q",
            changed,
            112,
            struct.unpack_from("<Q", changed, 112)[0] ^ 1,
        )
        refresh_header_fnv(changed)
        mutations.append(
            expect_failure(
                "payload_fnv",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "payload_fnv",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        changed[120] ^= 1
        refresh_header_fnv(changed)
        mutations.append(
            expect_failure(
                "payload_sha256",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "payload_sha256",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        changed[152] ^= 1
        mutations.append(
            expect_failure(
                "header_fnv",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "header_fnv",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        mutations.append(
            expect_failure(
                "truncated",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "truncated",
                        raw[:-1],
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        mutations.append(
            expect_failure(
                "trailing",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "trailing",
                        raw + b"\x00",
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        changed[-1] ^= 1
        mutations.append(
            expect_failure(
                "payload_byte",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "payload_byte",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        second = HEADER_BYTES + RECORD_BYTES
        third = second + RECORD_BYTES
        final = third + RECORD_BYTES

        changed = bytearray(raw)
        struct.pack_into("<Q", changed, second, 5)
        refresh_payload_identity(changed)
        mutations.append(
            expect_failure(
                "checkpoint_position",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "checkpoint_position",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        first_count = HEADER_BYTES + 24
        struct.pack_into("<I", changed, first_count, 1)
        refresh_payload_identity(changed)
        mutations.append(
            expect_failure(
                "counter_sum",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "counter_sum",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        final_counts = final + 24
        sentinel_offset = final_counts + 256 * 4
        sentinel = struct.unpack_from(
            "<I",
            changed,
            sentinel_offset,
        )[0]
        require(sentinel == 1, "baseline sentinel mismatch")

        donor = None

        for symbol in range(256):
            offset = final_counts + symbol * 4
            count = struct.unpack_from(
                "<I",
                changed,
                offset,
            )[0]

            if count > 0:
                donor = offset
                break

        require(donor is not None, "sentinel donor missing")

        struct.pack_into("<I", changed, donor, (
            struct.unpack_from("<I", changed, donor)[0] - 1
        ))
        struct.pack_into("<I", changed, sentinel_offset, 2)
        refresh_payload_identity(changed)

        mutations.append(
            expect_failure(
                "sentinel_cardinality",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "sentinel_cardinality",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        previous_counts = third + 24
        current_counts = final + 24
        target = None

        for symbol in range(256):
            previous = struct.unpack_from(
                "<I",
                changed,
                previous_counts + symbol * 4,
            )[0]

            if previous > 0:
                target = symbol
                break

        require(target is not None, "regression target missing")

        target_offset = current_counts + target * 4
        old_target = struct.unpack_from(
            "<I",
            changed,
            target_offset,
        )[0]
        previous_target = struct.unpack_from(
            "<I",
            changed,
            previous_counts + target * 4,
        )[0]
        new_target = previous_target - 1
        delta = old_target - new_target

        compensation = (
            0 if target != 0 else 1
        )
        compensation_offset = (
            current_counts + compensation * 4
        )
        compensation_value = struct.unpack_from(
            "<I",
            changed,
            compensation_offset,
        )[0]

        struct.pack_into(
            "<I",
            changed,
            target_offset,
            new_target,
        )
        struct.pack_into(
            "<I",
            changed,
            compensation_offset,
            compensation_value + delta,
        )
        refresh_payload_identity(changed)

        mutations.append(
            expect_failure(
                "counter_regression",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "counter_regression",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        source_bytes = struct.unpack_from(
            "<Q",
            raw,
            64,
        )[0]

        changed = bytearray(raw)
        struct.pack_into(
            "<Q",
            changed,
            final + 8,
            source_bytes + 1,
        )
        refresh_payload_identity(changed)
        mutations.append(
            expect_failure(
                "stream_offset_outside",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "stream_offset_outside",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        struct.pack_into("<Q", changed, second + 16, 12)
        refresh_payload_identity(changed)
        mutations.append(
            expect_failure(
                "run_offset_outside",
                [
                    sys.executable,
                    tool,
                    "inspect",
                    write_candidate(
                        work,
                        "run_offset_outside",
                        changed,
                    ),
                ],
                work,
                "GLYPH RLR2 INSPECT OK",
            )
        )

        changed = bytearray(raw)
        struct.pack_into(
            "<Q",
            changed,
            64,
            source_bytes + 1,
        )
        refresh_header_fnv(changed)
        source_bytes_candidate = write_candidate(
            work,
            "source_bytes",
            changed,
        )
        inspect_success(
            tool,
            source_bytes_candidate,
            work,
        )
        mutations.append(
            expect_failure(
                "source_bytes_verify",
                [
                    sys.executable,
                    tool,
                    "verify",
                    source_rlb2,
                    source_bytes_candidate,
                ],
                work,
                "GLYPH RLR2 VERIFY OK",
            )
        )

        changed = bytearray(raw)
        changed[72] ^= 1
        refresh_header_fnv(changed)
        source_sha_candidate = write_candidate(
            work,
            "source_sha",
            changed,
        )
        inspect_success(
            tool,
            source_sha_candidate,
            work,
        )
        mutations.append(
            expect_failure(
                "source_sha_verify",
                [
                    sys.executable,
                    tool,
                    "verify",
                    source_rlb2,
                    source_sha_candidate,
                ],
                work,
                "GLYPH RLR2 VERIFY OK",
            )
        )

        mutations.append(
            expect_failure(
                "build_existing_output",
                [
                    sys.executable,
                    tool,
                    "build",
                    source_rlb2,
                    baseline,
                    "--rank-step",
                    "4",
                ],
                work,
                "GLYPH RLR2 BUILD OK",
            )
        )

    require(
        len(mutations) == 29,
        f"mutation count mismatch: {len(mutations)}",
    )
    require(
        all(item["rejected"] for item in mutations),
        "not all mutations rejected",
    )

    payload = canonical_bytes({
        "format": "GLYPH_RLR2_HOSTILE_GATE_V1",
        "ok": True,
        "tool_sha256":
            hashlib.sha256(
                tool.read_bytes()
            ).hexdigest(),
        "source_rlb2_sha256":
            hashlib.sha256(
                source_rlb2.read_bytes()
            ).hexdigest(),
        "positive_case_count": 4,
        "mutation_count": len(mutations),
        "all_mutations_rejected": True,
        "deterministic_build_verified": True,
        "different_cwd_verified": True,
        "source_binding_verified": True,
        "semantic_checkpoint_validation_verified": True,
        "mutations": mutations,
        "success_marker": SUCCESS,
    })

    descriptor = os.open(
        output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )

    try:
        with os.fdopen(
            descriptor,
            "wb",
            closefd=True,
        ) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor is not None:
            os.close(descriptor)

    print(SUCCESS)
    print("positive_case_count=4")
    print(f"mutation_count={len(mutations)}")
    print("all_mutations_rejected=true")
    print(
        "output_sha256="
        + hashlib.sha256(payload).hexdigest()
    )


def main():
    if len(sys.argv) != 4:
        print(
            "usage: checker TOOL SOURCE_RLB2 OUTPUT",
            file=sys.stderr,
        )
        return 2

    try:
        run_checker(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            Path(sys.argv[3]),
        )
        return 0
    except CheckError as error:
        print(
            f"RLR2 CHECK ERROR: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
