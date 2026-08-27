#!/usr/bin/env python3

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import stat
import struct
import sys


FORMAT = "GLYPH_RLBWT_BINARY_SAFE_V2_PROBE_V1"
BWT_MAGIC = b"GLYBWT1\x00"
BWT_VERSION = 1
ALPHABET_SIZE = 257
LOGICAL_SENTINEL = 256
SYMBOL_WIDTH_BITS = 16
HEADER = struct.Struct("<8sIQQIIIQQ")

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1

LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")


class ProbeError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ProbeError(message)


def uleb128_size(value):
    require(value > 0, "run length must be positive")
    size = 1

    while value >= 128:
        value >>= 7
        size += 1

    return size


def ratio_record(payload_bytes, corpus_bytes):
    decimal = None

    if corpus_bytes:
        with localcontext() as context:
            context.prec = 40
            decimal = format(
                Decimal(payload_bytes)
                / Decimal(corpus_bytes),
                ".12f",
            )

    return {
        "payload_bytes": payload_bytes,
        "ratio_denominator_source_bytes":
            corpus_bytes,
        "ratio_decimal": decimal,
        "ratio_numerator_payload_bytes":
            payload_bytes,
    }


def parse_item(value):
    if ":" not in value:
        raise ProbeError(
            "--item must be LABEL:/path/to/bwt.bin"
        )

    label, raw_path = value.split(":", 1)

    require(
        LABEL_RE.fullmatch(label) is not None,
        f"invalid item label: {label!r}",
    )
    require(raw_path != "", "empty item path")

    return label, Path(raw_path)


def inspect_regular_file(path):
    try:
        metadata = os.lstat(path)
    except FileNotFoundError as error:
        raise ProbeError(
            f"input does not exist: {path}"
        ) from error

    require(
        not stat.S_ISLNK(metadata.st_mode),
        f"symbolic-link input rejected: {path}",
    )
    require(
        stat.S_ISREG(metadata.st_mode),
        f"input must be a regular file: {path}",
    )


def analyze_bwt(label, path):
    inspect_regular_file(path)

    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())

        require(
            stat.S_ISREG(opened.st_mode),
            "opened input is not a regular file",
        )
        require(
            opened.st_size >= HEADER.size,
            "BWT file is shorter than header",
        )

        mapped = mmap.mmap(
            stream.fileno(),
            0,
            access=mmap.ACCESS_READ,
        )

        try:
            file_sha256 = hashlib.sha256(
                mapped
            ).hexdigest()

            fields = HEADER.unpack(
                mapped[:HEADER.size]
            )

            (
                magic,
                version,
                corpus_bytes,
                row_count,
                alphabet_size,
                sentinel,
                width_bits,
                payload_bytes,
                expected_checksum,
            ) = fields

            require(
                magic == BWT_MAGIC,
                "BWT magic mismatch",
            )
            require(
                version == BWT_VERSION,
                "BWT version mismatch",
            )
            require(
                alphabet_size == ALPHABET_SIZE,
                "BWT alphabet size mismatch",
            )
            require(
                sentinel == LOGICAL_SENTINEL,
                "BWT sentinel mismatch",
            )
            require(
                width_bits == SYMBOL_WIDTH_BITS,
                "BWT symbol width mismatch",
            )
            require(
                row_count == corpus_bytes + 1,
                "BWT row count mismatch",
            )
            require(
                payload_bytes == row_count * 2,
                "BWT payload size field mismatch",
            )
            require(
                opened.st_size
                == HEADER.size + payload_bytes,
                "BWT file size mismatch",
            )

            payload = memoryview(
                mapped
            )[HEADER.size:]

            native_values = None

            try:
                if sys.byteorder == "little":
                    native_values = payload.cast("H")
                    values = native_values
                else:
                    values = (
                        item[0]
                        for item in struct.iter_unpack(
                            "<H",
                            payload,
                        )
                    )

                run_counts = [0] * ALPHABET_SIZE
                symbol_counts = [0] * ALPHABET_SIZE

                run_count = 0
                run_length_bytes = 0
                maximum_run_length = 0

                previous = None
                current_length = 0
                checksum = FNV_OFFSET

                def finish_run(symbol, length):
                    nonlocal run_count
                    nonlocal run_length_bytes
                    nonlocal maximum_run_length

                    run_counts[symbol] += 1
                    run_count += 1
                    run_length_bytes += (
                        uleb128_size(length)
                    )
                    maximum_run_length = max(
                        maximum_run_length,
                        length,
                    )

                for raw_symbol in values:
                    symbol = int(raw_symbol)

                    require(
                        0 <= symbol < ALPHABET_SIZE,
                        "BWT symbol outside 0..256",
                    )

                    symbol_counts[symbol] += 1

                    checksum ^= symbol & 0xff
                    checksum = (
                        checksum * FNV_PRIME
                    ) & U64_MASK
                    checksum ^= (symbol >> 8) & 0xff
                    checksum = (
                        checksum * FNV_PRIME
                    ) & U64_MASK

                    if previous is None:
                        previous = symbol
                        current_length = 1
                    elif symbol == previous:
                        current_length += 1
                    else:
                        finish_run(
                            previous,
                            current_length,
                        )
                        previous = symbol
                        current_length = 1

                if previous is not None:
                    finish_run(
                        previous,
                        current_length,
                    )

            finally:
                if native_values is not None:
                    native_values.release()
                payload.release()

        finally:
            mapped.close()

    require(
        checksum == expected_checksum,
        "BWT payload checksum mismatch",
    )
    require(
        sum(symbol_counts) == row_count,
        "BWT symbol-count sum mismatch",
    )
    require(
        symbol_counts[LOGICAL_SENTINEL] == 1,
        "logical sentinel occurrence count mismatch",
    )
    require(
        run_counts[LOGICAL_SENTINEL] == 1,
        "logical sentinel run count mismatch",
    )
    require(run_count > 0, "empty BWT run sequence")

    escape_symbol = min(
        range(256),
        key=lambda symbol: (
            run_counts[symbol],
            symbol,
        ),
    )
    escape_runs = run_counts[escape_symbol]
    sentinel_runs = run_counts[LOGICAL_SENTINEL]

    byte_only_reference = (
        run_count + run_length_bytes
    )

    escape_payload = (
        run_count
        + escape_runs
        + sentinel_runs
        + run_length_bytes
    )

    uvarint_symbol_bytes = sum(
        count * (1 if symbol < 128 else 2)
        for symbol, count
        in enumerate(run_counts)
    )
    uvarint_payload = (
        uvarint_symbol_bytes
        + run_length_bytes
    )

    packed9_payload = (
        (run_count * 9 + 7) // 8
        + run_length_bytes
    )

    uint16_payload = (
        run_count * 2
        + run_length_bytes
    )

    escape_overhead = (
        escape_payload - byte_only_reference
    )
    escape_bound = (
        (run_count - 1) // 256 + 1
    )

    require(
        escape_overhead <= escape_bound,
        "adaptive-escape bound violated",
    )

    candidates = {}

    for name, payload_size, note in (
        (
            "adaptive_escape",
            escape_payload,
            "one-byte normal run head; "
            "two-byte escaped run head",
        ),
        (
            "packed_9bit",
            packed9_payload,
            "theoretical separate packed "
            "run-head stream",
        ),
        (
            "symbol_uvarint",
            uvarint_payload,
            "ULEB128 run head and run length",
        ),
        (
            "symbol_uint16_le",
            uint16_payload,
            "fixed uint16 little-endian "
            "run head",
        ),
    ):
        candidate = ratio_record(
            payload_size,
            corpus_bytes,
        )
        candidate["note"] = note
        candidate[
            "overhead_vs_byte_only_reference"
        ] = payload_size - byte_only_reference
        candidates[name] = candidate

    return {
        "alphabet_size": alphabet_size,
        "all_256_byte_values_present":
            all(
                count > 0
                for count in symbol_counts[:256]
            ),
        "bwt_file_bytes": opened.st_size,
        "bwt_sha256": file_sha256,
        "byte_only_reference_payload_bytes":
            byte_only_reference,
        "candidates": candidates,
        "corpus_bytes": corpus_bytes,
        "escape_choice": {
            "policy":
                "fewest_runs_then_lowest_symbol",
            "symbol": escape_symbol,
            "symbol_runs": escape_runs,
        },
        "fnv1a64_payload_checksum":
            expected_checksum,
        "fnv1a64_payload_checksum_verified":
            True,
        "header_bytes": HEADER.size,
        "label": label,
        "logical_sentinel": sentinel,
        "logical_sentinel_occurrences":
            symbol_counts[LOGICAL_SENTINEL],
        "logical_sentinel_runs":
            sentinel_runs,
        "maximum_run_length":
            maximum_run_length,
        "payload_bytes": payload_bytes,
        "row_count": row_count,
        "run_count": run_count,
        "run_counts_by_symbol": run_counts,
        "run_length_uleb128_bytes":
            run_length_bytes,
        "source_byte_value_count":
            sum(
                count > 0
                for count in symbol_counts[:256]
            ),
        "symbol_width_bits": width_bits,
        "theorem_candidate": {
            "actual_escape_overhead_bytes":
                escape_overhead,
            "adaptive_escape_bound_holds":
                True,
            "upper_bound_bytes":
                escape_bound,
        },
        "version": version,
    }


def canonical_bytes(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_atomic(path, payload):
    require(
        not path.exists() and not path.is_symlink(),
        f"output already exists: {path}",
    )
    require(
        path.parent.is_dir(),
        f"output parent is not a directory: "
        f"{path.parent}",
    )

    temporary = (
        path.parent
        / f".{path.name}.tmp-{os.getpid()}"
    )

    descriptor = None

    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL,
            0o644,
        )

        with os.fdopen(
            descriptor,
            "wb",
            closefd=True,
        ) as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ProbeError(
                f"output already exists: {path}"
            ) from error

        temporary.unlink()

        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY

        directory_descriptor = os.open(
            path.parent,
            directory_flags,
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Measure candidate binary-safe RLBWT "
            "run encodings from canonical "
            "GLYPH_BINARY_BWT_V1 files. "
            "This validates container parsing and "
            "run-cost models, not suffix-BWT "
            "semantics."
        )
    )
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        metavar="LABEL:BWT",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
    )
    arguments = parser.parse_args()

    parsed = [
        parse_item(value)
        for value in arguments.item
    ]

    labels = [label for label, _ in parsed]

    require(
        len(labels) == len(set(labels)),
        "duplicate item label",
    )

    items = [
        analyze_bwt(label, path)
        for label, path in sorted(parsed)
    ]

    result = {
        "decision_status":
            "MEASUREMENT_ONLY_NO_RLB2_SELECTED",
        "format": FORMAT,
        "item_count": len(items),
        "items": items,
        "non_claims": [
            "No RLB2 format is selected.",
            "Rank, locate, metadata and evidence "
            "bytes are excluded.",
            "No runtime code is changed.",
            "No complete sub-1x runtime is claimed.",
        ],
        "ok": True,
        "probe_version": 1,
    }

    payload = canonical_bytes(result)
    output_sha256 = hashlib.sha256(
        payload
    ).hexdigest()

    write_atomic(arguments.out, payload)

    print(
        "GLYPH RLBWT BINARY SAFE V2 "
        "PROBE OK"
    )
    print(f"item_count={len(items)}")
    print(f"output_sha256={output_sha256}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProbeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
