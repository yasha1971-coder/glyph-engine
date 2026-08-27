#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import tempfile

import rlbwt_container_v2 as rlb2


MAGIC = b"GLYRLR2\x00"
VERSION = 2
HEADER_BYTES = 160
ALPHABET_SIZE = 257
LOGICAL_SENTINEL = 256

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1

HEADER = struct.Struct(
    "<8sIIQQIIIIQIIQ32sQQ32sQ"
)
PREFIX = struct.Struct("<QQQ")

SUCCESS_BUILD = "GLYPH RLR2 BUILD OK"
SUCCESS_INSPECT = "GLYPH RLR2 INSPECT OK"
SUCCESS_VERIFY = "GLYPH RLR2 VERIFY OK"


class RankError(Exception):
    pass


class Fnv1a64:
    def __init__(self):
        self.value = FNV_OFFSET

    def update(self, payload):
        for byte in payload:
            self.value ^= byte
            self.value = (
                self.value * FNV_PRIME
            ) & U64_MASK


class CountingReader:
    def __init__(self, stream, payload_bytes):
        self.inner = rlb2.BufferedPayloadReader(
            stream,
            payload_bytes,
        )
        self.consumed = 0

    def read_byte(self):
        value = self.inner.read_byte()
        self.consumed += 1
        return value

    def finish(self):
        self.inner.finish()


def require(condition, message):
    if not condition:
        raise RankError(message)


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
    return rlb2.sha256_file(path)


def inspect_regular_file(path):
    require(
        not path.is_symlink(),
        f"symlink rejected: {path}",
    )

    try:
        status = path.stat()
    except FileNotFoundError as error:
        raise RankError(
            f"file not found: {path}"
        ) from error

    require(
        stat.S_ISREG(status.st_mode),
        f"not a regular file: {path}",
    )

    return status


def deterministic_counter_width(raw_length):
    require(
        0 <= raw_length <= U64_MASK,
        "raw length outside uint64",
    )

    if raw_length <= UINT32_MAX:
        return 32

    return 64


def record_bytes(counter_width_bits):
    require(
        counter_width_bits in (32, 64),
        "counter width must be 32 or 64",
    )

    return (
        PREFIX.size
        + ALPHABET_SIZE
        * (counter_width_bits // 8)
    )


def checkpoint_count(raw_length, rank_step):
    require(rank_step > 0, "rank step must be positive")

    return (
        (raw_length + rank_step - 1)
        // rank_step
        + 1
    )


def fnv1a64(payload):
    digest = Fnv1a64()
    digest.update(payload)
    return digest.value


def pack_header(
    *,
    raw_length,
    run_count,
    rank_step,
    counter_width_bits,
    checkpoints,
    record_size,
    source_rlb2_bytes,
    source_rlb2_sha256,
    payload_bytes,
    payload_fnv,
    payload_sha256,
):
    require(
        len(source_rlb2_sha256) == 32,
        "source SHA-256 must be 32 bytes",
    )
    require(
        len(payload_sha256) == 32,
        "payload SHA-256 must be 32 bytes",
    )

    raw = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        raw_length,
        run_count,
        rank_step,
        ALPHABET_SIZE,
        LOGICAL_SENTINEL,
        counter_width_bits,
        checkpoints,
        record_size,
        0,
        source_rlb2_bytes,
        source_rlb2_sha256,
        payload_bytes,
        payload_fnv,
        payload_sha256,
        0,
    )

    require(
        len(raw) == HEADER_BYTES,
        "internal header size mismatch",
    )

    header_fnv = fnv1a64(raw[:152])

    return raw[:152] + struct.pack(
        "<Q",
        header_fnv,
    )


def parse_header(stream, file_bytes):
    raw = stream.read(HEADER_BYTES)

    require(
        len(raw) == HEADER_BYTES,
        "truncated RLR2 header",
    )

    values = HEADER.unpack(raw)

    (
        magic,
        version,
        header_bytes,
        raw_length,
        run_count,
        rank_step,
        alphabet_size,
        logical_sentinel,
        counter_width_bits,
        checkpoints,
        record_size,
        reserved,
        source_rlb2_bytes,
        source_rlb2_sha256,
        payload_bytes,
        payload_fnv,
        payload_sha256,
        header_fnv,
    ) = values

    require(magic == MAGIC, "RLR2 magic mismatch")
    require(version == VERSION, "RLR2 version mismatch")
    require(
        header_bytes == HEADER_BYTES,
        "RLR2 header size mismatch",
    )
    require(
        alphabet_size == ALPHABET_SIZE,
        "RLR2 alphabet mismatch",
    )
    require(
        logical_sentinel == LOGICAL_SENTINEL,
        "RLR2 sentinel mismatch",
    )
    require(reserved == 0, "RLR2 reserved field nonzero")
    require(rank_step > 0, "RLR2 rank step is zero")
    require(
        counter_width_bits
        == deterministic_counter_width(raw_length),
        "RLR2 counter width is not deterministic",
    )
    require(
        record_size
        == record_bytes(counter_width_bits),
        "RLR2 record size mismatch",
    )
    require(
        checkpoints
        == checkpoint_count(raw_length, rank_step),
        "RLR2 checkpoint count mismatch",
    )
    require(
        payload_bytes == checkpoints * record_size,
        "RLR2 payload size mismatch",
    )
    require(
        file_bytes == HEADER_BYTES + payload_bytes,
        "RLR2 file size mismatch",
    )
    require(
        fnv1a64(raw[:152]) == header_fnv,
        "RLR2 header checksum mismatch",
    )

    return {
        "raw_header": raw,
        "raw_length": raw_length,
        "run_count": run_count,
        "rank_step": rank_step,
        "alphabet_size": alphabet_size,
        "logical_sentinel": logical_sentinel,
        "counter_width_bits": counter_width_bits,
        "checkpoint_count": checkpoints,
        "record_bytes": record_size,
        "source_rlb2_bytes": source_rlb2_bytes,
        "source_rlb2_sha256": source_rlb2_sha256,
        "payload_bytes": payload_bytes,
        "payload_fnv": payload_fnv,
        "payload_sha256": payload_sha256,
        "header_fnv": header_fnv,
    }


def decode_run(reader, escape_symbol):
    run_stream_offset = reader.consumed
    head = reader.read_byte()

    if head != escape_symbol:
        symbol = head
    else:
        tag = reader.read_byte()

        if tag == 0:
            symbol = escape_symbol
        elif tag == 1:
            symbol = LOGICAL_SENTINEL
        else:
            raise RankError("invalid RLB2 escape tag")

    try:
        length = rlb2.decode_uleb128(reader)
    except rlb2.Rlb2Error as error:
        raise RankError(
            f"invalid RLB2 run length: {error}"
        ) from error

    require(length > 0, "zero RLB2 run length")

    return (
        run_stream_offset,
        symbol,
        length,
    )


def publish_temporary(temporary, output):
    require(
        not output.exists() and not output.is_symlink(),
        f"output already exists: {output}",
    )

    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise RankError(
            f"output already exists: {output}"
        ) from error

    temporary.unlink()

    flags = os.O_RDONLY

    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY

    descriptor = os.open(output.parent, flags)

    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_rank(source, output, rank_step):
    source_status = inspect_regular_file(source)

    require(
        output.parent.is_dir(),
        f"output parent is not a directory: {output.parent}",
    )
    require(
        not output.exists() and not output.is_symlink(),
        f"output already exists: {output}",
    )
    require(
        rank_step > 0 and rank_step <= UINT32_MAX,
        "rank step outside uint32",
    )

    # This performs the committed RLB2 structural and payload
    # integrity checks before rank construction.
    try:
        rlb2.inspect_file(source)
    except rlb2.Rlb2Error as error:
        raise RankError(
            f"source RLB2 verification failed: {error}"
        ) from error

    source_sha256 = sha256_file(source)

    require(
        isinstance(source_sha256, bytes)
        and len(source_sha256) == 32,
        "source SHA-256 API contract mismatch",
    )

    source_sha256_hex = source_sha256.hex()

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    published = False

    try:
        with os.fdopen(
            temporary_fd,
            "w+b",
            closefd=True,
        ) as writer:
            descriptor_open = False
            writer.write(b"\x00" * HEADER_BYTES)

            payload_fnv = Fnv1a64()
            payload_sha = hashlib.sha256()
            payload_written = 0
            records_written = 0

            with source.open("rb") as source_stream:
                source_header = rlb2.parse_rlb2_header(
                    source_stream,
                    source_status.st_size,
                )

                raw_length = source_header["row_count"]
                run_count = source_header["run_count"]
                source_payload_bytes = source_header["payload_bytes"]
                escape_symbol = source_header["escape_symbol"]

                counter_width_bits = (
                    deterministic_counter_width(raw_length)
                )
                record_size = record_bytes(
                    counter_width_bits
                )
                expected_checkpoints = checkpoint_count(
                    raw_length,
                    rank_step,
                )

                if counter_width_bits == 32:
                    counters_struct = struct.Struct(
                        f"<{ALPHABET_SIZE}I"
                    )
                    counter_max = UINT32_MAX
                else:
                    counters_struct = struct.Struct(
                        f"<{ALPHABET_SIZE}Q"
                    )
                    counter_max = U64_MASK

                counts = [0] * ALPHABET_SIZE

                def write_payload(blob):
                    nonlocal payload_written
                    writer.write(blob)
                    payload_fnv.update(blob)
                    payload_sha.update(blob)
                    payload_written += len(blob)

                def write_record(
                    raw_position,
                    stream_offset,
                    run_offset,
                    record_counts,
                ):
                    nonlocal records_written

                    require(
                        0 <= raw_position <= raw_length,
                        "checkpoint raw position invalid",
                    )
                    require(
                        0 <= stream_offset
                        <= source_payload_bytes,
                        "checkpoint stream offset invalid",
                    )
                    require(
                        0 <= run_offset <= raw_length,
                        "checkpoint run offset invalid",
                    )
                    require(
                        len(record_counts) == ALPHABET_SIZE,
                        "checkpoint counter count invalid",
                    )
                    require(
                        all(
                            0 <= value <= counter_max
                            for value in record_counts
                        ),
                        "checkpoint counter overflow",
                    )
                    require(
                        sum(record_counts) == raw_position,
                        "checkpoint counter sum mismatch",
                    )

                    blob = (
                        PREFIX.pack(
                            raw_position,
                            stream_offset,
                            run_offset,
                        )
                        + counters_struct.pack(
                            *record_counts
                        )
                    )

                    require(
                        len(blob) == record_size,
                        "checkpoint record size mismatch",
                    )

                    write_payload(blob)
                    records_written += 1

                write_record(0, 0, 0, counts)
                next_checkpoint = rank_step
                raw_position = 0
                runs_seen = 0

                reader = CountingReader(
                    source_stream,
                    source_payload_bytes,
                )

                while runs_seen < run_count:
                    (
                        run_stream_offset,
                        symbol,
                        run_length,
                    ) = decode_run(
                        reader,
                        escape_symbol,
                    )

                    require(
                        0 <= symbol < ALPHABET_SIZE,
                        "decoded symbol outside alphabet",
                    )

                    run_start = raw_position
                    run_end = run_start + run_length

                    require(
                        run_end <= raw_length,
                        "decoded runs exceed raw length",
                    )

                    while (
                        next_checkpoint < raw_length
                        and next_checkpoint < run_end
                    ):
                        offset = (
                            next_checkpoint - run_start
                        )

                        require(
                            offset >= 0,
                            "checkpoint passed before current run",
                        )

                        checkpoint_counts = counts.copy()
                        checkpoint_counts[symbol] += offset

                        write_record(
                            next_checkpoint,
                            run_stream_offset,
                            offset,
                            checkpoint_counts,
                        )

                        next_checkpoint += rank_step

                    counts[symbol] += run_length
                    raw_position = run_end
                    runs_seen += 1

                reader.finish()

                require(
                    reader.consumed == source_payload_bytes,
                    "RLB2 payload consumption mismatch",
                )
                require(
                    runs_seen == run_count,
                    "RLB2 run count mismatch",
                )
                require(
                    raw_position == raw_length,
                    "RLB2 decoded length mismatch",
                )
                require(
                    sum(counts) == raw_length,
                    "final counter sum mismatch",
                )
                require(
                    counts[LOGICAL_SENTINEL] == 1,
                    "logical sentinel count mismatch",
                )

                write_record(
                    raw_length,
                    source_payload_bytes,
                    0,
                    counts,
                )

                require(
                    records_written == expected_checkpoints,
                    "written checkpoint count mismatch",
                )

                expected_payload_bytes = (
                    expected_checkpoints * record_size
                )

                require(
                    payload_written == expected_payload_bytes,
                    "written payload size mismatch",
                )

                header = pack_header(
                    raw_length=raw_length,
                    run_count=run_count,
                    rank_step=rank_step,
                    counter_width_bits=
                        counter_width_bits,
                    checkpoints=expected_checkpoints,
                    record_size=record_size,
                    source_rlb2_bytes=
                        source_status.st_size,
                    source_rlb2_sha256=
                        source_sha256,
                    payload_bytes=payload_written,
                    payload_fnv=payload_fnv.value,
                    payload_sha256=
                        payload_sha.digest(),
                )

            writer.seek(0)
            writer.write(header)
            writer.flush()
            os.fsync(writer.fileno())

        publish_temporary(temporary, output)
        published = True

    finally:
        if descriptor_open:
            os.close(temporary_fd)

        if temporary.exists():
            temporary.unlink()

    require(published, "RLR2 publication failed")

    result = inspect_rank(output)
    result.update({
        "operation": "build",
        "source_rlb2_sha256":
            source_sha256_hex,
    })

    return result


def inspect_rank(path):
    status = inspect_regular_file(path)

    with path.open("rb") as stream:
        header = parse_header(
            stream,
            status.st_size,
        )

        if header["counter_width_bits"] == 32:
            counters_struct = struct.Struct(
                f"<{ALPHABET_SIZE}I"
            )
        else:
            counters_struct = struct.Struct(
                f"<{ALPHABET_SIZE}Q"
            )

        payload_fnv = Fnv1a64()
        payload_sha = hashlib.sha256()
        previous_counts = [0] * ALPHABET_SIZE
        previous_stream_offset = 0

        for index in range(
            header["checkpoint_count"]
        ):
            blob = stream.read(header["record_bytes"])

            require(
                len(blob) == header["record_bytes"],
                "truncated RLR2 record",
            )

            payload_fnv.update(blob)
            payload_sha.update(blob)

            (
                raw_position,
                stream_offset,
                run_offset,
            ) = PREFIX.unpack(
                blob[:PREFIX.size]
            )

            counts = list(
                counters_struct.unpack(
                    blob[PREFIX.size:]
                )
            )

            if index + 1 == header["checkpoint_count"]:
                expected_position = header["raw_length"]
            else:
                expected_position = (
                    index * header["rank_step"]
                )

            require(
                raw_position == expected_position,
                "RLR2 checkpoint position mismatch",
            )
            require(
                stream_offset >= previous_stream_offset,
                "RLR2 stream offsets decrease",
            )
            require(
                stream_offset
                <= header["source_rlb2_bytes"],
                "RLR2 stream offset outside source",
            )
            require(
                run_offset <= header["raw_length"],
                "RLR2 run offset outside source",
            )
            require(
                sum(counts) == raw_position,
                "RLR2 checkpoint counter sum mismatch",
            )
            require(
                counts[LOGICAL_SENTINEL] <= 1,
                "RLR2 sentinel count exceeds one",
            )
            require(
                all(
                    current >= previous
                    for current, previous in zip(
                        counts,
                        previous_counts,
                    )
                ),
                "RLR2 counters decrease",
            )

            previous_counts = counts
            previous_stream_offset = stream_offset

        require(
            stream.read(1) == b"",
            "trailing RLR2 data",
        )
        require(
            payload_fnv.value == header["payload_fnv"],
            "RLR2 payload FNV mismatch",
        )
        require(
            payload_sha.digest()
            == header["payload_sha256"],
            "RLR2 payload SHA-256 mismatch",
        )
        require(
            sum(previous_counts)
            == header["raw_length"],
            "RLR2 final counter sum mismatch",
        )
        require(
            previous_counts[LOGICAL_SENTINEL] == 1,
            "RLR2 final sentinel count mismatch",
        )

    return {
        "format": "GLYPH_RLR2_INSPECT_V1",
        "ok": True,
        "operation": "inspect",
        "version": VERSION,
        "file_bytes": status.st_size,
        "file_sha256": sha256_file(path).hex(),
        "raw_length": header["raw_length"],
        "run_count": header["run_count"],
        "rank_step": header["rank_step"],
        "alphabet_size": ALPHABET_SIZE,
        "logical_sentinel": LOGICAL_SENTINEL,
        "counter_width_bits":
            header["counter_width_bits"],
        "checkpoint_count":
            header["checkpoint_count"],
        "record_bytes": header["record_bytes"],
        "header_bytes": HEADER_BYTES,
        "payload_bytes": header["payload_bytes"],
        "source_rlb2_bytes":
            header["source_rlb2_bytes"],
        "source_rlb2_sha256":
            header["source_rlb2_sha256"].hex(),
        "payload_fnv1a64":
            header["payload_fnv"],
        "payload_sha256":
            header["payload_sha256"].hex(),
        "header_fnv1a64":
            header["header_fnv"],
        "semantic_checkpoints_verified": True,
    }


def files_equal(left, right):
    left_status = inspect_regular_file(left)
    right_status = inspect_regular_file(right)

    if left_status.st_size != right_status.st_size:
        return False

    with left.open("rb") as left_stream:
        with right.open("rb") as right_stream:
            while True:
                left_chunk = left_stream.read(1024 * 1024)
                right_chunk = right_stream.read(1024 * 1024)

                if left_chunk != right_chunk:
                    return False

                if not left_chunk:
                    return True


def verify_rank(source, rank):
    source_status = inspect_regular_file(source)
    actual = inspect_rank(rank)

    source_digest = sha256_file(source)

    require(
        isinstance(source_digest, bytes)
        and len(source_digest) == 32,
        "source SHA-256 API contract mismatch",
    )
    require(
        actual["source_rlb2_bytes"]
        == source_status.st_size,
        "RLR2 source byte length mismatch",
    )
    require(
        actual["source_rlb2_sha256"]
        == source_digest.hex(),
        "RLR2 source SHA-256 mismatch",
    )

    try:
        rlb2.inspect_file(source)
    except rlb2.Rlb2Error as error:
        raise RankError(
            f"source RLB2 verification failed: {error}"
        ) from error

    with source.open("rb") as stream:
        source_header = rlb2.parse_rlb2_header(
            stream,
            source_status.st_size,
        )

    require(
        actual["raw_length"]
        == source_header["row_count"],
        "RLR2 source raw length mismatch",
    )
    require(
        actual["run_count"]
        == source_header["run_count"],
        "RLR2 source run count mismatch",
    )

    with tempfile.TemporaryDirectory(
        prefix="glyph-rlr2-verify-"
    ) as temporary_name:
        temporary = Path(temporary_name)
        expected = temporary / "expected.rlr2"

        build_rank(
            source,
            expected,
            actual["rank_step"],
        )

        require(
            files_equal(expected, rank),
            "RLR2 deterministic rebuild mismatch",
        )

        expected_sha256 = sha256_file(expected)

        require(
            isinstance(expected_sha256, bytes)
            and len(expected_sha256) == 32,
            "expected SHA-256 API contract mismatch",
        )

    result = dict(actual)
    result.update({
        "format": "GLYPH_RLR2_VERIFY_V1",
        "operation": "verify",
        "source_rlb2_verified": True,
        "deterministic_rebuild_verified": True,
        "expected_rlr2_sha256":
            expected_sha256.hex(),
    })

    return result


def emit(value):
    print(
        canonical_bytes(value).decode("utf-8"),
        end="",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental additive binary-safe RLR2 "
            "rank container for canonical RLB2 files. "
            "Does not redefine RLR1."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build = subparsers.add_parser("build")
    build.add_argument("rlb2", type=Path)
    build.add_argument("rlr2", type=Path)
    build.add_argument(
        "--rank-step",
        type=int,
        required=True,
    )

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("rlr2", type=Path)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("rlb2", type=Path)
    verify_parser.add_argument("rlr2", type=Path)

    arguments = parser.parse_args()

    try:
        if arguments.command == "build":
            result = build_rank(
                arguments.rlb2,
                arguments.rlr2,
                arguments.rank_step,
            )
            emit(result)
            print(SUCCESS_BUILD)
        elif arguments.command == "inspect":
            result = inspect_rank(arguments.rlr2)
            emit(result)
            print(SUCCESS_INSPECT)
        elif arguments.command == "verify":
            result = verify_rank(
                arguments.rlb2,
                arguments.rlr2,
            )
            emit(result)
            print(SUCCESS_VERIFY)
        else:
            raise RankError("unknown command")

        return 0

    except (
        RankError,
        rlb2.Rlb2Error,
        OSError,
        ValueError,
        struct.error,
    ) as error:
        print(f"RLR2 ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
