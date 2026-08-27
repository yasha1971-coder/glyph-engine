#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile


BWT_MAGIC = b"GLYBWT1\x00"
BWT_VERSION = 1
BWT_HEADER = struct.Struct("<8sIQQIIIQQ")

RLB2_MAGIC = b"GLYRLB2\x00"
RLB2_VERSION = 2
RLB2_HEADER = struct.Struct(
    "<8sIIQQQIIIIIIQQQQ32s32s"
)
RLB2_HEADER_BYTES = RLB2_HEADER.size

ALPHABET_SIZE = 257
LOGICAL_SENTINEL = 256
SYMBOL_WIDTH_BITS = 16
HEAD_ENCODING_ADAPTIVE_ESCAPE = 1
LENGTH_ENCODING_CANONICAL_ULEB128 = 1

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
U64_MASK = (1 << 64) - 1


class Rlb2Error(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Rlb2Error(message)


class Fnv1a64:
    def __init__(self):
        self.value = FNV_OFFSET

    def update(self, payload):
        value = self.value
        for byte in payload:
            value ^= byte
            value = (value * FNV_PRIME) & U64_MASK
        self.value = value


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def regular_signature(path):
    metadata = path.lstat()

    require(
        stat.S_ISREG(metadata.st_mode),
        f"not a regular file: {path}",
    )

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.digest()


def read_exact(stream, count, message):
    payload = stream.read(count)
    require(len(payload) == count, message)
    return payload


def parse_bwt_header(stream, file_bytes):
    raw = read_exact(
        stream,
        BWT_HEADER.size,
        "truncated canonical BWT header",
    )

    (
        magic,
        version,
        corpus_bytes,
        row_count,
        alphabet_size,
        logical_sentinel,
        symbol_width_bits,
        payload_bytes,
        payload_checksum,
    ) = BWT_HEADER.unpack(raw)

    require(magic == BWT_MAGIC, "canonical BWT magic mismatch")
    require(
        version == BWT_VERSION,
        "canonical BWT version mismatch",
    )
    require(
        alphabet_size == ALPHABET_SIZE,
        "canonical BWT alphabet mismatch",
    )
    require(
        logical_sentinel == LOGICAL_SENTINEL,
        "canonical BWT sentinel mismatch",
    )
    require(
        symbol_width_bits == SYMBOL_WIDTH_BITS,
        "canonical BWT symbol width mismatch",
    )
    require(
        row_count == corpus_bytes + 1,
        "canonical BWT row count mismatch",
    )
    require(
        payload_bytes == row_count * 2,
        "canonical BWT payload size mismatch",
    )
    require(
        file_bytes == BWT_HEADER.size + payload_bytes,
        "canonical BWT file size mismatch",
    )

    return {
        "raw_header": raw,
        "corpus_bytes": corpus_bytes,
        "row_count": row_count,
        "payload_bytes": payload_bytes,
        "payload_checksum": payload_checksum,
    }


def iter_bwt_symbols(stream, payload_bytes):
    remaining = payload_bytes
    carry = b""

    while remaining:
        block = stream.read(min(8 * 1024 * 1024, remaining))
        require(block, "truncated canonical BWT payload")
        remaining -= len(block)

        block = carry + block
        complete = len(block) & ~1
        payload = block[:complete]
        carry = block[complete:]

        for (symbol,) in struct.iter_unpack("<H", payload):
            yield symbol

    require(not carry, "odd canonical BWT payload size")
    require(
        stream.read(1) == b"",
        "trailing canonical BWT bytes",
    )


def scan_bwt(path):
    signature_before = regular_signature(path)
    file_bytes = signature_before[2]

    counts = [0] * ALPHABET_SIZE
    run_count = 0
    run_length_bytes = 0
    maximum_run_length = 0
    sentinel_occurrences = 0
    sentinel_runs = 0

    checksum = Fnv1a64()

    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        header = parse_bwt_header(stream, file_bytes)

        previous = None
        length = 0
        symbol_count = 0

        for symbol in iter_bwt_symbols(
            stream,
            header["payload_bytes"],
        ):
            require(
                0 <= symbol <= LOGICAL_SENTINEL,
                "canonical BWT symbol out of range",
            )

            checksum.update(struct.pack("<H", symbol))
            symbol_count += 1

            if symbol == LOGICAL_SENTINEL:
                sentinel_occurrences += 1

            if previous is None:
                previous = symbol
                length = 1
            elif symbol == previous:
                length += 1
            else:
                counts[previous] += 1
                run_count += 1
                run_length_bytes += len(encode_uleb128(length))
                maximum_run_length = max(
                    maximum_run_length,
                    length,
                )
                if previous == LOGICAL_SENTINEL:
                    sentinel_runs += 1

                previous = symbol
                length = 1

        require(previous is not None, "empty canonical BWT")

        counts[previous] += 1
        run_count += 1
        run_length_bytes += len(encode_uleb128(length))
        maximum_run_length = max(maximum_run_length, length)

        if previous == LOGICAL_SENTINEL:
            sentinel_runs += 1

    require(
        symbol_count == header["row_count"],
        "canonical BWT decoded row mismatch",
    )
    require(
        checksum.value == header["payload_checksum"],
        "canonical BWT checksum mismatch",
    )
    require(
        sentinel_occurrences == 1,
        "canonical BWT must contain one sentinel",
    )
    require(
        sentinel_runs == 1,
        "canonical BWT sentinel run mismatch",
    )
    require(
        regular_signature(path) == signature_before,
        "canonical BWT changed during scan",
    )

    escape_symbol = min(
        range(256),
        key=lambda symbol: (counts[symbol], symbol),
    )

    payload_bytes = (
        run_count
        + run_length_bytes
        + counts[escape_symbol]
        + sentinel_runs
    )

    return {
        **header,
        "file_bytes": file_bytes,
        "signature": signature_before,
        "run_counts": counts,
        "run_count": run_count,
        "run_length_bytes": run_length_bytes,
        "maximum_run_length": maximum_run_length,
        "sentinel_occurrences": sentinel_occurrences,
        "sentinel_runs": sentinel_runs,
        "escape_symbol": escape_symbol,
        "payload_bytes_rlb2": payload_bytes,
    }


def encode_uleb128(value):
    require(value > 0, "run length must be positive")
    result = bytearray()

    while True:
        byte = value & 0x7f
        value >>= 7

        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


class BufferedPayloadWriter:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()
        self.bytes_written = 0
        self.sha256 = hashlib.sha256()
        self.fnv = Fnv1a64()

    def append(self, payload):
        self.buffer.extend(payload)

        if len(self.buffer) >= 1024 * 1024:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        payload = bytes(self.buffer)
        self.stream.write(payload)
        self.sha256.update(payload)
        self.fnv.update(payload)
        self.bytes_written += len(payload)
        self.buffer.clear()


def append_run(writer, symbol, length, escape_symbol):
    if symbol == LOGICAL_SENTINEL:
        writer.append(bytes((escape_symbol, 1)))
    elif symbol == escape_symbol:
        writer.append(bytes((escape_symbol, 0)))
    else:
        require(0 <= symbol <= 255, "run head out of range")
        writer.append(bytes((symbol,)))

    writer.append(encode_uleb128(length))


def encode_payload(bwt_path, stream, analysis):
    with bwt_path.open(
        "rb",
        buffering=8 * 1024 * 1024,
    ) as source:
        header = parse_bwt_header(
            source,
            analysis["file_bytes"],
        )

        writer = BufferedPayloadWriter(stream)
        previous = None
        length = 0
        emitted_runs = 0

        for symbol in iter_bwt_symbols(
            source,
            header["payload_bytes"],
        ):
            if previous is None:
                previous = symbol
                length = 1
            elif symbol == previous:
                length += 1
            else:
                append_run(
                    writer,
                    previous,
                    length,
                    analysis["escape_symbol"],
                )
                emitted_runs += 1
                previous = symbol
                length = 1

        require(previous is not None, "empty canonical BWT")

        append_run(
            writer,
            previous,
            length,
            analysis["escape_symbol"],
        )
        emitted_runs += 1
        writer.flush()

    require(
        emitted_runs == analysis["run_count"],
        "encoded run count mismatch",
    )
    require(
        writer.bytes_written
        == analysis["payload_bytes_rlb2"],
        "encoded payload size mismatch",
    )

    return {
        "payload_bytes": writer.bytes_written,
        "payload_sha256": writer.sha256.digest(),
        "payload_fnv1a64": writer.fnv.value,
    }


def pack_rlb2_header(analysis, source_sha256, payload):
    return RLB2_HEADER.pack(
        RLB2_MAGIC,
        RLB2_VERSION,
        RLB2_HEADER_BYTES,
        analysis["corpus_bytes"],
        analysis["row_count"],
        analysis["run_count"],
        ALPHABET_SIZE,
        LOGICAL_SENTINEL,
        analysis["escape_symbol"],
        HEAD_ENCODING_ADAPTIVE_ESCAPE,
        LENGTH_ENCODING_CANONICAL_ULEB128,
        0,
        payload["payload_bytes"],
        analysis["payload_bytes"],
        analysis["payload_checksum"],
        payload["payload_fnv1a64"],
        source_sha256,
        payload["payload_sha256"],
    )


def publish_temporary(temporary, output):
    require(
        not output.exists() and not output.is_symlink(),
        f"output already exists: {output}",
    )
    require(
        output.parent.is_dir(),
        "output parent is not a directory",
    )

    try:
        os.link(temporary, output)
    except FileExistsError as error:
        raise Rlb2Error(
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


def encode_file(bwt_path, output):
    require(
        not output.exists() and not output.is_symlink(),
        f"output already exists: {output}",
    )

    analysis = scan_bwt(bwt_path)
    source_sha256 = sha256_file(bwt_path)

    require(
        regular_signature(bwt_path) == analysis["signature"],
        "canonical BWT changed before encode",
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)

    try:
        with os.fdopen(
            descriptor,
            "w+b",
            buffering=8 * 1024 * 1024,
        ) as stream:
            stream.write(b"\x00" * RLB2_HEADER_BYTES)
            payload = encode_payload(
                bwt_path,
                stream,
                analysis,
            )

            require(
                regular_signature(bwt_path)
                == analysis["signature"],
                "canonical BWT changed during encode",
            )

            header = pack_rlb2_header(
                analysis,
                source_sha256,
                payload,
            )
            stream.seek(0)
            stream.write(header)
            stream.flush()
            os.fsync(stream.fileno())

        publish_temporary(temporary, output)

    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "format": "GLYPH_RLBWT_BINARY_SAFE_CONTAINER_V2",
        "operation": "encode",
        "ok": True,
        "header_bytes": RLB2_HEADER_BYTES,
        "corpus_bytes": analysis["corpus_bytes"],
        "row_count": analysis["row_count"],
        "run_count": analysis["run_count"],
        "escape_symbol": analysis["escape_symbol"],
        "payload_bytes": payload["payload_bytes"],
        "file_bytes": output.stat().st_size,
        "ratio_vs_corpus":
            output.stat().st_size
            / analysis["corpus_bytes"],
        "source_bwt_sha256": source_sha256.hex(),
        "payload_sha256":
            payload["payload_sha256"].hex(),
    }


def parse_rlb2_header(stream, file_bytes):
    raw = read_exact(
        stream,
        RLB2_HEADER_BYTES,
        "truncated RLB2 header",
    )
    values = RLB2_HEADER.unpack(raw)

    (
        magic,
        version,
        header_bytes,
        corpus_bytes,
        row_count,
        run_count,
        alphabet_size,
        sentinel,
        escape_symbol,
        head_encoding,
        length_encoding,
        reserved,
        payload_bytes,
        decoded_payload_bytes,
        source_payload_fnv,
        payload_fnv,
        source_bwt_sha256,
        payload_sha256,
    ) = values

    require(magic == RLB2_MAGIC, "RLB2 magic mismatch")
    require(version == RLB2_VERSION, "RLB2 version mismatch")
    require(
        header_bytes == RLB2_HEADER_BYTES,
        "RLB2 header size mismatch",
    )
    require(row_count == corpus_bytes + 1, "RLB2 row mismatch")
    require(run_count > 0, "RLB2 run count invalid")
    require(
        alphabet_size == ALPHABET_SIZE,
        "RLB2 alphabet mismatch",
    )
    require(sentinel == LOGICAL_SENTINEL, "RLB2 sentinel mismatch")
    require(0 <= escape_symbol <= 255, "RLB2 escape invalid")
    require(
        head_encoding == HEAD_ENCODING_ADAPTIVE_ESCAPE,
        "RLB2 head encoding mismatch",
    )
    require(
        length_encoding
        == LENGTH_ENCODING_CANONICAL_ULEB128,
        "RLB2 length encoding mismatch",
    )
    require(reserved == 0, "RLB2 reserved field nonzero")
    require(
        decoded_payload_bytes == row_count * 2,
        "RLB2 decoded size mismatch",
    )
    require(
        file_bytes == RLB2_HEADER_BYTES + payload_bytes,
        "RLB2 file size mismatch",
    )

    return {
        "raw_header": raw,
        "corpus_bytes": corpus_bytes,
        "row_count": row_count,
        "run_count": run_count,
        "escape_symbol": escape_symbol,
        "payload_bytes": payload_bytes,
        "decoded_payload_bytes": decoded_payload_bytes,
        "source_payload_fnv": source_payload_fnv,
        "payload_fnv": payload_fnv,
        "source_bwt_sha256": source_bwt_sha256,
        "payload_sha256": payload_sha256,
    }


class BufferedPayloadReader:
    def __init__(self, stream, payload_bytes):
        self.stream = stream
        self.remaining = payload_bytes
        self.buffer = b""
        self.position = 0
        self.sha256 = hashlib.sha256()
        self.fnv = Fnv1a64()

    def read_byte(self):
        if self.position == len(self.buffer):
            require(self.remaining > 0, "truncated RLB2 payload")
            block = self.stream.read(
                min(1024 * 1024, self.remaining)
            )
            require(block, "truncated RLB2 payload")
            self.remaining -= len(block)
            self.buffer = block
            self.position = 0
            self.sha256.update(block)
            self.fnv.update(block)

        value = self.buffer[self.position]
        self.position += 1
        return value

    def finish(self):
        require(
            self.remaining == 0
            and self.position == len(self.buffer),
            "trailing RLB2 payload bytes",
        )
        require(
            self.stream.read(1) == b"",
            "trailing RLB2 file bytes",
        )


def decode_uleb128(reader):
    encoded = bytearray()
    value = 0
    shift = 0

    for _ in range(10):
        byte = reader.read_byte()
        encoded.append(byte)

        payload = byte & 0x7f
        require(
            shift < 64 or payload == 0,
            "RLB2 run length overflow",
        )
        value |= payload << shift

        if not byte & 0x80:
            require(value > 0, "RLB2 zero run length")
            require(
                bytes(encoded) == encode_uleb128(value),
                "RLB2 noncanonical ULEB128",
            )
            require(value <= U64_MASK, "RLB2 run length overflow")
            return value

        shift += 7

    raise Rlb2Error("RLB2 run length too long")


def write_repeated_symbol(
    stream,
    symbol,
    count,
    source_sha256,
    source_fnv,
):
    unit = struct.pack("<H", symbol)

    while count:
        chunk_count = min(count, 32768)
        payload = unit * chunk_count
        stream.write(payload)
        source_sha256.update(payload)
        source_fnv.update(payload)
        count -= chunk_count


def decode_file(rlb2_path, output):
    require(
        not output.exists() and not output.is_symlink(),
        f"output already exists: {output}",
    )

    signature_before = regular_signature(rlb2_path)
    file_bytes = signature_before[2]

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    temporary = Path(temporary_name)

    try:
        with rlb2_path.open(
            "rb",
            buffering=8 * 1024 * 1024,
        ) as source:
            header = parse_rlb2_header(source, file_bytes)
            reader = BufferedPayloadReader(
                source,
                header["payload_bytes"],
            )

            canonical_header = BWT_HEADER.pack(
                BWT_MAGIC,
                BWT_VERSION,
                header["corpus_bytes"],
                header["row_count"],
                ALPHABET_SIZE,
                LOGICAL_SENTINEL,
                SYMBOL_WIDTH_BITS,
                header["decoded_payload_bytes"],
                header["source_payload_fnv"],
            )

            decoded_sha256 = hashlib.sha256()
            decoded_sha256.update(canonical_header)
            decoded_fnv = Fnv1a64()

            decoded_rows = 0
            decoded_runs = 0
            sentinel_occurrences = 0
            previous_symbol = None

            with os.fdopen(
                descriptor,
                "wb",
                buffering=8 * 1024 * 1024,
            ) as destination:
                descriptor = None
                destination.write(canonical_header)

                while decoded_runs < header["run_count"]:
                    head = reader.read_byte()

                    if head == header["escape_symbol"]:
                        tag = reader.read_byte()

                        if tag == 0:
                            symbol = header["escape_symbol"]
                        elif tag == 1:
                            symbol = LOGICAL_SENTINEL
                        else:
                            raise Rlb2Error(
                                "RLB2 escape tag invalid"
                            )
                    else:
                        symbol = head

                    require(
                        symbol != previous_symbol,
                        "RLB2 adjacent equal runs",
                    )

                    length = decode_uleb128(reader)
                    require(
                        length
                        <= header["row_count"] - decoded_rows,
                        "RLB2 rows exceed declared count",
                    )

                    write_repeated_symbol(
                        destination,
                        symbol,
                        length,
                        decoded_sha256,
                        decoded_fnv,
                    )

                    if symbol == LOGICAL_SENTINEL:
                        sentinel_occurrences += length

                    decoded_rows += length
                    decoded_runs += 1
                    previous_symbol = symbol

                reader.finish()

                require(
                    reader.sha256.digest()
                    == header["payload_sha256"],
                    "RLB2 payload SHA-256 mismatch",
                )
                require(
                    reader.fnv.value == header["payload_fnv"],
                    "RLB2 payload FNV mismatch",
                )
                require(
                    decoded_rows == header["row_count"],
                    "RLB2 decoded row count mismatch",
                )
                require(
                    sentinel_occurrences == 1,
                    "RLB2 decoded sentinel count mismatch",
                )
                require(
                    decoded_fnv.value
                    == header["source_payload_fnv"],
                    "RLB2 decoded BWT checksum mismatch",
                )
                require(
                    decoded_sha256.digest()
                    == header["source_bwt_sha256"],
                    "RLB2 decoded BWT SHA-256 mismatch",
                )

                destination.flush()
                os.fsync(destination.fileno())

        require(
            regular_signature(rlb2_path) == signature_before,
            "RLB2 changed during decode",
        )
        publish_temporary(temporary, output)

    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()

    return {
        "format": "GLYPH_RLBWT_BINARY_SAFE_CONTAINER_V2",
        "operation": "decode",
        "ok": True,
        "row_count": header["row_count"],
        "run_count": header["run_count"],
        "decoded_bwt_bytes": output.stat().st_size,
        "decoded_bwt_sha256":
            header["source_bwt_sha256"].hex(),
    }


def inspect_file(path):
    signature_before = regular_signature(path)

    with path.open("rb", buffering=8 * 1024 * 1024) as stream:
        header = parse_rlb2_header(
            stream,
            signature_before[2],
        )
        digest = hashlib.sha256()
        fnv = Fnv1a64()
        remaining = header["payload_bytes"]

        while remaining:
            block = stream.read(
                min(8 * 1024 * 1024, remaining)
            )
            require(block, "truncated RLB2 payload")
            remaining -= len(block)
            digest.update(block)
            fnv.update(block)

        require(stream.read(1) == b"", "trailing RLB2 bytes")

    require(
        digest.digest() == header["payload_sha256"],
        "RLB2 payload SHA-256 mismatch",
    )
    require(
        fnv.value == header["payload_fnv"],
        "RLB2 payload FNV mismatch",
    )
    require(
        regular_signature(path) == signature_before,
        "RLB2 changed during inspect",
    )

    return {
        "format": "GLYPH_RLBWT_BINARY_SAFE_CONTAINER_V2",
        "operation": "inspect",
        "ok": True,
        "header_bytes": RLB2_HEADER_BYTES,
        "corpus_bytes": header["corpus_bytes"],
        "row_count": header["row_count"],
        "run_count": header["run_count"],
        "escape_symbol": header["escape_symbol"],
        "payload_bytes": header["payload_bytes"],
        "file_bytes": signature_before[2],
        "ratio_vs_corpus":
            signature_before[2] / header["corpus_bytes"],
        "source_bwt_sha256":
            header["source_bwt_sha256"].hex(),
        "payload_sha256":
            header["payload_sha256"].hex(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental additive binary-safe RLBWT "
            "container. Does not redefine RLB1/RLR1."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("bwt", type=Path)
    encode_parser.add_argument("rlb2", type=Path)

    decode_parser = subparsers.add_parser("decode")
    decode_parser.add_argument("rlb2", type=Path)
    decode_parser.add_argument("bwt", type=Path)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("rlb2", type=Path)

    arguments = parser.parse_args()

    if arguments.command == "encode":
        result = encode_file(
            arguments.bwt.resolve(),
            arguments.rlb2.resolve(),
        )
    elif arguments.command == "decode":
        result = decode_file(
            arguments.rlb2.resolve(),
            arguments.bwt.resolve(),
        )
    else:
        result = inspect_file(arguments.rlb2.resolve())

    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Rlb2Error as error:
        print(f"RLB2 ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
