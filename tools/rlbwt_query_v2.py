#!/usr/bin/env python3

import argparse
import hashlib
import json
import mmap
from pathlib import Path
import stat
import struct
import sys
import time

import rlbwt_container_v2 as rlb2


RLR2_MAGIC = b"GLYRLR2\x00"
RLR2_HEADER_BYTES = 160
RLR2_VERSION = 2
ALPHABET_SIZE = 257
LOGICAL_SENTINEL = 256

LOCATE_MAGIC = b"LOC1"
LOCATE_HEADER_BYTES = 24
LOCATE_RECORD_BYTES = 16

SUCCESS = "GLYPH RLBWT BINARY SAFE QUERY V2 OK"


class QueryError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise QueryError(message)


def encode_uleb128(value):
    require(value > 0, "run length must be positive")
    payload = bytearray()

    while True:
        byte = value & 0x7f
        value >>= 7

        if value:
            byte |= 0x80

        payload.append(byte)

        if not value:
            return bytes(payload)


class RankV2:
    def __init__(self, rlb2_path, rlr2_path):
        self.rlb2_path = rlb2_path
        self.rlr2_path = rlr2_path

        self.rlb2_stream = rlb2_path.open("rb")
        self.rlr2_stream = rlr2_path.open("rb")

        self.rlb2_file_bytes = rlb2_path.stat().st_size
        self.rlr2_file_bytes = rlr2_path.stat().st_size

        self.rlb2_header = rlb2.parse_rlb2_header(
            self.rlb2_stream,
            self.rlb2_file_bytes,
        )
        self.rlb2_header_bytes = self.rlb2_stream.tell()

        require(
            self.rlb2_header_bytes == 160,
            "unexpected RLB2 header size",
        )

        self.rlr2_map = mmap.mmap(
            self.rlr2_stream.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        )

        raw = self.rlr2_map[:RLR2_HEADER_BYTES]

        require(raw[:8] == RLR2_MAGIC, "bad RLR2 magic")

        self.version = struct.unpack_from("<I", raw, 8)[0]
        header_bytes = struct.unpack_from("<I", raw, 12)[0]
        self.raw_length = struct.unpack_from("<Q", raw, 16)[0]
        self.run_count = struct.unpack_from("<Q", raw, 24)[0]
        self.rank_step = struct.unpack_from("<I", raw, 32)[0]
        self.alphabet_size = struct.unpack_from("<I", raw, 36)[0]
        self.logical_sentinel = struct.unpack_from("<I", raw, 40)[0]
        self.counter_width_bits = struct.unpack_from("<I", raw, 44)[0]
        self.checkpoint_count = struct.unpack_from("<Q", raw, 48)[0]
        self.record_bytes = struct.unpack_from("<I", raw, 56)[0]
        reserved = struct.unpack_from("<I", raw, 60)[0]
        self.source_rlb2_bytes = struct.unpack_from("<Q", raw, 64)[0]
        self.source_rlb2_sha256 = raw[72:104]
        self.payload_bytes = struct.unpack_from("<Q", raw, 104)[0]

        require(self.version == RLR2_VERSION, "bad RLR2 version")
        require(
            header_bytes == RLR2_HEADER_BYTES,
            "bad RLR2 header size",
        )
        require(
            self.alphabet_size == ALPHABET_SIZE,
            "bad RLR2 alphabet size",
        )
        require(
            self.logical_sentinel == LOGICAL_SENTINEL,
            "bad logical sentinel",
        )
        require(
            self.counter_width_bits == 32,
            "query prototype requires u32 counters",
        )
        require(
            self.record_bytes == 24 + ALPHABET_SIZE * 4,
            "bad RLR2 record size",
        )
        require(reserved == 0, "RLR2 reserved field nonzero")
        require(
            self.payload_bytes
            == self.checkpoint_count * self.record_bytes,
            "RLR2 payload geometry mismatch",
        )
        require(
            self.rlr2_file_bytes
            == RLR2_HEADER_BYTES + self.payload_bytes,
            "RLR2 file geometry mismatch",
        )
        require(
            self.source_rlb2_bytes == self.rlb2_file_bytes,
            "RLR2 source byte binding mismatch",
        )

        actual_rlb2_sha256 = rlb2.sha256_file(rlb2_path)

        require(
            actual_rlb2_sha256 == self.source_rlb2_sha256,
            "RLR2 source SHA-256 binding mismatch",
        )
        require(
            self.rlb2_header["row_count"] == self.raw_length,
            "RLB2/RLR2 row count mismatch",
        )
        require(
            self.rlb2_header["run_count"] == self.run_count,
            "RLB2/RLR2 run count mismatch",
        )

        self.escape_symbol = self.rlb2_header["escape_symbol"]
        self.rlb2_payload_bytes = self.rlb2_header["payload_bytes"]

        self.rank_calls = 0
        self.lf_calls = 0
        self.decoded_runs = 0
        self.scanned_symbols = 0

        final = self._record(self.checkpoint_count - 1)
        final_position, _, _, final_counts = final

        require(
            final_position == self.raw_length,
            "RLR2 final checkpoint position mismatch",
        )
        require(
            sum(final_counts) == self.raw_length,
            "RLR2 final histogram mismatch",
        )
        require(
            final_counts[LOGICAL_SENTINEL] == 1,
            "sentinel cardinality mismatch",
        )

        self.frequencies = final_counts
        self.C = [0] * ALPHABET_SIZE
        self.C[LOGICAL_SENTINEL] = 0

        running = 1

        for symbol in range(256):
            self.C[symbol] = running
            running += self.frequencies[symbol]

        require(
            running == self.raw_length,
            "binary-safe C-array total mismatch",
        )

    def close(self):
        if getattr(self, "rlr2_map", None) is not None:
            self.rlr2_map.close()
            self.rlr2_map = None

        if getattr(self, "rlr2_stream", None) is not None:
            self.rlr2_stream.close()
            self.rlr2_stream = None

        if getattr(self, "rlb2_stream", None) is not None:
            self.rlb2_stream.close()
            self.rlb2_stream = None

    def _record_offset(self, index):
        require(
            0 <= index < self.checkpoint_count,
            "checkpoint index outside range",
        )

        return RLR2_HEADER_BYTES + index * self.record_bytes

    def _raw_position(self, index):
        return struct.unpack_from(
            "<Q",
            self.rlr2_map,
            self._record_offset(index),
        )[0]

    def _record(self, index):
        offset = self._record_offset(index)

        raw_position, stream_offset, run_offset = (
            struct.unpack_from("<QQQ", self.rlr2_map, offset)
        )
        counts = list(
            struct.unpack_from(
                "<257I",
                self.rlr2_map,
                offset + 24,
            )
        )

        return raw_position, stream_offset, run_offset, counts

    def _checkpoint_for(self, position):
        require(
            0 <= position <= self.raw_length,
            "rank position outside range",
        )

        index = min(
            position // self.rank_step,
            self.checkpoint_count - 1,
        )

        while (
            index + 1 < self.checkpoint_count
            and self._raw_position(index + 1) <= position
        ):
            index += 1

        while self._raw_position(index) > position:
            require(index > 0, "no checkpoint before position")
            index -= 1

        return self._record(index)

    def _read_byte(self):
        payload_position = (
            self.rlb2_stream.tell() - self.rlb2_header_bytes
        )

        require(
            0 <= payload_position < self.rlb2_payload_bytes,
            "RLB2 payload read outside range",
        )

        raw = self.rlb2_stream.read(1)
        require(len(raw) == 1, "RLB2 payload EOF")
        return raw[0]

    def _read_uleb128(self):
        raw = bytearray()
        value = 0
        shift = 0

        for _ in range(10):
            byte = self._read_byte()
            raw.append(byte)
            value |= (byte & 0x7f) << shift

            if not (byte & 0x80):
                require(value > 0, "zero run length")
                require(
                    bytes(raw) == encode_uleb128(value),
                    "noncanonical ULEB128",
                )
                return value

            shift += 7

        raise QueryError("ULEB128 too long")

    def _read_run(self, stream_offset):
        require(
            0 <= stream_offset < self.rlb2_payload_bytes,
            "run stream offset outside payload",
        )

        self.rlb2_stream.seek(
            self.rlb2_header_bytes + stream_offset
        )

        head = self._read_byte()

        if head != self.escape_symbol:
            symbol = head
        else:
            tag = self._read_byte()

            if tag == 0:
                symbol = self.escape_symbol
            elif tag == 1:
                symbol = LOGICAL_SENTINEL
            else:
                raise QueryError("invalid escaped run tag")

        length = self._read_uleb128()
        next_offset = (
            self.rlb2_stream.tell() - self.rlb2_header_bytes
        )

        require(
            next_offset <= self.rlb2_payload_bytes,
            "run extends beyond payload",
        )

        self.decoded_runs += 1
        return symbol, length, next_offset

    def rank(self, symbol, position):
        require(
            0 <= symbol < ALPHABET_SIZE,
            "rank symbol outside alphabet",
        )

        self.rank_calls += 1

        (
            checkpoint_position,
            stream_offset,
            run_offset,
            counts,
        ) = self._checkpoint_for(position)

        result = counts[symbol]
        current = checkpoint_position
        first = True

        while current < position:
            run_symbol, run_length, next_offset = (
                self._read_run(stream_offset)
            )

            skip = run_offset if first else 0
            first = False

            require(skip <= run_length, "run offset exceeds run")
            available = run_length - skip
            stream_offset = next_offset

            if available == 0:
                continue

            take = min(available, position - current)

            if run_symbol == symbol:
                result += take

            current += take
            self.scanned_symbols += take

        require(current == position, "rank scan stopped early")
        return result

    def symbol_and_rank(self, position):
        require(
            0 <= position < self.raw_length,
            "symbol position outside BWT",
        )

        (
            checkpoint_position,
            stream_offset,
            run_offset,
            counts,
        ) = self._checkpoint_for(position)

        local_counts = [0] * ALPHABET_SIZE
        current = checkpoint_position
        first = True

        while current <= position:
            run_symbol, run_length, next_offset = (
                self._read_run(stream_offset)
            )

            skip = run_offset if first else 0
            first = False

            require(skip <= run_length, "run offset exceeds run")
            available = run_length - skip
            stream_offset = next_offset

            if available == 0:
                continue

            if position < current + available:
                self.scanned_symbols += position - current
                return (
                    run_symbol,
                    counts[run_symbol]
                    + local_counts[run_symbol]
                    + (position - current),
                )

            local_counts[run_symbol] += available
            current += available
            self.scanned_symbols += available

        raise QueryError("symbol scan stopped early")

    def lf(self, row):
        symbol, rank_before = self.symbol_and_rank(row)
        self.lf_calls += 1
        return self.C[symbol] + rank_before

    def backward_search(self, pattern):
        require(pattern, "empty patterns are not supported")

        left = 0
        right = self.raw_length

        for symbol in reversed(pattern):
            left = self.C[symbol] + self.rank(symbol, left)
            right = self.C[symbol] + self.rank(symbol, right)

            if left >= right:
                return left, left

        return left, right


class LocateCore:
    def __init__(self, path):
        self.path = path
        self.stream = path.open("rb")
        self.file_bytes = path.stat().st_size
        self.map = mmap.mmap(
            self.stream.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        )

        require(
            self.file_bytes >= LOCATE_HEADER_BYTES,
            "LOC1 too small",
        )
        require(
            self.map[:4] == LOCATE_MAGIC,
            "bad LOC1 magic",
        )

        self.sa_size = struct.unpack_from("<Q", self.map, 4)[0]
        self.sample_step = struct.unpack_from("<I", self.map, 12)[0]
        self.sampled_count = struct.unpack_from("<Q", self.map, 16)[0]

        require(self.sample_step > 0, "zero locate sample step")
        require(
            self.file_bytes
            == LOCATE_HEADER_BYTES
            + self.sampled_count * LOCATE_RECORD_BYTES,
            "LOC1 geometry mismatch",
        )

    def close(self):
        if getattr(self, "map", None) is not None:
            self.map.close()
            self.map = None

        if getattr(self, "stream", None) is not None:
            self.stream.close()
            self.stream = None

    def sampled_sa(self, row):
        if row % self.sample_step:
            return None

        index = row // self.sample_step

        if index >= self.sampled_count:
            return None

        offset = LOCATE_HEADER_BYTES + index * LOCATE_RECORD_BYTES
        stored_row, suffix_offset = struct.unpack_from(
            "<QQ",
            self.map,
            offset,
        )

        require(stored_row == row, "LOC1 sampled row mismatch")
        require(
            suffix_offset < self.sa_size,
            "LOC1 suffix offset outside SA",
        )

        return suffix_offset


class QueryRuntime:
    def __init__(self, rlb2_path, rlr2_path, locate_path):
        self.rank = RankV2(rlb2_path, rlr2_path)
        self.locate = LocateCore(locate_path)

        require(
            self.locate.sa_size == self.rank.raw_length,
            "LOC1/RLR2 row count mismatch",
        )

        self.corpus_bytes = self.rank.raw_length - 1

    def close(self):
        self.locate.close()
        self.rank.close()

    def locate_row(self, row):
        require(
            0 <= row < self.rank.raw_length,
            "FM row outside range",
        )

        current = row
        steps = 0

        while True:
            sampled = self.locate.sampled_sa(current)

            if sampled is not None:
                suffix_offset = (
                    sampled + steps
                ) % self.locate.sa_size

                return suffix_offset, steps

            current = self.rank.lf(current)
            steps += 1

            require(
                steps <= self.locate.sa_size,
                "locate LF walk exceeded SA size",
            )

    def query(self, pattern, max_offsets):
        require(max_offsets >= -1, "max offsets must be >= -1")

        started = time.perf_counter_ns()
        left, right = self.rank.backward_search(pattern)
        count = right - left

        if max_offsets < 0:
            locate_count = count
        else:
            locate_count = min(count, max_offsets)

        offsets = []
        total_lf_steps = 0
        maximum_lf_steps = 0

        for row in range(left, left + locate_count):
            suffix_offset, steps = self.locate_row(row)

            require(
                suffix_offset < self.corpus_bytes,
                "ordinary pattern resolved to terminal suffix",
            )

            offsets.append(suffix_offset)
            total_lf_steps += steps
            maximum_lf_steps = max(maximum_lf_steps, steps)

        offsets.sort()
        elapsed_ns = time.perf_counter_ns() - started

        return {
            "count": count,
            "decoded_runs": self.rank.decoded_runs,
            "fm_interval": [left, right],
            "format": "GLYPH_RLBWT_BINARY_SAFE_QUERY_V2",
            "lf_calls": self.rank.lf_calls,
            "locate_offsets": offsets,
            "locate_offsets_complete": locate_count == count,
            "located_count": locate_count,
            "maximum_lf_steps": maximum_lf_steps,
            "ok": True,
            "pattern_bytes": len(pattern),
            "pattern_hex": pattern.hex(),
            "query_elapsed_ns": elapsed_ns,
            "rank_calls": self.rank.rank_calls,
            "rank_step": self.rank.rank_step,
            "sample_step": self.locate.sample_step,
            "scanned_symbols": self.rank.scanned_symbols,
            "total_lf_steps": total_lf_steps,
        }



def reject_duplicate_keys(pairs):
    value = {}

    for key, child in pairs:
        if key in value:
            raise QueryError(f"duplicate JSON key: {key}")
        value[key] = child

    return value


def is_lower_hex(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def inspect_and_hash_regular(path):
    require(
        path.is_file() and not path.is_symlink(),
        f"runtime artifact is not a regular file: {path.name}",
    )

    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode),
        f"runtime artifact is not regular: {path.name}",
    )

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    after = path.stat()

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )

    require(
        identity_before == identity_after,
        f"runtime artifact changed while hashing: {path.name}",
    )

    return {
        "bytes": before.st_size,
        "sha256": digest.hexdigest(),
    }


def load_runtime_manifest(path):
    require(
        path.is_file() and not path.is_symlink(),
        "runtime manifest is not a regular file",
    )

    raw = path.read_bytes()

    try:
        manifest = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryError(
            f"invalid runtime manifest JSON: {error}"
        ) from error

    require(
        raw == canonical_json_bytes(manifest),
        "runtime manifest JSON is not canonical",
    )
    require(
        set(manifest) == {
            "corpus_identity",
            "files",
            "format",
            "rank_step",
            "row_count",
            "runtime_data_bytes",
            "sample_step",
            "version",
        },
        "runtime manifest top-level fields mismatch",
    )
    require(
        manifest["format"]
        == "GLYPH_RLBWT_BINARY_SAFE_RUNTIME_V2",
        "runtime manifest format mismatch",
    )
    require(
        manifest["version"] == 1,
        "runtime manifest version mismatch",
    )

    corpus = manifest["corpus_identity"]

    require(
        set(corpus) == {
            "bytes",
            "md5",
            "reference_id",
            "sha256",
        },
        "runtime corpus identity fields mismatch",
    )
    require(
        isinstance(corpus["reference_id"], str)
        and corpus["reference_id"]
        and all(
            character in
            "abcdefghijklmnopqrstuvwxyz0123456789-._"
            for character in corpus["reference_id"]
        ),
        "runtime corpus reference ID invalid",
    )
    require(
        isinstance(corpus["bytes"], int)
        and corpus["bytes"] >= 0,
        "runtime corpus byte length invalid",
    )
    require(
        is_lower_hex(corpus["md5"], 32),
        "runtime corpus MD5 invalid",
    )
    require(
        is_lower_hex(corpus["sha256"], 64),
        "runtime corpus SHA-256 invalid",
    )
    require(
        manifest["row_count"] == corpus["bytes"] + 1,
        "runtime row count mismatch",
    )
    require(
        isinstance(manifest["rank_step"], int)
        and manifest["rank_step"] > 0,
        "runtime rank step invalid",
    )
    require(
        isinstance(manifest["sample_step"], int)
        and manifest["sample_step"] > 0,
        "runtime sample step invalid",
    )

    files = manifest["files"]

    require(
        isinstance(files, dict)
        and set(files) == {"locate", "rlb2", "rlr2"},
        "runtime file roles mismatch",
    )

    expected_formats = {
        "locate": "LOC1",
        "rlb2": "GLYPH_RLB2_EXPERIMENTAL_V2",
        "rlr2": "GLYPH_RLR2_V2",
    }

    paths = {}
    total = 0

    for role in sorted(files):
        record = files[role]

        require(
            set(record) == {
                "bytes",
                "format",
                "name",
                "sha256",
            },
            f"runtime file fields mismatch: {role}",
        )
        require(
            isinstance(record["name"], str)
            and record["name"]
            and Path(record["name"]).name == record["name"]
            and record["name"] not in {".", ".."},
            f"unsafe runtime filename: {role}",
        )
        require(
            record["format"] == expected_formats[role],
            f"runtime file format mismatch: {role}",
        )
        require(
            isinstance(record["bytes"], int)
            and record["bytes"] > 0,
            f"runtime file byte length invalid: {role}",
        )
        require(
            is_lower_hex(record["sha256"], 64),
            f"runtime file SHA-256 invalid: {role}",
        )

        artifact_path = path.parent / record["name"]
        actual = inspect_and_hash_regular(artifact_path)

        require(
            actual["bytes"] == record["bytes"],
            f"runtime file byte length mismatch: {role}",
        )
        require(
            actual["sha256"] == record["sha256"],
            f"runtime file SHA-256 mismatch: {role}",
        )

        paths[role] = artifact_path.resolve()
        total += actual["bytes"]

    require(
        manifest["runtime_data_bytes"] == total,
        "runtime data byte total mismatch",
    )
    # Size superiority is an evidence claim, not a
    # validity condition. Tiny conformance fixtures may be larger
    # than their source while remaining structurally correct.

    return {
        "manifest": manifest,
        "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "paths": paths,
    }



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental exact count/locate query over "
            "binary-safe RLB2 + RLR2 + LOC1."
        )
    )
    parser.add_argument("--runtime-manifest", type=Path)
    parser.add_argument("--rlb2", type=Path)
    parser.add_argument("--rank-index", type=Path)
    parser.add_argument("--locate-core", type=Path)
    parser.add_argument("--pattern-hex", required=True)
    parser.add_argument("--max-offsets", type=int, default=100)
    arguments = parser.parse_args()

    try:
        pattern = bytes.fromhex(arguments.pattern_hex)
        require(pattern, "pattern must not be empty")

        explicit_paths = (
            arguments.rlb2,
            arguments.rank_index,
            arguments.locate_core,
        )

        if arguments.runtime_manifest is not None:
            require(
                all(item is None for item in explicit_paths),
                "runtime manifest cannot be mixed with "
                "explicit artifact paths",
            )

            binding = load_runtime_manifest(
                arguments.runtime_manifest.resolve()
            )

            rlb2_path = binding["paths"]["rlb2"]
            rank_path = binding["paths"]["rlr2"]
            locate_path = binding["paths"]["locate"]
            binding_mode = "canonical_manifest"
        else:
            require(
                all(item is not None for item in explicit_paths),
                "either --runtime-manifest or all three "
                "explicit artifact paths are required",
            )

            binding = None
            rlb2_path = arguments.rlb2.resolve()
            rank_path = arguments.rank_index.resolve()
            locate_path = arguments.locate_core.resolve()
            binding_mode = "explicit_research_paths"

        runtime = QueryRuntime(
            rlb2_path,
            rank_path,
            locate_path,
        )

        try:
            result = runtime.query(
                pattern,
                arguments.max_offsets,
            )

            if binding is not None:
                manifest = binding["manifest"]

                require(
                    runtime.rank.raw_length
                    == manifest["row_count"],
                    "manifest row count mismatch",
                )
                require(
                    runtime.rank.rank_step
                    == manifest["rank_step"],
                    "manifest rank step mismatch",
                )
                require(
                    runtime.locate.sample_step
                    == manifest["sample_step"],
                    "manifest sample step mismatch",
                )
                require(
                    runtime.corpus_bytes
                    == manifest["corpus_identity"]["bytes"],
                    "manifest corpus size mismatch",
                )
        finally:
            runtime.close()

        result["binding_mode"] = binding_mode

        if binding is not None:
            result["runtime_manifest_bytes"] = (
                binding["manifest_bytes"]
            )
            result["runtime_manifest_sha256"] = (
                binding["manifest_sha256"]
            )
            result["runtime_data_bytes"] = (
                binding["manifest"]["runtime_data_bytes"]
            )

        print(
            json.dumps(
                result,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        print(SUCCESS)
        return 0
    except (OSError, ValueError, QueryError) as error:
        print(f"RLBWT QUERY V2 ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
