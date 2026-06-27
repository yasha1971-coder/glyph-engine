#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path

MAGIC = b"RLB1"
VERSION = 1


def write_u32(f, x: int):
    f.write(struct.pack("<I", x))


def write_u64(f, x: int):
    f.write(struct.pack("<Q", x))


def read_u32(f) -> int:
    b = f.read(4)
    if len(b) != 4:
        raise EOFError("failed to read u32")
    return struct.unpack("<I", b)[0]


def read_u64(f) -> int:
    b = f.read(8)
    if len(b) != 8:
        raise EOFError("failed to read u64")
    return struct.unpack("<Q", b)[0]


def write_varint(f, x: int):
    if x <= 0:
        raise ValueError("run length must be positive")
    while x >= 0x80:
        f.write(bytes([(x & 0x7F) | 0x80]))
        x >>= 7
    f.write(bytes([x]))


def read_varint(f) -> int:
    shift = 0
    x = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("unexpected EOF in varint")
        v = b[0]
        x |= (v & 0x7F) << shift
        if (v & 0x80) == 0:
            return x
        shift += 7
        if shift > 63:
            raise ValueError("varint too large")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def encode_bwt(bwt_path: Path, out_path: Path):
    data = bwt_path.read_bytes()
    if not data:
        raise ValueError("empty BWT")

    raw_sha = hashlib.sha256(data).digest()

    runs = []
    last = data[0]
    run_len = 1

    for b in data[1:]:
        if b == last:
            run_len += 1
        else:
            runs.append((last, run_len))
            last = b
            run_len = 1
    runs.append((last, run_len))

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        f.write(MAGIC)
        write_u32(f, VERSION)
        write_u64(f, len(data))
        write_u64(f, len(runs))
        f.write(raw_sha)

        for sym, length in runs:
            f.write(bytes([sym]))
            write_varint(f, length)

    encoded_bytes = out_path.stat().st_size

    result = {
        "ok": True,
        "mode": "encode",
        "bwt_path": str(bwt_path),
        "out_path": str(out_path),
        "raw_bwt_bytes": len(data),
        "encoded_bytes": encoded_bytes,
        "runs": len(runs),
        "avg_run_len": len(data) / len(runs),
        "ratio_vs_bwt": encoded_bytes / len(data),
        "raw_sha256": raw_sha.hex(),
    }
    print(json.dumps(result, indent=2))
    return result


def decode_rlbwt(rlbwt_path: Path, out_path: Path):
    with rlbwt_path.open("rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")

        version = read_u32(f)
        if version != VERSION:
            raise ValueError(f"unsupported version: {version}")

        raw_len = read_u64(f)
        run_count = read_u64(f)
        expected_sha = f.read(32)
        if len(expected_sha) != 32:
            raise EOFError("failed to read sha256")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with out_path.open("wb") as out:
            for _ in range(run_count):
                sym_b = f.read(1)
                if len(sym_b) != 1:
                    raise EOFError("unexpected EOF reading symbol")
                run_len = read_varint(f)
                out.write(sym_b * run_len)
                written += run_len

            trailing = f.read(1)
            if trailing:
                raise ValueError("trailing bytes after RLBWT stream")

        if written != raw_len:
            raise ValueError(f"decoded length mismatch: {written} != {raw_len}")

    got_sha = bytes.fromhex(sha256_file(out_path))
    if got_sha != expected_sha:
        raise ValueError("decoded sha256 mismatch")

    result = {
        "ok": True,
        "mode": "decode",
        "rlbwt_path": str(rlbwt_path),
        "out_path": str(out_path),
        "decoded_bytes": written,
        "decoded_sha256": got_sha.hex(),
    }
    print(json.dumps(result, indent=2))
    return result


def verify_roundtrip(bwt_path: Path, rlbwt_path: Path):
    with tempfile.TemporaryDirectory() as td:
        decoded = Path(td) / "decoded.bwt.bin"
        decode_rlbwt(rlbwt_path, decoded)

        src_sha = sha256_file(bwt_path)
        dec_sha = sha256_file(decoded)
        src_size = bwt_path.stat().st_size
        dec_size = decoded.stat().st_size

        ok = src_sha == dec_sha and src_size == dec_size

        result = {
            "ok": ok,
            "mode": "verify",
            "bwt_path": str(bwt_path),
            "rlbwt_path": str(rlbwt_path),
            "source_bytes": src_size,
            "decoded_bytes": dec_size,
            "source_sha256": src_sha,
            "decoded_sha256": dec_sha,
            "encoded_bytes": rlbwt_path.stat().st_size,
            "ratio_vs_bwt": rlbwt_path.stat().st_size / src_size,
        }

        print(json.dumps(result, indent=2))

        if not ok:
            raise SystemExit(1)

        return result


def main():
    ap = argparse.ArgumentParser(description="RLBWT Container V1 encode/decode/verify.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode")
    enc.add_argument("--bwt", required=True)
    enc.add_argument("--out", required=True)

    dec = sub.add_parser("decode")
    dec.add_argument("--rlbwt", required=True)
    dec.add_argument("--out", required=True)

    ver = sub.add_parser("verify")
    ver.add_argument("--bwt", required=True)
    ver.add_argument("--rlbwt", required=True)

    args = ap.parse_args()

    if args.cmd == "encode":
        encode_bwt(Path(args.bwt), Path(args.out))
    elif args.cmd == "decode":
        decode_rlbwt(Path(args.rlbwt), Path(args.out))
    elif args.cmd == "verify":
        verify_roundtrip(Path(args.bwt), Path(args.rlbwt))
    else:
        raise SystemExit("unknown command")


if __name__ == "__main__":
    main()
