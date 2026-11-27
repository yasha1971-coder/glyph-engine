#!/usr/bin/env python3
"""Experimental pinned-index result profile, shared by GLYPH and native controls.

The trusted caller supplies corpus SHA, index ID, query bytes and limit. This
semantic/replay profile does NOT replace glyph-evidence's canonical full mode.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
import tempfile

PROFILE = "glyph-index-result-v1"
PARTS = ("bwt.rlbwt", "bwt.rlbwt.rank", "locate_core_s128.bin")
MAX_BYTES = 128 * 1024 * 1024


def require(ok, error):
    if not ok:
        raise ValueError(error)


def pin(value):
    require(isinstance(value, str) and len(value) == 64 and
            set(value) <= set("0123456789abcdef"), "invalid SHA-256 pin")
    return value


def read_regular(path, maximum):
    path = Path(path)
    with path.open("rb") as f:
        s = os.fstat(f.fileno())
        require(stat.S_ISREG(s.st_mode) and s.st_size <= maximum, "invalid regular file size")
        data = f.read(maximum + 1)
    require(len(data) <= maximum, "file grew beyond limit")
    return data


def copy_pinned(source, dest, expected, maximum):
    h = hashlib.sha256()
    size = 0
    with Path(source).open("rb") as f, Path(dest).open("xb") as out:
        require(stat.S_ISREG(os.fstat(f.fileno()).st_mode), "index is not regular")
        for data in iter(lambda: f.read(1024 * 1024), b""):
            size += len(data)
            require(size <= maximum, "index file outside size bound")
            h.update(data)
            out.write(data)
    require(h.hexdigest() == pin(expected), "index SHA-256 mismatch")
    return size


def snapshot_index(args, work):
    """Parse/replay only private bytes copied and hashed in the same read."""
    index = Path(args.index)
    expected = pin(args.expect_index_sha256)
    dest = Path(work) / "index"
    dest.mkdir()
    if args.engine != "glyph":
        copy_pinned(index, dest / "native", expected, 1024 * 1024 * 1024)
        return dest / "native", args.corpus_bytes
    raw = read_regular(index / "manifest.json", 8192)
    require(hashlib.sha256(raw).hexdigest() == expected, "index manifest SHA-256 mismatch")
    m = json.loads(raw)
    require(m["format"] in ("glyph-search-index-v1", "glyph-search-index-v2"), "unsupported index format")
    size = m["corpus_bytes"]
    require(type(size) is int and 1 <= size <= MAX_BYTES and size == args.corpus_bytes,
            "index corpus length mismatch")
    require(m["corpus_sha256"] == args.expect_corpus_sha256, "index corpus SHA-256 mismatch")
    require(set(m["files"]) == set(PARTS), "wrong index file coverage")
    (dest / "manifest.json").write_bytes(raw)
    for name in PARTS:
        spec = m["files"][name]
        require(type(spec["bytes"]) is int and spec["bytes"] > 0, "invalid index file length")
        actual = copy_pinned(index / name, dest / name, spec["sha256"], 1024 * 1024 * 1024)
        require(actual == spec["bytes"], "index part length mismatch")
    v2 = m["format"].endswith("v2")
    count = 1 + size // 128
    p = dest / PARTS[2]
    with p.open("rb") as f:
        require(f.read(24) == struct.pack("<4sQIQ", b"LOC2" if v2 else b"LOC1", size + 1, 128, count)
                and p.stat().st_size == 24 + count * (4 if v2 else 16), "invalid locate layout")
    return dest, size


def query_bytes(args):
    q = read_regular(args.query_file, 1024)
    require(1 <= len(q) <= 1024 and b"\0" not in q, "unsupported query")
    require(type(args.limit) is int and 1 <= args.limit <= 65536, "invalid query limit")
    return q


def replay(args, index, q, work):
    tool = Path(args.tool)
    env = dict(os.environ, LD_LIBRARY_PATH=str(tool / "lib"), OMP_NUM_THREADS="1")
    if args.engine == "glyph":
        command = [tool / "build/rlbwt_full_query_locate_server_v1", *[index / n for n in PARTS]]
        out = subprocess.check_output(list(map(str, command)), input=f"{q.hex()}\t{args.limit}\n".encode(), env=env)
        fields = out.decode("ascii").strip("\n").split("\t")
        require(len(fields) == 9 and fields[0] == "OK", "unexpected GLYPH response")
        count, n = int(fields[3]), int(fields[4])
        offsets = list(map(int, fields[8].split(","))) if fields[8] else []
        require(n == len(offsets) and fields[5] == str(n < count).lower(), "inconsistent GLYPH response")
    else:
        query = Path(work) / "query.hex"
        query.write_text(q.hex() + "\n", encoding="ascii")
        out = subprocess.check_output([str(tool / "engine"), "query", str(index), str(query), str(args.limit)], env=env)
        lines = out.splitlines()
        require(len(lines) == 1, "unexpected native response count")
        row = json.loads(lines[0])
        require(row["query_hex"] == q.hex(), "native query mismatch")
        count, offsets = row["count"], row["offsets"]
    require(type(count) is int and 0 <= count <= args.corpus_bytes and isinstance(offsets, list)
            and len(offsets) == min(count, args.limit) and offsets == sorted(set(offsets))
            and all(type(k) is int and 0 <= k <= args.corpus_bytes - len(q) for k in offsets),
            "invalid native result bounds")
    return count, offsets


def request(args, q):
    return dict(profile=PROFILE, engine=args.engine, corpus_sha256=pin(args.expect_corpus_sha256),
                corpus_bytes=args.corpus_bytes, index_sha256=pin(args.expect_index_sha256),
                query_hex=q.hex(), query_sha256=hashlib.sha256(q).hexdigest(), limit=args.limit)


def create(args, work):
    q = query_bytes(args)
    expected = request(args, q)
    index, _ = snapshot_index(args, work)
    count, offsets = replay(args, index, q, work)
    row = dict(expected, count=count, offsets=offsets, bounded=count > args.limit)
    with Path(args.out).open("x", encoding="ascii") as f:
        json.dump(row, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    return dict(ok=True, profile=PROFILE, count=count, offsets=offsets,
                result_bytes=Path(args.out).stat().st_size)


def corpus_snapshot(args, index, work):
    cache = Path(args.cache)
    binding = dict(profile=PROFILE, corpus_sha256=pin(args.expect_corpus_sha256),
                   index_sha256=pin(args.expect_index_sha256), corpus_bytes=args.corpus_bytes)
    if cache.exists():
        require(json.loads(read_regular(cache / "binding.json", 4096)) == binding, "cache binding mismatch")
    else:
        # Publish only complete, independently pinned recovery. Failed attempts
        # leave no cache authorizing a later shortcut.
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache.parent, prefix=".recovery-") as t:
            stage = Path(t) / "cache"
            stage.mkdir()
            body = stage / "corpus.bin"
            tool = Path(args.tool)
            if args.engine == "glyph":
                from recover_search_index_v1 import recover
                recover(index / PARTS[0], args.corpus_bytes, args.expect_corpus_sha256, body,
                        tool / "build/recover_bwt_libsais", work)
            else:
                env = dict(os.environ, LD_LIBRARY_PATH=str(tool / "lib"), OMP_NUM_THREADS="1")
                subprocess.run([str(tool / "engine"), "extract", str(index), str(body)], check=True, env=env)
            data = read_regular(body, MAX_BYTES)
            require(len(data) == args.corpus_bytes and hashlib.sha256(data).hexdigest() == binding["corpus_sha256"],
                    "recovered corpus SHA-256 or length mismatch")
            del data
            (stage / "binding.json").write_text(json.dumps(binding, sort_keys=True) + "\n")
            # renameat2 publication is used by search build; here never replace
            # existing recipient state, including a racing empty directory.
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(cache), 1):
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
    data = read_regular(cache / "corpus.bin", MAX_BYTES)
    require(len(data) == args.corpus_bytes and hashlib.sha256(data).hexdigest() == binding["corpus_sha256"],
            "corpus SHA-256 or length mismatch")
    return data


def verify(args, work):
    q = query_bytes(args)
    expected = request(args, q)
    row = json.loads(read_regular(args.result, 2621440))
    require(set(row) == set(expected) | {"count", "offsets", "bounded"}, "wrong result coverage")
    for key, value in expected.items():
        require(type(row[key]) is type(value) and row[key] == value, "expected request mismatch: " + key)
    index, _ = snapshot_index(args, work)
    data = corpus_snapshot(args, index, work)
    # All SHA, count and byte decisions use this same immutable bytes object.
    count = 0
    at = data.find(q)
    while at >= 0:
        count += 1
        at = data.find(q, at + 1)
    offsets = row["offsets"]
    require(isinstance(offsets, list) and all(type(k) is int and 0 <= k <= len(data) - len(q)
            and data[k:k + len(q)] == q for k in offsets) and len(set(offsets)) == len(offsets), "offset byte mismatch")
    require(type(row["count"]) is int and row["count"] == count
            and len(offsets) == min(count, args.limit) and row["bounded"] is (count > args.limit),
            "count or coverage mismatch")
    actual_count, actual_offsets = replay(args, index, q, work)
    require(actual_count == count and actual_offsets == offsets, "native replay mismatch")
    return dict(ok=True, profile=PROFILE, corpus_sha256=expected["corpus_sha256"],
                index_sha256=expected["index_sha256"], count=count, offsets=offsets,
                byte_check=True, native_replay=True, canonical_runtime_rebuilt=False)


def main(argv=None, default_tool=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("create", "verify"):
        p = sub.add_parser(cmd)
        p.add_argument("--index", required=True, type=Path)
        p.add_argument("--tool", type=Path, default=default_tool, required=default_tool is None)
        p.add_argument("--engine", choices=("glyph", "sdsl", "mover", "rindex"), default="glyph")
        p.add_argument("--expect-index-sha256", required=True)
        p.add_argument("--expect-corpus-sha256", required=True)
        p.add_argument("--corpus-bytes", type=int, required=True)
        p.add_argument("--query-file", required=True, type=Path)
        p.add_argument("--limit", type=int, default=10)
        if cmd == "create":
            p.add_argument("--out", required=True, type=Path)
        else:
            p.add_argument("--result", required=True, type=Path)
            p.add_argument("--cache", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        require(1 <= args.corpus_bytes <= MAX_BYTES, "unsupported corpus length")
        with tempfile.TemporaryDirectory(prefix="glyph-result-", dir="/tmp") as t:
            result = (create if args.cmd == "create" else verify)(args, Path(t))
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, EOFError,
            subprocess.CalledProcessError) as exc:
        print(json.dumps(dict(ok=False, error=str(exc))), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
