#!/usr/bin/env python3
import argparse
import json
import math
import mmap
import struct
import subprocess
import time
from collections import Counter


def byte_entropy(b: bytes):
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def candidate_starts(query_len, frag_len, step, max_windows):
    return list(range(0, max(0, query_len - frag_len + 1), step))[:max_windows]


class MMapChunkMap:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = self.mm.size() // 4

    def get_range(self, l, r):
        out = set()
        if l >= r:
            return out

        for i in range(l, r):
            off = i * 4
            chunk = struct.unpack_from("<I", self.mm, off)[0]
            out.add(chunk)

        return out

    def close(self):
        self.mm.close()
        self.f.close()


class FMServer:
    def __init__(self, server_bin, fm, bwt):
        self.proc = subprocess.Popen(
            [server_bin, fm, bwt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.proc.stderr.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"FM server did not start: {ready}")

    def query_many(self, hex_patterns):
        for pat in hex_patterns:
            self.proc.stdin.write(pat + "\n")
        self.proc.stdin.flush()

        out = []
        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            l, r, cnt = map(int, line.split())
            out.append((l, r, cnt))
        return out

    def close(self):
        try:
            self.proc.stdin.write("__EXIT__\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


def retrieve_bytes(query_bytes, server, chunk_map, args):
    t0 = time.time()

    qlen = len(query_bytes)

    if qlen == 0:
        return {"outcome": "EMPTY_QUERY"}

    if qlen > args.max_query_bytes:
        return {"outcome": "QUERY_TOO_LONG"}

    if qlen < args.frag_len:
        return {"outcome": "TOO_SHORT"}

    starts = candidate_starts(qlen, args.frag_len, args.window_step, args.max_windows)

    windows = []
    for s in starts:
        frag = query_bytes[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        ent = byte_entropy(frag)
        if ent >= args.entropy_min:
            windows.append((s, frag, ent))

    if not windows:
        return {"outcome": "NON_SELECTIVE"}

    hex_patterns = [frag.hex() for _, frag, _ in windows]

    t1 = time.time()
    fm_results = server.query_many(hex_patterns)
    t2 = time.time()

    scored = []
    for (start, _, ent), (l, r, cnt) in zip(windows, fm_results):
        chunks = chunk_map.get_range(l, r)
        scored.append((cnt, start, chunks))

    scored.sort()
    selected = scored[:args.pick_k]

    shortlist = set()
    for _, _, ch in selected:
        shortlist |= ch

    outcome = "EXACT_UNIQUE" if len(shortlist) == 1 else "EXACT_MULTI"

    return {
        "outcome": outcome,
        "shortlist_size": len(shortlist),
        "timings": {
            "total_time_sec": time.time() - t0,
            "server_time_sec": t2 - t1,
        }
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--query-file", required=True)

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--entropy-min", type=float, default=2.0)
    ap.add_argument("--max-query-bytes", type=int, default=1048576)

    args = ap.parse_args()

    with open(args.query_file, "rb") as f:
        query = f.read()

    chunk_map = MMapChunkMap(args.chunk_map)
    server = FMServer(args.server_bin, args.fm, args.bwt)

    try:
        out = retrieve_bytes(query, server, chunk_map, args)
        print(json.dumps(out, indent=2))
    finally:
        server.close()
        chunk_map.close()


if __name__ == "__main__":
    main()
