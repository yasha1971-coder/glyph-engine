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
    def __init__(self, path, max_range_scan=10000):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = self.mm.size() // 4
        self.max_range_scan = max_range_scan

    def get_range(self, l, r):
        if l >= r:
            return set(), False

        n = r - l
        if n > self.max_range_scan:
            return set(), True

        start = l * 4
        end = r * 4

        if n == 1:
            return {struct.unpack_from("<I", self.mm, start)[0]}, False

        raw = self.mm[start:end]
        vals = struct.unpack("<" + "I" * n, raw)
        return set(vals), False

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
        return base("EMPTY_QUERY", t0)

    if qlen > args.max_query_bytes:
        return base("QUERY_TOO_LONG", t0)

    if qlen < args.frag_len:
        return base("TOO_SHORT", t0)

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
        return base("NON_SELECTIVE", t0)

    hex_patterns = [frag.hex() for _, frag, _ in windows]

    t1 = time.time()
    fm_results = server.query_many(hex_patterns)
    t2 = time.time()

    if all(cnt == 0 for _, _, cnt in fm_results):
        return base("NO_HIT", t0, t2 - t1, len(hex_patterns))

    scored = []
    range_truncated = False

    for (start, _, ent), (l, r, cnt) in zip(windows, fm_results):
        chunks, truncated = chunk_map.get_range(l, r)
        if truncated:
            range_truncated = True

        scored.append({
            "start": start,
            "entropy": ent,
            "sa_hits": cnt,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "interval": (l, r),
            "range_truncated": truncated,
        })

    scored.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = scored[:args.pick_k]

    if any(w["range_truncated"] for w in selected):
        return {
            "outcome": "NON_SELECTIVE",
            "reason": "max_range_scan_exceeded",
            "shortlist_size": 0,
            "total_count": 0,
            "truncated": True,
            "shortlist_top": [],
            "timings": {
                "total_time_sec": time.time() - t0,
                "server_time_sec": t2 - t1,
                "fm_calls": len(hex_patterns),
            },
            "selected_anchors": [
                {
                    "start": w["start"],
                    "entropy": round(w["entropy"], 3),
                    "sa_hits": w["sa_hits"],
                    "chunk_count": w["chunk_count"],
                    "interval": w["interval"],
                    "range_truncated": w["range_truncated"],
                }
                for w in selected
            ],
        }

    shortlist_set = set()
    for w in selected:
        shortlist_set |= w["chunks"]

    total_count = len(shortlist_set)
    shortlist_top = sorted(shortlist_set)[:args.limit]
    truncated = total_count > args.limit

    any_zero = any(w["sa_hits"] == 0 for w in selected)
    singletons = [next(iter(w["chunks"])) for w in selected if len(w["chunks"]) == 1]
    all_same_single_chunk = len(singletons) == len(selected) and len(set(singletons)) == 1

    if total_count == 0:
        outcome = "NO_HIT"
    elif any_zero:
        outcome = "INVALID_PARTIAL_HIT"
    elif total_count > args.non_selective_threshold:
        outcome = "NON_SELECTIVE"
    elif total_count == 1 and all_same_single_chunk:
        outcome = "EXACT_UNIQUE"
    else:
        outcome = "EXACT_MULTI"

    return {
        "outcome": outcome,
        "shortlist_size": len(shortlist_top),
        "total_count": total_count,
        "truncated": truncated,
        "shortlist_top": shortlist_top,
        "timings": {
            "total_time_sec": time.time() - t0,
            "server_time_sec": t2 - t1,
            "fm_calls": len(hex_patterns),
        },
        "selected_anchors": [
            {
                "start": w["start"],
                "entropy": round(w["entropy"], 3),
                "sa_hits": w["sa_hits"],
                "chunk_count": w["chunk_count"],
                "interval": w["interval"],
                "range_truncated": w["range_truncated"],
            }
            for w in selected
        ],
    }


def base(outcome, t0, server_time=0.0, fm_calls=0):
    return {
        "outcome": outcome,
        "shortlist_size": 0,
        "total_count": 0,
        "truncated": False,
        "shortlist_top": [],
        "timings": {
            "total_time_sec": time.time() - t0,
            "server_time_sec": server_time,
            "fm_calls": fm_calls,
        },
    }


def load_query_bytes(args):
    if args.query_text is not None:
        return args.query_text.encode("utf-8")
    if args.query_file is not None:
        with open(args.query_file, "rb") as f:
            return f.read()
    raise ValueError("provide --query-text or --query-file")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--query-text")
    ap.add_argument("--query-file")

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--entropy-min", type=float, default=2.0)
    ap.add_argument("--non-selective-threshold", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-query-bytes", type=int, default=1048576)
    ap.add_argument("--max-range-scan", type=int, default=10000)

    args = ap.parse_args()

    query = load_query_bytes(args)

    chunk_map = MMapChunkMap(args.chunk_map, max_range_scan=args.max_range_scan)
    server = FMServer(args.server_bin, args.fm, args.bwt)

    try:
        out = retrieve_bytes(query, server, chunk_map, args)
        print(json.dumps(out, indent=2))
    finally:
        server.close()
        chunk_map.close()


if __name__ == "__main__":
    main()
