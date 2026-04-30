import argparse
import math
import struct
import subprocess
import time
from collections import Counter


def load_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not multiple of 4")
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def candidate_starts(query_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, query_len - frag_len + 1), step))
    return starts[:max_windows]


def interval_to_chunks(chunk_map, l, r):
    if l >= r:
        return set()
    return set(chunk_map[l:r])


def byte_entropy(b: bytes):
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def load_query_bytes(args):
    sources = sum(x is not None for x in [args.query_text, args.query_hex, args.query_file])
    if sources != 1:
        raise ValueError("provide exactly one of: --query-text, --query-hex, --query-file")

    if args.query_text is not None:
        return args.query_text.encode("utf-8")
    if args.query_hex is not None:
        return bytes.fromhex(args.query_hex)
    with open(args.query_file, "rb") as f:
        return f.read()


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
            err = self.proc.stderr.read()
            raise RuntimeError(f"FM server failed to start: {ready}\n{err}")

    def query_many(self, hex_patterns):
        for pat in hex_patterns:
            self.proc.stdin.write(pat + "\n")
        self.proc.stdin.flush()

        out = []
        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("empty line from FM server")
            l, r, cnt = map(int, line.split())
            out.append((l, r, cnt))
        return out

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("__EXIT__\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def print_header(args, query_len):
    print("=" * 72)
    print(" RARE ANCHOR RETRIEVAL STRICT V3")
    print("=" * 72)
    print(f"query_len = {query_len}")
    print(f"frag_len = {args.frag_len}")
    print(f"window_step = {args.window_step}")
    print(f"max_windows = {args.max_windows}")
    print(f"pick_k = {args.pick_k}")
    print(f"entropy_min = {args.entropy_min}")
    print(f"non_selective_threshold = {args.non_selective_threshold}")
    print(f"limit = {args.limit}")
    print(f"max_query_bytes = {args.max_query_bytes}")


def print_result(outcome, shortlist=None, total_time=0.0, server_time=0.0, fm_calls=0, explain=None, total_count=None, truncated=False):
    if shortlist is None:
        shortlist = []
    if total_count is None:
        total_count = len(shortlist)

    print("\nSTATUS:")
    print(f"  outcome = {outcome}")

    print("\nRESULT:")
    print(f"  shortlist_size = {len(shortlist)}")
    print(f"  total_count = {total_count}")
    print(f"  truncated = {truncated}")
    print(f"  shortlist_top = {shortlist}")

    print("\nTIMINGS:")
    print(f"  total_time_sec = {total_time:.6f}")
    print(f"  server_time_sec = {server_time:.6f}")
    print(f"  fm_calls = {fm_calls}")

    if explain:
        print("\nEXPLAIN:")
        for line in explain:
            print(f"  - {line}")


def classify(shortlist_size, any_zero, all_same_single_chunk, non_selective_threshold):
    if shortlist_size == 0:
        return "NO_HIT"
    if any_zero:
        return "INVALID_PARTIAL_HIT"
    if shortlist_size > non_selective_threshold:
        return "NON_SELECTIVE"
    if shortlist_size == 1 and all_same_single_chunk:
        return "EXACT_UNIQUE"
    return "EXACT_MULTI"


def retrieve(args):
    query_bytes = load_query_bytes(args)
    query_len = len(query_bytes)
    print_header(args, query_len)

    t0_all = time.time()

    if query_len == 0:
        explain = ["query length is zero"] if args.explain else None
        print_result("EMPTY_QUERY", explain=explain)
        return

    if query_len > args.max_query_bytes:
        explain = [f"query length {query_len} > max_query_bytes {args.max_query_bytes}"] if args.explain else None
        print_result("QUERY_TOO_LONG", explain=explain)
        return

    if query_len < args.frag_len:
        explain = [f"query length {query_len} < frag_len {args.frag_len}"] if args.explain else None
        print_result("TOO_SHORT", explain=explain)
        return

    starts = candidate_starts(query_len, args.frag_len, args.window_step, args.max_windows)

    windows_pre = []
    for s in starts:
        frag = query_bytes[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        ent = byte_entropy(frag)
        windows_pre.append({
            "start": s,
            "frag": frag,
            "entropy": ent,
            "accepted": ent >= args.entropy_min,
        })

    accepted = [w for w in windows_pre if w["accepted"]]

    if not accepted:
        explain = None
        if args.explain:
            explain = [
                "no anchors passed entropy filter",
                f"dropped_windows = {len(windows_pre)}",
            ]
            if windows_pre:
                explain.append(f"max_entropy_seen = {max(w['entropy'] for w in windows_pre):.3f}")
        print_result("NON_SELECTIVE", total_time=time.time() - t0_all, explain=explain)
        return

    hex_patterns = [w["frag"].hex() for w in accepted]

    server = FMServer(args.server_bin, args.fm, args.bwt)
    try:
        t1 = time.time()
        results = server.query_many(hex_patterns)
        t2 = time.time()
    finally:
        server.close()

    # Early NO_HIT: no need to load 3.8GB chunk_map if every accepted anchor missed.
    if all(cnt == 0 for _, _, cnt in results):
        explain = None
        if args.explain:
            explain = [
                f"accepted_anchors = {len(accepted)}",
                "all accepted anchors returned zero hits",
            ]
        print_result(
            "NO_HIT",
            total_time=time.time() - t0_all,
            server_time=t2 - t1,
            fm_calls=len(hex_patterns),
            explain=explain,
        )
        return

    # Now chunk_map is actually needed.
    chunk_map = load_u32(args.chunk_map)

    scored = []
    for w, (l, r, cnt) in zip(accepted, results):
        chunks = interval_to_chunks(chunk_map, l, r)
        scored.append({
            "start": w["start"],
            "entropy": w["entropy"],
            "sa_hits": cnt,
            "interval": (l, r),
            "chunks": chunks,
            "chunk_count": len(chunks),
        })

    scored.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = scored[:args.pick_k]

    shortlist_set = set()
    for w in selected:
        shortlist_set |= w["chunks"]

    total_count = len(shortlist_set)
    shortlist_top = sorted(shortlist_set)[:args.limit]
    truncated = total_count > args.limit

    any_zero = any(w["sa_hits"] == 0 for w in selected)
    singletons = [next(iter(w["chunks"])) for w in selected if len(w["chunks"]) == 1]
    all_same_single_chunk = len(singletons) == len(selected) and len(set(singletons)) == 1

    outcome = classify(
        shortlist_size=total_count,
        any_zero=any_zero,
        all_same_single_chunk=all_same_single_chunk,
        non_selective_threshold=args.non_selective_threshold,
    )

    print("\nSELECTED ANCHORS:")
    for w in selected:
        print(
            f"  start={w['start']} "
            f"entropy={w['entropy']:.3f} "
            f"sa_hits={w['sa_hits']} "
            f"chunks={w['chunk_count']} "
            f"interval={w['interval']}"
        )

    explain = None
    if args.explain:
        explain = [
            f"accepted_anchors = {len(accepted)}",
            f"dropped_by_entropy = {len(windows_pre) - len(accepted)}",
            f"selected_anchors = {len(selected)}",
            f"min_selected_sa_hits = {min(w['sa_hits'] for w in selected) if selected else None}",
            f"max_selected_sa_hits = {max(w['sa_hits'] for w in selected) if selected else None}",
        ]

        if outcome == "INVALID_PARTIAL_HIT":
            explain.append(f"anchors_with_zero_hits = {[w['start'] for w in selected if w['sa_hits'] == 0]}")
        if outcome == "EXACT_MULTI":
            explain.append("multiple chunks survived strict exact filtering")
        if outcome == "NON_SELECTIVE":
            explain.append(f"shortlist_size {total_count} > threshold {args.non_selective_threshold}")

    print_result(
        outcome,
        shortlist=shortlist_top,
        total_count=total_count,
        truncated=truncated,
        total_time=time.time() - t0_all,
        server_time=t2 - t1,
        fm_calls=len(hex_patterns),
        explain=explain,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--query-text")
    ap.add_argument("--query-hex")
    ap.add_argument("--query-file")

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--entropy-min", type=float, default=2.0)
    ap.add_argument("--non-selective-threshold", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-query-bytes", type=int, default=1048576)
    ap.add_argument("--explain", action="store_true")

    args = ap.parse_args()
    retrieve(args)