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
    probs = [v / n for v in c.values()]
    return -sum(p * math.log2(p) for p in probs)


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

        results = []
        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("empty line from FM server")
            l, r, cnt = map(int, line.split())
            results.append((l, r, cnt))
        return results

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


def load_query_bytes(args):
    sources = 0
    if args.query_text is not None:
        sources += 1
    if args.query_hex is not None:
        sources += 1
    if args.query_file is not None:
        sources += 1

    if sources != 1:
        raise ValueError("provide exactly one of: --query-text, --query-hex, --query-file")

    if args.query_text is not None:
        return args.query_text.encode("utf-8")

    if args.query_hex is not None:
        return bytes.fromhex(args.query_hex)

    with open(args.query_file, "rb") as f:
        return f.read()


def classify(shortlist_size, any_zero, all_same_single_chunk, non_selective_threshold, too_short):
    if too_short:
        return "TOO_SHORT"
    if shortlist_size == 0:
        return "NO_HIT"
    if any_zero:
        return "INVALID_PARTIAL_HIT"
    if shortlist_size > non_selective_threshold:
        return "NON_SELECTIVE"
    if shortlist_size == 1 and all_same_single_chunk:
        return "EXACT_UNIQUE"
    return "EXACT_MULTI"


def print_explain(explain_lines):
    print("\nEXPLAIN:")
    for line in explain_lines:
        print(f"  - {line}")


def retrieve(args):
    query_bytes = load_query_bytes(args)

    print("=" * 72)
    print(" RARE ANCHOR RETRIEVAL STRICT V2")
    print("=" * 72)

    explain = []

    if len(query_bytes) < args.frag_len:
        print(f"query_len = {len(query_bytes)}")
        print(f"frag_len = {args.frag_len}")

        print("\nSTATUS:")
        print("  outcome = TOO_SHORT")

        print("\nRESULT:")
        print("  shortlist_size = 0")
        print("  total_count = 0")
        print("  truncated = False")
        print("  shortlist_top = []")

        print("\nTIMINGS:")
        print("  total_time_sec = 0.000000")
        print("  server_time_sec = 0.000000")
        print("  fm_calls = 0")

        if args.explain:
            explain.append(f"query length {len(query_bytes)} < frag_len {args.frag_len}")
            print_explain(explain)
        return

    chunk_map = load_u32(args.chunk_map)
    starts = candidate_starts(len(query_bytes), args.frag_len, args.window_step, args.max_windows)

    windows_pre = []
    for s in starts:
        frag = query_bytes[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        ent = byte_entropy(frag)
        accepted = ent >= args.entropy_min
        windows_pre.append({
            "start": s,
            "frag": frag,
            "entropy": ent,
            "accepted": accepted
        })

    accepted = [w for w in windows_pre if w["accepted"]]

    print(f"query_len = {len(query_bytes)}")
    print(f"frag_len = {args.frag_len}")
    print(f"window_step = {args.window_step}")
    print(f"max_windows = {args.max_windows}")
    print(f"pick_k = {args.pick_k}")
    print(f"entropy_min = {args.entropy_min}")
    print(f"non_selective_threshold = {args.non_selective_threshold}")
    print(f"limit = {args.limit}")

    if not accepted:
        print("\nSTATUS:")
        print("  outcome = NON_SELECTIVE")

        print("\nRESULT:")
        print("  shortlist_size = 0")
        print("  total_count = 0")
        print("  truncated = False")
        print("  shortlist_top = []")

        print("\nTIMINGS:")
        print("  total_time_sec = 0.000000")
        print("  server_time_sec = 0.000000")
        print("  fm_calls = 0")

        if args.explain:
            dropped = len(windows_pre)
            explain.append("no anchors passed entropy filter")
            explain.append(f"dropped_windows = {dropped}")
            if windows_pre:
                explain.append(
                    f"max_entropy_seen = {max(w['entropy'] for w in windows_pre):.3f}"
                )
            print_explain(explain)
        return

    hex_patterns = [w["frag"].hex() for w in accepted]

    server = FMServer(args.server_bin, args.fm, args.bwt)

    try:
        t0 = time.time()
        t1 = time.time()
        results = server.query_many(hex_patterns)
        t2 = time.time()

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

        shortlist = set()
        for w in selected:
            shortlist |= w["chunks"]
        shortlist = sorted(shortlist)

        any_zero = any(w["sa_hits"] == 0 for w in selected)
        singletons = [next(iter(w["chunks"])) for w in selected if len(w["chunks"]) == 1]
        all_same_single_chunk = (
            len(singletons) == len(selected) and len(set(singletons)) == 1
        )

        outcome = classify(
            shortlist_size=len(shortlist),
            any_zero=any_zero,
            all_same_single_chunk=all_same_single_chunk,
            non_selective_threshold=args.non_selective_threshold,
            too_short=False,
        )

        total_count = len(shortlist)
        shortlist_top = shortlist[:args.limit]
        truncated = total_count > args.limit

        t3 = time.time()
    finally:
        server.close()

    print("\nSELECTED ANCHORS:")
    for w in selected:
        print(
            f"  start={w['start']} "
            f"entropy={w['entropy']:.3f} "
            f"sa_hits={w['sa_hits']} "
            f"chunks={w['chunk_count']} "
            f"interval={w['interval']}"
        )

    print("\nSTATUS:")
    print(f"  outcome = {outcome}")
    print(f"  any_zero_anchor = {any_zero}")
    print(f"  all_same_single_chunk = {all_same_single_chunk}")

    print("\nRESULT:")
    print(f"  shortlist_size = {len(shortlist)}")
    print(f"  total_count = {total_count}")
    print(f"  truncated = {truncated}")
    print(f"  shortlist_top = {shortlist_top}")

    print("\nTIMINGS:")
    print(f"  total_time_sec = {t3 - t0:.6f}")
    print(f"  server_time_sec = {t2 - t1:.6f}")
    print(f"  fm_calls = {len(hex_patterns)}")

    if args.explain:
        dropped = len(windows_pre) - len(accepted)
        explain.append(f"accepted_anchors = {len(accepted)}")
        explain.append(f"dropped_by_entropy = {dropped}")
        explain.append(f"selected_anchors = {len(selected)}")
        explain.append(f"min_selected_sa_hits = {min(w['sa_hits'] for w in selected) if selected else None}")
        explain.append(f"max_selected_sa_hits = {max(w['sa_hits'] for w in selected) if selected else None}")

        if outcome == "INVALID_PARTIAL_HIT":
            bad = [w["start"] for w in selected if w["sa_hits"] == 0]
            explain.append(f"anchors_with_zero_hits = {bad}")

        if outcome == "EXACT_MULTI":
            explain.append("multiple chunks survived strict exact filtering")

        if outcome == "NON_SELECTIVE":
            explain.append(f"shortlist_size {len(shortlist)} > threshold {args.non_selective_threshold}")

        print_explain(explain)


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
    ap.add_argument("--explain", action="store_true")

    args = ap.parse_args()
    retrieve(args)