import argparse
import struct
import subprocess
import time


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


def retrieve(args):
    query_bytes = load_query_bytes(args)
    if len(query_bytes) < args.frag_len:
        raise ValueError(
            f"query too short: len={len(query_bytes)}, frag_len={args.frag_len}"
        )

    chunk_map = load_u32(args.chunk_map)
    starts = candidate_starts(len(query_bytes), args.frag_len, args.window_step, args.max_windows)

    hex_patterns = []
    valid_starts = []
    for s in starts:
        frag = query_bytes[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        hex_patterns.append(frag.hex())
        valid_starts.append(s)

    server = FMServer(args.server_bin, args.fm, args.bwt)

    try:
        t0 = time.time()
        t1 = time.time()
        results = server.query_many(hex_patterns)
        t2 = time.time()

        windows = []
        for start, (l, r, cnt) in zip(valid_starts, results):
            chunks = interval_to_chunks(chunk_map, l, r)
            windows.append({
                "start": start,
                "sa_hits": cnt,
                "interval": (l, r),
                "chunks": chunks,
                "chunk_count": len(chunks),
            })

        windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
        selected = windows[:args.pick_k]

        shortlist = set()
        for w in selected:
            shortlist |= w["chunks"]
        shortlist = sorted(shortlist)

        t3 = time.time()
    finally:
        server.close()

    print("=" * 60)
    print(" RARE ANCHOR RETRIEVAL RAW V1")
    print("=" * 60)
    print(f"query_len = {len(query_bytes)}")
    print(f"frag_len = {args.frag_len}")
    print(f"window_step = {args.window_step}")
    print(f"max_windows = {args.max_windows}")
    print(f"pick_k = {args.pick_k}")

    print("\nSELECTED ANCHORS:")
    for w in selected:
        print(
            f"  start={w['start']} "
            f"sa_hits={w['sa_hits']} "
            f"chunks={w['chunk_count']} "
            f"interval={w['interval']}"
        )

    print("\nRESULT:")
    print(f"  shortlist_size = {len(shortlist)}")
    print(f"  shortlist[:16] = {shortlist[:16]}")

    print("\nTIMINGS:")
    print(f"  total_time_sec = {t3 - t0:.6f}")
    print(f"  server_time_sec = {t2 - t1:.6f}")
    print(f"  fm_calls = {len(hex_patterns)}")


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

    args = ap.parse_args()
    retrieve(args)