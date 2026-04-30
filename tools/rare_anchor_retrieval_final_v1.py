import argparse
import struct
import subprocess
import time


def load_chunk(corpus_path, qid, chunk_size):
    with open(corpus_path, "rb") as f:
        f.seek(qid * chunk_size)
        return f.read(chunk_size)


def load_u32(path):
    with open(path, "rb") as f:
        data = f.read()
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def candidate_starts(chunk_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, chunk_len - frag_len + 1), step))
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
            raise RuntimeError("FM server failed to start")

    def query_many(self, hex_patterns):
        for pat in hex_patterns:
            self.proc.stdin.write(pat + "\n")
        self.proc.stdin.flush()

        results = []
        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            l, r, cnt = map(int, line.split())
            results.append((l, r, cnt))
        return results

    def close(self):
        try:
            self.proc.stdin.write("__EXIT__\n")
            self.proc.stdin.flush()
        except:
            pass


def retrieve(args):
    print("=" * 60)
    print(" RARE ANCHOR RETRIEVAL FINAL V1")
    print("=" * 60)

    chunk = load_chunk(args.corpus, args.qid, args.chunk_size)
    chunk_map = load_u32(args.chunk_map)

    server = FMServer(args.server_bin, args.fm, args.bwt)

    starts = candidate_starts(len(chunk), args.frag_len, args.window_step, args.max_windows)

    hex_patterns = []
    valid_starts = []

    for s in starts:
        frag = chunk[s:s + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        hex_patterns.append(frag.hex())
        valid_starts.append(s)

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
            "chunks": chunks,
            "chunk_count": len(chunks)
        })

    # сортировка по редкости
    windows.sort(key=lambda x: (x["sa_hits"], x["start"]))

    selected = windows[:args.pick_k]

    shortlist = set()
    for w in selected:
        shortlist |= w["chunks"]

    shortlist = sorted(shortlist)

    t3 = time.time()

    server.close()

    print(f"\nQUERY qid = {args.qid}")
    print(f"frag_len = {args.frag_len}")
    print(f"pick_k = {args.pick_k}")

    print("\nSELECTED ANCHORS:")
    for w in selected:
        print(f"  start={w['start']} sa_hits={w['sa_hits']} chunks={w['chunk_count']}")

    print("\nRESULT:")
    print(f"  shortlist_size = {len(shortlist)}")
    print(f"  shortlist[:16] = {shortlist[:16]}")
    print(f"  hit = {args.qid in shortlist}")

    print("\nTIMINGS:")
    print(f"  total_time_sec = {t3 - t0:.6f}")
    print(f"  server_time_sec = {t2 - t1:.6f}")
    print(f"  fm_calls = {len(hex_patterns)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--qid", type=int, required=True)

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=16384)

    args = ap.parse_args()
    retrieve(args)