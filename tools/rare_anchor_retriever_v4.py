import argparse
import pickle
import struct
import subprocess
import time


def load_chunks(corpus_path: str, chunk_size: int = 16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def load_u32(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not multiple of 4")
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def normalize_rows(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        if "queries" in obj and isinstance(obj["queries"], list):
            return obj["queries"]
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError("unsupported results format")


def find_row_by_qid(rows, qid: int):
    for row in rows:
        if int(row["qid"]) == int(qid):
            return row
    raise ValueError(f"qid={qid} not found in gap_results")


def candidate_starts(chunk_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, chunk_len - frag_len + 1), step))
    return starts[:max_windows]


def interval_to_chunk_candidates(chunk_map, interval):
    l, r = interval
    if l >= r:
        return set()
    return set(chunk_map[l:r])


class FMServer:
    def __init__(self, server_bin, fm, bwt):
        self.proc = subprocess.Popen(
            [str(server_bin), str(fm), str(bwt)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self.proc.stderr.readline().strip()
        if ready != "READY":
            err_rest = self.proc.stderr.read()
            raise RuntimeError(f"server failed to start: first line={ready!r}\n{err_rest}")

    def query_many(self, hex_patterns):
        results = []
        for pat in hex_patterns:
            self.proc.stdin.write(pat + "\n")
        self.proc.stdin.flush()

        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("empty line from FM server")
            parts = line.split()
            if len(parts) != 3:
                raise RuntimeError(f"bad server line: {line}")
            l, r, cnt = map(int, parts)
            results.append(((l, r), cnt))
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


def retrieve(args):
    t0_total = time.time()

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    row = find_row_by_qid(rows, args.qid)
    chunk_id = int(row["qid"])
    query_chunk = chunks[chunk_id]

    starts = candidate_starts(len(query_chunk), args.frag_len, args.window_step, args.max_windows)

    hex_patterns = []
    frag_meta = []

    for start in starts:
        frag = query_chunk[start:start + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        hex_patterns.append(frag.hex())
        frag_meta.append(start)

    server = FMServer(args.server_bin, args.fm, args.bwt)

    try:
        t1 = time.time()
        batch_results = server.query_many(hex_patterns)
        t2 = time.time()
    finally:
        server.close()

    all_windows = []
    for start, ((l, r), count) in zip(frag_meta, batch_results):
        cand_chunks = interval_to_chunk_candidates(chunk_map, (l, r))
        all_windows.append({
            "start": start,
            "sa_hits": count,
            "interval": (l, r),
            "chunks": len(cand_chunks),
            "cand_chunks": cand_chunks,
        })

    all_windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = all_windows[:args.pick_k]

    shortlist_set = set()
    for s in selected:
        shortlist_set |= s["cand_chunks"]

    shortlist = sorted(shortlist_set)
    total_time = time.time() - t0_total
    server_query_time = t2 - t1

    print("============================================")
    print(" RARE ANCHOR RETRIEVER V4")
    print("============================================")
    print(f"qid: {chunk_id}")
    print(f"frag_len: {args.frag_len}")
    print(f"window_step: {args.window_step}")
    print(f"max_windows: {args.max_windows}")
    print(f"pick_k: {args.pick_k}")

    print("\nSELECTED ANCHORS:")
    for s in selected:
        print(
            f"  start={s['start']} "
            f"sa_hits={s['sa_hits']} "
            f"chunks={s['chunks']} "
            f"interval={s['interval']}"
        )

    print("\nRESULTS:")
    print(f"  shortlist_size: {len(shortlist)}")
    print(f"  shortlist[:16]: {shortlist[:16]}")
    print(f"  truth_in_shortlist: {chunk_id in shortlist_set}")

    print("\nTIMINGS:")
    print(f"  total_time_sec: {total_time:.6f}")
    print(f"  server_query_time_sec: {server_query_time:.6f}")
    print(f"  fm_calls_server: {len(hex_patterns)}")
    if hex_patterns:
        print(f"  avg_fm_call_equiv_sec: {server_query_time / len(hex_patterns):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gap-results", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--fm", required=True)
    parser.add_argument("--bwt", required=True)
    parser.add_argument("--chunk-map", required=True)
    parser.add_argument("--server-bin", required=True)

    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--frag-len", type=int, default=48)
    parser.add_argument("--window-step", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--pick-k", type=int, default=3)

    args = parser.parse_args()
    retrieve(args)