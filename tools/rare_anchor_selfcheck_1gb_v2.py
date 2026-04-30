import argparse
import random
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)

    chunks = load_chunks(args.corpus, chunk_size=args.chunk_size)
    chunk_map = load_u32(args.chunk_map)

    valid_qids = [i for i, ch in enumerate(chunks) if len(ch) >= args.frag_len]
    picked_qids = random.sample(valid_qids, min(args.limit, len(valid_qids)))

    server = FMServer(args.server_bin, args.fm, args.bwt)

    total = 0
    hit = 0
    shortlist_sizes = []
    selected_sa_hits = []
    total_times = []
    server_times = []
    examples = []

    print("=" * 60)
    print(" RARE ANCHOR SELFCHECK 1GB V2")
    print("=" * 60)

    try:
        for qid in picked_qids:
            query_chunk = chunks[qid]
            starts = candidate_starts(len(query_chunk), args.frag_len, args.window_step, args.max_windows)

            hex_patterns = []
            frag_meta = []
            for start in starts:
                frag = query_chunk[start:start + args.frag_len]
                if len(frag) != args.frag_len:
                    continue
                hex_patterns.append(frag.hex())
                frag_meta.append(start)

            t0 = time.time()
            t1 = time.time()
            batch_results = server.query_many(hex_patterns)
            t2 = time.time()

            all_windows = []
            for start, ((l, r), count) in zip(frag_meta, batch_results):
                cand_chunks = interval_to_chunk_candidates(chunk_map, (l, r))
                all_windows.append({
                    "start": start,
                    "sa_hits": count,
                    "chunks": len(cand_chunks),
                    "cand_chunks": cand_chunks,
                })

            all_windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
            selected = all_windows[:args.pick_k]

            shortlist_set = set()
            for s in selected:
                shortlist_set |= s["cand_chunks"]

            shortlist = sorted(shortlist_set)[:16]
            t3 = time.time()

            total += 1
            hit += int(qid in shortlist)
            shortlist_sizes.append(len(shortlist_set))
            selected_sa_hits.append(sum(s["sa_hits"] for s in selected) / len(selected))
            total_times.append(t3 - t0)
            server_times.append(t2 - t1)

            if len(examples) < args.show:
                examples.append({
                    "qid": qid,
                    "shortlist_size": len(shortlist_set),
                    "shortlist_hit": qid in shortlist,
                    "selected": selected,
                    "total_time": t3 - t0,
                    "server_time": t2 - t1,
                })
    finally:
        server.close()

    print("\nRESULTS:")
    print(f"  queries                 = {total}")
    print(f"  shortlist_hit@16        = {hit / total:.2%}")
    print(f"  avg_shortlist_size      = {sum(shortlist_sizes) / len(shortlist_sizes):.2f}")
    print(f"  avg_selected_sa_hits    = {sum(selected_sa_hits) / len(selected_sa_hits):.2f}")
    print(f"  avg_total_time_sec      = {sum(total_times) / len(total_times):.6f}")
    print(f"  avg_server_time_sec     = {sum(server_times) / len(server_times):.6f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(
            f"  qid={ex['qid']} "
            f"shortlist_size={ex['shortlist_size']} "
            f"shortlist_hit={ex['shortlist_hit']} "
            f"total_time={ex['total_time']:.6f} "
            f"server_time={ex['server_time']:.6f}"
        )
        for s in ex["selected"]:
            print(f"    start={s['start']} sa_hits={s['sa_hits']} chunks={s['chunks']}")


if __name__ == "__main__":
    main()