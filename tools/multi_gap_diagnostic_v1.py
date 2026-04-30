import argparse
import pickle
import random
import struct
from collections import defaultdict

import numpy as np


UINT32_MAX = np.uint32(0xFFFFFFFF)


def load_raw_chunks(corpus_path, chunk_size=16384, limit_chunks=None):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            if limit_chunks is not None and len(chunks) >= limit_chunks:
                break
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragment_set(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    frags = [chunk[s:s + frag_len] for s in starts]
    return starts, frags


class FMIndex:
    def __init__(self, fm_meta, bwt_path):
        self.C = fm_meta["C"]
        self.checkpoint_step = fm_meta["checkpoint_step"]
        self.rank_checkpoints = fm_meta["rank_checkpoints"]
        self.n = fm_meta["corpus_bytes"]

        with open(bwt_path, "rb") as f:
            self.bwt = f.read()

    def occ(self, c, pos):
        if pos <= 0:
            return 0
        if pos > self.n:
            pos = self.n

        step = self.checkpoint_step
        block = pos // step
        offset = pos % step

        if block >= len(self.rank_checkpoints):
            block = len(self.rank_checkpoints) - 1
            offset = pos - block * step

        base = self.rank_checkpoints[block][c]
        start = block * step
        end = start + offset
        extra = self.bwt[start:end].count(c)
        return base + extra

    def backward_search(self, pattern: bytes):
        l = 0
        r = self.n
        for c in reversed(pattern):
            c_int = c if isinstance(c, int) else ord(c)
            l = self.C[c_int] + self.occ(c_int, l)
            r = self.C[c_int] + self.occ(c_int, r)
            if l >= r:
                return 0, 0
        return l, r  # half-open


def load_chunk_map_payload(chunk_map_bin):
    with open(chunk_map_bin, "rb") as f:
        magic = f.read(8)
        if magic != b"CHMAPV1\x00":
            raise ValueError(f"bad chunk_map magic: {magic!r}")
        sa_len = struct.unpack("<Q", f.read(8))[0]
        chunk_size = struct.unpack("<I", f.read(4))[0]
        num_chunks = struct.unpack("<I", f.read(4))[0]

    payload_offset = 24
    mm = np.memmap(
        chunk_map_bin,
        dtype=np.uint32,
        mode="r",
        offset=payload_offset,
        shape=(sa_len,),
    )
    return mm, sa_len, chunk_size, num_chunks


def range_contains(l, r, x):
    return l <= x < r


def pair_gap_hits(range_a, range_b, jump_rank, chunk_map, max_scan=None):
    l_a, r_a = range_a
    l_b, r_b = range_b

    hits = defaultdict(int)

    span = r_a - l_a
    if span <= 0:
        return hits
    if max_scan is not None and span > max_scan:
        return hits

    for rank in range(l_a, r_a):
        jr = int(jump_rank[rank])
        if jr == 0xFFFFFFFF:
            continue
        if range_contains(l_b, r_b, jr):
            cid = int(chunk_map[rank])
            if cid != 0xFFFFFFFF:
                hits[cid] += 1

    return hits


def aggregate_scores(hit_dicts):
    gap_support = defaultdict(int)
    raw_support = defaultdict(int)

    for d in hit_dicts:
        for cid, cnt in d.items():
            gap_support[cid] += 1
            raw_support[cid] += cnt

    ranked = []
    for cid in set(gap_support.keys()) | set(raw_support.keys()):
        ranked.append((cid, gap_support[cid], raw_support[cid]))

    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--jump176", required=True)
    ap.add_argument("--jump352", required=True)
    ap.add_argument("--jump528", required=True)
    ap.add_argument("--jump704", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument(
        "--results-out",
        default="/home/glyph/GLYPH_CPP_BACKEND/tools/gap_results.pkl"
    )
    args = ap.parse_args()

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)

    fm = FMIndex(fm_meta, args.bwt)

    jump176 = np.memmap(args.jump176, dtype=np.uint32, mode="r")
    jump352 = np.memmap(args.jump352, dtype=np.uint32, mode="r")
    jump528 = np.memmap(args.jump528, dtype=np.uint32, mode="r")
    jump704 = np.memmap(args.jump704, dtype=np.uint32, mode="r")

    for arr, name in [(jump176, "176"), (jump352, "352"), (jump528, "528"), (jump704, "704")]:
        if len(arr) != sa_len:
            raise ValueError(f"jump{name} length mismatch")

    eligible_ids = [
        cid for cid in range(min(len(chunks), num_chunks))
        if len(chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
    ]

    rng = random.Random(args.seed)
    if args.trials >= len(eligible_ids):
        bench_ids = eligible_ids[:args.trials]
    else:
        bench_ids = rng.sample(eligible_ids, args.trials)

    hit = 0
    examples = []
    results = []

    print("=" * 60)
    print(" MULTI GAP DIAGNOSTIC V1")
    print("=" * 60)
    print(f" trials={len(bench_ids)}")
    print(f" top_k={args.top_k}")
    print(f" max_scan={args.max_scan}")
    print(f" results_out={args.results_out}")

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid)
        starts, frags = build_fragment_set(
            chunks[qid],
            args.frag_len,
            args.nfrag,
            args.min_gap,
            local_rng
        )
        if frags is None:
            continue

        ranges = [fm.backward_search(frag) for frag in frags]

        p12 = pair_gap_hits(ranges[0], ranges[1], jump176, chunk_map, max_scan=args.max_scan)
        p23 = pair_gap_hits(ranges[1], ranges[2], jump176, chunk_map, max_scan=args.max_scan)
        p34 = pair_gap_hits(ranges[2], ranges[3], jump176, chunk_map, max_scan=args.max_scan)
        p45 = pair_gap_hits(ranges[3], ranges[4], jump176, chunk_map, max_scan=args.max_scan)

        p13 = pair_gap_hits(ranges[0], ranges[2], jump352, chunk_map, max_scan=args.max_scan)
        p24 = pair_gap_hits(ranges[1], ranges[3], jump352, chunk_map, max_scan=args.max_scan)
        p35 = pair_gap_hits(ranges[2], ranges[4], jump352, chunk_map, max_scan=args.max_scan)

        p14 = pair_gap_hits(ranges[0], ranges[3], jump528, chunk_map, max_scan=args.max_scan)
        p25 = pair_gap_hits(ranges[1], ranges[4], jump528, chunk_map, max_scan=args.max_scan)

        p15 = pair_gap_hits(ranges[0], ranges[4], jump704, chunk_map, max_scan=args.max_scan)

        ranked = aggregate_scores([p12, p23, p34, p45, p13, p24, p35, p14, p25, p15])
        shortlist = [cid for cid, _gs, _rs in ranked[:args.top_k]]

        ok = qid in shortlist
        if ok:
            hit += 1

        results.append({
            "qid": qid,
            "starts": starts,
            "top": ranked[:args.top_k],
            "ranges": ranges,
            "hit": ok,
        })

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "ok": ok,
                "top5": ranked[:5],
                "ranges": ranges,
            })

    with open(args.results_out, "wb") as f:
        pickle.dump(results, f)

    print("\nRESULTS:")
    print(f"  multi_gap_shortlist_hit@{args.top_k} = {hit / len(bench_ids):.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} hit={ex['ok']}")
        print(f"    ranges={ex['ranges']}")
        print(f"    top5={ex['top5']}")


if __name__ == "__main__":
    main()