import argparse
import pickle
import random
import struct
from collections import defaultdict

import numpy as np


UINT32_MAX = np.uint32(0xFFFFFFFF)


# -----------------------------
# LOAD RAW CORPUS CHUNKS
# -----------------------------
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


# -----------------------------
# VARIABLE-LENGTH FRAGMENTS
# -----------------------------
def build_variable_fragment_set(chunk, lengths, min_gap, rng):
    total_need = sum(lengths) + (len(lengths) - 1) * min_gap
    if len(chunk) < total_need:
        return None, None

    base = rng.randint(0, len(chunk) - total_need)

    starts = []
    frags = []
    pos = base

    for i, L in enumerate(lengths):
        starts.append(pos)
        frags.append(chunk[pos:pos + L])
        pos += L
        if i + 1 < len(lengths):
            pos += min_gap

    return starts, frags


# -----------------------------
# FM INDEX
# -----------------------------
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


# -----------------------------
# CHUNK MAP BIN
# -----------------------------
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


# -----------------------------
# PAIR GAP
# -----------------------------
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


# -----------------------------
# EXACT RERANK
# -----------------------------
def exact_score(query_frags, candidate_chunk):
    present = 0
    total = 0
    for frag in query_frags:
        cnt = candidate_chunk.count(frag)
        if cnt > 0:
            present += 1
            total += cnt
    return present, total


def rerank_exact(chunks, query_frags, candidates):
    scored = []
    for cid in candidates:
        present, total = exact_score(query_frags, chunks[cid])
        scored.append((cid, present, total))
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=True)

    ap.add_argument("--jump176", required=True)
    ap.add_argument("--jump184", required=True)
    ap.add_argument("--jump192", required=True)
    ap.add_argument("--jump360", required=True)
    ap.add_argument("--jump376", required=True)
    ap.add_argument("--jump552", required=True)
    ap.add_argument("--jump560", required=True)
    ap.add_argument("--jump736", required=True)

    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--max-scan", type=int, default=200000)
    args = ap.parse_args()

    lengths = [48, 56, 64, 56, 48]

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)
    fm = FMIndex(fm_meta, args.bwt)

    jump176 = np.memmap(args.jump176, dtype=np.uint32, mode="r")
    jump184 = np.memmap(args.jump184, dtype=np.uint32, mode="r")
    jump192 = np.memmap(args.jump192, dtype=np.uint32, mode="r")
    jump360 = np.memmap(args.jump360, dtype=np.uint32, mode="r")
    jump376 = np.memmap(args.jump376, dtype=np.uint32, mode="r")
    jump552 = np.memmap(args.jump552, dtype=np.uint32, mode="r")
    jump560 = np.memmap(args.jump560, dtype=np.uint32, mode="r")
    jump736 = np.memmap(args.jump736, dtype=np.uint32, mode="r")

    for arr, name in [
        (jump176, "176"), (jump184, "184"), (jump192, "192"),
        (jump360, "360"), (jump376, "376"),
        (jump552, "552"), (jump560, "560"), (jump736, "736")
    ]:
        if len(arr) != sa_len:
            raise ValueError(f"jump{name} length mismatch")

    eligible_ids = [
        cid for cid in range(min(len(chunks), num_chunks))
        if len(chunks[cid]) >= sum(lengths) + (len(lengths) - 1) * args.min_gap
    ]

    rng = random.Random(args.seed)
    if args.trials >= len(eligible_ids):
        bench_ids = eligible_ids[:args.trials]
    else:
        bench_ids = rng.sample(eligible_ids, args.trials)

    shortlist_hit = 0
    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    mrr = 0.0

    examples = []

    print("=" * 60)
    print(" VARIABLE LEN GAP DIAGNOSTIC V1")
    print("=" * 60)
    print(f" trials={len(bench_ids)}")
    print(f" lengths={lengths}")
    print(f" top_k={args.top_k}")
    print(f" max_scan={args.max_scan}")

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid)
        starts, frags = build_variable_fragment_set(
            chunks[qid],
            lengths,
            args.min_gap,
            local_rng
        )
        if frags is None:
            continue

        ranges = [fm.backward_search(frag) for frag in frags]

        # adjacent
        p12 = pair_gap_hits(ranges[0], ranges[1], jump176, chunk_map, max_scan=args.max_scan)
        p23 = pair_gap_hits(ranges[1], ranges[2], jump184, chunk_map, max_scan=args.max_scan)
        p34 = pair_gap_hits(ranges[2], ranges[3], jump192, chunk_map, max_scan=args.max_scan)
        p45 = pair_gap_hits(ranges[3], ranges[4], jump184, chunk_map, max_scan=args.max_scan)

        # skip-1
        p13 = pair_gap_hits(ranges[0], ranges[2], jump360, chunk_map, max_scan=args.max_scan)
        p24 = pair_gap_hits(ranges[1], ranges[3], jump376, chunk_map, max_scan=args.max_scan)
        p35 = pair_gap_hits(ranges[2], ranges[4], jump376, chunk_map, max_scan=args.max_scan)

        # skip-2
        p14 = pair_gap_hits(ranges[0], ranges[3], jump552, chunk_map, max_scan=args.max_scan)
        p25 = pair_gap_hits(ranges[1], ranges[4], jump560, chunk_map, max_scan=args.max_scan)

        # skip-3
        p15 = pair_gap_hits(ranges[0], ranges[4], jump736, chunk_map, max_scan=args.max_scan)

        ranked = aggregate_scores([p12, p23, p34, p45, p13, p24, p35, p14, p25, p15])
        shortlist = [cid for cid, _gs, _rs in ranked[:args.top_k]]

        if qid in shortlist:
            shortlist_hit += 1

        reranked = rerank_exact(chunks, frags, shortlist)

        rank = None
        for i, (cid, _present, _total) in enumerate(reranked):
            if cid == qid:
                rank = i + 1
                break

        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 4:
                hit4 += 1
            if rank <= 8:
                hit8 += 1
            if rank <= 16:
                hit16 += 1
            mrr += 1.0 / rank

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "ranges": ranges,
                "shortlist_hit": qid in shortlist,
                "rank": rank,
                "top5": ranked[:5],
                "rerank_top5": reranked[:5],
            })

    n = len(bench_ids)

    print("\nRESULTS:")
    print(f"  variable_shortlist_hit@{args.top_k} = {shortlist_hit / n:.2%}")
    print(f"  variable_hit@1  = {hit1 / n:.2%}")
    print(f"  variable_hit@4  = {hit4 / n:.2%}")
    print(f"  variable_hit@8  = {hit8 / n:.2%}")
    print(f"  variable_hit@16 = {hit16 / n:.2%}")
    print(f"  variable_MRR    = {mrr / n:.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} shortlist_hit={ex['shortlist_hit']} rank={ex['rank']}")
        print(f"    ranges={ex['ranges']}")
        print(f"    top5={ex['top5']}")
        print(f"    rerank_top5={ex['rerank_top5']}")


if __name__ == "__main__":
    main()