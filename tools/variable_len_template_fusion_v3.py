import argparse
import pickle
import random
import struct
from collections import defaultdict

import numpy as np


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
# VARIABLE FRAGMENTS WITH SHARED BASE
# -----------------------------
def choose_base_start(chunk_len, total_need, rng):
    if chunk_len < total_need:
        return None
    return rng.randint(0, chunk_len - total_need)


def build_variable_fragment_set_from_base(chunk, lengths, min_gap, base):
    total_need = sum(lengths) + (len(lengths) - 1) * min_gap
    if base is None or base < 0 or base + total_need > len(chunk):
        return None, None

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
        return l, r


# -----------------------------
# CHUNK MAP
# -----------------------------
def load_chunk_map_payload(chunk_map_bin):
    with open(chunk_map_bin, "rb") as f:
        magic = f.read(8)
        if magic != b"CHMAPV1\x00":
            raise ValueError("bad chunk_map magic")

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
# GAP MATCH
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
def exact_score(frags, chunk):
    present = 0
    total = 0
    for f in frags:
        c = chunk.count(f)
        if c > 0:
            present += 1
            total += c
    return present, total


# -----------------------------
# TEMPLATE EVAL
# -----------------------------
def eval_template(ranges, chunk_map, jumps, max_scan):
    jump176, jump184, jump192, jump360, jump376, jump552, jump560, jump736 = jumps

    p12 = pair_gap_hits(ranges[0], ranges[1], jump176, chunk_map, max_scan)
    p23 = pair_gap_hits(ranges[1], ranges[2], jump184, chunk_map, max_scan)
    p34 = pair_gap_hits(ranges[2], ranges[3], jump192, chunk_map, max_scan)
    p45 = pair_gap_hits(ranges[3], ranges[4], jump184, chunk_map, max_scan)

    p13 = pair_gap_hits(ranges[0], ranges[2], jump360, chunk_map, max_scan)
    p24 = pair_gap_hits(ranges[1], ranges[3], jump376, chunk_map, max_scan)
    p35 = pair_gap_hits(ranges[2], ranges[4], jump376, chunk_map, max_scan)

    p14 = pair_gap_hits(ranges[0], ranges[3], jump552, chunk_map, max_scan)
    p25 = pair_gap_hits(ranges[1], ranges[4], jump560, chunk_map, max_scan)

    p15 = pair_gap_hits(ranges[0], ranges[4], jump736, chunk_map, max_scan)

    return aggregate_scores([p12, p23, p34, p45, p13, p24, p35, p14, p25, p15])


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

    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--fusion-top-k", type=int, default=32)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument("--show", type=int, default=5)

    args = ap.parse_args()

    T1 = [48, 56, 64, 56, 48]
    T2 = [64, 48, 56, 64, 48]
    min_gap = 128

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size, num_chunks)
    fm = FMIndex(fm_meta, args.bwt)

    jump176 = np.memmap(args.jump176, dtype=np.uint32, mode="r")
    jump184 = np.memmap(args.jump184, dtype=np.uint32, mode="r")
    jump192 = np.memmap(args.jump192, dtype=np.uint32, mode="r")
    jump360 = np.memmap(args.jump360, dtype=np.uint32, mode="r")
    jump376 = np.memmap(args.jump376, dtype=np.uint32, mode="r")
    jump552 = np.memmap(args.jump552, dtype=np.uint32, mode="r")
    jump560 = np.memmap(args.jump560, dtype=np.uint32, mode="r")
    jump736 = np.memmap(args.jump736, dtype=np.uint32, mode="r")

    jumps = (jump176, jump184, jump192, jump360, jump376, jump552, jump560, jump736)

    need = max(sum(T1) + (len(T1) - 1) * min_gap, sum(T2) + (len(T2) - 1) * min_gap)
    eligible_ids = [cid for cid in range(num_chunks) if len(chunks[cid]) >= need]

    rng = random.Random(args.seed)
    bench_ids = rng.sample(eligible_ids, args.trials)

    t1_hit1 = 0
    t2_hit1 = 0
    fusion_shortlist_hit = 0
    fusion_hit1 = 0
    examples = []

    print("=" * 60)
    print(" VARIABLE LEN TEMPLATE FUSION V3")
    print("=" * 60)
    print(f"trials={len(bench_ids)}")
    print(f"T1={T1}")
    print(f"T2={T2}")
    print("shared_base=True weighted_fusion=True")

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid + 7777)
        base = choose_base_start(len(chunks[qid]), need, local_rng)
        if base is None:
            continue

        starts1, frags1 = build_variable_fragment_set_from_base(chunks[qid], T1, min_gap, base)
        starts2, frags2 = build_variable_fragment_set_from_base(chunks[qid], T2, min_gap, base)
        if frags1 is None or frags2 is None:
            continue

        ranges1 = [fm.backward_search(f) for f in frags1]
        ranges2 = [fm.backward_search(f) for f in frags2]

        ranked1 = eval_template(ranges1, chunk_map, jumps, args.max_scan)
        ranked2 = eval_template(ranges2, chunk_map, jumps, args.max_scan)

        shortlist1 = [cid for cid, _, _ in ranked1[:args.top_k]]
        shortlist2 = [cid for cid, _, _ in ranked2[:args.top-k]] if False else [cid for cid, _, _ in ranked2[:args.top_k]]

        # individual rerank
        exact1 = []
        for cid in shortlist1:
            p, t = exact_score(frags1, chunks[cid])
            exact1.append((cid, p, t))
        exact1.sort(key=lambda x: (-x[1], -x[2], x[0]))

        exact2 = []
        for cid in shortlist2:
            p, t = exact_score(frags2, chunks[cid])
            exact2.append((cid, p, t))
        exact2.sort(key=lambda x: (-x[1], -x[2], x[0]))

        if len(exact1) > 0 and exact1[0][0] == qid:
            t1_hit1 += 1
        if len(exact2) > 0 and exact2[0][0] == qid:
            t2_hit1 += 1

        # weighted fusion map
        # T1 stronger than T2 by measured results, so weight T1 more
        r1_map = {cid: (gs, rs) for cid, gs, rs in ranked1[:args.top_k]}
        r2_map = {cid: (gs, rs) for cid, gs, rs in ranked2[:args.top_k]}

        all_cands = set(r1_map) | set(r2_map)
        fusion_ranked = []

        for cid in all_cands:
            gs1, rs1 = r1_map.get(cid, (0, 0))
            gs2, rs2 = r2_map.get(cid, (0, 0))

            # asymmetrical weighting
            score = (
                4.0 * rs1 +
                2.0 * rs2 +
                1.0 * gs1 +
                0.5 * gs2
            )

            votes = int(cid in r1_map) + int(cid in r2_map)
            fusion_ranked.append((cid, score, votes, rs1, rs2, gs1, gs2))

        fusion_ranked.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], -x[6], x[0]))
        fusion_shortlist = [cid for cid, *_ in fusion_ranked[:args.fusion_top_k]]

        if qid in fusion_shortlist:
            fusion_shortlist_hit += 1

        fusion_exact = []
        for cid in fusion_shortlist:
            p1, t1 = exact_score(frags1, chunks[cid])
            p2, t2 = exact_score(frags2, chunks[cid])
            fusion_exact.append((cid, p1 + p2, t1 + t2))
        fusion_exact.sort(key=lambda x: (-x[1], -x[2], x[0]))

        if len(fusion_exact) > 0 and fusion_exact[0][0] == qid:
            fusion_hit1 += 1

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "base": base,
                "starts1": starts1,
                "starts2": starts2,
                "t1_top5": ranked1[:5],
                "t2_top5": ranked2[:5],
                "fusion_top5": fusion_ranked[:5],
                "fusion_exact_top5": fusion_exact[:5],
            })

    n = len(bench_ids)

    print("\nRESULTS:")
    print(f"  T1_hit@1                   = {t1_hit1 / n:.2%}")
    print(f"  T2_hit@1                   = {t2_hit1 / n:.2%}")
    print(f"  fusion_shortlist@{args.fusion_top_k}     = {fusion_shortlist_hit / n:.2%}")
    print(f"  fusion_hit@1               = {fusion_hit1 / n:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} base={ex['base']}")
        print(f"    starts1={ex['starts1']}")
        print(f"    starts2={ex['starts2']}")
        print(f"    T1_top5={ex['t1_top5']}")
        print(f"    T2_top5={ex['t2_top5']}")
        print(f"    fusion_top5={ex['fusion_top5']}")
        print(f"    fusion_exact_top5={ex['fusion_exact_top5']}")


if __name__ == "__main__":
    main()   