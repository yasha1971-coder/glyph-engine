import argparse
import pickle
import random
import struct
import zlib
from collections import defaultdict

import numpy as np


UINT32_MAX = np.uint32(0xFFFFFFFF)


# -----------------------------
# LOADERS
# -----------------------------
def load_glyph_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph magic")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


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
# FRAGMENTS
# -----------------------------
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


# -----------------------------
# HYBRID
# -----------------------------
def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def shortlist_hybrid_scored(filter_obj, frags, top_k):
    dense_inv = filter_obj["dense"]["inv"]
    rare_map = build_rare_map(filter_obj)

    dense_sig_len = filter_obj["dense"].get("sig_len", 8)
    rare_sig_len = filter_obj["rare"].get("sig_len", 8)
    if dense_sig_len != rare_sig_len:
        raise ValueError(f"sig_len mismatch: dense={dense_sig_len} rare={rare_sig_len}")
    sig_len = dense_sig_len

    chunk_scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
        "rare_present": 0,
        "rare_matches": 0,
        "rare_score": 0.0
    })

    for frag in frags:
        frag_seen_chunks = set()
        for i in range(len(frag) - sig_len + 1):
            sig = frag[i:i + sig_len].hex()
            chunk_map = dense_inv.get(sig)
            if not chunk_map:
                continue
            for chunk_id_key, cnt in chunk_map.items():
                cid = int(chunk_id_key)
                chunk_scores[cid]["dense_total"] += cnt
                frag_seen_chunks.add(cid)
        for cid in frag_seen_chunks:
            chunk_scores[cid]["dense_present"] += 1

    for frag in frags:
        frag_seen_chunks = set()
        for i in range(len(frag) - sig_len + 1):
            sig = frag[i:i + sig_len].hex()
            for cid, cmap in rare_map.items():
                hit = cmap.get(sig)
                if hit is None:
                    continue
                cnt, df = hit
                score = 1.0 / (df + 1.0)
                chunk_scores[cid]["rare_score"] += score * cnt
                chunk_scores[cid]["rare_matches"] += 1
                frag_seen_chunks.add(cid)
        for cid in frag_seen_chunks:
            chunk_scores[cid]["rare_present"] += 1

    scored = []
    for cid, s in chunk_scores.items():
        scored.append((
            cid,
            s["rare_present"],
            s["dense_present"],
            s["rare_score"],
            s["dense_total"],
            s["rare_matches"]
        ))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], x[0]))
    shortlist = [cid for cid, *_ in scored[:top_k]]
    score_map = {cid: tuple(row) for cid, *row in scored}
    return shortlist, score_map


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
# FM
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
# GAP
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


def aggregate_gap_scores(hit_dicts):
    gap_support = defaultdict(int)
    raw_support = defaultdict(int)

    for d in hit_dicts:
        for cid, cnt in d.items():
            gap_support[cid] += 1
            raw_support[cid] += cnt

    return gap_support, raw_support


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-filter", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--jump176", required=True)
    ap.add_argument("--jump352", required=True)
    ap.add_argument("--jump528", required=True)
    ap.add_argument("--jump704", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--hybrid-top-k", type=int, default=16)
    ap.add_argument("--gap-top-k", type=int, default=16)
    ap.add_argument("--fusion-top-k", type=int, default=32)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--max-scan", type=int, default=200000)
    args = ap.parse_args()

    with open(args.baseline_filter, "rb") as f:
        baseline = pickle.load(f)
    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    glyph_chunks = load_glyph_chunks(args.glyph)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    raw_chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)

    fm = FMIndex(fm_meta, args.bwt)

    jump176 = np.memmap(args.jump176, dtype=np.uint32, mode="r")
    jump352 = np.memmap(args.jump352, dtype=np.uint32, mode="r")
    jump528 = np.memmap(args.jump528, dtype=np.uint32, mode="r")
    jump704 = np.memmap(args.jump704, dtype=np.uint32, mode="r")

    indexed_ids = sorted(row["chunk_id"] for row in baseline["rare"]["chunks"])
    eligible_ids = [
        cid for cid in indexed_ids
        if cid < len(glyph_chunks)
        and len(glyph_chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
        and cid < len(raw_chunks)
        and len(raw_chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
    ]

    rng = random.Random(args.seed)
    if args.trials >= len(eligible_ids):
        bench_ids = eligible_ids[:args.trials]
    else:
        bench_ids = rng.sample(eligible_ids, args.trials)

    hybrid_hit = 0
    gap_hit = 0
    fusion_hit = 0
    examples = []

    print("=" * 60)
    print(" FUSION DIAGNOSTIC V1")
    print("=" * 60)
    print(f"trials={len(bench_ids)}")

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid)

        # same starts for both representations
        starts = choose_fragment_starts(len(glyph_chunks[qid]), args.frag_len, args.nfrag, args.min_gap, local_rng)
        if starts is None:
            continue

        glyph_frags = [glyph_chunks[qid][s:s + args.frag_len] for s in starts]
        raw_frags = [raw_chunks[qid][s:s + args.frag_len] for s in starts]

        # hybrid
        hybrid_shortlist, hybrid_score_map = shortlist_hybrid_scored(baseline, glyph_frags, args.hybrid_top_k)
        hybrid_ok = qid in hybrid_shortlist
        if hybrid_ok:
            hybrid_hit += 1

        # gap
        ranges = [fm.backward_search(frag) for frag in raw_frags]
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

        gap_support, raw_support = aggregate_gap_scores([p12, p23, p34, p45, p13, p24, p35, p14, p25, p15])
        gap_ranked = sorted(
            [(cid, gap_support[cid], raw_support[cid]) for cid in set(gap_support) | set(raw_support)],
            key=lambda x: (-x[1], -x[2], x[0])
        )
        gap_shortlist = [cid for cid, _, _ in gap_ranked[:args.gap_top_k]]
        gap_ok = qid in gap_shortlist
        if gap_ok:
            gap_hit += 1

        # fusion candidate set
        fusion_candidates = []
        seen = set()
        for cid in hybrid_shortlist + gap_shortlist:
            if cid not in seen:
                seen.add(cid)
                fusion_candidates.append(cid)

        # fusion score
        fusion_rows = []
        for cid in fusion_candidates:
            gsup = gap_support.get(cid, 0)
            rsup = raw_support.get(cid, 0)

            if cid in hybrid_score_map:
                rare_present, dense_present, rare_score, dense_total, rare_matches = hybrid_score_map[cid]
            else:
                rare_present = dense_present = rare_score = dense_total = rare_matches = 0

            fusion_rows.append((
                cid,
                gsup,
                rsup,
                rare_present,
                dense_present,
                rare_score,
                dense_total,
                rare_matches,
            ))

        fusion_rows.sort(
            key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], -x[6], -x[7], x[0])
        )

        fusion_shortlist = [cid for cid, *_ in fusion_rows[:args.fusion_top_k]]
        fusion_reranked = rerank_exact(glyph_chunks, glyph_frags, fusion_shortlist)
        fusion_ok = len(fusion_reranked) > 0 and fusion_reranked[0][0] == qid
        if fusion_ok:
            fusion_hit += 1

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "hybrid_ok": hybrid_ok,
                "gap_ok": gap_ok,
                "fusion_ok": fusion_ok,
                "fusion_top5": fusion_rows[:5],
                "rerank_top5": fusion_reranked[:5],
            })

    n = len(bench_ids)
    print("\nRESULTS:")
    print(f"  hybrid_hit@1 = {hybrid_hit / n:.2%}")
    print(f"  gap_hit@1    = {gap_hit / n:.2%}")
    print(f"  fusion_hit@1 = {fusion_hit / n:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    hybrid_ok={ex['hybrid_ok']} gap_ok={ex['gap_ok']} fusion_ok={ex['fusion_ok']}")
        print(f"    fusion_top5={ex['fusion_top5']}")
        print(f"    rerank_top5={ex['rerank_top5']}")


if __name__ == "__main__":
    main()