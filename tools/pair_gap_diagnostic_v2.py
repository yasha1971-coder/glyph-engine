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
# BASELINE HYBRID
# uses raw chunk bytes, but baseline filter is still hybrid substring logic
# -----------------------------
def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def shortlist_hybrid(filter_obj, frags, top_k):
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

    # dense
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

    # rare
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
    return [cid for cid, *_ in scored[:top_k]]


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

        if len(self.bwt) != fm_meta["bwt_bytes"]:
            raise ValueError(
                f"BWT size mismatch: loaded={len(self.bwt)} meta={fm_meta['bwt_bytes']}"
            )

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
        return l, r  # half-open [l, r)


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


def aggregate_pair_scores(pair_hit_dicts):
    pair_support = defaultdict(int)
    raw_support = defaultdict(int)

    for d in pair_hit_dicts:
        for cid, cnt in d.items():
            pair_support[cid] += 1
            raw_support[cid] += cnt

    ranked = []
    for cid in set(pair_support.keys()) | set(raw_support.keys()):
        ranked.append((cid, pair_support[cid], raw_support[cid]))

    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return ranked


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-filter", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--jump-bin", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--baseline-top-k", type=int, default=16)
    ap.add_argument("--pair-top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--max-scan", type=int, default=200000)
    args = ap.parse_args()

    with open(args.baseline_filter, "rb") as f:
        baseline = pickle.load(f)

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    jump_rank = np.memmap(args.jump_bin, dtype=np.uint32, mode="r")
    if len(jump_rank) != sa_len:
        raise ValueError(f"jump length mismatch: jump={len(jump_rank)} chunk_map={sa_len}")

    chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)
    fm = FMIndex(fm_meta, args.bwt)

    rng = random.Random(args.seed)

    indexed_ids = sorted(row["chunk_id"] for row in baseline["rare"]["chunks"])
    eligible_ids = [
        cid for cid in indexed_ids
        if cid < len(chunks)
        and len(chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
    ]
    if not eligible_ids:
        raise ValueError("no eligible indexed chunks")

    if args.trials >= len(eligible_ids):
        bench_ids = eligible_ids[:args.trials]
    else:
        bench_ids = rng.sample(eligible_ids, args.trials)

    pair_shortlist_hit = 0
    baseline_missed = 0
    recovered = 0
    executed = 0

    examples = []

    print("=" * 60)
    print(" PAIR GAP DIAGNOSTIC V2")
    print("=" * 60)
    print(f" trials={len(bench_ids)}")
    print(f" baseline_top_k={args.baseline_top_k}")
    print(f" pair_top_k={args.pair_top_k}")
    print(f" step={args.frag_len + args.min_gap}")
    print(f" corpus_chunk_size={chunk_size}")
    print(f" max_scan={args.max_scan}")

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

        executed += 1

        baseline_shortlist = shortlist_hybrid(baseline, frags, args.baseline_top_k)
        baseline_hit = qid in baseline_shortlist
        if not baseline_hit:
            baseline_missed += 1

        ranges = [fm.backward_search(frag) for frag in frags]

        pair12 = pair_gap_hits(ranges[0], ranges[1], jump_rank, chunk_map, max_scan=args.max_scan)
        pair23 = pair_gap_hits(ranges[1], ranges[2], jump_rank, chunk_map, max_scan=args.max_scan)
        pair34 = pair_gap_hits(ranges[2], ranges[3], jump_rank, chunk_map, max_scan=args.max_scan)
        pair45 = pair_gap_hits(ranges[3], ranges[4], jump_rank, chunk_map, max_scan=args.max_scan)

        ranked = aggregate_pair_scores([pair12, pair23, pair34, pair45])
        pair_shortlist = [cid for cid, _pair_support, _raw_support in ranked[:args.pair_top_k]]

        pair_hit = qid in pair_shortlist
        if pair_hit:
            pair_shortlist_hit += 1
        if (not baseline_hit) and pair_hit:
            recovered += 1

        if len(examples) < args.show and ((not baseline_hit) or pair_hit):
            examples.append({
                "qid": qid,
                "starts": starts,
                "ranges": ranges,
                "baseline_hit": baseline_hit,
                "pair_hit": pair_hit,
                "pair12_top": sorted(pair12.items(), key=lambda x: (-x[1], x[0]))[:5],
                "pair23_top": sorted(pair23.items(), key=lambda x: (-x[1], x[0]))[:5],
                "pair34_top": sorted(pair34.items(), key=lambda x: (-x[1], x[0]))[:5],
                "pair45_top": sorted(pair45.items(), key=lambda x: (-x[1], x[0]))[:5],
                "pair_top5": ranked[:5],
            })

    if executed == 0:
        raise ValueError("no executed trials")

    print("\nRESULTS:")
    print(f"  pair_shortlist_hit@{args.pair_top_k} = {pair_shortlist_hit / executed:.2%}")
    print(f"  baseline_missed = {baseline_missed}")
    if baseline_missed > 0:
        print(f"  recovered_from_baseline_missed = {recovered / baseline_missed:.2%} ({recovered}/{baseline_missed})")
    else:
        print("  recovered_from_baseline_missed = NA")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    baseline_hit={ex['baseline_hit']} pair_hit={ex['pair_hit']}")
        print(f"    ranges={ex['ranges']}")
        print(f"    pair12_top={ex['pair12_top']}")
        print(f"    pair23_top={ex['pair23_top']}")
        print(f"    pair34_top={ex['pair34_top']}")
        print(f"    pair45_top={ex['pair45_top']}")
        print(f"    pair_top5={ex['pair_top5']}")


if __name__ == "__main__":
    main()