import argparse
import itertools
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
    # Adjacent distances
    d12, d23, d34, d45 = jumps["adj"]
    # Skip-1 distances
    d13, d24, d35 = jumps["skip1"]
    # Skip-2 distances
    d14, d25 = jumps["skip2"]
    # Skip-3 distance
    d15 = jumps["skip3"]

    p12 = pair_gap_hits(ranges[0], ranges[1], d12, chunk_map, max_scan)
    p23 = pair_gap_hits(ranges[1], ranges[2], d23, chunk_map, max_scan)
    p34 = pair_gap_hits(ranges[2], ranges[3], d34, chunk_map, max_scan)
    p45 = pair_gap_hits(ranges[3], ranges[4], d45, chunk_map, max_scan)

    p13 = pair_gap_hits(ranges[0], ranges[2], d13, chunk_map, max_scan)
    p24 = pair_gap_hits(ranges[1], ranges[3], d24, chunk_map, max_scan)
    p35 = pair_gap_hits(ranges[2], ranges[4], d35, chunk_map, max_scan)

    p14 = pair_gap_hits(ranges[0], ranges[3], d14, chunk_map, max_scan)
    p25 = pair_gap_hits(ranges[1], ranges[4], d25, chunk_map, max_scan)

    p15 = pair_gap_hits(ranges[0], ranges[4], d15, chunk_map, max_scan)

    return aggregate_scores([p12, p23, p34, p45, p13, p24, p35, p14, p25, p15])


def template_distances(lengths, min_gap):
    adj = []
    for i in range(4):
        adj.append(lengths[i] + min_gap)

    skip1 = [
        adj[0] + adj[1],
        adj[1] + adj[2],
        adj[2] + adj[3],
    ]
    skip2 = [
        adj[0] + adj[1] + adj[2],
        adj[1] + adj[2] + adj[3],
    ]
    skip3 = adj[0] + adj[1] + adj[2] + adj[3]

    return {
        "adj": tuple(adj),
        "skip1": tuple(skip1),
        "skip2": tuple(skip2),
        "skip3": skip3,
    }


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
    ap.add_argument("--jump368", required=False)
    ap.add_argument("--jump376", required=True)
    ap.add_argument("--jump544", required=False)
    ap.add_argument("--jump552", required=True)
    ap.add_argument("--jump560", required=True)
    ap.add_argument("--jump736", required=True)

    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument("--top-out", type=int, default=15)

    args = ap.parse_args()

    min_gap = 128
    allowed_lengths = [48, 56, 64]

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size, num_chunks)
    fm = FMIndex(fm_meta, args.bwt)

    jump_map = {
        176: np.memmap(args.jump176, dtype=np.uint32, mode="r"),
        184: np.memmap(args.jump184, dtype=np.uint32, mode="r"),
        192: np.memmap(args.jump192, dtype=np.uint32, mode="r"),
        360: np.memmap(args.jump360, dtype=np.uint32, mode="r"),
        376: np.memmap(args.jump376, dtype=np.uint32, mode="r"),
        552: np.memmap(args.jump552, dtype=np.uint32, mode="r"),
        560: np.memmap(args.jump560, dtype=np.uint32, mode="r"),
        736: np.memmap(args.jump736, dtype=np.uint32, mode="r"),
    }

    if args.jump368:
        jump_map[368] = np.memmap(args.jump368, dtype=np.uint32, mode="r")
    if args.jump544:
        jump_map[544] = np.memmap(args.jump544, dtype=np.uint32, mode="r")

    # Validate jump lengths
    for dist, arr in jump_map.items():
        if len(arr) != sa_len:
            raise ValueError(f"jump{dist} length mismatch")

    # Candidate templates
    all_templates = list(itertools.product(allowed_lengths, repeat=5))

    # Keep only templates whose required jumps are available
    valid_templates = []
    for tpl in all_templates:
        d = template_distances(tpl, min_gap)
        needed = set(d["adj"]) | set(d["skip1"]) | set(d["skip2"]) | {d["skip3"]}
        if all(x in jump_map for x in needed):
            valid_templates.append(tuple(tpl))

    need = max(sum(t) + (len(t) - 1) * min_gap for t in valid_templates)
    eligible_ids = [cid for cid in range(num_chunks) if len(chunks[cid]) >= need]

    rng = random.Random(args.seed)
    bench_ids = rng.sample(eligible_ids, args.trials)

    print("=" * 60)
    print(" VARIABLE LEN TEMPLATE SEARCH V1")
    print("=" * 60)
    print(f"trials={len(bench_ids)}")
    print(f"valid_templates={len(valid_templates)}")
    print(f"available_jumps={sorted(jump_map.keys())}")

    results = []

    for tpl in valid_templates:
        d = template_distances(tpl, min_gap)
        jumps = {
            "adj": tuple(jump_map[x] for x in d["adj"]),
            "skip1": tuple(jump_map[x] for x in d["skip1"]),
            "skip2": tuple(jump_map[x] for x in d["skip2"]),
            "skip3": jump_map[d["skip3"]],
        }

        shortlist_hit = 0
        hit1 = 0
        mrr = 0.0

        for qid in bench_ids:
            local_rng = random.Random(args.seed + qid + 7777)
            total_need = sum(tpl) + (len(tpl) - 1) * min_gap
            base = choose_base_start(len(chunks[qid]), total_need, local_rng)
            if base is None:
                continue

            starts, frags = build_variable_fragment_set_from_base(chunks[qid], tpl, min_gap, base)
            if frags is None:
                continue

            ranges = [fm.backward_search(f) for f in frags]
            ranked = eval_template(ranges, chunk_map, jumps, args.max_scan)
            shortlist = [cid for cid, _, _ in ranked[:args.top_k]]

            if qid in shortlist:
                shortlist_hit += 1

            exact_rows = []
            for cid in shortlist:
                p, t = exact_score(frags, chunks[cid])
                exact_rows.append((cid, p, t))
            exact_rows.sort(key=lambda x: (-x[1], -x[2], x[0]))

            rank = None
            for i, row in enumerate(exact_rows):
                if row[0] == qid:
                    rank = i + 1
                    break

            if rank == 1:
                hit1 += 1
            if rank is not None:
                mrr += 1.0 / rank

        n = len(bench_ids)
        results.append({
            "template": tpl,
            "shortlist": shortlist_hit / n,
            "hit1": hit1 / n,
            "mrr": mrr / n,
            "distances": d,
        })

    results.sort(key=lambda x: (-x["hit1"], -x["shortlist"], -x["mrr"], x["template"]))

    print("\nTOP TEMPLATES:")
    for row in results[:args.top_out]:
        print(
            f"  template={list(row['template'])} "
            f"shortlist@{args.top_k}={row['shortlist']:.2%} "
            f"hit@1={row['hit1']:.2%} "
            f"MRR={row['mrr']:.4f} "
            f"dist={row['distances']}"
        )
if __name__ == "__main__":
    main()