import argparse
import itertools
import json
import os
import pickle
import random
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


# ============================================================
# GLOBALS PER WORKER
# ============================================================
G = {}


# ============================================================
# LOADERS
# ============================================================
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


# ============================================================
# FRAGMENT BUILD
# ============================================================
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


# ============================================================
# GAP MATCH
# ============================================================
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


def exact_score(frags, chunk):
    present = 0
    total = 0
    for f in frags:
        c = chunk.count(f)
        if c > 0:
            present += 1
            total += c
    return present, total


# ============================================================
# TEMPLATE DISTANCES
# ============================================================
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


def eval_template(ranges, chunk_map, jumps, max_scan):
    d12, d23, d34, d45 = jumps["adj"]
    d13, d24, d35 = jumps["skip1"]
    d14, d25 = jumps["skip2"]
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


# ============================================================
# WORKER INIT
# ============================================================
def init_worker(config):
    global G
    G = {}

    with open(config["fm"], "rb") as f:
        fm_meta = pickle.load(f)

    chunk_map, sa_len, chunk_size, num_chunks = load_chunk_map_payload(config["chunk_map"])
    chunks = load_raw_chunks(config["corpus"], chunk_size, num_chunks)
    fm = FMIndex(fm_meta, config["bwt"])

    jump_map = {}
    for dist_str, path in config["jumps"].items():
        dist = int(dist_str)
        arr = np.memmap(path, dtype=np.uint32, mode="r")
        if len(arr) != sa_len:
            raise ValueError(f"jump{dist} length mismatch")
        jump_map[dist] = arr

    G["fm"] = fm
    G["chunk_map"] = chunk_map
    G["chunks"] = chunks
    G["jump_map"] = jump_map
    G["seed"] = config["seed"]
    G["top_k"] = config["top_k"]
    G["max_scan"] = config["max_scan"]
    G["bench_ids"] = config["bench_ids"]
    G["min_gap"] = config["min_gap"]


# ============================================================
# WORKER TASK
# ============================================================
def evaluate_template_worker(template):
    fm = G["fm"]
    chunk_map = G["chunk_map"]
    chunks = G["chunks"]
    jump_map = G["jump_map"]
    seed = G["seed"]
    top_k = G["top_k"]
    max_scan = G["max_scan"]
    bench_ids = G["bench_ids"]
    min_gap = G["min_gap"]

    tpl = tuple(template)
    d = template_distances(tpl, min_gap)
    needed = set(d["adj"]) | set(d["skip1"]) | set(d["skip2"]) | {d["skip3"]}

    if not all(x in jump_map for x in needed):
        return {
            "template": list(tpl),
            "valid": False,
            "reason": f"missing jumps: {sorted(needed - set(jump_map.keys()))}"
        }

    jumps = {
        "adj": tuple(jump_map[x] for x in d["adj"]),
        "skip1": tuple(jump_map[x] for x in d["skip1"]),
        "skip2": tuple(jump_map[x] for x in d["skip2"]),
        "skip3": jump_map[d["skip3"]],
    }

    shortlist_hit = 0
    hit1 = 0
    hit4 = 0
    mrr = 0.0

    total_need = sum(tpl) + (len(tpl) - 1) * min_gap

    for qid in bench_ids:
        local_rng = random.Random(seed + qid + 7777)
        base = choose_base_start(len(chunks[qid]), total_need, local_rng)
        if base is None:
            continue

        starts, frags = build_variable_fragment_set_from_base(chunks[qid], tpl, min_gap, base)
        if frags is None:
            continue

        ranges = [fm.backward_search(f) for f in frags]
        ranked = eval_template(ranges, chunk_map, jumps, max_scan)
        shortlist = [cid for cid, _, _ in ranked[:top_k]]

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
        if rank is not None and rank <= 4:
            hit4 += 1
        if rank is not None:
            mrr += 1.0 / rank

    n = len(bench_ids)
    return {
        "template": list(tpl),
        "valid": True,
        "shortlist": shortlist_hit / n,
        "hit1": hit1 / n,
        "hit4": hit4 / n,
        "mrr": mrr / n,
        "distances": d,
    }


# ============================================================
# MAIN
# ============================================================
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

    ap.add_argument("--jump368", required=False)
    ap.add_argument("--jump544", required=False)
    ap.add_argument("--jump728", required=False)
    ap.add_argument("--jump744", required=False)
    ap.add_argument("--jump752", required=False)

    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-scan", type=int, default=200000)
    ap.add_argument("--top-out", type=int, default=20)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out-json", default="/home/glyph/GLYPH_CPP_BACKEND/out/template_batch_results_v1.json")

    args = ap.parse_args()

    # Load once in parent to select bench ids deterministically
    _, _, chunk_size, num_chunks = load_chunk_map_payload(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size, num_chunks)

    min_gap = 128
    allowed_lengths = [48, 56, 64]

    need = 5 * max(allowed_lengths) + 4 * min_gap
    eligible_ids = [cid for cid in range(num_chunks) if len(chunks[cid]) >= need]

    rng = random.Random(args.seed)
    bench_ids = rng.sample(eligible_ids, args.trials)

    jumps = {
        "176": args.jump176,
        "184": args.jump184,
        "192": args.jump192,
        "360": args.jump360,
        "376": args.jump376,
        "552": args.jump552,
        "560": args.jump560,
        "736": args.jump736,
    }
    if args.jump368:
        jumps["368"] = args.jump368
    if args.jump544:
        jumps["544"] = args.jump544
    if args.jump728:
        jumps["728"] = args.jump728
    if args.jump744:
        jumps["744"] = args.jump744
    if args.jump752:
        jumps["752"] = args.jump752

    config = {
        "fm": args.fm,
        "bwt": args.bwt,
        "corpus": args.corpus,
        "chunk_map": args.chunk_map,
        "jumps": jumps,
        "seed": args.seed,
        "top_k": args.top_k,
        "max_scan": args.max_scan,
        "bench_ids": bench_ids,
        "min_gap": min_gap,
    }

    all_templates = list(itertools.product(allowed_lengths, repeat=5))

    print("=" * 60)
    print(" TEMPLATE BATCH RUNNER V1")
    print("=" * 60)
    print(f"trials={args.trials}")
    print(f"workers={args.workers}")
    print(f"candidate_templates={len(all_templates)}")
    print(f"available_jumps={sorted(int(k) for k in jumps.keys())}")

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(config,)) as ex:
        futs = [ex.submit(evaluate_template_worker, tpl) for tpl in all_templates]

        for fut in as_completed(futs):
            row = fut.result()
            if row.get("valid"):
                results.append(row)
            completed += 1
            if completed % 20 == 0 or completed == len(futs):
                print(f"  progress: {completed}/{len(futs)}")

    results.sort(key=lambda x: (-x["hit1"], -x["shortlist"], -x["hit4"], -x["mrr"], x["template"]))

    out = {
        "trials": args.trials,
        "workers": args.workers,
        "available_jumps": sorted(int(k) for k in jumps.keys()),
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nTOP TEMPLATES:")
    for row in results[:args.top_out]:
        print(
            f"  template={row['template']} "
            f"shortlist@{args.top_k}={row['shortlist']:.2%} "
            f"hit@1={row['hit1']:.2%} "
            f"hit@4={row['hit4']:.2%} "
            f"MRR={row['mrr']:.4f}"
        )

    print(f"\nsaved: {args.out_json}")


if __name__ == "__main__":
    main()