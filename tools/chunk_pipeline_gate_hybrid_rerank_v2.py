import argparse
import math
import pickle
import random
import struct
import zlib
from collections import defaultdict


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph")
        _v, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# FRAGMENTS
# ------------------------------------------------------------
def choose_fragment_starts(L, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if L < need:
        return None
    base = rng.randint(0, L - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragments(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    return [chunk[s:s+frag_len] for s in starts], starts


def iter_sigs(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len].hex()


# ------------------------------------------------------------
# WEIGHTED COARSE GATE
# ------------------------------------------------------------
def coarse_gate_v2(query_frags, gate_obj, gate_top_m):
    sig_len = gate_obj["sig_len"]
    stride = gate_obj["stride"]
    N = gate_obj["num_chunks"]
    global_df = gate_obj["global_df"]

    frag_sig_sets = []
    for frag in query_frags:
        frag_sig_sets.append(set(iter_sigs(frag, sig_len, stride)))

    scored = []
    for row in gate_obj["chunks"]:
        cid = row["chunk_id"]
        chunk_sigs = set(row["sigs"])

        coverage = 0
        idf_sum = 0.0

        for sig_set in frag_sig_sets:
            frag_hit = False
            for sig in sig_set:
                if sig in chunk_sigs:
                    frag_hit = True
                    df = global_df.get(sig, N)
                    idf_sum += math.log(1.0 + N / max(1, df))
            if frag_hit:
                coverage += 1

        if coverage > 0:
            score = coverage * 10.0 + idf_sum
            scored.append((cid, coverage, idf_sum, score))

    scored.sort(key=lambda x: (-x[3], -x[1], -x[2], x[0]))
    return [cid for cid, *_ in scored[:gate_top_m]], scored[:gate_top_m]


# ------------------------------------------------------------
# HYBRID SHORTLIST (restricted)
# ------------------------------------------------------------
def build_dense_inv(filter_obj):
    return filter_obj["dense"]["inv"]


def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        anchors = row.get("rare_anchors", row.get("anchors", []))
        for sig_hex, cnt, df in anchors:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def shortlist_hybrid_restricted(filter_obj, frags, top_k, allowed_cids):
    allowed = set(allowed_cids)

    dense_inv = build_dense_inv(filter_obj)
    rare_map = build_rare_map(filter_obj)

    scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
        "rare_present": 0,
        "rare_score": 0.0
    })

    for frag in frags:
        seen_dense = set()
        seen_rare = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()

            inv = dense_inv.get(sig)
            if inv:
                for cid_key, cnt in inv.items():
                    cid = int(cid_key)
                    if cid not in allowed:
                        continue
                    scores[cid]["dense_total"] += cnt
                    seen_dense.add(cid)

            for cid in allowed:
                cmap = rare_map.get(cid)
                if not cmap:
                    continue
                hit = cmap.get(sig)
                if hit:
                    cnt, df = hit
                    scores[cid]["rare_score"] += cnt / (df + 1.0)
                    seen_rare.add(cid)

        for cid in seen_dense:
            scores[cid]["dense_present"] += 1
        for cid in seen_rare:
            scores[cid]["rare_present"] += 1

    ranked = []
    for cid, s in scores.items():
        score = (
            s["rare_present"] * 5.0 +
            s["dense_present"] * 2.0 +
            s["rare_score"]
        )
        ranked.append((cid, score))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ranked[:top_k]]


# ------------------------------------------------------------
# EXACT
# ------------------------------------------------------------
def exact_score(frags, chunk):
    present = 0
    total = 0
    for frag in frags:
        c = chunk.count(frag)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank_exact(chunks, frags, shortlist):
    out = []
    for cid in shortlist:
        p, t = exact_score(frags, chunks[cid])
        out.append((cid, p, t))
    out.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--gate-top-m", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    gate_obj = load_pickle(args.gate)
    filt = load_pickle(args.filter)
    chunks = load_chunks(args.glyph)[:len(filt["rare"]["chunks"])]

    gate_hit = 0
    shortlist_hit = 0
    hit1 = 0
    total = 0

    print("=" * 60)
    print(" GATE V2 + HYBRID + EXACT")
    print("=" * 60)
    print(f" indexed_chunks={len(chunks)}")
    print(f" gate_top_m={args.gate_top_m}")
    print(f" top_k={args.top_k}")

    ids = list(range(len(chunks)))
    if args.trials < len(ids):
        ids = rng.sample(ids, args.trials)

    for qid in ids:
        frags, _starts = build_fragments(
            chunks[qid],
            args.frag_len,
            args.nfrag,
            args.min_gap,
            random.Random(args.seed + qid),
        )
        if frags is None:
            continue

        gated_ids, _gate_rows = coarse_gate_v2(frags, gate_obj, args.gate_top_m)
        if qid in gated_ids:
            gate_hit += 1

        shortlist = shortlist_hybrid_restricted(filt, frags, args.top_k, gated_ids)
        if qid in shortlist:
            shortlist_hit += 1

        rr = rerank_exact(chunks, frags, shortlist)
        if rr and rr[0][0] == qid:
            hit1 += 1

        total += 1

    print()
    print("RESULT:")
    print(f"  trials               = {total}")
    print(f"  gate_hit@{args.gate_top_m:<3}      = {gate_hit / total:.4f}")
    print(f"  shortlist_hit@{args.top_k:<2} = {shortlist_hit / total:.4f}")
    print(f"  hit@1                = {hit1 / total:.4f}")


if __name__ == "__main__":
    main()