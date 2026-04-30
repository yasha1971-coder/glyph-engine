import argparse
import math
import pickle
import random
import struct
import zlib
from collections import defaultdict


# =========================
# IO
# =========================
def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        magic = f.read(6)
        if magic != b"GLYPH1":
            raise ValueError(f"Bad glyph magic: {magic!r}")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


# =========================
# SIG ITER
# =========================
def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


# =========================
# FRAGMENTS
# =========================
def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragment_set(chunk: bytes, frag_len: int, nfrag: int, min_gap: int, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    frags = [chunk[s:s + frag_len] for s in starts]
    return starts, frags


# =========================
# DENSE (v2b)
# =========================
def dense_vote(filt_dense, query_frags):
    sig_len = filt_dense["sig_len"]
    stride = filt_dense["stride"]
    inv = filt_dense["inv"]

    chunk_votes = defaultdict(lambda: [0] * len(query_frags))

    for fi, frag in enumerate(query_frags):
        for sig in iter_signatures(frag, sig_len, stride):
            sig_hex = sig.hex()
            chunk_map = inv.get(sig_hex)
            if not chunk_map:
                continue
            for chunk_id_str, cnt in chunk_map.items():
                chunk_id = int(chunk_id_str)
                chunk_votes[chunk_id][fi] += cnt

    dense_stats = {}
    for chunk_id, votes in chunk_votes.items():
        present = sum(v > 0 for v in votes)
        total = sum(votes)
        dense_stats[chunk_id] = (present, total, votes)

    return dense_stats


# =========================
# RARE (v3)
# =========================
def build_rare_maps(filt_rare):
    chunk_anchor_maps = {}
    for row in filt_rare["chunks"]:
        amap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            amap[sig_hex] = (cnt, df)
        chunk_anchor_maps[row["chunk_id"]] = amap
    return chunk_anchor_maps


def rare_vote(filt_rare, query_frags, chunk_anchor_maps):
    sig_len = filt_rare["sig_len"]
    stride = filt_rare["stride"]
    global_df = filt_rare["global_df"]
    N = len(filt_rare["chunks"])

    frag_votes = defaultdict(lambda: [0] * len(query_frags))
    weighted = defaultdict(float)

    for fi, frag in enumerate(query_frags):
        seen = set()

        for sig in iter_signatures(frag, sig_len, stride):
            sig_hex = sig.hex()
            if sig_hex in seen:
                continue
            seen.add(sig_hex)

            df = global_df.get(sig_hex)
            if df is None:
                continue

            idf = math.log(1.0 + N / max(1, df))

            for chunk_id, amap in chunk_anchor_maps.items():
                hit = amap.get(sig_hex)
                if hit is None:
                    continue
                local_cnt, _ = hit
                frag_votes[chunk_id][fi] += 1
                weighted[chunk_id] += local_cnt * idf

    rare_stats = {}
    for chunk_id, votes in frag_votes.items():
        present = sum(v > 0 for v in votes)
        matches = sum(votes)
        score = weighted[chunk_id]
        rare_stats[chunk_id] = (present, matches, score, votes)

    return rare_stats


# =========================
# HYBRID SHORTLIST
# =========================
def hybrid_shortlist(filt_dense, filt_rare, query_frags, top_k):
    dense_stats = dense_vote(filt_dense, query_frags)
    chunk_anchor_maps = build_rare_maps(filt_rare)
    rare_stats = rare_vote(filt_rare, query_frags, chunk_anchor_maps)

    all_chunks = set(dense_stats.keys()) | set(rare_stats.keys())

    scored = []

    for cid in all_chunks:
        # dense
        d_present, d_total, d_votes = dense_stats.get(cid, (0, 0, [0]*len(query_frags)))

        # rare
        r_present, r_matches, r_score, r_votes = rare_stats.get(cid, (0, 0, 0.0, [0]*len(query_frags)))

        scored.append((
            cid,
            r_present,
            d_present,
            r_score,
            d_total,
            r_matches,
            d_votes,
            r_votes
        ))

    scored.sort(key=lambda x: (
        -x[1],  # rare present
        -x[2],  # dense present
        -x[3],  # rare score
        -x[4],  # dense total
        -x[5],  # rare matches
        x[0]
    ))

    return scored[:top_k]


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-dense", required=True)
    ap.add_argument("--filter-rare", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--query-chunk-id", type=int, required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.filter_dense, "rb") as f:
        filt_dense = pickle.load(f)

    with open(args.filter_rare, "rb") as f:
        filt_rare = pickle.load(f)

    chunks = load_chunks(args.glyph)
    qid = args.query_chunk_id

    rng = random.Random(args.seed + qid)
    query_chunk = chunks[qid]

    starts, frags = build_fragment_set(
        query_chunk, args.frag_len, args.nfrag, args.min_gap, rng
    )

    if frags is None:
        raise ValueError("query chunk too short")

    scored = hybrid_shortlist(filt_dense, filt_rare, frags, args.top_k)

    print("=" * 60)
    print("  CHUNK FILTER QUERY HYBRID V1")
    print("=" * 60)
    print(f"  query_chunk_id={qid}")
    print(f"  fragment_starts={starts}")

    print("\n  shortlist:")
    for rank, (cid, r_present, d_present, r_score, d_total, r_matches, d_votes, r_votes) in enumerate(scored, 1):
        marker = " <== TRUE" if cid == qid else ""
        print(
            f"    #{rank:>2}: chunk={cid} "
            f"rare_present={r_present}/{args.nfrag} "
            f"dense_present={d_present}/{args.nfrag} "
            f"rare_score={r_score:.2f} "
            f"dense_total={d_total} "
            f"rare_matches={r_matches} "
            f"{marker}"
        )


if __name__ == "__main__":
    main()