import argparse
import math
import pickle
import random
import struct
import zlib
from collections import defaultdict


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


def iter_kmers(data: bytes, k: int):
    if len(data) < k:
        return
    for i in range(0, len(data) - k + 1):
        yield data[i:i + k]


def compute_minimizers(data: bytes, k: int, w: int):
    kmers = list(iter_kmers(data, k))
    if not kmers:
        return []

    if len(kmers) <= w:
        return [min(kmers)]

    mins = []
    for i in range(0, len(kmers) - w + 1):
        window = kmers[i:i + w]
        mins.append(min(window))
    return mins


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


def build_chunk_minimizer_maps(filter_obj):
    chunk_maps = {}
    for row in filter_obj["chunks"]:
        cmap = {}
        for tok_hex, cnt, df in row["minimizers"]:
            cmap[tok_hex] = (cnt, df)
        chunk_maps[row["chunk_id"]] = cmap
    return chunk_maps


def shortlist_minimizer(filter_obj, query_frags, top_k, query_max_df):
    N = len(filter_obj["chunks"])
    k = filter_obj["k"]
    w = filter_obj["w"]
    global_df = filter_obj["global_df"]
    chunk_maps = build_chunk_minimizer_maps(filter_obj)

    # chunk_id -> votes per fragment
    chunk_votes = defaultdict(lambda: [0] * len(query_frags))
    chunk_weighted = defaultdict(float)

    for fi, frag in enumerate(query_frags):
        seen = set()
        mins = compute_minimizers(frag, k, w)

        for tok in mins:
            tok_hex = tok.hex()
            if tok_hex in seen:
                continue
            seen.add(tok_hex)

            df = global_df.get(tok_hex)
            if df is None:
                continue
            if df > query_max_df:
                continue

            idf = math.log(1.0 + N / max(1, df))

            for chunk_id, cmap in chunk_maps.items():
                hit = cmap.get(tok_hex)
                if hit is None:
                    continue
                local_cnt, _local_df = hit
                chunk_votes[chunk_id][fi] += 1
                chunk_weighted[chunk_id] += local_cnt * idf

    scored = []
    for chunk_id, votes in chunk_votes.items():
        present_fragments = sum(v > 0 for v in votes)
        total_votes = sum(votes)
        idf_weighted_score = chunk_weighted[chunk_id]
        max_fragment_votes = max(votes) if votes else 0
        min_fragment_votes = min((v for v in votes if v > 0), default=0)

        scored.append((
            chunk_id,
            present_fragments,
            idf_weighted_score,
            total_votes,
            max_fragment_votes,
            min_fragment_votes,
            votes,
        ))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], x[0]))
    return scored[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--query-chunk-id", type=int, required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-max-df", type=int, default=16,
                    help="Ignore overly-common minimizers at query time")
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    chunks = load_chunks(args.glyph)
    qid = args.query_chunk_id
    if qid < 0 or qid >= len(chunks):
        raise ValueError(f"query chunk id out of range: {qid}")

    rng = random.Random(args.seed + qid)
    starts, frags = build_fragment_set(chunks[qid], args.frag_len, args.nfrag, args.min_gap, rng)
    if frags is None:
        raise ValueError("query chunk too short for fragment regime")

    scored = shortlist_minimizer(
        filt,
        frags,
        args.top_k,
        args.query_max_df,
    )

    print("=" * 60)
    print("  CHUNK FILTER QUERY V4 (MINIMIZER)")
    print("=" * 60)
    print(f"  filter={args.filter}")
    print(f"  glyph={args.glyph}")
    print(f"  query_chunk_id={qid}")
    print(f"  frag_len={args.frag_len}")
    print(f"  nfrag={args.nfrag}")
    print(f"  min_gap={args.min_gap}")
    print(f"  top_k={args.top_k}")
    print(f"  query_max_df={args.query_max_df}")
    print(f"  fragment_starts={starts}")

    print("")
    print("  shortlist:")
    for rank, (chunk_id, present_fragments, idf_weighted_score, total_votes,
               max_fragment_votes, min_fragment_votes, votes) in enumerate(scored, 1):
        marker = " <== TRUE" if chunk_id == qid else ""
        print(
            f"    #{rank:>2}: chunk={chunk_id} "
            f"present_fragments={present_fragments}/{args.nfrag} "
            f"idf_weighted_score={idf_weighted_score:.3f} "
            f"total_votes={total_votes} "
            f"max_fragment_votes={max_fragment_votes} "
            f"min_fragment_votes={min_fragment_votes} "
            f"votes={votes}{marker}"
        )


if __name__ == "__main__":
    main()