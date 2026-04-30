import argparse
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


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


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


def shortlist_fragment_voting(filter_obj, query_frags, top_k):
    sig_len = filter_obj["sig_len"]
    stride = filter_obj["stride"]
    inv = filter_obj["inv"]

    # chunk_id -> [votes per fragment]
    chunk_votes = defaultdict(lambda: [0] * len(query_frags))

    for fi, frag in enumerate(query_frags):
        for sig in iter_signatures(frag, sig_len, stride):
            sig_hex = sig.hex()
            chunk_map = inv.get(sig_hex)
            if not chunk_map:
                continue

            for chunk_id_key, cnt in chunk_map.items():
                chunk_id = int(chunk_id_key)
                chunk_votes[chunk_id][fi] += cnt

    scored = []
    for chunk_id, votes in chunk_votes.items():
        present_fragments = sum(v > 0 for v in votes)
        total_votes = sum(votes)
        max_fragment_votes = max(votes) if votes else 0
        min_fragment_votes = min((v for v in votes if v > 0), default=0)

        scored.append((
            chunk_id,
            present_fragments,
            total_votes,
            max_fragment_votes,
            min_fragment_votes,
            votes,
        ))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], x[0]))
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
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    chunks = load_chunks(args.glyph)
    qid = args.query_chunk_id
    if qid < 0 or qid >= len(chunks):
        raise ValueError(f"query chunk id out of range: {qid}")

    rng = random.Random(args.seed + qid)
    query_chunk = chunks[qid]
    starts, frags = build_fragment_set(
        query_chunk,
        args.frag_len,
        args.nfrag,
        args.min_gap,
        rng,
    )
    if frags is None:
        raise ValueError("query chunk too short for fragment regime")

    scored = shortlist_fragment_voting(filt, frags, args.top_k)

    print("=" * 60)
    print("  CHUNK FILTER QUERY V2 (FRAGMENT-VOTING)")
    print("=" * 60)
    print(f"  filter={args.filter}")
    print(f"  glyph={args.glyph}")
    print(f"  query_chunk_id={qid}")
    print(f"  frag_len={args.frag_len}")
    print(f"  nfrag={args.nfrag}")
    print(f"  min_gap={args.min_gap}")
    print(f"  top_k={args.top_k}")
    print(f"  fragment_starts={starts}")

    print("")
    print("  shortlist:")
    for rank, (chunk_id, present_fragments, total_votes, max_fragment_votes, min_fragment_votes, votes) in enumerate(scored, 1):
        marker = " <== TRUE" if chunk_id == qid else ""
        print(
            f"    #{rank:>2}: chunk={chunk_id} "
            f"present_fragments={present_fragments}/{args.nfrag} "
            f"total_votes={total_votes} "
            f"max_fragment_votes={max_fragment_votes} "
            f"min_fragment_votes={min_fragment_votes} "
            f"votes={votes}{marker}"
        )


if __name__ == "__main__":
    main()