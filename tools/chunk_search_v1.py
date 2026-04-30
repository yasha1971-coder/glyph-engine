import argparse
import json
import pickle
import struct
import zlib
from collections import Counter


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


def shortlist_chunks(filter_obj, query: bytes, top_k: int):
    sig_len = filter_obj["sig_len"]
    stride = filter_obj["stride"]
    qsig = Counter(iter_signatures(query, sig_len, stride))

    scored = []
    for row in filter_obj["chunks"]:
        chunk_id = row["chunk_id"]
        top_map = {bytes.fromhex(sig_hex): cnt for sig_hex, cnt in row["top_signatures"]}

        overlap = 0
        weighted = 0
        for sig, qcnt in qsig.items():
            if sig in top_map:
                overlap += 1
                weighted += qcnt * top_map[sig]

        scored.append((chunk_id, overlap, weighted))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored[:top_k]


def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = 0
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def count_occurrences(haystack: bytes, needle: bytes) -> int:
    if not needle or len(needle) > len(haystack):
        return 0
    cnt = 0
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            break
        cnt += 1
        start = pos + 1
    return cnt


def rerank_shortlist(manifest_records, shortlist, query_chunk, frag_len, nfrag, min_gap):
    starts = choose_fragment_starts(len(query_chunk), frag_len, nfrag, min_gap)
    if starts is None:
        raise ValueError("query chunk too short for requested fragment regime")

    query_frags = [query_chunk[s:s + frag_len] for s in starts]
    results = []

    for chunk_id, overlap, weighted in shortlist:
        rec = manifest_records[chunk_id]
        raw_path = rec["chunk_dir"] + "/chunk.raw.bin"
        with open(raw_path, "rb") as f:
            raw = f.read()

        frag_counts = [count_occurrences(raw, frag) for frag in query_frags]
        present = sum(c > 0 for c in frag_counts)
        mn = min((c for c in frag_counts if c > 0), default=0)
        total = sum(frag_counts)

        score = (present, mn, total, weighted, overlap)
        results.append({
            "chunk_id": chunk_id,
            "overlap": overlap,
            "weighted": weighted,
            "frag_counts": frag_counts,
            "present": present,
            "mn": mn,
            "total": total,
            "score": score,
        })

    results.sort(key=lambda r: (-r["present"], -r["mn"], -r["total"], -r["weighted"], -r["overlap"], r["chunk_id"]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--query-chunk-id", type=int, required=True)
    ap.add_argument("--shortlist-k", type=int, default=16)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filter_obj = pickle.load(f)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_records = [json.loads(line) for line in f]
    manifest_by_id = {r["chunk_id"]: r for r in manifest_records}

    chunks = load_chunks(args.glyph)
    qid = args.query_chunk_id
    if qid < 0 or qid >= len(chunks):
        raise ValueError(f"query chunk id out of range: {qid}")
    query_chunk = chunks[qid]

    shortlist = shortlist_chunks(filter_obj, query_chunk, args.shortlist_k)
    reranked = rerank_shortlist(manifest_by_id, shortlist, query_chunk, args.frag_len, args.nfrag, args.min_gap)

    print("=" * 60)
    print("  CHUNK SEARCH V1")
    print("=" * 60)
    print(f"  query_chunk_id={qid}")
    print(f"  shortlist_k={args.shortlist_k}")
    print(f"  frag_len={args.frag_len}")
    print(f"  nfrag={args.nfrag}")
    print(f"  min_gap={args.min_gap}")

    print("")
    print("  shortlist (filter stage):")
    for rank, (chunk_id, overlap, weighted) in enumerate(shortlist, 1):
        marker = " <== TRUE" if chunk_id == qid else ""
        print(f"    #{rank:>2}: chunk={chunk_id} overlap={overlap} weighted={weighted}{marker}")

    print("")
    print("  reranked (exact stage):")
    for rank, row in enumerate(reranked[:args.shortlist_k], 1):
        marker = " <== TRUE" if row["chunk_id"] == qid else ""
        print(
            f"    #{rank:>2}: chunk={row['chunk_id']} "
            f"present={row['present']}/{args.nfrag} "
            f"mn={row['mn']} total={row['total']} "
            f"weighted={row['weighted']} overlap={row['overlap']} "
            f"frag_counts={row['frag_counts']}{marker}"
        )


if __name__ == "__main__":
    main()