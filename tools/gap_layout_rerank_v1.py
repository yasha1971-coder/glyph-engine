import argparse
import pickle
import struct
import zlib


# -----------------------------
# LOAD GLYPH
# -----------------------------
def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph magic")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


# -----------------------------
# EXACT POSITIONS
# -----------------------------
def find_all_positions(chunk: bytes, frag: bytes):
    pos = []
    start = 0
    while True:
        i = chunk.find(frag, start)
        if i == -1:
            break
        pos.append(i)
        start = i + 1
    return pos


def present_total_from_positions(pos_lists):
    present = 0
    total = 0
    for lst in pos_lists:
        if lst:
            present += 1
            total += len(lst)
    return present, total


# -----------------------------
# CHAIN SEARCH
# -----------------------------
def best_chain(pos_lists, target_gap=176, tol=32):
    """
    Greedy DP-like longest chain:
    choose p0 from frag0 positions, then the nearest valid next position
    within [prev + target_gap - tol, prev + target_gap + tol].
    Returns:
      best_len, pair_matches, gap_error_sum, best_chain_positions
    """
    if not pos_lists or not pos_lists[0]:
        return 0, 0, 10**9, []

    best_len = 0
    best_pair_matches = 0
    best_gap_error_sum = 10**9
    best_chain_positions = []

    for start_pos in pos_lists[0]:
        chain = [start_pos]
        pair_matches = 0
        gap_error_sum = 0
        prev = start_pos

        for k in range(1, len(pos_lists)):
            cand = pos_lists[k]
            lo = prev + target_gap - tol
            hi = prev + target_gap + tol

            best_next = None
            best_err = None

            # linear scan is okay here: shortlist only 16, positions typically small enough
            for p in cand:
                if p < lo:
                    continue
                if p > hi:
                    break
                err = abs((p - prev) - target_gap)
                if best_next is None or err < best_err or (err == best_err and p < best_next):
                    best_next = p
                    best_err = err

            if best_next is None:
                break

            chain.append(best_next)
            pair_matches += 1
            gap_error_sum += best_err
            prev = best_next

        chain_len = len(chain)

        if (
            chain_len > best_len
            or (chain_len == best_len and pair_matches > best_pair_matches)
            or (chain_len == best_len and pair_matches == best_pair_matches and gap_error_sum < best_gap_error_sum)
        ):
            best_len = chain_len
            best_pair_matches = pair_matches
            best_gap_error_sum = gap_error_sum
            best_chain_positions = chain

    return best_len, best_pair_matches, best_gap_error_sum, best_chain_positions


def skip_pair_matches(chain_positions, target_gap=176, tol=32):
    """
    Count skip-gap confirmations inside the best chain:
      0->2 = 352
      1->3 = 352
      2->4 = 352
      0->3 = 528
      1->4 = 528
      0->4 = 704
    """
    n = len(chain_positions)
    if n < 3:
        return 0

    score = 0

    def ok(i, j):
        d = chain_positions[j] - chain_positions[i]
        target = (j - i) * target_gap
        return abs(d - target) <= tol * (j - i)

    pairs = [
        (0, 2), (1, 3), (2, 4),
        (0, 3), (1, 4),
        (0, 4),
    ]

    for i, j in pairs:
        if i < n and j < n and ok(i, j):
            score += 1

    return score


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=int, default=32)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks(args.glyph)

    with open(args.gap_results, "rb") as f:
        gap_data = pickle.load(f)

    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    mrr = 0.0
    executed = 0
    examples = []

    print("=" * 60)
    print(" GAP LAYOUT RERANK V1")
    print("=" * 60)
    print(f" top_k={args.top_k}")
    print(f" tol={args.tol}")
    print(f" queries={len(gap_data)}")

    for item in gap_data:
        qid = item["qid"]
        starts = item["starts"]
        gap_top = item["top"][:args.top_k]

        if qid >= len(chunks):
            continue

        query_chunk = chunks[qid]
        frags = [query_chunk[s:s + args.frag_len] for s in starts]

        candidates = [cid for cid, *_ in gap_top]
        if not candidates:
            executed += 1
            continue

        scored = []

        for cid in candidates:
            if cid >= len(chunks):
                continue

            cand_chunk = chunks[cid]
            pos_lists = [find_all_positions(cand_chunk, frag) for frag in frags]

            present, total = present_total_from_positions(pos_lists)
            best_len, pair_matches, gap_error_sum, chain_positions = best_chain(
                pos_lists,
                target_gap=args.frag_len + args.min_gap,
                tol=args.tol
            )
            skip_matches = skip_pair_matches(
                chain_positions,
                target_gap=args.frag_len + args.min_gap,
                tol=args.tol
            )

            scored.append((
                cid,
                best_len,
                pair_matches,
                skip_matches,
                -gap_error_sum,   # larger is better in sort
                present,
                total,
                chain_positions,
            ))

        scored.sort(
            key=lambda x: (
                -x[1],   # best_len desc
                -x[2],   # pair_matches desc
                -x[3],   # skip_matches desc
                -x[4],   # -gap_error_sum desc => smaller error wins
                -x[5],   # present desc
                -x[6],   # total desc
                x[0],    # cid asc
            )
        )

        rank = None
        for i, row in enumerate(scored):
            if row[0] == qid:
                rank = i + 1
                break

        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 4:
                hit4 += 1
            if rank <= 8:
                hit8 += 1
            if rank <= 16:
                hit16 += 1
            mrr += 1.0 / rank

        executed += 1

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "rank": rank,
                "top5": scored[:5],
            })

    print("\nRESULTS:")
    print(f"  gap_layout_hit@1  = {hit1 / executed:.2%}")
    print(f"  gap_layout_hit@4  = {hit4 / executed:.2%}")
    print(f"  gap_layout_hit@8  = {hit8 / executed:.2%}")
    print(f"  gap_layout_hit@16 = {hit16 / executed:.2%}")
    print(f"  gap_layout_MRR    = {mrr / executed:.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} rank={ex['rank']}")
        for row in ex["top5"]:
            cid, best_len, pair_matches, skip_matches, neg_gap_error, present, total, chain_positions = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"    chunk={cid} "
                f"best_len={best_len} "
                f"pair_matches={pair_matches} "
                f"skip_matches={skip_matches} "
                f"gap_error_sum={-neg_gap_error} "
                f"present={present} "
                f"total={total} "
                f"chain={chain_positions}{marker}"
            )


if __name__ == "__main__":
    main()