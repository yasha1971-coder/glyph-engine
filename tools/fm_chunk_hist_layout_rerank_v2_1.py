import argparse
import pickle
import struct
import subprocess
import tempfile
import random
from statistics import median
from typing import List, Tuple


# ============================================================
# FAIR SAMPLER
# ============================================================
def choose_fragment_starts(chunk_len: int, frag_len: int, nfrag: int, min_gap: int, rng: random.Random):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def sample_fragments(corpus_path: str, chunk_id: int, frag_len: int = 48, nfrag: int = 5, min_gap: int = 128,
                     seed: int = 42) -> Tuple[List[bytes], List[int]]:
    chunk_size = 16384
    start = chunk_id * chunk_size

    with open(corpus_path, "rb") as f:
        f.seek(start)
        chunk = f.read(chunk_size)

    if len(chunk) < frag_len:
        return [], []

    rng = random.Random(seed + chunk_id)
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return [], []

    frags = [chunk[s:s + frag_len] for s in starts]
    return frags, starts


# ============================================================
# LOAD RAW CHUNKS
# ============================================================
def load_chunks(corpus_path: str, num_chunks: int = 512, chunk_size: int = 16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        for _ in range(num_chunks):
            chunk = f.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
    return chunks


# ============================================================
# FM RANK / BACKWARD SEARCH
# ============================================================
class FMIndex:
    def __init__(self, fm_meta, bwt_path: str):
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

    def occ(self, c: int, pos: int) -> int:
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


# ============================================================
# IPC
# ============================================================
def write_request(path: str, ranges, topk: int):
    with open(path, "wb") as f:
        f.write(b"FMHREQ1\0")
        f.write(struct.pack("<I", len(ranges)))
        f.write(struct.pack("<I", topk))
        for l, r, w in ranges:
            f.write(struct.pack("<I", l))
            f.write(struct.pack("<I", r))
            f.write(struct.pack("<I", w))


def read_response_bytes(data: bytes):
    if data[:8] != b"FMHRES1\0":
        raise ValueError(f"bad response magic: {data[:8]!r}")

    pos = 8
    n = struct.unpack("<I", data[pos:pos + 4])[0]
    pos += 4

    rows = []
    for _ in range(n):
        cid = struct.unpack("<I", data[pos:pos + 4])[0]
        pos += 4
        score = struct.unpack("<I", data[pos:pos + 4])[0]
        pos += 4
        rows.append((cid, score))

    return rows


def run_backend(backend: str, chunk_map: str, ranges, topk: int):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        write_request(tmp.name, ranges, topk)
        tmp.seek(0)
        req_bytes = tmp.read()

    proc = subprocess.run(
        [backend, chunk_map],
        input=req_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    return read_response_bytes(proc.stdout)


# ============================================================
# POSITION HELPERS
# ============================================================
def find_all_positions(data: bytes, sub: bytes):
    positions = []
    start = 0
    while True:
        pos = data.find(sub, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    return positions


def mad(values):
    if not values:
        return 10**9
    m = median(values)
    dev = [abs(x - m) for x in values]
    return median(dev)


# ============================================================
# CHAIN SEARCH
# ============================================================
def build_valid_chains(position_lists, expected_gap=176, tol=32, min_chain_len=3):
    n = len(position_lists)
    if n == 0 or not position_lists[0]:
        return []

    chains = [[p] for p in position_lists[0]]

    for frag_idx in range(1, n):
        next_positions = position_lists[frag_idx]
        if not next_positions:
            break

        new_chains = []
        for chain in chains:
            prev = chain[-1]
            for p in next_positions:
                gap = p - prev
                if gap <= 0:
                    continue
                if abs(gap - expected_gap) <= tol:
                    new_chains.append(chain + [p])

        if not new_chains:
            break
        chains = new_chains

    chains = [c for c in chains if len(c) >= min_chain_len]
    return chains


def score_chain(chain, expected_gap=176):
    if len(chain) < 2:
        return len(chain), 10**9, 10**9

    gaps = [chain[i + 1] - chain[i] for i in range(len(chain) - 1)]
    gap_errors = [abs(g - expected_gap) for g in gaps]
    total_gap_error = sum(gap_errors)
    gap_mad = mad(gaps)
    return len(chain), total_gap_error, gap_mad


def exact_score(frags, chunk: bytes):
    present = 0
    total = 0
    for frag in frags:
        c = chunk.count(frag)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank_layout(chunks, frags, shortlist_rows, expected_gap=176, tol=32):
    out = []

    for cid, fm_score in shortlist_rows:
        if cid < 0 or cid >= len(chunks):
            continue

        chunk = chunks[cid]
        position_lists = [find_all_positions(chunk, frag) for frag in frags]
        present, total = exact_score(frags, chunk)

        chains = build_valid_chains(
            position_lists,
            expected_gap=expected_gap,
            tol=tol,
            min_chain_len=3
        )

        best_len = 0
        best_total_gap_error = 10**9
        best_gap_mad = 10**9
        best_chain = []

        for ch in chains:
            ln, gap_err, gap_mad_val = score_chain(ch, expected_gap=expected_gap)
            cand = (ln, -gap_err, -gap_mad_val)
            best = (best_len, -best_total_gap_error, -best_gap_mad)
            if cand > best:
                best_len = ln
                best_total_gap_error = gap_err
                best_gap_mad = gap_mad_val
                best_chain = ch

        num_chains = len(chains)

        # v2.1 kept, but softened by exact signals
        score_tuple = (
            best_len,
            -best_total_gap_error,
            -best_gap_mad,
            present,
            total,
            -num_chains,
            fm_score,
        )

        out.append({
            "cid": cid,
            "fm_score": fm_score,
            "present": present,
            "total": total,
            "best_len": best_len,
            "best_total_gap_error": best_total_gap_error,
            "best_gap_mad": best_gap_mad,
            "num_chains": num_chains,
            "best_chain": best_chain,
            "score_tuple": score_tuple,
        })

    out.sort(key=lambda x: x["score_tuple"], reverse=True)
    return out


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=int, default=32)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    fm = FMIndex(fm_meta, args.bwt)
    chunks = load_chunks(args.corpus, num_chunks=512)

    expected_gap = 48 + 128

    shortlist_hit = 0
    hit1 = 0
    valid = 0
    empty_ranges = 0
    examples = []

    for _ in range(args.trials):
        q_chunk = random.randint(0, 511)

        frags, starts = sample_fragments(
            args.corpus,
            q_chunk,
            frag_len=48,
            nfrag=5,
            min_gap=128,
            seed=args.seed,
        )

        ranges = []
        for frag in frags:
            l, r = fm.backward_search(frag)
            if r > l:
                ranges.append((l, r, 1))

        if not ranges:
            empty_ranges += 1
            continue

        valid += 1

        rows = run_backend(
            args.backend,
            args.chunk_map,
            ranges,
            args.top_k
        )

        shortlist_rows = rows
        shortlist = [cid for cid, _score in shortlist_rows]

        if q_chunk in shortlist:
            shortlist_hit += 1

        rr = rerank_layout(
            chunks,
            frags,
            shortlist_rows,
            expected_gap=expected_gap,
            tol=args.tol
        )

        if rr and rr[0]["cid"] == q_chunk:
            hit1 += 1

        if len(examples) < 5:
            examples.append({
                "q_chunk": q_chunk,
                "starts": starts,
                "ranges": ranges[:5],
                "shortlist_top5": shortlist_rows[:5],
                "rerank_top5": [
                    (
                        x["cid"],
                        x["fm_score"],
                        x["present"],
                        x["total"],
                        x["best_len"],
                        x["best_total_gap_error"],
                        x["best_gap_mad"],
                        x["num_chains"],
                        x["best_chain"],
                    )
                    for x in rr[:5]
                ],
            })

    print("=" * 60)
    print(" FM CHUNK HIST + LAYOUT RERANK V2.1")
    print("=" * 60)
    print(f" trials={args.trials}")
    print(f" valid_queries={valid}")
    print(f" empty_ranges={empty_ranges}")
    print(f" shortlist_hit@{args.top_k} = {shortlist_hit / max(1, valid):.4f}")
    print(f" hit@1 = {hit1 / max(1, valid):.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  q_chunk={ex['q_chunk']} starts={ex['starts']}")
        print(f"    ranges={ex['ranges']}")
        print(f"    shortlist_top5={ex['shortlist_top5']}")
        print(f"    rerank_top5={ex['rerank_top5']}")


if __name__ == "__main__":
    main()