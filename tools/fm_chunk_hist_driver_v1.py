import argparse
import pickle
import struct
import subprocess
import tempfile
import random
from typing import List, Tuple


# ============================================================
# FAIR SAMPLER (matches old benchmark regime)
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
# FM RANK / BACKWARD SEARCH
# ============================================================
class FMIndex:
    def __init__(self, fm_meta, bwt_path: str):
        self.C = fm_meta["C"]                  # list[256]
        self.checkpoint_step = fm_meta["checkpoint_step"]
        self.rank_checkpoints = fm_meta["rank_checkpoints"]   # list of 256-lists
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

    def backward_search(self, pattern: bytes) -> Tuple[int, int]:
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
# IPC HELPERS
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

    return read_response_bytes(proc.stdout), proc.stderr.decode("utf-8", errors="replace")


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
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.fm, "rb") as f:
        fm_meta = pickle.load(f)

    fm = FMIndex(fm_meta, args.bwt)

    hit = 0
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

        rows, _stderr_text = run_backend(
            args.backend,
            args.chunk_map,
            ranges,
            args.top_k
        )

        shortlist = [cid for cid, _score in rows]

        if q_chunk in shortlist:
            hit += 1

        if len(examples) < 5:
            examples.append({
                "q_chunk": q_chunk,
                "starts": starts,
                "ranges": ranges[:5],
                "shortlist": rows[:5],
            })

    print("=" * 60)
    print(" FM CHUNK HIST DRIVER V1")
    print("=" * 60)
    print(f" trials={args.trials}")
    print(f" valid_queries={valid}")
    print(f" empty_ranges={empty_ranges}")
    print(f" hit@{args.top_k} = {hit / max(1, valid):.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  q_chunk={ex['q_chunk']} starts={ex['starts']}")
        print(f"    ranges={ex['ranges']}")
        print(f"    shortlist_top5={ex['shortlist']}")


if __name__ == "__main__":
    main()