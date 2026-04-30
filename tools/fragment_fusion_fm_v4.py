import argparse
import math
import pickle
import random
import statistics
import struct
import subprocess
import time
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


def offset_to_chunk(offset, chunk_starts, chunks):
    lo, hi = 0, len(chunk_starts)
    while lo < hi:
        mid = (lo + hi) // 2
        start = chunk_starts[mid]
        end = start + len(chunks[mid])
        if offset < start:
            hi = mid
        elif offset >= end:
            lo = mid + 1
        else:
            return mid
    return None


class FMIndex:
    def __init__(self, bwt: bytes, fm_obj: dict):
        self.bwt = bwt
        self.C = fm_obj["C"]
        self.freq = fm_obj["freq"]
        self.step = fm_obj["checkpoint_step"]
        self.rank_ckpts = fm_obj["rank_checkpoints"]
        self.n = len(bwt)

    def rank(self, c: int, pos: int) -> int:
        if pos <= 0:
            return 0
        block = pos // self.step
        offset = pos % self.step
        base = self.rank_ckpts[block][c]
        start = block * self.step
        end = start + offset
        cnt = 0
        for b in self.bwt[start:end]:
            if b == c:
                cnt += 1
        return base + cnt

    def backward_search(self, pattern: bytes):
        if not pattern:
            return 0, len(self.bwt)

        c = pattern[-1]
        l = self.C[c]
        r = self.C[c] + self.freq[c]

        for ch in reversed(pattern[:-1]):
            l = self.C[ch] + self.rank(ch, l)
            r = self.C[ch] + self.rank(ch, r)
            if l >= r:
                return 0, 0
        return l, r


def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def positions_to_counts(positions, chunk_starts, chunks):
    ctr = Counter()
    for p in positions:
        ci = offset_to_chunk(p, chunk_starts, chunks)
        if ci is not None:
            ctr[ci] += 1
    return ctr


def idf_from_hits(num_chunks_hit, total_chunks):
    return math.log(1.0 + total_chunks / max(1, num_chunks_hit))


def rank_multi_fast(counts_list, total_chunks):
    keys = set()
    for c in counts_list:
        keys |= set(c.keys())

    frag_idf = [idf_from_hits(len(c), total_chunks) for c in counts_list]
    scored = []

    for ci in keys:
        vals = [c.get(ci, 0) for c in counts_list]
        present = sum(v > 0 for v in vals)
        mn = min((v for v in vals if v > 0), default=0)
        total = sum(vals)

        weighted_total = 0.0
        weighted_presence = 0.0
        for v, w in zip(vals, frag_idf):
            if v > 0:
                weighted_total += v * w
                weighted_presence += w

        nonzero = [v for v in vals if v > 0]
        balance = (min(nonzero) / max(nonzero)) if len(nonzero) >= 2 and max(nonzero) > 0 else (1.0 if len(nonzero) == 1 else 0.0)

        scored.append((ci, present, weighted_presence, mn, weighted_total, balance, total))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], -x[6], x[0]))
    return [x[0] for x in scored], scored


def confidence_from_top(row, nfrag):
    _ci, present, weighted_presence, mn, weighted_total, balance, total = row
    coverage = present / max(1, nfrag)
    min_support = min(1.0, mn / 4.0)
    weighted = min(1.0, weighted_total / (nfrag * 10.0))
    raw = 0.45 * coverage + 0.25 * min_support + 0.20 * weighted + 0.10 * balance
    if raw >= 0.88:
        label = "high"
    elif raw >= 0.68:
        label = "medium"
    else:
        label = "low"
    return raw, label


class PersistentLocateClient:
    def __init__(self, backend_bin, fm_core_bin, locate_core_bin, bwt_bin):
        self.proc = subprocess.Popen(
            [backend_bin, fm_core_bin, locate_core_bin, bwt_bin],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _read_exact(self, n: int) -> bytes:
        data = self.proc.stdout.read(n)
        if len(data) != n:
            raise RuntimeError(f"short read: wanted {n}, got {len(data)}")
        return data

    def locate_ranges(self, ranges):
        req = bytearray()
        req += b"REQ1"
        req += struct.pack("<I", len(ranges))
        for l, r in ranges:
            req += struct.pack("<Q", l)
            req += struct.pack("<Q", r)

        self.proc.stdin.write(req)
        self.proc.stdin.flush()

        magic = self._read_exact(4)
        if magic != b"RES1":
            raise RuntimeError(f"bad response magic: {magic!r}")

        num_ranges = struct.unpack("<I", self._read_exact(4))[0]
        out = []
        for _ in range(num_ranges):
            count = struct.unpack("<Q", self._read_exact(8))[0]
            total_steps = struct.unpack("<Q", self._read_exact(8))[0]
            max_steps = struct.unpack("<Q", self._read_exact(8))[0]
            positions = [struct.unpack("<Q", self._read_exact(8))[0] for _ in range(count)]
            out.append((positions, total_steps, max_steps))
        return out

    def close(self):
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        stderr = ""
        try:
            stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        ret = self.proc.wait()
        return ret, stderr


def eval_fragment_fusion_fm_v4(
    chunks,
    chunk_starts,
    fm,
    client,
    trials,
    seed,
    frag_len,
    nfrag,
    min_gap,
    topk,
    show_examples,
):
    rng = random.Random(seed)
    eligible = [i for i, ch in enumerate(chunks) if len(ch) >= nfrag * frag_len + (nfrag - 1) * min_gap]
    hits = {k: 0 for k in topk}
    avg_interval_total = []
    avg_total_positions = []
    avg_candidate_chunks = []
    avg_locate_steps = []
    avg_locate_max = []
    conf_values = []
    examples = []

    t0 = time.perf_counter()

    for _ in range(trials):
        ci = rng.choice(eligible)
        ch = chunks[ci]
        starts = choose_fragment_starts(len(ch), frag_len, nfrag, min_gap, rng)
        if starts is None:
            continue

        ranges = []
        for s in starts:
            q = bytes((b + 1) for b in ch[s:s + frag_len])
            l, r = fm.backward_search(q)
            ranges.append((l, r))

        located = client.locate_ranges(ranges)

        counts_list = []
        total_interval = 0
        total_positions = 0
        total_locate_steps = 0
        trial_max_steps = 0

        for (l, r), (positions, locate_steps, max_steps) in zip(ranges, located):
            total_interval += (r - l)
            total_positions += len(positions)
            total_locate_steps += locate_steps
            trial_max_steps = max(trial_max_steps, max_steps)
            counts = positions_to_counts(positions, chunk_starts, chunks)
            counts_list.append(counts)

        ranked, scored = rank_multi_fast(counts_list, len(chunks))

        avg_interval_total.append(total_interval)
        avg_total_positions.append(total_positions)
        avg_candidate_chunks.append(len(ranked))
        avg_locate_steps.append(total_locate_steps)
        avg_locate_max.append(trial_max_steps)

        if scored:
            conf, label = confidence_from_top(scored[0], nfrag)
            conf_values.append(conf)
            if len(examples) < show_examples:
                examples.append({
                    "true": ci,
                    "top": scored[0][0],
                    "present": scored[0][1],
                    "mn": scored[0][3],
                    "wtotal": round(scored[0][4], 3),
                    "conf": round(conf, 3),
                    "label": label,
                })

        for k in topk:
            if ci in ranked[:k]:
                hits[k] += 1

    elapsed = time.perf_counter() - t0

    print("=" * 60)
    print("  FRAGMENT FUSION FM V4 (PERSISTENT C++ LOCATE)")
    print("=" * 60)
    print(f"  trials={trials}")
    print(f"  elapsed_sec={elapsed:.3f}")
    print(f"  sec_per_trial={elapsed / max(1, trials):.6f}")
    print(f"  avg_interval_total={statistics.mean(avg_interval_total):.2f}")
    print(f"  avg_total_positions={statistics.mean(avg_total_positions):.2f}")
    print(f"  avg_candidate_chunks={statistics.mean(avg_candidate_chunks):.2f}")
    print(f"  avg_locate_steps={statistics.mean(avg_locate_steps):.2f}")
    print(f"  avg_locate_max={statistics.mean(avg_locate_max):.2f}")
    if conf_values:
        print(f"  avg_confidence={statistics.mean(conf_values):.3f}")
    for k in topk:
        print(f"  top{k}: {100.0 * hits[k] / max(1, trials):.1f}%")

    if examples:
        print("")
        print("  examples:")
        for ex in examples:
            print(
                f"    true={ex['true']} top={ex['top']} "
                f"present={ex['present']}/{nfrag} mn={ex['mn']} "
                f"wt={ex['wtotal']} conf={ex['conf']} {ex['label']}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--backend-bin", required=True)
    ap.add_argument("--fm-core-bin", required=True)
    ap.add_argument("--locate-core-bin", required=True)
    ap.add_argument("--bwt-bin", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--show-examples", type=int, default=5)
    args = ap.parse_args()

    t0 = time.perf_counter()
    chunks = load_chunks(args.glyph)
    t1 = time.perf_counter()

    with open(args.prefix + ".bwt.bin", "rb") as f:
        bwt = f.read()
    with open(args.prefix + ".fm.pkl", "rb") as f:
        fm_obj = pickle.load(f)
    with open(args.prefix + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    t2 = time.perf_counter()

    chunk_starts = meta["chunk_starts"]
    fm = FMIndex(bwt, fm_obj)

    print("=" * 60)
    print("  FM V4 LOAD")
    print("=" * 60)
    print(f"  glyph={args.glyph}")
    print(f"  prefix={args.prefix}")
    print(f"  backend_bin={args.backend_bin}")
    print(f"  load_chunks_sec={t1 - t0:.3f}")
    print(f"  load_artifacts_sec={t2 - t1:.3f}")
    print("")

    client = PersistentLocateClient(
        args.backend_bin,
        args.fm_core_bin,
        args.locate_core_bin,
        args.bwt_bin,
    )
    try:
        eval_fragment_fusion_fm_v4(
            chunks=chunks,
            chunk_starts=chunk_starts,
            fm=fm,
            client=client,
            trials=args.trials,
            seed=args.seed,
            frag_len=args.frag_len,
            nfrag=args.nfrag,
            min_gap=args.min_gap,
            topk=(1, 4, 8, 16, 32, 64),
            show_examples=args.show_examples,
        )
    finally:
        ret, stderr = client.close()
        print("")
        print(f"  backend_returncode={ret}")
        if stderr.strip():
            print("  backend_stderr:")
            print(stderr.strip())


if __name__ == "__main__":
    main()