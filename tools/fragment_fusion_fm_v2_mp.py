import argparse
import math
import pickle
import random
import statistics
import struct
import time
import zlib
from collections import Counter
from multiprocessing import Pool


_GLOBAL = {}


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

    def lf(self, i: int) -> int:
        c = self.bwt[i]
        return self.C[c] + self.rank(c, i)


class SampledLocate:
    def __init__(self, fm: FMIndex, locate_obj: dict):
        self.fm = fm
        self.sample_step = locate_obj["sample_step"]
        self.sampled_sa = locate_obj["sampled_sa"]

    def locate_one(self, i: int):
        steps = 0
        cur = i
        n = self.fm.n
        while cur not in self.sampled_sa:
            cur = self.fm.lf(cur)
            steps += 1
        return (self.sampled_sa[cur] + steps) % n, steps

    def locate_range(self, l: int, r: int):
        pos = []
        total_steps = 0
        max_steps = 0
        for i in range(l, r):
            p, s = self.locate_one(i)
            pos.append(p)
            total_steps += s
            if s > max_steps:
                max_steps = s
        return pos, total_steps, max_steps


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


def init_worker(glyph, prefix, sample_step):
    chunks = load_chunks(glyph)
    with open(prefix + ".bwt.bin", "rb") as f:
        bwt = f.read()
    with open(prefix + ".fm.pkl", "rb") as f:
        fm_obj = pickle.load(f)
    with open(prefix + ".meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(prefix + f".locate_s{sample_step}.pkl", "rb") as f:
        locate_obj = pickle.load(f)

    _GLOBAL["chunks"] = chunks
    _GLOBAL["chunk_starts"] = meta["chunk_starts"]
    _GLOBAL["fm"] = FMIndex(bwt, fm_obj)
    _GLOBAL["locator"] = SampledLocate(_GLOBAL["fm"], locate_obj)


def run_one_trial(args):
    trial_idx, base_seed, frag_len, nfrag, min_gap = args
    rng = random.Random(base_seed + trial_idx)

    chunks = _GLOBAL["chunks"]
    chunk_starts = _GLOBAL["chunk_starts"]
    fm = _GLOBAL["fm"]
    locator = _GLOBAL["locator"]

    eligible = [i for i, ch in enumerate(chunks) if len(ch) >= nfrag * frag_len + (nfrag - 1) * min_gap]
    ci = rng.choice(eligible)
    ch = chunks[ci]
    starts = choose_fragment_starts(len(ch), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None

    counts_list = []
    total_interval = 0
    total_positions = 0
    total_locate_steps = 0
    trial_max_steps = 0

    for s in starts:
        q = bytes((b + 1) for b in ch[s:s + frag_len])
        l, r = fm.backward_search(q)
        total_interval += (r - l)

        positions, locate_steps, max_steps = locator.locate_range(l, r)
        counts = positions_to_counts(positions, chunk_starts, chunks)
        counts_list.append(counts)

        total_positions += len(positions)
        total_locate_steps += locate_steps
        trial_max_steps = max(trial_max_steps, max_steps)

    ranked, scored = rank_multi_fast(counts_list, len(chunks))

    conf = None
    label = None
    top = None
    present = None
    mn = None
    wtotal = None
    if scored:
        conf, label = confidence_from_top(scored[0], nfrag)
        top = scored[0][0]
        present = scored[0][1]
        mn = scored[0][3]
        wtotal = scored[0][4]

    return {
        "true": ci,
        "ranked": ranked,
        "total_interval": total_interval,
        "total_positions": total_positions,
        "candidate_chunks": len(ranked),
        "locate_steps": total_locate_steps,
        "locate_max": trial_max_steps,
        "conf": conf,
        "label": label,
        "top": top,
        "present": present,
        "mn": mn,
        "wtotal": wtotal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sample-step", type=int, default=32)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--show-examples", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60)
    print("  FM V2 MP LOAD")
    print("=" * 60)
    print(f"  glyph={args.glyph}")
    print(f"  prefix={args.prefix}")
    print(f"  sample_step={args.sample_step}")
    print(f"  trials={args.trials}")
    print(f"  workers={args.workers}")
    print("")

    t0 = time.perf_counter()
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.glyph, args.prefix, args.sample_step),
    ) as pool:
        work = [(i, args.seed, args.frag_len, args.nfrag, args.min_gap) for i in range(args.trials)]
        results = pool.map(run_one_trial, work)
    elapsed = time.perf_counter() - t0

    results = [r for r in results if r is not None]

    hits = {k: 0 for k in (1, 4, 8, 16, 32, 64)}
    avg_interval_total = []
    avg_total_positions = []
    avg_candidate_chunks = []
    avg_locate_steps = []
    avg_locate_max = []
    conf_values = []
    examples = []

    for r in results:
        avg_interval_total.append(r["total_interval"])
        avg_total_positions.append(r["total_positions"])
        avg_candidate_chunks.append(r["candidate_chunks"])
        avg_locate_steps.append(r["locate_steps"])
        avg_locate_max.append(r["locate_max"])

        if r["conf"] is not None:
            conf_values.append(r["conf"])
            if len(examples) < args.show_examples:
                examples.append(r)

        for k in hits:
            if r["true"] in r["ranked"][:k]:
                hits[k] += 1

    print("=" * 60)
    print("  FRAGMENT FUSION FM V2 MP (TRUE BACKEND + SAMPLED LOCATE)")
    print("=" * 60)
    print(f"  completed_trials={len(results)}")
    print(f"  elapsed_sec={elapsed:.3f}")
    print(f"  sec_per_trial={elapsed / max(1, len(results)):.6f}")
    print(f"  avg_interval_total={statistics.mean(avg_interval_total):.2f}")
    print(f"  avg_total_positions={statistics.mean(avg_total_positions):.2f}")
    print(f"  avg_candidate_chunks={statistics.mean(avg_candidate_chunks):.2f}")
    print(f"  avg_locate_steps={statistics.mean(avg_locate_steps):.2f}")
    print(f"  avg_locate_max={statistics.mean(avg_locate_max):.2f}")
    if conf_values:
        print(f"  avg_confidence={statistics.mean(conf_values):.3f}")
    for k in (1, 4, 8, 16, 32, 64):
        print(f"  top{k}: {100.0 * hits[k] / max(1, len(results)):.1f}%")

    if examples:
        print("")
        print("  examples:")
        for ex in examples:
            print(
                f"    true={ex['true']} top={ex['top']} "
                f"present={ex['present']}/{args.nfrag} mn={ex['mn']} "
                f"wt={round(ex['wtotal'], 3) if ex['wtotal'] is not None else None} "
                f"conf={round(ex['conf'], 3) if ex['conf'] is not None else None} {ex['label']}"
            )


if __name__ == "__main__":
    main()