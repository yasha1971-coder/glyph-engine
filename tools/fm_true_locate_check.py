import argparse
import pickle
import random
import struct
import time


def read_i32_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("SA bin size is not divisible by 4")
    n = len(data) // 4
    return list(struct.unpack(f"<{n}i", data))


class FMIndexV1:
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

    def lf(self, i: int) -> int:
        c = self.bwt[i]
        return self.C[c] + self.rank(c, i)


class SampledLocateV1:
    def __init__(self, fm: FMIndexV1, locate_obj: dict):
        self.fm = fm
        self.sample_step = locate_obj["sample_step"]
        self.sampled_sa = locate_obj["sampled_sa"]

    def locate(self, i: int):
        steps = 0
        cur = i
        n = self.fm.n
        while cur not in self.sampled_sa:
            cur = self.fm.lf(cur)
            steps += 1
        return (self.sampled_sa[cur] + steps) % n, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sample-step", type=int, default=32)
    ap.add_argument("--checks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    with open(args.prefix + ".bwt.bin", "rb") as f:
        bwt = f.read()
    with open(args.prefix + ".fm.pkl", "rb") as f:
        fm_obj = pickle.load(f)
    sa = read_i32_bin(args.prefix + ".sa.bin")
    with open(args.prefix + f".locate_s{args.sample_step}.pkl", "rb") as f:
        locate_obj = pickle.load(f)
    t1 = time.perf_counter()

    fm = FMIndexV1(bwt, fm_obj)
    locator = SampledLocateV1(fm, locate_obj)

    rng = random.Random(args.seed)
    indices = [rng.randrange(len(sa)) for _ in range(args.checks)]

    print("=" * 60)
    print("  FM TRUE LOCATE CHECK")
    print("=" * 60)
    print(f"  prefix={args.prefix}")
    print(f"  sample_step={args.sample_step}")
    print(f"  checks={args.checks}")
    print(f"  load_sec={t1 - t0:.3f}")

    ok = 0
    total_steps = 0
    max_steps = 0
    mismatches = []

    t2 = time.perf_counter()
    for idx in indices:
        got, steps = locator.locate(idx)
        want = sa[idx]
        total_steps += steps
        max_steps = max(max_steps, steps)
        if got == want:
            ok += 1
        elif len(mismatches) < 10:
            mismatches.append((idx, want, got, steps))
    t3 = time.perf_counter()

    print(f"  locate_sec={t3 - t2:.3f}")
    print(f"  correct={ok}/{args.checks}")
    print(f"  avg_steps={total_steps / max(1, args.checks):.3f}")
    print(f"  max_steps={max_steps}")

    if mismatches:
        print("")
        print("  mismatches:")
        for idx, want, got, steps in mismatches:
            print(f"    idx={idx} want={want} got={got} steps={steps}")
    else:
        print("")
        print("  all checks passed")


if __name__ == "__main__":
    main()