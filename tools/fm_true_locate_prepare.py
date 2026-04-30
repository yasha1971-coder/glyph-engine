import argparse
import pickle
import struct
import time

def read_i32_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError("SA bin size is not divisible by 4")
    n = len(data) // 4
    return struct.unpack(f"<{n}i", data)

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sample-step", type=int, default=32)
    args = ap.parse_args()

    sa_path = args.prefix + ".sa.bin"
    out_path = args.prefix + f".locate_s{args.sample_step}.pkl"

    t0 = time.perf_counter()
    sa = read_i32_bin(sa_path)
    t1 = time.perf_counter()

    sample_step = max(1, args.sample_step)

    print("=" * 60)
    print("  FM TRUE LOCATE PREPARE")
    print("=" * 60)
    print(f"  prefix={args.prefix}")
    print(f"  sa_size={len(sa):,}")
    print(f"  sample_step={sample_step}")
    print(f"  load_sec={t1 - t0:.3f}")

    sampled = {}
    for i, pos in enumerate(sa):
        if i % sample_step == 0:
            sampled[i] = pos
    t2 = time.perf_counter()

    obj = {
        "sample_step": sample_step,
        "sampled_sa": sampled,
        "sa_size": len(sa),
    }

    print(f"  sampled_entries={len(sampled):,}")
    print(f"  keep_ratio={len(sampled) / len(sa):.6f}")
    print(f"  build_sec={t2 - t1:.3f}")

    save_pickle(out_path, obj)
    t3 = time.perf_counter()

    print(f"  save_sec={t3 - t2:.3f}")
    print(f"  total_sec={t3 - t0:.3f}")
    print("")
    print("  written:")
    print(f"    {out_path}")

if __name__ == "__main__":
    main()