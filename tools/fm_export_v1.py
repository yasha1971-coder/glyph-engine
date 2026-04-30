import argparse
import pickle
import struct
import time
from pathlib import Path


MAGIC_FM = b"FMV1"
MAGIC_LOC = b"LOC1"


def write_u32(f, x):
    f.write(struct.pack("<I", x))


def write_u64(f, x):
    f.write(struct.pack("<Q", x))


def export_fm_core(prefix: str, out_path: str):
    with open(prefix + ".fm.pkl", "rb") as f:
        fm = pickle.load(f)

    C = fm["C"]
    freq = fm["freq"]
    checkpoint_step = fm["checkpoint_step"]
    rank_checkpoints = fm["rank_checkpoints"]
    bwt_bytes = fm["bwt_bytes"]

    num_checkpoints = len(rank_checkpoints)

    with open(out_path, "wb") as f:
        f.write(MAGIC_FM)
        write_u64(f, bwt_bytes)
        write_u32(f, checkpoint_step)
        write_u64(f, num_checkpoints)

        for x in C:
            write_u64(f, int(x))
        for x in freq:
            write_u64(f, int(x))

        for row in rank_checkpoints:
            if len(row) != 256:
                raise ValueError("rank checkpoint row length != 256")
            for x in row:
                write_u64(f, int(x))

    return {
        "bwt_bytes": bwt_bytes,
        "checkpoint_step": checkpoint_step,
        "num_checkpoints": num_checkpoints,
        "C_len": len(C),
        "freq_len": len(freq),
    }


def export_locate_core(prefix: str, sample_step: int, out_path: str):
    with open(prefix + f".locate_s{sample_step}.pkl", "rb") as f:
        loc = pickle.load(f)

    step = loc["sample_step"]
    sa_size = loc["sa_size"]
    sampled_sa = loc["sampled_sa"]

    items = sorted((int(k), int(v)) for k, v in sampled_sa.items())
    sampled_count = len(items)

    with open(out_path, "wb") as f:
        f.write(MAGIC_LOC)
        write_u64(f, sa_size)
        write_u32(f, step)
        write_u64(f, sampled_count)

        for idx, pos in items:
            write_u64(f, idx)
            write_u64(f, pos)

    return {
        "sa_size": sa_size,
        "sample_step": step,
        "sampled_count": sampled_count,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--sample-step", type=int, default=32)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fm_out = out_dir / "fm_core.bin"
    loc_out = out_dir / f"locate_core_s{args.sample_step}.bin"

    print("=" * 60)
    print("  FM EXPORT V1")
    print("=" * 60)
    print(f"  prefix={args.prefix}")
    print(f"  sample_step={args.sample_step}")
    print(f"  out_dir={out_dir}")

    t0 = time.perf_counter()
    fm_info = export_fm_core(args.prefix, str(fm_out))
    t1 = time.perf_counter()
    loc_info = export_locate_core(args.prefix, args.sample_step, str(loc_out))
    t2 = time.perf_counter()

    print("")
    print("  fm_core:")
    for k, v in fm_info.items():
        print(f"    {k}={v}")
    print(f"    path={fm_out}")
    print(f"    export_sec={t1 - t0:.3f}")

    print("")
    print("  locate_core:")
    for k, v in loc_info.items():
        print(f"    {k}={v}")
    print(f"    path={loc_out}")
    print(f"    export_sec={t2 - t1:.3f}")

    print("")
    print(f"  total_sec={t2 - t0:.3f}")


if __name__ == "__main__":
    main()