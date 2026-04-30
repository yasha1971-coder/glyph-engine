import argparse
import struct

def read_u32(f):
    data = f.read(4)
    if len(data) != 4:
        raise EOFError("cannot read u32")
    return struct.unpack("<I", data)[0]

def read_u64(f):
    data = f.read(8)
    if len(data) != 8:
        raise EOFError("cannot read u64")
    return struct.unpack("<Q", data)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", required=True)
    args = ap.parse_args()

    with open(args.response, "rb") as f:
        magic = f.read(4)
        if magic != b"RES1":
            raise ValueError(f"bad magic: {magic!r}")

        num_ranges = read_u32(f)
        print("=" * 60)
        print("  READ RESPONSE V1")
        print("=" * 60)
        print(f"  response={args.response}")
        print(f"  num_ranges={num_ranges}")

        for i in range(num_ranges):
            count = read_u64(f)
            total_steps = read_u64(f)
            max_steps = read_u64(f)
            pos = [read_u64(f) for _ in range(count)]

            print("")
            print(f"  range[{i}]")
            print(f"    count={count}")
            print(f"    total_steps={total_steps}")
            print(f"    max_steps={max_steps}")
            print(f"    positions[:10]={pos[:10]}")

if __name__ == "__main__":
    main()