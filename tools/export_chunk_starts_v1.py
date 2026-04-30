import argparse
import pickle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    starts = meta["chunk_starts"]

    with open(args.out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(str(x) for x in starts))

    print("saved:", args.out_csv)
    print("num_chunk_starts:", len(starts))
    print("first3:", starts[:3])

if __name__ == "__main__":
    main()