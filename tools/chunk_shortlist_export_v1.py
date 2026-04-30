import argparse
import pickle
import struct


def write_u32(f, x):
    f.write(struct.pack("<I", x))


def write_f64(f, x):
    f.write(struct.pack("<d", x))


def hex_to_bytes(h):
    return bytes.fromhex(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--out-dense", required=True)
    ap.add_argument("--out-rare", required=True)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    dense = filt["dense"]["inv"]
    rare_chunks = filt["rare"]["chunks"]

    # ---------- dense ----------
    # format:
    # magic[8] = DENSEV1\0
    # num_keys u32
    # for each key:
    #   sig[8]
    #   num_postings u32
    #   repeated:
    #       cid u32
    #       cnt u32
    with open(args.out_dense, "wb") as f:
        f.write(b"DENSEV1\0")
        write_u32(f, len(dense))
        for sig_hex, posting_map in dense.items():
            sig = hex_to_bytes(sig_hex)
            if len(sig) != 8:
                continue
            f.write(sig)
            write_u32(f, len(posting_map))
            for cid_key, cnt in posting_map.items():
                write_u32(f, int(cid_key))
                write_u32(f, int(cnt))

    # ---------- rare ----------
    # format:
    # magic[8] = RAREV1\0\0
    # num_chunks u32
    # for each chunk:
    #   cid u32
    #   num_anchors u32
    #   repeated:
    #       sig[8]
    #       cnt u32
    #       df u32
    with open(args.out_rare, "wb") as f:
        f.write(b"RAREV1\0\0")
        write_u32(f, len(rare_chunks))
        for row in rare_chunks:
            cid = int(row["chunk_id"])
            anchors = row.get("rare_anchors", row.get("anchors", []))
            write_u32(f, cid)
            write_u32(f, len(anchors))
            for sig_hex, cnt, df in anchors:
                sig = hex_to_bytes(sig_hex)
                if len(sig) != 8:
                    continue
                f.write(sig)
                write_u32(f, int(cnt))
                write_u32(f, int(df))

    print("=" * 60)
    print(" SHORTLIST EXPORT V1")
    print("=" * 60)
    print(f" filter={args.filter}")
    print(f" dense_keys={len(dense)}")
    print(f" rare_chunks={len(rare_chunks)}")
    print(f" out_dense={args.out_dense}")
    print(f" out_rare={args.out_rare}")


if __name__ == "__main__":
    main()