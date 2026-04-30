#!/usr/bin/env python3
import struct

out = "/home/glyph/GLYPH_CPP_BACKEND/out/core_bin/test_request.bin"
ranges = [
    (53582742, 53582752),
    (64037689, 64037699),
]

with open(out, "wb") as f:
    f.write(b"REQ1")
    f.write(struct.pack("<I", len(ranges)))
    for l, r in ranges:
        f.write(struct.pack("<Q", l))
        f.write(struct.pack("<Q", r))

print("written:", out)
