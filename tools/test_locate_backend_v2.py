import struct
import subprocess

backend = "/home/glyph/GLYPH_CPP_BACKEND/build/locate_backend_v2"
fm_core = "/home/glyph/GLYPH_CPP_BACKEND/out/core_bin_s16/fm_core.bin"
loc_core = "/home/glyph/GLYPH_CPP_BACKEND/out/core_bin_s16/locate_core_s16.bin"
bwt = "/home/glyph/GLYPH_CPP_BACKEND/out/py_true/corpus2_true_backend.bwt.bin"

proc = subprocess.Popen(
    [backend, fm_core, loc_core, bwt],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

ranges = [
    (53582742, 53582752),
    (64037689, 64037699),
]

req = bytearray()
req += b"REQ1"
req += struct.pack("<I", len(ranges))
for l, r in ranges:
    req += struct.pack("<Q", l)
    req += struct.pack("<Q", r)

proc.stdin.write(req)
proc.stdin.flush()

def read_exact(n):
    data = proc.stdout.read(n)
    if len(data) != n:
        raise RuntimeError(f"short read: wanted {n}, got {len(data)}")
    return data

magic = read_exact(4)
if magic != b"RES1":
    raise RuntimeError(f"bad magic: {magic!r}")

num_ranges = struct.unpack("<I", read_exact(4))[0]
print("num_ranges:", num_ranges)

for i in range(num_ranges):
    count = struct.unpack("<Q", read_exact(8))[0]
    total_steps = struct.unpack("<Q", read_exact(8))[0]
    max_steps = struct.unpack("<Q", read_exact(8))[0]
    pos = [struct.unpack("<Q", read_exact(8))[0] for _ in range(count)]
    print(f"range[{i}] count={count} total_steps={total_steps} max_steps={max_steps} pos[:10]={pos[:10]}")

proc.stdin.close()
stderr = proc.stderr.read().decode("utf-8", errors="replace")
ret = proc.wait()
print("returncode:", ret)
print("stderr:")
print(stderr)