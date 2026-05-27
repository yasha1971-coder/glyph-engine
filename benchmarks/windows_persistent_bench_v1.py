import subprocess
import time
from pathlib import Path

patterns = [
    "746865",
    "616161",
    "7177787a",
    "74686520616e64",
]

queries = []
for i in range(100):
    queries.append(patterns[i % len(patterns)])

qpath = Path("win_enwik8_queries.txt")
qpath.write_text("\n".join(queries) + "\n", encoding="ascii")

t0 = time.perf_counter()

p = subprocess.run(
    [
        r".\build\Release\query_fm_server_v1.exe",
        r"win_enwik8\fm.bin",
        r"win_enwik8\bwt.bin",
    ],
    input=qpath.read_text(encoding="ascii"),
    text=True,
    capture_output=True,
)

dt = (time.perf_counter() - t0) * 1000.0

out_lines = [x for x in p.stdout.splitlines() if x.strip()]

print("bench: WINDOWS_PERSISTENT_BATCH_BENCH_V1")
print("returncode:", p.returncode)
print("stdout_lines:", len(out_lines))
print("elapsed_ms:", round(dt, 3))
print("queries:", len(queries))
print("avg_ms_per_query_including_startup:", round(dt / len(queries), 4))
print("first_lines:")
for line in out_lines[:6]:
    print(line)