#!/usr/bin/env python3
import subprocess, time
from pathlib import Path

queries = [q.strip() for q in Path("bench_1gb/queries.txt").read_text().splitlines() if q.strip()]

p = subprocess.Popen(
    ["./build/query_fm_server_v1", "bench_1gb/out/hdfs_1gb.fm.bin", "bench_1gb/out/hdfs_1gb.bwt.bin"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

print("server:", p.stderr.readline().strip())

t0 = time.time()
for q in queries:
    p.stdin.write(q.encode().hex() + "\n")
    p.stdin.flush()
    out = p.stdout.readline()
    if not out:
        raise RuntimeError("no output")

dt = time.time() - t0
p.kill()

print("queries:", len(queries))
print("total_time_sec:", round(dt, 6))
print("avg_ms_per_query:", round(dt / len(queries) * 1000, 6))
