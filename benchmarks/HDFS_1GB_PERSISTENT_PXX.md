# HDFS 1GB Persistent FM Benchmark — p50/p95/p99

## Benchmark

Tool:

    benchmarks/persistent_fm_v1.py

Mode:

    persistent_cpp_backend

Corpus:

    bench_1gb/HDFS_1GB.log

Artifacts:

    bench_1gb/out/hdfs_1gb.fm.bin
    bench_1gb/out/hdfs_1gb.bwt.bin

Query set:

    bench_1gb/queries.txt

Query count:

    100

Warm runs:

    10

Warm measurements:

    1000

---

## Result

Backend startup / index load:

    7576.852983 ms

Cold query batch:

    min:   0.012595 ms
    p50:   0.013735 ms
    p95:   0.017700 ms
    p99:   0.021775 ms
    max:   0.048982 ms
    mean:  0.014639 ms

Warm query batch:

    min:   0.008186 ms
    p50:   0.009578 ms
    p95:   0.010399 ms
    p99:   0.010468 ms
    max:   0.013484 ms
    mean:  0.009799 ms

Sample response:

    715386381 715386399 18

---

## Interpretation

This benchmark measures persistent C++ FM backend latency with the index
already loaded into a long-lived process.

It does not include:

- Python startup per query
- manifest verification
- HTTP overhead
- OS-level cold-cache drop
- reboot-level cold mmap behavior

The result shows a stable warm latency envelope on the benchmark machine:

    warm p99 / p50 ≈ 1.09

This is substantially more informative than a single average latency number.

---

## Machine

See:

    benchmarks/MACHINE_SPEC.md
