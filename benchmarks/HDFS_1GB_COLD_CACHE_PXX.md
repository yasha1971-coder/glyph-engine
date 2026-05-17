# HDFS 1GB Persistent FM Benchmark — controlled cold-cache runs

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

Warm runs per cold-cache run:

    3

Cache reset before each run:

    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches

---

## Run 1

Backend startup / index load:

    7333.136344 ms

Cold query batch:

    min:   0.012651 ms
    p50:   0.013659 ms
    p95:   0.018244 ms
    p99:   0.022719 ms
    max:   0.052441 ms
    mean:  0.014355 ms

Warm query batch:

    min:   0.009328 ms
    p50:   0.012513 ms
    p95:   0.012763 ms
    p99:   0.014712 ms
    max:   0.047950 ms
    mean:  0.012111 ms

---

## Run 2

Backend startup / index load:

    7374.850554 ms

Cold query batch:

    min:   0.012714 ms
    p50:   0.015465 ms
    p95:   0.019207 ms
    p99:   0.023550 ms
    max:   0.049943 ms
    mean:  0.015518 ms

Warm query batch:

    min:   0.009647 ms
    p50:   0.012375 ms
    p95:   0.012612 ms
    p99:   0.012736 ms
    max:   0.015037 ms
    mean:  0.012031 ms

---

## Interpretation

Controlled cold-cache runs show that the dominant cold cost is backend
startup / index load, not individual FM query execution.

After the persistent backend is started, query latency remains stable.

Observed startup/load range:

    ~7.33–7.37 s

Observed warm p50 range after cache reset:

    ~0.0124–0.0125 ms

Observed warm p99 range after cache reset:

    ~0.0127–0.0147 ms

This measurement does not represent reboot-level cold storage behavior.
It represents Linux page-cache drop followed by persistent backend startup.

---

## Machine

See:

    benchmarks/MACHINE_SPEC.md
