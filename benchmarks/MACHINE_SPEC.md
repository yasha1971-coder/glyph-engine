# GLYPH Benchmark Machine Specification

All benchmark results in this repository were produced on the following machine.

Numbers from different hardware are not directly comparable.

---

## Hardware

| Component | Value |
|---|---|
| CPU | AMD EPYC 4344P 8-Core Processor |
| Logical CPUs | 16 |
| Physical cores | 8 |
| Threads per core | 2 |
| Sockets | 1 |
| RAM | 125 GiB |
| Storage | 2 × NVMe disks, 894.3 GiB each |
| Storage rotation | non-rotational (`ROTA=0`) |
| NIC | not relevant for local benchmarks |

---

## Software

| Component | Value |
|---|---|
| OS / kernel | Linux ace-core 5.15.0-163-generic x86_64 |
| Kernel build | #173-Ubuntu SMP Tue Oct 14 17:51:00 UTC 2025 |
| Python | Python 3.10.12 |
| GCC | gcc 11.4.0 |
| CMake | cmake 3.22.1 |
| libsais | vendored (`third_party/libsais`) |

---

## Memory state at measurement time

| Metric | Value |
|---|---|
| Total RAM | 125 GiB |
| Used RAM | 39 GiB |
| Free RAM | 1.5 GiB |
| Buff/cache | 84 GiB |
| Available RAM | 84 GiB |
| Swap | 1.0 GiB total / 1.0 GiB used |

Interpretation:

The system had a large active page cache during benchmark work.

Warm-query results should be interpreted as warm-cache measurements.

---

## Page cache state

| Benchmark type | Cache state |
|---|---|
| cold query | first measured query in current benchmark process |
| warm query | index accessed at least once before measurement |

Current benchmark scripts do not yet perform OS-level cache dropping.

Linux cache-drop procedure for future controlled cold measurements:

```bash
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
This requires root access and should be documented whenever used.

⸻

Why this matters

Benchmark numbers without a machine specification are not reproducible.

Common failure modes:

* comparing warm RAM numbers from a high-RAM server against a laptop
* comparing page-cache-warm measurements against cold storage measurements
* mixing operational wrapper latency with raw persistent backend latency
* omitting CPU, RAM, kernel, and Python version

GLYPH benchmark numbers must be interpreted together with this document.

## Cache residency note

HDFS 1GB FM artifact:

    bench_1gb/out/hdfs_1gb.fm.bin
    size: 8.1 GiB

CPU L3 cache:

    32 MiB

Conclusion:

    The full FM artifact is not L3-cache-resident.

Therefore stable warm-query p50/p95/p99 behavior should not be explained
as full-index cache residency.

More likely contributors:

- OS page cache residency
- mmap-backed access
- small number of touched pages per query
- simple persistent backend path
- low orchestration overhead in single-backend mode

## Perf observation: persistent FM backend

Command:

    perf stat -e page-faults,cache-misses,cache-references \
      python3 benchmarks/persistent_fm_v1.py \
      --fm bench_1gb/out/hdfs_1gb.fm.bin \
      --bwt bench_1gb/out/hdfs_1gb.bwt.bin \
      --queries-file bench_1gb/queries.txt \
      --warm-runs 3

Result:

    startup_ms:       ~3704 ms
    warm p50:         ~0.014 ms
    warm p99:         ~0.015 ms
    page-faults:      ~2.36M
    cache-misses:     ~52M
    cache-references: ~1.67B

Interpretation:

    perf stat covers the full benchmark process, including backend startup,
    mmap loading, and warm queries.

    Therefore page-faults mostly describe startup/load behavior, not
    individual warm-query behavior.

    Warm query latency remains stable after startup.

Current model:

    The full FM artifact is not L3-cache-resident.
    Stable warm behavior is likely caused by OS page cache residency,
    mmap-backed access, small touched working set per query, and a simple
    persistent backend path.
