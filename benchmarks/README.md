# GLYPH Benchmark Methodology

## What is measured

GLYPH currently exposes two distinct latency layers.

These layers measure different operational realities and must not be
compared directly.

---

## Layer 1 — End-to-end verified query

Tool:

    benchmarks/cold_warm_v1.py

Measures the complete verified operational path:

    Python startup
    + manifest integrity verification
    + verified query wrapper
    + query_fm_v1 subprocess launch
    + FM query execution
    + result parsing

This benchmark measures what a real CLI user experiences when using
the verified query path.

Current mini corpus result
(56-byte corpus, 2 occurrences of "error"):

    cold:      ~19.2 ms
    warm p50:  ~19.8 ms
    warm p95:  ~20.2 ms
    warm p99:  ~20.3 ms

Important:

The dominant cost here is process startup and verification overhead,
not FM computation itself.

At mini scale, the FM backward-search portion is effectively negligible
relative to Python/subprocess startup cost.

---

## Layer 2 — Persistent FM backend query

Tool:

    benchmarks/persistent_fm_v1.py

Measures persistent in-memory FM querying:

    mmap-loaded FM index
    + persistent C++ backend
    + backward search
    + count return

This benchmark excludes:

    per-query Python startup
    per-query subprocess startup
    manifest verification overhead

The backend process is started once and reused for all warm queries.

Current mini benchmark result:

    startup:   ~1.0 ms
    cold:      ~0.025 ms
    warm p50:  ~0.007 ms
    warm p95:  ~0.009 ms
    warm p99:  ~0.010 ms

Example response:

    20 22 2

Interpretation:

The persistent backend measures actual FM query latency once the index
is already resident in memory.

This isolates FM search cost from operational wrapper overhead.

---

## What is NOT measured

The current benchmark suite does not yet measure:

- cold mmap page-fault behavior after reboot
- persistent backend latency under memory pressure
- concurrent query contention
- network/HTTP overhead
- index build time
- cross-machine reproducibility
- persistent backend p99 on large corpora
- shard fan-out overhead for segmented retrieval

---

## Hardware disclaimer

All benchmark results are machine-local measurements.

Numbers are not portable across machines.

Reproducible benchmark methodology requires documenting:

- CPU model
- RAM size
- storage type
- OS/kernel version
- Python version
- warm vs cold page cache state

Current benchmark machine specification is not yet committed.

This is a known documentation gap.

---

## Why cold/warm separation matters

Cold and warm queries measure different system behavior.

Warm query:
    FM algorithm cost with data already resident in memory.

Cold query:
    process startup
    + mmap initialization
    + page loading
    + cache population

Reporting only warm numbers hides first-query operational cost.

GLYPH benchmarks intentionally separate these layers.

---

## Why p50/p95/p99 matter

Average latency alone is insufficient.

Tail latency exposes:

- scheduler jitter
- page-cache misses
- process startup variance
- GC/runtime noise
- storage stalls

Interpretation guideline:

    p99 >> p50
        unstable latency envelope

    p99 ≈ p50
        predictable behavior

Current persistent backend behavior:

    p50 ≈ 0.007 ms
    p99 ≈ 0.010 ms

This indicates stable warm-query behavior at mini scale.

---

## Known gaps

- [ ] persistent backend benchmark on HDFS 1GB
- [ ] fixed reproducible query set committed to repo
- [ ] cold-start measurements after cache drop/reboot
- [ ] documented benchmark hardware spec
- [ ] segmented retrieval benchmark methodology
- [ ] shard fan-out p95/p99
- [ ] HTTP server overhead benchmark
- [ ] concurrent query benchmark

---

## Benchmark files

| File | Purpose |
|---|---|
| `benchmarks/cold_warm_v1.py` | End-to-end verified query benchmark |
| `benchmarks/persistent_fm_v1.py` | Persistent in-memory FM latency benchmark |
| `benchmarks/bench_1gb_persistent.py` | Legacy persistent 1GB benchmark |
| `benchmarks/bench_hdfs_1gb.sh` | Legacy HDFS 1GB benchmark pipeline |
| `benchmarks/HDFS_1GB_BENCHMARK.md` | Historical 1GB benchmark notes |

---

## Interpretation

GLYPH is not designed as a replacement for one-off grep scans.

The architecture targets deterministic repeated exact retrieval over
prepared static corpora.

The two latency layers serve different operational models:

Persistent backend (~0.007 ms warm):

    long-lived resident service
    repeated exact queries
    mmap-resident indexes
    low-latency retrieval systems

Verified wrapper (~19 ms):

    integrity-first workflows
    CLI tooling
    fail-fast artifact verification
    operational correctness boundaries

These are different engineering tradeoffs and should not be compared
as equivalent latency paths.
