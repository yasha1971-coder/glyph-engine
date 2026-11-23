# CORE-PERF-P03 preregistered native AC experiment — BEFORE performance results

Status: **PREDICTIONS ONLY / NO CI EVIDENCE YET**. Existing [Draft PR #29](https://github.com/yasha1971-coder/glyph-engine/pull/29), [Issue #28](https://github.com/yasha1971-coder/glyph-engine/issues/28), research branch only. Frozen V1, original L9-S, P01 and P02 remain unchanged.

## Algorithmic hypothesis

P02 eliminated redundant Q full physical reads, but still invokes Q native byte scans per in-memory block. P03 is a **bounded C++ Aho–Corasick** byte automaton for at most Q=16 nonempty byte patterns, sum of pattern lengths <=4096, <=4097 states, full SHA256+complete EOF in one source read. Each state has 256 u32 transitions and a <=16-bit match mask; a shared forward automaton state continues across 1MiB read blocks, so no carry buffer and no cross-boundary duplicate counting. Full hit count and first 256 globally ordered offsets per pattern, all evidence held until source SHA+fd unchanged; truncated offsets means UNTRUSTED/LIMITED_COMPLETENESS. The standalone implementation is **not a proof of FM/rank/EF semantics, version authority or currentness**; source buffer I/O is mandatory.

Strict bounded allocation: 4097 × sizeof(Node) ~4.2 MiB automaton, 1MiB block, 16×256 u64 offset prefixes, OpenSSL SHA context; measure actual peak RSS. No threading, SIMD instruction assumptions, GPU, prefetch or cache tuning yet.

## Predictions fixed BEFORE GitHub benchmark

- **P03-A correctness hard gate:** on deterministic random/binary/repetitive inputs and boundary/overlap fixture, all native counts/prefix offsets/statuses equal Python P02 and an independent raw byte oracle. Missing/corrupt source, wrong SHA/size, symlink, invalid/duplicate/budget-exceeding patterns are rejected with code 3 and no authoritative output.
- **P03-B boundedness hard gate:** no more than 4097 states and automaton allocation <= 5 MiB; admitted source <=100MB, Q<=16, max total pattern bytes 4096. Report native and Python process RSS separately.
- **P03-C performance hypothesis:** on hash-pinned canonical 100MB enwik8, Q=16 native C++ AC median wall < P02 Q=16 Python batch median; **FAIL if slower**. Compare Q=1,2,4,8 as well; retain negative findings, no silent rerun to optimize a failed prediction.
- **P03-D repeated corpus hypothesis:** on >=4MB repetitive synthetic data with many overlapping occurrences, native count/offset completeness remains correct and bounded, regardless of speed. This scenario may be slower than P02's optimized bytes.count and is an intentional falsifier.
- **P03-E trust boundary:** no FOUND/PROVEN_EMPTY before full source SHA equality, EOF and stable fd identity. External source pin itself unsigned. Full product speed/storage reduction is NOT implied.

## Science/research input

Aho–Corasick (1975) computes a failure-link finite automaton over all patterns with shared O(n+z) streaming recognition after building the automaton. Because full count of frequent patterns requires reporting/updating each matched pattern, cost includes z; actual CPU, memory and cache locality may be inferior to highly optimized native bytes.count for Q=1. Follow Daniel Lemire's data parallelism principle only after measuring the algorithmic cost; the P03 native C++ improvement is not in itself proof of explicit SIMD.

Formal work: P01/P02 Mathlib proves carry arithmetic for Python shared-buffer streaming. P03 instead carries **automaton state**, a new proof obligation (state after a prefix is the longest recognized suffix prefix; failure transitions preserve correctness). Do not apply P02's Lean proof as though it certified C++ AC. First independent oracle+negative tests; full machine checked C++ proof is open.

Use one existing CORE-L10 tracker, no new GitHub Issue/PR, no merging or OVH workloads. Keep exact baseline and failure records.
