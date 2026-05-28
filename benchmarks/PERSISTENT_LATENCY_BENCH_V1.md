# PERSISTENT LATENCY BENCH V1

Purpose:

Measure steady-state retrieval latency of the persistent FM runtime.

Runtime:
query_fm_server_v1

Focus:
resident retrieval behavior

Not measured:
- build time
- startup time
- artifact generation
- HTTP layer
- external orchestration

## Benchmark dimensions

1. Warmup behavior
2. Steady-state latency
3. Repeated query latency
4. Random query latency
5. Absent-pattern latency
6. Long-pattern latency
7. Throughput (queries/sec)
8. p50 / p95 / p99 latency
9. Linux/Windows parity

## Canonical corpora

- examples/mini
- enwik8
- Silesia
- enwik9 step=256

## Query classes

Short:
- 3-byte
- 4-byte

Medium:
- 8-byte
- 16-byte

Long:
- 32-byte
- 64-byte

Classes:
- high-frequency
- medium-frequency
- rare
- absent

## Runtime rules

- FM/BWT loaded once
- runtime remains resident
- queries streamed via stdin
- startup separated from steady-state timing

## Required outputs

- avg latency
- p50
- p95
- p99
- max latency
- queries/sec
- total queries
- corpus size
- checkpoint step
- hardware info
- OS/runtime info

## Goal

Establish canonical retrieval behavior surface for GLYPH persistent runtime.

## Success condition

Stable, reproducible steady-state latency measurements across platforms.
