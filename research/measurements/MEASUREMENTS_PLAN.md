# Measurements plan

Status: planned measurements.

## Focus

GLYPH identity should be based on reproducible retrieval behavior, not marketing claims.

Measure:
- warm vs cold lookup latency
- mmap page fault behavior
- minor vs major faults
- RSS / VMS during index load
- shard size vs latency
- memory pressure behavior
- repeated query stability

## Kernel / mmap note

Recent Linux mmap/mm discussions and CVEs show that memory-mapped behavior is an active systems area.

Do not claim NVMe Gen5 conclusions without measurement.

## Goal

Same query over same data should produce:
- same result
- stable latency envelope
- predictable hardware behavior
