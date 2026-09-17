# Measured codec frontier — 2026-09-14

Public Silesia, 12 files, 211,938,580 bytes; official member sizes/MD5 checked.
All 12 SHA-256/size pairs also match the user-supplied laptop microscope report.
Three repeats per policy, two workers, 288 complete byte-exact restorations.
Timings below are from the assistant execution environment, NOT the laptop.

| Policy | Payload bytes | Median seconds | Min–max seconds | Speedup | Extra payload vs legacy |
|---|---:|---:|---:|---:|---:|
| legacy-best | 47,886,536 | 80.193 | 79.871–80.220 | 1.00x | 0.00% |
| bzip2-9 | 54,506,769 | 10.965 | 10.924–11.204 | 7.31x | 13.82% |
| xz-3 | 55,752,304 | 21.510 | 20.986–21.567 | 3.73x | 16.43% |
| xz-6 | 49,233,340 | 51.361 | 51.190–52.214 | 1.56x | 2.81% |
| xz-9 | 48,795,480 | 61.775 | 61.081–61.871 | 1.30x | 1.90% |
| sample-bz-xz6 | 50,095,977 | 19.235 | 19.174–19.380 | 4.17x | 4.61% |
| zstd-3 | 66,512,901 | 0.911 | 0.905–0.929 | 88.01x | 38.90% |
| zstd-9 | 59,371,345 | 2.347 | 2.322–2.509 | 34.16x | 23.98% |

The fixed sample policy selected xz-6 for one file (mozilla), and bzip2-9
for the other eleven, without reading names or prior reports. Against legacy,
it adds 2,209,441 bytes (4.61%) and reduces median measured pipeline time from
80.19 to 19.23 seconds (4.17x). This meets the proposed 5% size-cost gate here.
It is a measured engineering tradeoff, not a new universal compression algorithm.

Highest observed worker RSS: adaptive 195.8 MiB, legacy 642.3 MiB.
These are process high-water marks including input buffers, not the total
concurrent memory footprint. No sum of independent peaks is claimed.

## Negative control and limitations

A four-file synthetic case (seeded random, generated JSON, compressed JSON,
and a mixed file) gave adaptive and bzip2-only the same 1,773,725-byte payload.
Adaptive took 0.395 s median; bzip2-only took 0.274 s. Probing added cost without
a density benefit. The adaptive rule therefore must not be treated as always faster.

Silesia is a known development corpus. VIKA is the next independent test;
no VIKA measurements of this policy are available yet. Precomp is absent here.
The prior VIKA 21.25% saving cannot be carried over to these new modes.

Wall time includes process startup, source read/hash, probe, compression,
writing and fsync, rereading, decoding and full comparison. The OS cache was
not flushed. It excludes inventory construction before timing, persistent
catalogue updates and durable directory commit. Payload is not total vault size.
No application ingestion, 1 TB scaling, SSD wear, power-loss or iPhone claim.

## Decision

Use sample-bz-xz6 as the candidate for density-focused laptop validation.
Keep zstd-3/zstd-9 as fast-mode candidates with their explicitly larger payloads.
Keep native CDC exact and independently measured; its speedup must not be
multiplied into this result because this path did not call CDC.

Do not replace the working vault codec policy before the independent corpus
and application-path validation. For future integration, preserve logical object
identity across representations and verify/atomically publish a denser copy
before reclaiming its prior representation. Shared version blocks require their
existing reference/liveness accounting, regardless of physical codec.

## Reproduction and evidence

Runner: codec_frontier.py. Tests: tests/test_codec_frontier.py.
Raw evidence: CODEC_FRONTIER_SILESIA_RESULT.json and CODEC_FRONTIER_SYNTHETIC_RESULT.json.
Machine/library versions and exact derived values: CODEC_FRONTIER_SUMMARY.json.
Primary source links and protocol: CODEC_FRONTIER.md.

No private source file bytes or private filenames are included in this change.
