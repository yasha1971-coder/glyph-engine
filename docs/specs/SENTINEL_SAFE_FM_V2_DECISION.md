# SENTINEL_SAFE_FM_V2_DECISION

Status:
OPEN

Problem:

Current GLYPH v0.x FM pipeline builds:

SA over text length n

but BWT injects a synthetic sentinel byte when SA[i] == 0.

This creates a mismatch:

SA has n suffixes.
BWT/FM histogram has synthetic sentinel semantics.
C/Occ/backward search operate as if the sentinel participates in FM order.

Observed Effect:

FM intervals may have correct sizes but shifted boundaries.
Longer patterns can undercount or disappear.

Confirmed on:

Pizza & Chili english 2GB sanitized prefix

Failing pattern:

Ten Days th

Python/direct SA count:
1

FM count:
0

Root Cause:

Synthetic sentinel without real appended sentinel suffix is not a strict FM-index model.

Decision:

Introduce a canonical sentinel-safe FM v2 pipeline.

Required Model:

text must be transformed into:

text + real sentinel

before SA construction.

Constraints:

1. Input corpus for v0.x must not contain 0x00.
2. The appended sentinel is 0x00.
3. SA must include n+1 suffixes.
4. BWT must be built from SA over text+sentinel.
5. FM n must match BWT length n+1.
6. Query results must exclude sentinel-only artifacts.
7. Locate/snippet extraction must map back to original text offsets.
8. Existing synthetic-sentinel artifacts must not be used for strict correctness claims.

Migration:

Do not silently change old artifact semantics.

Create new builders or artifact version:

FMBINv3

or explicit SENTINEL_SAFE mode.

Status:

Required before further Pizza & Chili latency claims.
