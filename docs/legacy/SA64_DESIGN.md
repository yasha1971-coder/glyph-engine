# SA64 / SA32u Design Note — GLYPH_CPP_BACKEND
## 1. Why this exists
Current GLYPH exact layer is validated up to ~2GB.
4GB test is blocked at SA construction:
```text
ERROR: corpus too large for int32 SA

This is not a hardware limit.

It is a format/type limit:

current SA = int32
4GB corpus = 4,000,000,001 positions
int32 max  = 2,147,483,647

⸻

2. Current validated scale

512MB

* corpus: 536,870,913 bytes
* BWT validation: OK
* retrieval: EXACT_UNIQUE 100/100
* total p99: ~2.67 ms

1GB

* corpus: 1,000,000,001 bytes
* strict exact layer: READY
* fuzz: 12/12 passed
* total p99: ~3.86 ms

2GB

* corpus: 2,000,000,001 bytes
* SA: built OK
* BWT: built OK
* FM: built OK
* chunk_map: built OK
* retrieval p99: ~4.30 ms
* note: EXACT_MULTI expected because corpus repeats 1GB data

4GB

* corpus: 4,000,000,001 bytes
* meta/chunk_starts: OK
* SA: blocked by int32 format

⸻

3. Design options

Option A — int64 SA

Use signed 64-bit integers for SA positions.

Pros:

* simple conceptually
* supports very large corpora

Cons:

* SA size doubles:
    * 4GB corpus → ~32GB SA
    * 16GB corpus → ~128GB SA
* slower IO
* more memory pressure

Use only if needed.

⸻

Option B — uint32 SA

Use unsigned 32-bit positions.

Supports:

0 .. 4,294,967,295

This is enough for:

4GB corpus = 4,000,000,001 positions

Pros:

* same disk size as current int32 SA
* enough for 4GB boundary test
* minimal disruption

Cons:

* does not support >4GB
* requires careful handling near uint32 max
* sentinel and corpus size must stay below 2^32

Recommended next step:

SA32u for 4GB validation

⸻

Option C — segmented SA

Split corpus into segments and build separate SA/BWT/FM per segment.

Pros:

* scalable beyond 4GB
* avoids giant monolithic SA
* easier memory management

Cons:

* retrieval must merge results across segments
* exact cross-segment queries require overlap
* more complex query routing

Potential future direction for 16GB+.

⸻

4. Recommended path

Phase 1 — SA32u

Goal:

validate 4GB exact layer

Change:

* build_sa output type: uint32_t
* build_bwt SA reader: uint32_t
* build_chunk_map SA reader: uint32_t
* ensure all SA positions are treated as uint32_t / uint64_t during arithmetic
* avoid signed int comparisons

Do not change:

* FM format
* chunk_map format
* query_fm_server
* retrieval Python logic

Reason:

FM intervals are counts/positions and already printed as integer ranges.
chunk_map stores chunk ids, not corpus offsets, so u32 is still enough.

⸻

5. Files likely affected

src/build_sa.cpp
src/build_bwt.cpp
src/build_chunk_map.cpp

Possibly:

src/query_fm_v1.cpp
src/query_fm_batch_v1.cpp
src/query_fm_server_v1.cpp

Only if they assume signed int32 internally.

⸻

6. Safety rules

1. Do not overwrite working 1GB/2GB artifacts.
2. Use new artifact directory:

out_4gb_u32/

3. Prefer new binaries if needed:

build_sa_u32
build_bwt_u32
build_chunk_map_u32

4. Keep old binaries intact until 4GB passes.
5. Do not change retrieval semantics while changing SA format.

⸻

7. Validation checklist

For 4GB SA32u:

Build

* SA builds successfully
* SA size approx 16GB
* BWT builds successfully
* BWT size equals corpus size
* BWT zero count = 1
* FM builds successfully
* chunk_map builds successfully

Retrieval

* live benchmark runs
* no overflow/crash
* outcomes expected:
    * likely EXACT_MULTI due repeated corpus
* latency target:
    * total p99 < 10 ms
    * server p99 < 3 ms

⸻

8. Risks

Risk 1 — libsais limitation

If libsais only returns int32 SA, SA32u may require different API or alternate builder.

Risk 2 — BWT arithmetic overflow

Expressions like:

sa[i] - 1

must handle zero carefully.

Correct pattern:

pos == 0 ? sentinel : corpus[pos - 1]

Risk 3 — signed/unsigned bugs

Avoid:

int x = sa[i];

Use:

uint32_t x;
uint64_t pos;
size_t i;

Risk 4 — 4GB edge

4,000,000,001 is below uint32 max but close enough that accidental signed casts will fail.

⸻

9. Decision

Current decision:

Do SA32u first.
Do not jump to int64 unless SA32u is blocked.
Do not implement segmented SA yet.

Reason:

SA32u is the minimal change that unlocks 4GB validation.

⸻

10. Next concrete step

Inspect current SA type assumptions:

grep -n "int32\\|uint32\\|int \\*\\|fread\\|fwrite\\|sizeof" src/build_sa.cpp src/build_bwt.cpp src/build_chunk_map.cpp

Then patch carefully.
