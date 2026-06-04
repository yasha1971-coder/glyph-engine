# SENTINEL_SMOKE_V1

Status:
PASS

Purpose:

Verify that the new sentinel-safe pipeline can retrieve a pattern
that failed under the legacy synthetic-sentinel model.

Pipeline:

build_sa_sentinel_v1
build_bwt_sentinel_v1
build_fm
query_fm_batch_v1

Corpus:

Ten Days that Shook the World
John Reed
Ten Days they waited
Ten Days to remember

Original bytes:
82

Indexed bytes with sentinel:
83

Query:

Ten Days that Shook the World

Result:

23 24 1

Conclusion:

The sentinel-safe pipeline returns count=1 for the target phrase.

This supports the diagnosis that the Pizza & Chili undercount was caused
by the legacy synthetic sentinel model.

Next:

Rebuild Pizza & Chili english 2GB sanitized prefix using sentinel-safe pipeline.
