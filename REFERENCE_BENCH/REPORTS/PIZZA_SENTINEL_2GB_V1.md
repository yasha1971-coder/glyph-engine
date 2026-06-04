# PIZZA_SENTINEL_2GB_V1

Status:
PASS

Corpus:
Pizza & Chili english 2GB sanitized prefix

Original bytes:
2,000,000,000

Indexed bytes with sentinel:
2,000,000,001

Pipeline:
build_sa_sentinel_v1
build_bwt_sentinel_v1
build_fm
query_fm_batch_v1

Build times:

SA:
2m03.786s

BWT:
21.388s

FM:
18.802s

Critical comparison:

Legacy synthetic-sentinel 2GB pipeline missed several valid patterns.

Sentinel-safe 2GB pipeline returns the expected counts.

Query results:

Ten Days that Shook the World:
1

John Reed:
12

Normal Wolcott:
1

Graphics:
2

Harvard University:
185

Roosevelt administration:
3

William Bullitt:
1

Louise Bryant:
3

Adams House:
1

Stalin:
64

Raw output:

491231628 491231629 1
474919089 474919101 12
481507432 481507433 1
463927230 463927232 2
464929283 464929468 185
486721523 486721526 3
498080782 498080783 1
477214099 477214102 3
450504058 450504059 1
489699118 489699182 64

Conclusion:

The sentinel-safe pipeline fixes the confirmed Pizza & Chili undercount bug at full 2GB scale.

The root cause was the legacy synthetic sentinel model.

Next:
Promote sentinel-safe pipeline as the canonical GLYPH build path.
