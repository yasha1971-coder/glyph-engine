# PIZZA_CHILI_QUERY_V1

Corpus:
Pizza & Chili english 2GB sanitized prefix

Query mode:
HEX byte patterns

Plain-text queries were converted to Latin-1 hex before search.

Important finding:

GLYPH query tools are byte/hex-first.

Tools tested:
query_fm_batch_v1

Input:
REFERENCE_BENCH/QUERIES/pizza_queries_v1.txt
REFERENCE_BENCH/QUERIES/pizza_queries_v1.hex.txt

Results:

Ten Days that Shook the World:
0 hits

John Reed:
11 hits

Normal Wolcott:
0 hits

Graphics:
2 hits

Harvard University:
184 hits

Roosevelt administration:
2 hits

William Bullitt:
0 hits

Louise Bryant:
2 hits

Adams House:
0 hits

Stalin:
64 hits

Raw output:

0 0 0
474919091 474919102 11
0 0 0
463927230 463927232 2
464929285 464929469 184
486721525 486721527 2
0 0 0
477214100 477214102 2
0 0 0
489699118 489699182 64

Status:
QUERY TEST PASSED

Next:
Run latency benchmark using the same hex query set.
