# PIZZA_SENTINEL_50MB_V1

Status:
PASS

Purpose:

Validate sentinel-safe FM pipeline on a real Pizza & Chili english slice.

Corpus:

REFERENCE_BENCH/OUT/pizza_sentinel_test/english_50mb.txt

Original bytes:
50,000,000

Indexed bytes with sentinel:
50,000,001

Pipeline:

build_sa_sentinel_v1
build_bwt_sentinel_v1
build_fm
query_fm_batch_v1

Build times:

SA:
2.245s

BWT:
0.409s

FM:
0.476s

Queries:

Ten Days that Shook the World
John Reed
Normal Wolcott
Graphics
Harvard University
Roosevelt administration
William Bullitt
Louise Bryant
Adams House
Stalin

Results:

Ten Days that Shook the World:
1

John Reed:
6

Normal Wolcott:
1

Graphics:
1

Harvard University:
5

Roosevelt administration:
1

William Bullitt:
1

Louise Bryant:
3

Adams House:
1

Stalin:
2

Raw output:

12587658 12587659 1
12153799 12153805 6
12339845 12339846 1
11897683 11897684 1
11919248 11919253 5
12472953 12472954 1
12772155 12772156 1
12222366 12222369 3
11551855 11551856 1
12555813 12555815 2

Conclusion:

The sentinel-safe pipeline successfully retrieves patterns that failed
under the legacy synthetic-sentinel model.

This supports the root-cause diagnosis:

legacy synthetic sentinel caused FM interval mismatch.

Next:

Rebuild full Pizza & Chili 2GB sanitized prefix using sentinel-safe pipeline.
