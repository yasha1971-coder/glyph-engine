# PIZZA_SENTINEL_500MB_V1

Status:
PASS

Corpus:
Pizza & Chili english 500MB prefix, sanitized no-null source.

Original bytes:
500,000,000

Indexed bytes with sentinel:
500,000,001

Pipeline:
build_sa_sentinel_v1
build_bwt_sentinel_v1
build_fm
query_fm_batch_v1

Build times:

SA:
29.077s

BWT:
5.071s

FM:
4.706s

Query result summary:

All 10 Pizza query patterns returned non-zero counts.

Critical previously failing patterns:

Ten Days that Shook the World:
1

Normal Wolcott:
1

William Bullitt:
1

Adams House:
1

Raw output:

123701937 123701938 1
119479632 119479640 8
121234055 121234056 1
116711338 116711340 2
116972745 116972781 36
122564347 122564349 2
125413535 125413536 1
120073498 120073501 3
113311456 113311457 1
123321803 123321805 2

Conclusion:

Sentinel-safe pipeline scales from smoke test to 500MB real Pizza & Chili corpus and fixes the legacy synthetic-sentinel undercount class.

Next:
Rebuild full 2GB Pizza & Chili sanitized prefix using sentinel-safe pipeline.
