# TRUE STEADY STATE REPEATABILITY 2GB V1

Corpus:
bench_v2_2gb/corpus_2gb.bin

Corpus size:
2,000,000,000 bytes

Null bytes:
0

FM format:
FMBINv2

Checkpoint step:
256

Build timings:
SA: 1m58.935s
BWT: 19.761s
FM: 17.947s

Query set:
bench_queries/basic_1000_v1.txt

Warmup:
50

Runs:
10

Single steady-state run:
avg_ms=0.009034
p50_ms=0.009039
p95_ms=0.010259
p99_ms=0.010440
qps=110694.874

Repeatability summary:

avg_ms:
min=0.006164
max=0.008831
ratio=1.433

p50_ms:
min=0.006022
max=0.009077
ratio=1.507

p95_ms:
min=0.007235
max=0.010239
ratio=1.415

p99_ms:
min=0.007765
max=0.011172
ratio=1.439

max_ms:
min=0.012984
max=0.019696
ratio=1.517

qps:
min=113242.667
max=162226.694
ratio=1.433

Verdict basis:
p95_ms max/min

Verdict:
LAW_CONFIRMED

Interpretation:
2GB corpus preserves microsecond-scale persistent FM steady-state retrieval.

Scaling signal:
512MB, 1GB, and 2GB all remain in the same microsecond latency regime.
