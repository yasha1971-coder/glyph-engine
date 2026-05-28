# TRUE STEADY STATE REPEATABILITY 1GB V1

Corpus:
data_1gb/corpus_1gb.bin

Corpus size:
1,000,000,000 bytes

Null bytes:
0

FM format:
FMBINv2

Checkpoint step:
256

Build timings:
SA: 56.016s
BWT: 9.481s
FM: 9.062s

Query set:
bench_queries/basic_1000_v1.txt

Warmup:
50

Runs:
10

Single steady-state run:
avg_ms=0.009084
p50_ms=0.009127
p95_ms=0.010381
p99_ms=0.010667
qps=110084.372

Repeatability summary:

avg_ms:
min=0.006124
max=0.008795
ratio=1.436

p50_ms:
min=0.005990
max=0.009088
ratio=1.517

p95_ms:
min=0.006994
max=0.010440
ratio=1.493

p99_ms:
min=0.007665
max=0.010649
ratio=1.389

qps:
min=113703.607
max=163280.467
ratio=1.436

Verdict basis:
p95_ms max/min

Verdict:
LAW_CONFIRMED

Interpretation:
1GB corpus preserves microsecond-scale persistent FM steady-state retrieval.

Scaling signal:
512MB and 1GB both remain in the same microsecond regime.
