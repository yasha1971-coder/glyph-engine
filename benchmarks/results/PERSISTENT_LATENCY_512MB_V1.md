# PERSISTENT LATENCY 512MB V1

Runtime:
query_fm_server_v1

Mode:
batch-stdin

Corpus:
data_4gb/test_512mb.bin

Corpus size:
536,870,912 bytes

FM format:
FMBINv2

Checkpoint step:
256

Build timings:
SA: 29.355s
BWT: 4.985s
FM: 4.914s

Query set:
bench_queries/basic_v1.txt
7 queries

Result:
total_ms=2810.217293
avg_ms_including_startup_load=401.459613
qps=2.491

Query set:
bench_queries/basic_1000_v1.txt
1000 queries

Result:
total_ms=2748.811897
avg_ms_including_startup_load=2.748812
qps=363.794

Finding:
Increasing query count from 7 to 1000 massively amortizes process/artifact startup and load.

Conclusion:
The dominant cost is runtime startup/load, not per-query FM traversal.

Architecture signal:
persistent resident runtime is required for true retrieval latency measurement.
