# PERSISTENT FM RUNTIME V1

Status:
Runtime candidate.

Purpose:
Define query_fm_server_v1 as the first-class persistent retrieval runtime candidate.

Binary:
query_fm_server_v1

Mode:
stdin/stdout

Inputs:
- fm.bin
- bwt.bin

Startup:
- load FM
- load BWT
- print READY

Query protocol:
- one hex pattern per line on stdin

Response protocol:
- one line per query
- format:
  <l> <r> <count>

Example:
input:
746865

output:
90317800 91226254 908454

Known behavior:
- keeps index resident
- avoids one-process-per-query overhead
- supports repeated queries
- supports absent patterns
- currently not HTTP
- currently not binary protocol

Measured signal:
Windows enwik8:
CLI single query ~906-949ms
persistent batch 100 queries ~764.529ms total
~7.645ms/query including startup/load

Architectural conclusion:
Persistent runtime is the correct retrieval direction.

Next:
PERSISTENT_LATENCY_BENCH_V1
