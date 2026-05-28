# PLATFORM COMPARISON

Status:

Benchmark interpretation note.

Do not directly compare Linux/OVH and Windows laptop latency numbers without context.

## Linux / OVH

Machine:
AMD EPYC server-class environment

Known observations:
Persistent FM path on large corpus showed extremely low latency in prior Linux measurements.

Interpretation:
Server-grade CPU, memory bandwidth, Linux runtime, and different benchmark harness.

## Windows laptop

Machine:
Windows laptop

Corpus:
enwik8

Observed:
Single CLI query:
~906-949 ms/query

Persistent batch:
100 queries in 764.529 ms
~7.645 ms/query including startup/load

Interpretation:
This benchmark includes Windows process/runtime overhead and batch process lifecycle.
It is not directly comparable to Linux persistent hot-memory p50 numbers.

## Shared conclusion

Both platforms show the same architectural law:

one-process-per-query CLI is not the target retrieval architecture.

Persistent retrieval is the correct system direction.

## Main engineering decision

Focus next on:

Persistent daemon
+
hot index residency

Do not optimize zero-copy, batching, or protocol layers before the daemon is a first-class component.

## Reason

Measured improvement:

CLI query path:
slow due to process/artifact lifecycle

Persistent path:
dramatically faster because index stays resident

## Next milestone

PERSISTENT_DAEMON_V1

Requirements:
- load FM/BWT once
- keep index resident
- accept repeated queries
- provide stable request/response protocol
- expose benchmarkable latency
- support Linux and Windows
