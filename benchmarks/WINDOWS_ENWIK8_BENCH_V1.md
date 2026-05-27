WINDOWS ENWIK8 BENCH V1



Machine:

Windows laptop



Corpus:

enwik8



Corpus size:

100,000,000 bytes



Sentinel compatibility:

0 null bytes



Build:

MSVC Build Tools 2022

CMake 4.3.3



Results:



SA build:

12.26s

SA size:

400,000,000 bytes



BWT build:

1.65s

BWT size:

100,000,000 bytes



FM build:

checkpoint\_step=256

1.37s

FM size:

400,003,116 bytes



Query:

pattern hex:

746865



pattern:

the



Result:

count=908454

verified=true



Conclusion:

Windows laptop successfully builds and queries GLYPH index on enwik8.



Cold/Warm CLI query timing:



Command:

query\_fm\_v1 fm.bin bwt.bin 746865 --json



Cold-ish:

0.949s



Warm-ish:

0.906s



Interpretation:

For Windows enwik8 step=256 CLI query, cold/warm difference is small.

This measures full CLI path, not persistent in-memory query latency.



Pattern frequency timing:



the:

\~0.91-0.95s



aaa:

0.943s



qwxz:

0.908s



Finding:

CLI query time is almost independent of match frequency.



Interpretation:

Current Windows CLI query timing is dominated by process/artifact setup rather than match count.



Persistent batch benchmark:



Tool:

benchmarks/windows\_persistent\_bench\_v1.py



Mode:

query\_fm\_server\_v1 stdin batch



Queries:

100



Patterns:

the

aaa

qwxz

the and



Result:

returncode=0

stdout\_lines=100

elapsed\_ms=764.529

avg\_ms\_per\_query\_including\_startup=7.6453



Finding:

Batch/persistent-style stdin path is dramatically faster than one-process-per-query CLI.



Comparison:

CLI single query:

\~906-949 ms/query



Batch 100 queries through one server process:

\~7.65 ms/query including startup/load



Interpretation:

The main bottleneck is process/artifact lifecycle, not FM traversal.

