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

