FM Layout Next Signals

Observed:

checkpoint_step scaling dominates memory footprint.

Current layout:

checkpoint[block][256]
uint32 counters

Future layout directions:

1.
sparse checkpoint encoding

2.
compressed checkpoint delta

3.
wavelet tree Occ

4.
RRR bitvectors

5.
compressed BWT

6.
mixed layout profiles:

LATENCY
BALANCED
COMPACT

Current conclusion:

Optimization priority:

FM layout
>
SIMD micro-optimization

Evidence:

enwik9 reproducible benchmark
checkpoint scaling law
cold CLI measurements
