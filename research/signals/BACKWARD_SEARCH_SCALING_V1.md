BACKWARD SEARCH SCALING V1

Date:
2026-05

Observation:

pattern_len=4
p50=20ns
qps=26.9M

pattern_len=8
p50=20-30ns
qps=22.8M

pattern_len=16
p50=40ns
qps=17.2M

pattern_len=32
p50=70ns
qps=11.3M

Finding:

Backward search cost scales with pattern length.

Scaling not hidden completely.

Structural breakpoint begins near 16-32 symbols.

Implication:

Occ checkpoint optimization works.

Future optimization surface:

backward_search()

Possible future direction:

batch LF
bit-plane Occ
pattern pipeline optimization
