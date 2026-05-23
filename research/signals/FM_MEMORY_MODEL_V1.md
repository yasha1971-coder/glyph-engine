FM Memory Model V1

Current layout:

checkpoint[block][256]

counter_size=4 bytes

Approximate memory:

FM ≈
(
corpus_bytes
/
checkpoint_step
)
*
256
*
4

Observed:

enwik9

step=32
≈30G

step=64
≈15G

step=256
≈3.8G

Implication:

checkpoint table dominates memory.

Optimization leverage:

reduce:

blocks
or
counter width
or
alphabet density

Possible future:

uint24

delta checkpoints

sparse checkpoints

compressed checkpoint blocks

wavelet tree Occ

Goal:

memory law before implementation.
