# ENWIK9 PUBLIC BENCH RUNBOOK V1

Purpose:

Define reproducible public GLYPH benchmark procedure on enwik9.

Corpus:

Matt Mahoney LTCB enwik9

Download:

wget https://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip

Sentinel compatibility:

python3 -c "d=open('enwik9','rb').read(); print(d.count(b'\x00'), 'null bytes')"

Required result:

0 null bytes

Build SA:

time build_sa enwik9 out/sa.bin

Observed:

55.6s
SA size 3.8G

Build BWT:

time build_bwt enwik9 out/sa.bin out/bwt.bin 0x00

Observed:

9.5s
BWT size 954M

Build FM layouts:

checkpoint_step 32:

time build_fm out/bwt.bin out/fm.bin 32

Observed:

57.7s
FM size 30G

checkpoint_step 64:

time build_fm out/bwt.bin out64/fm.bin 64

Observed:

29.7s
FM size 15G

checkpoint_step 256:

time build_fm out/bwt.bin out256/fm.bin 256

Observed:

9.2s
FM size 3.8G

Cold CLI query test:

query_fm_v1 fm.bin bwt.bin HEX --json

Known caveat:

Cold CLI includes process startup and artifact load.

It is not persistent FM latency.

Current public benchmark interpretation:

checkpoint_step 256 is strongest cold CLI profile.

checkpoint_step 32 is not confirmed as latency profile until persistent raw FM benchmark exists.
