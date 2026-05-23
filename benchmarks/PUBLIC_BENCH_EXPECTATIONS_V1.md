PUBLIC BENCH EXPECTATIONS V1

Corpus:

enwik9

Expected invariants:

SA build:

must succeed

BWT build:

must succeed

FM build:

must succeed

Expected FM scaling:

step32

larger than step64

step64

larger than step256

Expected order:

FM32 > FM64 > FM256

Expected cold CLI order:

CLI32 slower than CLI64

CLI64 slower than CLI256

Expected sentinel compatibility:

0 null bytes

Expected benchmark stability:

relative ordering stable

Absolute timing may vary by hardware.

Failure conditions:

FM32 <= FM64

or

FM64 <= FM256

or

sentinel compatibility breaks

or

cold CLI ordering reverses unexpectedly

Action:

investigate layout regression
