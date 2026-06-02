# BUG_PIZZA_CHILI_PREFIX_UNDERCOUNT

Status:
OPEN

Corpus:
Pizza & Chili english 2GB sanitized prefix

File:
REFERENCE_BENCH/CORPORA/english_2gb_prefix_no_nulls.txt

Artifacts:
REFERENCE_BENCH/OUT/pizza_chili_english_2gb/fm.bin
REFERENCE_BENCH/OUT/pizza_chili_english_2gb/bwt.bin

Issue:

GLYPH FM query undercounts / misses a real exact match near the beginning of the corpus.

Minimal failing pattern:

Ten Days th

Hex:

54656e2044617973207468

Python count:

1

GLYPH count:

0

Full phrase:

Ten Days that Shook the World

Python count:

1

GLYPH count:

0

Known nearby behavior:

Ten Days t

Python count:
1

GLYPH count:
1

Ten Days 

Python count:
10

GLYPH count:
11

Observed divergence:

The failure begins when extending:

Ten Days t

to:

Ten Days th

Evidence:

The bytes exist at offset 53 in the corpus:

Ten Days that Shook the World

Hex:

54656e204461797320746861742053686f6f6b2074686520576f726c64

Interpretation:

This is not a query encoding issue.

The plain text bytes and hex query match exactly.

This is a correctness bug or invariant failure in the FM query path, BWT construction, sentinel handling, or corpus/artifact alignment.

Do not continue latency benchmarking on Pizza & Chili until this bug is understood.

Next debugging steps:

1. Validate SA ordering around offset 53.
2. Validate BWT character corresponding to suffix at offset 53.
3. Validate C table and Occ transitions for the failing pattern.
4. Compare backward search intervals step-by-step for:
   - Ten Days t
   - Ten Days th
5. Check whether the issue is related to sentinel, SA=0, or early-corpus suffixes.
--------------------------------------------------
UPDATE: DIRECT SA INTERVAL VALIDATION
--------------------------------------------------

Direct binary search over SA for pattern:

en Days th

returned:

866877230..866877233
count=3

FM backward search returned:

866877232..866877234
count=2

Therefore:

The SA is correct for this pattern.

The target suffix at offset 54 exists at:

SA index:
866877230

Suffix:

en Days that Shook the World

BWT at SA index 866877230 is correct:

expected previous char:
T

actual BWT char:
T

C table:
validated OK

BWT histogram:
validated OK

Occ checkpoint base:
validated OK around the failing interval

Conclusion:

The bug is now localized to the FM backward/Occ query path or to the exact way Occ(pos) is interpreted at interval boundaries.

This is not:

- query encoding issue
- corpus mismatch
- missing target bytes
- SA ordering issue
- BWT previous-char construction issue
- C table issue
