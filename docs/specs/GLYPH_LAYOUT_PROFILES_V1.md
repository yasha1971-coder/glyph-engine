# GLYPH Layout Profiles V1

Status:

Candidate profile definitions based on enwik9 measurements.

Important:

These are not final profiles.

Persistent query latency is not yet measured.

Current measurements are based on:

- artifact size
- build time
- cold CLI behavior

Corpus:

enwik9

Corpus size:

1,000,000,000 bytes

Artifact baseline:

SA32:

3.8G

BWT:

954M

Candidate profiles:

## LATENCY-CANDIDATE

checkpoint_step:

32

FM size:

30G

FM build time:

57.7s

Cold CLI query:

~44-49s

Notes:

Smallest scan length.

But cold CLI is worst due to huge FM artifact load.

Cannot be called final latency profile until persistent latency is measured.

## BALANCED-CANDIDATE

checkpoint_step:

64

FM size:

15G

FM build time:

29.7s

Cold CLI query:

~20-24s

Notes:

Middle memory footprint.

Needs persistent benchmark.

## COMPACT-CANDIDATE

checkpoint_step:

256

FM size:

3.8G

FM build time:

9.2s

Cold CLI query:

~5-7s

Notes:

Best cold CLI profile.

Likely strongest current public benchmark default.

Persistent latency still must be measured.

Key law:

FM checkpoint table dominates memory.

Approximate model:

FM ≈ (corpus_bytes / checkpoint_step) * 256 * 4 bytes

Current conclusion:

checkpoint_step is a layout contract parameter.

Do not optimize SIMD before understanding FM layout economics.

Next required benchmark:

raw persistent FM query latency for:

step 32
step 64
step 256
