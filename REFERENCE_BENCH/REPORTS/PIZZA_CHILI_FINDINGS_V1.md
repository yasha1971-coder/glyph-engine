# PIZZA_CHILI_FINDINGS_V1

Date:
2026-06-02

Corpus:
Pizza & Chili english

Working Corpus:
english_2gb_prefix_no_nulls.txt

--------------------------------------------------
FINDING 1
--------------------------------------------------

Official Pizza & Chili english corpus size:

2,210,395,553 bytes

This exceeds the current SA32-safe working boundary.

Implication:

Full Pizza & Chili english cannot be treated as a
safe SA32 corpus without additional architecture work.

--------------------------------------------------
FINDING 2
--------------------------------------------------

The original corpus contains embedded 0x00 bytes.

Observed:

13 null bytes

All null bytes were concentrated near ~787MB offset.

Null bytes appeared inside textual content.

Implication:

The official corpus is not directly sentinel-safe
for GLYPH v0.x.

--------------------------------------------------
FINDING 3
--------------------------------------------------

A sanitized benchmark corpus was created.

Name:

english_2gb_prefix_no_nulls.txt

Properties:

size:
2,000,000,000 bytes

null bytes:
0

Result:

Compatible with current GLYPH pipeline.

--------------------------------------------------
FINDING 4
--------------------------------------------------

GLYPH successfully built:

SA
BWT
FM

on the canonical Pizza & Chili derived corpus.

No build failures were observed.

--------------------------------------------------
FINDING 5
--------------------------------------------------

Observed index ratios:

SA:
4.0x

BWT:
1.0x

FM:
4.0x

Total stack:
~9.0x

Observation:

This is consistent with previous HDFS-scale measurements.

Hypothesis:

Current memory ratio is dominated by architecture
rather than corpus semantics.

--------------------------------------------------
STATUS
--------------------------------------------------

Pizza & Chili corpus integration:

SUCCESS

Next phase:

PIZZA_CHILI_QUERY_V1

Goals:

1. Retrieval correctness.
2. Query latency.
3. Compare with previous HDFS observations.
