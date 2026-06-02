# PIZZA_CHILI_INDEX_BUILD_V1

Corpus:
Pizza & Chili english 2GB sanitized prefix

Input:
REFERENCE_BENCH/CORPORA/english_2gb_prefix_no_nulls.txt

Corpus size:
2,000,000,000 bytes

Build outputs:

SA:
REFERENCE_BENCH/OUT/pizza_chili_english_2gb/sa.bin
Size:
8,000,000,000 bytes
Ratio:
4.0x
Build time:
2m03.883s

BWT:
REFERENCE_BENCH/OUT/pizza_chili_english_2gb/bwt.bin
Size:
2,000,000,000 bytes
Ratio:
1.0x
Build time:
20.634s

FM:
REFERENCE_BENCH/OUT/pizza_chili_english_2gb/fm.bin
Size:
8,000,003,116 bytes
Ratio:
4.0x
Build time:
18.745s

FM checkpoint step:
256

FM blocks:
7,812,501

Combined index without SA:
10,000,003,116 bytes
Ratio:
5.0x

Combined index with SA:
18,000,003,116 bytes
Ratio:
9.0x

Result:
GLYPH successfully built SA, BWT, and FM artifacts on a canonical Pizza & Chili derived corpus.

Status:
BUILD COMPLETE

Next:
Run exact retrieval correctness and latency tests on this corpus.
