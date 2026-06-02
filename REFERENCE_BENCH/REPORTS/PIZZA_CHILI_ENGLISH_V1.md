# PIZZA_CHILI_ENGLISH_V1

Corpus:
Pizza & Chili english

Source:https://pizzachili.dcc.uchile.cl/texts/nlang/

Compressed:
834,550,281 bytes

Original:
2,210,395,553 bytes

Working Corpus:

english_2gb_prefix_no_nulls.txt

Size:
2,000,000,000 bytes

Findings:

1. Corpus exceeds SA32-safe threshold if used in full.

2. A 2GB prefix was created for controlled benchmarking.

3. Original corpus contains 13 embedded 0x00 bytes.

4. Null bytes are clustered around ~787MB offset.

5. Null bytes appear inside textual content.

6. A sanitized benchmark corpus was created by replacing
   0x00 with ASCII space (0x20).

7. Sanitized corpus preserves byte count and removes
   sentinel conflicts.

Result:

english_2gb_prefix_no_nulls.txt becomes the canonical
Pizza & Chili benchmark corpus for GLYPH v0.x.

Status:

READY FOR INDEXING
