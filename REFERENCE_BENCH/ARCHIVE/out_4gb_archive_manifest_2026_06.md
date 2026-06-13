# OUT_4GB_ARCHIVE_MANIFEST_2026_06

Status:
READY_FOR_EXTERNAL_ARCHIVE

Path:
out_4gb

Size:
75G

Reason for archive:
Large historical 4GB index artifact from pre-provenance GLYPH phase.

Referenced by:
- RUNBOOK_4GB.md
- archive/CORE_FREEZE_v1.2.md
- manifests/segmented_8gb_demo.json
- config/shards_8gb_demo.json

Contents:
- bwt.bin
- chunk_map.bin
- chunk_map_u24.bin
- chunk_starts.csv
- fm.bin
- sa.bin

Archive destination:
External SSD / GLYPH_ARCHIVE/2026-06/out_4gb/

Restore path:
~/GLYPH_CPP_BACKEND/out_4gb

Restore rule:
Copy the directory back to ~/GLYPH_CPP_BACKEND/out_4gb exactly.

Important:
Do not edit files after archive.
Use checksum verification before deleting original.
