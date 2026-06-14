# STORAGE_CLEANUP_2026_06

Actions:

1. Removed fm_builder_test after verifying it was a full duplicate of fm_sentinel_fix.

Verified identical SHA256 for:
- fm.bin
- sa.bin
- bwt.bin
- corpus sentinel

Freed:
~7.1G

2. Removed duplicate SA test files:

- out_512mb/sa_u32_test.bin
- out_2gb/sa_u32_test.bin

Verified identical SHA256 against:
- out_512mb/sa.bin
- out_2gb/sa.bin

Freed:
~11.7G

Result:

Disk moved from approximately 99% used to 97% used.

No unique GLYPH source, spec, report, evidence, or commitment artifacts were deleted.

3. Archived and removed out_4gb.

Path:
out_4gb

Archive destination:
External SSD / GLYPH_ARCHIVE2026-06/out_4gb

Verified SHA256 match on:
- fm.bin
- sa.bin
- bwt.bin
- chunk_map.bin

Archive manifest:
REFERENCE_BENCH/ARCHIVE/out_4gb_archive_manifest_2026_06.md

Checksum list:
REFERENCE_BENCH/ARCHIVE/out_4gb_sha256_2026_06.txt

Freed:
~75G

Restore rule:
Copy the archived directory back to:
~/GLYPH_CPP_BACKEND/out_4gb

4. Archived and removed glyph-public-bench.

Path:
/tmp/glyph-public-bench

Archive destination:
External SSD / GLYPH_ARCHIVE2026-06/glyph-public-bench

Verified SHA256 match on:
- enwik9
- out/fm.bin
- out/sa.bin

Archive manifest:
REFERENCE_BENCH/ARCHIVE/glyph_public_bench_archive_manifest_2026_06.md

Checksum list:
REFERENCE_BENCH/ARCHIVE/glyph_public_bench_sha256_2026_06.txt

Freed:
~55G

Restore rule:
Copy archive back to:
/tmp/glyph-public-bench
