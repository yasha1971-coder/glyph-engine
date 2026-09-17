# GLYPH Personal Memory — Golden Demo RC1

Status: runnable review-candidate toolkit, not an industrial production release.
The actual laptop/GOLDEN measurements are pending. No corpus was downloaded or
benchmark payload published by this change. User supplied MANIFEST.md was read;
its historical claims are not substituted for verification of local inputs.

## Selection and primary evidence

Use all 12 Silesia files, 211,938,580 bytes, with exact official sizes and MD5.
Primary source actually read:
https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
The runtime also records SHA-256. MD5 is a compatibility corpus identity check,
not modern cryptographic authentication of the publisher. There is no cherry-picked
single-file score. Includes text, binary/executable, database and scientific files.
Important: ooffice is an executable DLL, not an office document. reymont is an
explicitly uncompressed PDF; results are not representative of modern compressed
PDF/media. Silesia excludes common lossy multimedia, so it cannot establish the
average personal-data savings. The prior VIKA result is a separate measurement.

Other sources read for selection:
https://corpus.canterbury.ac.nz/descriptions/
https://mattmahoney.net/dc/textdata.html
Canterbury is useful but much smaller; full enwik8/9 and repetitive corpora are
outside current per-file pilot budgets and not silently truncated for a headline.
No current runtime claim for Index Forest, RLBWT self-index or a local LLM.

## Reproduce on the laptop, WSL

Use a dedicated code checkout, keep the existing personal-memory installation.
From that checkout:

    python3 experiments/personal_memory_1tb_v2/golden_demo.py build --golden /mnt/d/GOLDEN --output /home/contour/GlyphPilot/GLYPH-GOLDEN-DEMO-RC1
    python3 experiments/personal_memory_1tb_v2/golden_demo.py serve --output /home/contour/GlyphPilot/GLYPH-GOLDEN-DEMO-RC1

The build requires 2 GiB free in the output parent. It refuses an existing output;
no destructive cleanup/retry is automatic. An incomplete run has no success report.
The serve command only opens a completed checksum-verified report and pinned base.
It uses localhost:8767, separate from personal-memory port 8766. Normal later use
runs only serve; do not rebuild. Existing dates/notes/versions in the personal vault
and all ACEAPEX code/data are untouched. GOLDEN is only read. Name discovery is
bounded to that directory (depth 6, 20000 entries); no disk-D inventory is repeated.
All 12 unpacked canonical filenames must exist and match. Compressed distributions
are not auto-extracted; missing/mismatching inputs stop before output creation.

## What is measured

1. Copy byte-identical verified specimens to isolated input. Only presentation
   aliases: dickens.txt, reymont.pdf, xml.xml; mappings and original names recorded.
2. Build existing raw/deflate9/bzip2-9/xz-9 hybrid archive. This self-contained demo
   needs no external Precomp binary; no claim that it is the best possible codec.
3. Restore all 12 files and independently compare complete bytes to staged source.
4. Make 100 explicitly synthetic XML edits: insertion, deletion and reversion.
   Verify each version, then cold-reopen and verify all 101 states again. Generated
   versions are never included in the original Silesia compression score.
5. Record overlay logical and allocated bytes including snapshots/recipes/objects;
   record a whole-file raw/deflate6 baseline with exact-identity deduplication.
   Baseline counts new payload only; overlay includes metadata: not equal scopes.
   Input staging copies, temporary verification and code are not archive bytes.
6. Add a deterministic generated gzip control, excluded from the Silesia score.
7. On a disposable overlay copy, corrupt that payload, require rejection, delete
   the corrupted record and verify remaining history. The live demo stays intact.
8. Prove one explicit selected-file byte scan. This is NOT an indexed search UI.

Output: GLYPH_GOLDEN_DEMO_RC1.json plus SHA-256 sidecar. Contains source identities,
per-file payloads/codecs, full-container ratio, timings, per-version identities,
code commit/clean flag/runtime source hashes, environment and limitation list.
Do not present results until this report exists. Self-checksum is integrity checking,
not an externally signed release. The report's initial current_snapshot describes
the build; subsequent intentional edits in the demo may advance CURRENT.

## Human demonstration (after local gates)

Open reymont.pdf for PDF view; xml.xml for readable text and 101-version history.
Choose a named historical version and download it. Add a disposable document,
update it, preview, then confirm deletion. Show the generated .gz as an already
compressed control (download only). Unsupported formats are not executed.
Text preview adds safe plain-text XML; no XML/HTML execution is enabled.

Review scope: exact preservation, economical versions, human-controlled actions,
corruption handling and reproducible evidence. UI full-text/semantic search, LLM,
key management/encryption, packaging with one-click Windows startup, 1 TB resource
qualification, independent security review and modern-media benchmark remain open.
Corpus redistribution licensing must be checked before shipping actual specimens;
the GitHub change contains only code/docs/synthetic tests, never the corpus itself.

## Validation here

117 synthetic tests pass, 36.958 seconds; targeted four-test rerun passes after the
baseline dedup accounting correction. GOLDEN_DEMO_TEST_OUTPUT.txt includes both.
The demo workflow test uses a tiny synthetic profile, not the real Silesia corpus.
Real D:/GOLDEN corpus execution and graphical browser acceptance are pending.
