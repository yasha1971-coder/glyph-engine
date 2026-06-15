# EXTERNAL_VERIFIER_DRY_RUN_V1
Status:
PASS
Date:
2026-06-15
Purpose:
Validate the external verification path after fixing verify.sh to build the canonical mini pipeline dependencies.
Test model:
Fresh clone in /tmp.
Commands:
```bash
cd /tmp
rm -rf glyph-verify-test
git clone https://github.com/yasha1971-coder/glyph-engine.git glyph-verify-test
cd glyph-verify-test
time ./verify.sh

Observed result:

[verify] GLYPH one-command verification
[verify] building required binaries
[100%] Built target build_sa_u32
[100%] Built target build_bwt
[100%] Built target build_fm
[100%] Built target query_fm_v1
VERIFY OK
real    0m3.759s
user    0m3.237s
sys     0m0.522s

Meaning:

A fresh clone successfully built the required canonical mini dependencies and completed the verification path.

Verified claim:

The canonical mini pipeline produced the expected result:

VERIFY OK

This confirms that the main external verification gate is repaired.

Previous blocker:

verify.sh previously built the wrong target set and failed in a clean clone.

Fixed by:

23fd7b2 verify: build canonical mini dependencies

Current status:

External verification infrastructure:
READY FOR HUMAN VERIFIER

External Verifiers:
0

Next target:

External Verifier #1

