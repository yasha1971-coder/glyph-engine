# GLYPH_EXTERNAL_VERIFIER_PLAN_V1

Status:
ACTIVE

Goal:

Move from:

External Verifiers = 0

to:

External Verifiers = 3

Core principle:

Do not ask people to review GLYPH.

Ask them to break one falsifiable claim.

Primary claim:

Same corpus + same build spec
produces the same evidence hash
and the same replay result
on another machine.

Verifier target:

A named human who runs the verification flow
and reports VERIFY OK or FAIL.

Current metric:

External Verifiers:
0

Target metric:

External Verifiers:
3

Required artifact:

VERIFY-IN-3-MINUTES

Minimal flow:

git clone glyph-engine
cd glyph-engine
./verify.sh

Expected output:

VERIFY OK

Rules:

1. No build required before first verification.
2. No dependency setup.
3. No long documentation.
4. No giant corpus.
5. No custom input from verifier.
6. No marketing language.
7. No claim of being a product.
8. No RAG positioning.
9. No speed claims before reproducibility claims.
10. No RAM discussion before verification succeeds.

Initial target:

One tiny fixed public corpus.

Preferred:

small enwik8 slice
zero 0x00 bytes
repo-pinned expected hash

Verification flow:

1. Load fixed corpus.
2. Run deterministic retrieval or replay.
3. Generate evidence object.
4. Compute evidence SHA256.
5. Compare against expected hash.
6. Run replay verifier.
7. Print VERIFY OK.

First verifier strategy:

Verifier #1:
warmest known infrastructure person.

Verifier #2:
already-engaged technical person.

Verifier #3:
public ask after two confirmations.

Do not contact cold academics before verifier #1 exists.

Failure modes:

1. Asking “what do you think?”
2. Requiring a build.
3. Leading with FM-index.
4. Leading with performance.
5. Mentioning market/compliance.
6. Using a large corpus.
7. Allowing custom corpus.
8. Posting publicly before one named verifier.

Next engineering task:

Create verify.sh for tiny fixed verification path.
