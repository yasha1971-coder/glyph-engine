GLYPH_VERIFICATION_CLAIM_V1

Status: Draft

Claim

GLYPH makes a reproducibility claim.

Given:

* the same corpus
* the same GLYPH version
* the same verification procedure

GLYPH should produce the same retrieval evidence.

The claim is intended to be independently testable.

No trust in the author is required.

⸻

Verification Procedure

Clone the repository:

git clone https://github.com/yasha1971-coder/glyph-engine.git
cd glyph-engine

Run:

./verify.sh

Expected result:

VERIFY OK

⸻

What VERIFY OK Means

VERIFY OK means:

1. The canonical mini corpus was indexed successfully.
2. The canonical sentinel-safe pipeline executed successfully.
3. FM-index construction completed successfully.
4. A known query produced the expected result.
5. The verification procedure completed without modification.

VERIFY OK does not prove:

* performance claims
* scalability claims
* production readiness
* superiority over other retrieval systems

It only verifies the reproducibility claim of the reference pipeline.

⸻

Falsification

The claim is considered falsified if an independent verifier can demonstrate that:

* the same verification procedure
* on a supported platform
* using the same repository state

produces a different result.

Examples:

* verification output differs
* expected query count differs
* deterministic evidence differs
* reproducibility cannot be achieved

A reproducible counterexample is sufficient.

⸻

Scope

This claim currently applies only to:

* the canonical mini verification pipeline
* the repository state associated with this specification

It does not automatically extend to:

* larger corpora
* segmented deployments
* benchmark configurations
* future releases

Those require independent verification.

⸻

Goal

The goal of this specification is simple:

Allow an independent engineer to test a concrete GLYPH claim in minutes.

Success metric:

External Verifiers > 0

A verifier is a person independent from the author who runs the verification procedure and reports the result.
