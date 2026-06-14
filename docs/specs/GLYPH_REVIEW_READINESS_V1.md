# GLYPH_REVIEW_READINESS_V1

Status:
DRAFT

Purpose:

Determine whether a first independent verifier can successfully evaluate GLYPH in under 3 minutes.

---

## Review Path

Expected verifier flow:

1. Open repository.
2. Read README.
3. Run:

```bash
./verify.sh
```

4. Observe:

```text
VERIFY OK
```

5. Understand what was verified.
6. Decide whether the claim is credible.

---

## Verified Claim

The canonical GLYPH mini corpus produces a deterministic, sentinel-safe FM-index and returns an exact match count of 2 for the query:

```text
error
```

This is the current minimal external verification claim.

---

## Readiness Checklist

### Repository Discovery

- [ ] Core claim visible near top of README
- [ ] Verification path visible near top of README
- [ ] No large documentation barrier before verification

### Verification

- [ ] verify.sh works on clean machine
- [ ] verify.sh reports clear success
- [ ] verify.sh reports clear failure
- [ ] verification requires minimal dependencies

### Understanding

- [ ] Verification claim documented
- [ ] Verification claim linked from README
- [ ] Core guarantees documented
- [ ] Known limitations documented

### Reproducibility

- [ ] Deterministic result
- [ ] Fixed mini corpus
- [ ] Sentinel-safe invariant documented
- [ ] Same verification result expected across machines

### External Reviewer Experience

Question:

Can a competent engineer determine what GLYPH is and what was verified in less than three minutes?

Status:

UNKNOWN

---

## Current Blockers

Unknown.

Awaiting first external verifier.

---

## Success Condition

A reviewer can:

1. Clone the repository.
2. Run verify.sh.
3. Observe VERIFY OK.
4. Understand the verified claim.
5. Decide whether to investigate further.

Without reading large amounts of documentation.