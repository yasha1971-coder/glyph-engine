# GLYPH

[![CI status](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yasha1971-coder/glyph-engine/actions/workflows/ci.yml)

**Verifiable exact-byte retrieval over committed corpora.**

GLYPH turns an exact binary query into a deterministic, replayable evidence result:

```text
committed corpus
→ exact query bytes
→ match count
→ canonical document coordinates
→ byte checks
→ portable evidence bundle
→ independent replay
```

GLYPH is designed for fixed corpora that have become archives, evidence
objects, reference datasets, or other immutable inputs where reproducibility
matters more than ranking or semantic similarity.

## Current status

| Layer | Status |
|---|---|
| Reference semantics P1–P12 | Verified by executable gates |
| Binary-safe compiled runtime R0–R6 | Verified by executable gates |
| Real-world operator workflow O1–O6 | Verified by executable gates |
| Embedded C ABI contract I0 | Draft pre-freeze contract; executable gate passes |
| Embedded shared-library implementation | Not implemented |
| Second external pre-freeze review | Pending |
| Production readiness | Not claimed |

The `main` branch requires every verified layer to pass before it prints
`VERIFY OK`.

## Start here

```bash
git clone https://github.com/yasha1971-coder/glyph-engine.git
cd glyph-engine
./verify.sh
```

The final closure must include:

```text
GLYPH PROOF GRAPH OK
GLYPH RUNTIME CONFORMANCE OK
GLYPH OPERATOR CONFORMANCE OK
GLYPH EMBEDDED I0 CONTRACT VERIFY PASS
VERIFY OK
```

`VERIFY OK` is not printed when a required proof, runtime, operator, or
embedded-contract gate fails.

## What GLYPH verifies

The narrow evidence claim is:

> These exact query bytes occurred at these exact coordinates in this exact
> committed corpus state, under the verified retrieval and replay rules.

Depending on the artifact and workflow, the verification chain binds or checks:

- corpus identity;
- source-document identity;
- exact binary query bytes;
- FM interval and match count;
- canonical document coordinates;
- source-byte checks;
- runtime and engine identity;
- deterministic manifests;
- portable evidence-bundle coverage;
- independent bundle replay.

Recorded coordinates can be checked directly against the committed source
snapshot. The repository also verifies the compiled runtime and operator
workflow that produce those coordinates.

## What GLYPH does not claim

GLYPH does not prove:

- that a source statement is true;
- authorship, intent, or attribution;
- legal admissibility;
- complete incident reconstruction;
- semantic relevance;
- freshness or absence of replay;
- protection from a privileged host attacker;
- production readiness.

GLYPH is not:

- semantic or vector search;
- fuzzy search;
- ranked retrieval;
- a SIEM or log-management platform;
- a general replacement for `grep`;
- a zero-knowledge proof system.

See [`WHAT_GLYPH_IS_NOT.md`](WHAT_GLYPH_IS_NOT.md).

## Verification architecture

```text
P1–P12
Reference semantics
    ↓
GLYPH PROOF GRAPH OK
    ↓
R0–R6
Binary-safe compiled runtime
    ↓
GLYPH RUNTIME CONFORMANCE OK
    ↓
O1–O6
Filesystem → index → query → bundle → replay
    ↓
GLYPH OPERATOR CONFORMANCE OK
    ↓
I0
Embedded C ABI pre-freeze contract
    ↓
GLYPH EMBEDDED I0 CONTRACT VERIFY PASS
    ↓
VERIFY OK
```

The layers are intentionally separate:

- the reference layer defines exact retrieval semantics;
- the runtime layer checks the compiled binary-safe implementation;
- the operator layer checks the complete filesystem-to-evidence workflow;
- the embedded layer defines integration, lifetime, resource, mmap, and
  authenticity contracts before implementation begins.

## Embedded API V1

The public C header and normative I0 contracts exist, but the shared-library
implementation has not started.

The pre-freeze contract defines:

- fixed-width C ABI types and frozen V1 layouts;
- caller-owned bounded locate output;
- deterministic document identity;
- exact failure and output semantics;
- concurrent read-only handle behavior;
- caller-side close barriers;
- resource and timeout boundaries;
- read-only mmap trust assumptions;
- hostile-file parsing requirements;
- signed-statement policy;
- resistance to signature stripping.

The I0 checker verifies normative-section hashes, cross-file invariants,
public-header layout, C and C++ consumer compatibility, negative mutations,
committed-result reproducibility, and the absence of a premature embedded ABI
implementation.

The ABI must not freeze and E1 implementation must not begin until the second
external pre-freeze review is resolved.

Relevant documents:

- [`docs/specs/GLYPH_C_ABI_V1.md`](docs/specs/GLYPH_C_ABI_V1.md)
- [`docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md`](docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md)
- [`docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md`](docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md)
- [`docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md`](docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md)
- [`docs/specs/GLYPH_SIGNED_STATEMENT_V1.md`](docs/specs/GLYPH_SIGNED_STATEMENT_V1.md)
- [`docs/reviews/GLYPH_EMBEDDED_I0_PREFREEZE_REVIEW_DISPOSITION_V1.md`](docs/reviews/GLYPH_EMBEDDED_I0_PREFREEZE_REVIEW_DISPOSITION_V1.md)

## Primary entry points

| Path | Purpose |
|---|---|
| [`verify.sh`](verify.sh) | Complete required verification chain |
| [`examples/mini/`](examples/mini/) | Small self-contained exact-retrieval example |
| [`examples/public-evidence-demo/`](examples/public-evidence-demo/) | Public-corpus evidence workflow |
| [`tools/run_glyph_operator_conformance_graph_v1.sh`](tools/run_glyph_operator_conformance_graph_v1.sh) | Operator conformance closure |
| [`docs/specs/`](docs/specs/) | Normative specifications |
| [`docs/reviews/`](docs/reviews/) | External reviews and dispositions |
| [`benchmarks/results/`](benchmarks/results/) | Committed deterministic verification artifacts |

## Suitable use cases

### Forensic evidence retrieval

Preserve and replay the exact source bytes supporting an investigative claim.

### Static log or archive verification

Query a committed corpus repeatedly without treating screenshots or copied
excerpts as the evidence record.

### RAG and AI provenance controls

Bind a generated claim to exact source fragments rather than only document
identifiers, embeddings, or similarity scores.

## Current limitations

- Primary workflows target immutable or committed corpora.
- Some legacy index paths require source corpora without embedded `0x00`.
- Current index representations are not optimized for minimal storage.
- Timeout enforcement is reserved by the ABI but is not implemented.
- The Embedded C ABI is a checked contract, not a working shared library.
- mmap-backed embedded querying is not yet implemented.
- Digital signing semantics are specified, but signing is not implemented.
- The trusted-local-filesystem mmap profile does not protect against a
  privileged writer mutating published inodes.
- HTTP, segmented-query, and compressed-runtime paths remain experimental
  unless explicitly covered by a committed verification gate.
- Broad production-throughput claims are not made.

## Additional research paths

The repository also contains:

- compressed RLBWT bounded-evidence experiments;
- Structural Fingerprint replay experiments;
- persistent HTTP and segmented-query prototypes;
- public benchmark runbooks;
- compact runtime and locate research.

These paths do not replace the required verification chain above.

## Documentation map

- Verification claim:
  [`docs/specs/GLYPH_VERIFICATION_CLAIM_V1.md`](docs/specs/GLYPH_VERIFICATION_CLAIM_V1.md)
- Engine overview:
  [`docs/architecture/ENGINE_OVERVIEW.md`](docs/architecture/ENGINE_OVERVIEW.md)
- Known limitations:
  [`docs/specs/KNOWN_LIMITATIONS.md`](docs/specs/KNOWN_LIMITATIONS.md)
- Project boundaries:
  [`WHAT_GLYPH_IS_NOT.md`](WHAT_GLYPH_IS_NOT.md)

## Project links

- Website: [glyph.rs](https://glyph.rs)
- Repository: [glyph-engine](https://github.com/yasha1971-coder/glyph-engine)
- Use-case issue: [Issue #3](https://github.com/yasha1971-coder/glyph-engine/issues/3)
- Discussion: [Discussion #4](https://github.com/yasha1971-coder/glyph-engine/discussions/4)

GLYPH is seeking independent reviewers and real-world users with fixed-corpus
exact-retrieval, provenance, forensic, or auditability problems.
