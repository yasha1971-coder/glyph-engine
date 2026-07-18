# GLYPH_EMBEDDED_I0_SECOND_REVIEW_BRIEF_V1

Status:

    READY FOR INDEPENDENT REVIEW

Review phase:

    SECOND_EXTERNAL_PREFREEZE_REVIEW

Review branch:

    embedded-i0-second-review-v1

Baseline repository identity:

    baseline_repository_commit:
    9f79fd5cf8f969ed46c2c1ad7945f6b5a2944edb

Review package identity:

    review_package_commit:
    recorded by GLYPH_EMBEDDED_I0_SECOND_REVIEW_MANIFEST_V1

Contract source identity:

    reviewed_contract_commit:
    fe94eb0314cff76191a7eaba20347c234069a8f2

Current contract status:

    DRAFT_NOT_FROZEN

Implementation status:

    Embedded E1 implementation has not started.

## Purpose

This review is the final independent line-by-line challenge of the
Embedded C ABI V1 contract before an ABI freeze decision.

The reviewer must determine whether the public header and the five
normative specifications form one coherent, implementable, bounded,
portable and security-conscious contract.

This review is not a request to approve the project in general.

It is a request to identify every remaining contract defect that could:

- make independent implementations observably incompatible;
- permit contradictory interpretations;
- cause unsafe lifetime or ownership behavior;
- expose unbounded resource behavior;
- invalidate deterministic document coordinates;
- weaken hostile-file handling;
- produce ambiguous status or failure semantics;
- make mmap-backed verification claims unsound;
- permit signature-policy downgrade or stripping;
- freeze an ABI layout that cannot be maintained;
- block reliable testing or fault injection later.

## Mandatory review inputs

The reviewer must inspect these files line by line:

1. `include/glyph/glyph.h`
2. `docs/specs/GLYPH_C_ABI_V1.md`
3. `docs/specs/GLYPH_EMBEDDED_THREAT_MODEL_V1.md`
4. `docs/specs/GLYPH_MMAP_TRUST_MODEL_V1.md`
5. `docs/specs/GLYPH_RESOURCE_FAILURE_MODEL_V1.md`
6. `docs/specs/GLYPH_SIGNED_STATEMENT_V1.md`

The reviewer must also inspect these supporting records:

7. `docs/reviews/GLYPH_EMBEDDED_I0_PREFREEZE_REVIEW_DISPOSITION_V1.md`
8. `tools/check_embedded_i0_contract_v1.py`
9. `benchmarks/results/GLYPH_EMBEDDED_I0_CONTRACT_V1.json`
10. `verify.sh`

## Exact review scope

The review covers:

- C ABI symbol namespace;
- function signatures;
- fixed-width public types;
- structure sizes, offsets and versioning;
- nullable and non-null argument rules;
- output initialization on every failure class;
- handle ownership and lifetime;
- close behavior and caller-side close barrier;
- concurrent read-only operations;
- stale-pointer and repeated-close boundaries;
- exact query-byte lifetime;
- bounded locate behavior;
- count and coordinate consistency;
- canonical coordinate ordering;
- deterministic document identity;
- source-path retrieval semantics;
- status-code partitioning;
- unsupported-feature behavior;
- checked arithmetic;
- allocation and memory-failure behavior;
- timeout and resource-limit semantics;
- interrupted open and I/O failures;
- malformed and hostile artifact handling;
- descriptor-relative opening;
- root-directory anchoring;
- mmap identity and mutation assumptions;
- integrity versus authenticity boundaries;
- signature policy;
- signature stripping and downgrade resistance;
- freshness and timestamp non-claims;
- compatibility and future-extension boundaries;
- testability of every normative requirement.

## Explicitly out of scope

The review must not evaluate:

- E1 implementation quality, because E1 does not exist;
- query-performance claims;
- compressed RLBWT performance;
- distributed deployment;
- HTTP service behavior;
- UI or website design;
- business viability;
- production readiness;
- legal admissibility;
- cryptographic algorithm implementation;
- assurance graph A0-A8 implementation.

The future SQLite-inspired GLYPH Assurance Graph A0-A8 is planned but
not part of this I0 freeze review.

However, the reviewer should flag any contract wording that would make
future requirement traceability, fault injection, mutation testing,
coverage, fuzzing or independent harnesses impossible or ambiguous.

## Required reviewer questions

The reviewer must answer every question below.

### Public ABI

1. Can a strict C99 consumer include and use the header?
2. Can a C11 and C++17 consumer use the same ABI without interpretation drift?
3. Are all public types platform-stable inside the supported host profile?
4. Can future structure growth occur without breaking frozen V1 callers?
5. Are all reserved fields and initialization requirements unambiguous?
6. Is every exported function uniquely identifiable in the C namespace?

### Ownership and lifetime

7. Is ownership of every pointer and handle unambiguous?
8. Is successful close behavior unambiguous?
9. Is `GLYPH_E_BUSY` close behavior recoverable and free of hidden latch state?
10. Are concurrent query and close obligations divided clearly between caller
    and library?
11. Are repeated close, null handle and stale-pointer cases distinguished?

### Query and locate

12. Are query bytes treated as exact binary data?
13. Is empty-query behavior explicit?
14. Are count and locate semantics consistent?
15. Is bounded incomplete locate distinguishable from operation failure?
16. Is canonical coordinate ordering defined strongly enough for independent
    implementations?
17. Are output buffers and required capacities unambiguous?

### Document identity and paths

18. Is `doc_id` assignment deterministic from committed source-manifest order?
19. Can the complete document domain be enumerated without hidden state?
20. Are out-of-range document identifiers handled consistently?
21. Are raw path bytes, lengths and size probes fully specified?
22. Are path traversal and absolute-path cases rejected at the correct layer?

### Resource and failure behavior

23. Is every integer calculation required to be checked before use?
24. Can any query operation hide an unbounded allocation or coordinate
    collection?
25. Are OOM, I/O, timeout, limit, verification and internal failures
    distinguishable?
26. Are failure outputs deterministic?
27. Is partial success permitted only where explicitly defined?
28. Are open-time resource boundaries and deadline non-claims honest?

### mmap and hostile files

29. Does the required open sequence bind verification to the same opened file
    description that is mapped?
30. Are descriptor-relative lookup and root anchoring sufficient for the stated
    threat model?
31. Are replacement races and size-preserving mutation addressed honestly?
32. Are privileged-writer and post-open mutation limits explicit?
33. Is the strongest permitted integrity claim narrower than the threat model?

### Signed statements

34. Is the unsigned-versus-required-signature policy explicit?
35. Can signature stripping be detected?
36. Is the signing preimage deterministic and canonical?
37. Are trust-store and key-selection responsibilities explicit?
38. Are authenticity, freshness and timestamps correctly separated?
39. Are mandatory rejection cases complete enough to prevent downgrade?

### Cross-document consistency

40. Does the header agree with the ABI specification?
41. Does the ABI specification agree with the threat model?
42. Does the mmap trust model agree with the hostile-filesystem boundary?
43. Does the resource model agree with every function’s failure semantics?
44. Does the signed-statement specification agree with the open policy?
45. Do any two normative passages assign contradictory obligations?

### Freeze readiness

46. Is any normative statement impossible to implement portably in the stated
    host profile?
47. Is any behavior observable by callers but left unspecified?
48. Is any future extension path likely to break frozen V1 layouts?
49. Is any requirement not objectively testable?
50. Should the ABI freeze now, freeze after named corrections, or remain
    blocked?

## Finding severity

Every finding must use exactly one severity.

### BLOCKER

The ABI must not freeze while the finding is open.

Examples:

- memory-unsafe ownership ambiguity;
- contradictory observable behavior;
- incompatible independent implementations;
- unbounded query behavior;
- unsound integrity claim;
- frozen layout defect;
- signature downgrade vulnerability;
- status or output behavior that cannot be tested deterministically.

### MAJOR

The finding is not necessarily an immediate memory-safety defect, but
it materially weakens portability, reliability, security, testability
or compatibility.

The freeze is blocked unless the reviewer explicitly explains why the
finding may be deferred.

### MINOR

The contract is implementable, but wording, cross-reference, examples
or non-observable details should be corrected before final publication.

### NOTE

Non-blocking observation, future work or explanatory suggestion.

A NOTE must not silently introduce a new V1 requirement.

## Mandatory finding format

Each finding must use this exact structure:

    finding_id:
    severity:
    title:
    affected_files:
    affected_sections:
    affected_lines_or_symbols:
    contract_text:
    problem:
    observable_consequence:
    minimal_reproduction_or_counterexample:
    required_resolution:
    freeze_effect:
    verification_test_required:
    confidence:

`freeze_effect` must be one of:

- `BLOCKS_FREEZE`
- `BLOCKS_FREEZE_UNLESS_JUSTIFIED`
- `DOES_NOT_BLOCK_FREEZE`

`confidence` must be one of:

- `HIGH`
- `MEDIUM`
- `LOW`

## Reviewer non-invention rule

The reviewer must distinguish:

- a defect in the written contract;
- a preferred alternative design;
- future functionality;
- implementation advice;
- speculation.

A preferred redesign is not automatically a blocker.

A blocker requires a demonstrated contradiction, unsafe behavior,
observable incompatibility, unverifiable requirement or broken trust
boundary.

## Required adversarial method

The reviewer should actively construct counterexamples, including:

- two independent implementations choosing different valid interpretations;
- maximum and zero-size arguments;
- truncated and size-preserving corrupted files;
- `doc_id` at both boundaries and beyond the domain;
- `max_offsets` below, equal to and above total matches;
- null pointers in every conditionally nullable position;
- close attempted with active operations;
- close retry after `GLYPH_E_BUSY`;
- integer overflow near `UINT64_MAX`;
- signature removed from an otherwise valid statement;
- unknown signing key;
- valid signature over mismatched corpus identity;
- file replacement between path lookup, verification and mapping;
- failure after outputs have been partially touched.

## Required review result

The review must end with exactly one verdict:

### FREEZE_ACCEPTED

Allowed only when:

- no BLOCKER remains;
- no unjustified MAJOR remains;
- all observable behavior is sufficiently specified;
- the reviewer finds the contract implementable and independently testable.

### FREEZE_ACCEPTED_WITH_MINOR_CORRECTIONS

Allowed only when:

- no BLOCKER remains;
- no MAJOR remains;
- listed MINOR corrections do not alter observable ABI semantics.

### FREEZE_BLOCKED

Required when:

- at least one BLOCKER remains;
- an unresolved MAJOR affects compatibility, safety, trust or testability;
- the review is incomplete;
- required files were not inspected;
- evidence is insufficient to justify freeze.

## Required review summary

The reviewer must report:

    verdict:
    reviewed_snapshot:
    reviewed_contract_commit:
    reviewer_identity_or_agent:
    review_method:
    files_reviewed:
    blocker_count:
    major_count:
    minor_count:
    note_count:
    unresolved_questions:
    freeze_conditions:
    implementation_may_begin:
    review_completed_at:

`implementation_may_begin` must be `false` unless the final accepted
disposition is committed and the I0 artifact is regenerated with a
frozen status.

## Project response rule

GLYPH must not respond to findings by immediately changing the contract
one item at a time.

The required sequence is:

1. collect the complete second review;
2. classify every finding;
3. create one review disposition document;
4. decide accepted, modified or rejected resolution for every finding;
5. apply coherent contract changes as one controlled group;
6. harden checker mutations and invariants;
7. regenerate the committed I0 artifact;
8. run the clean-environment verification chain;
9. obtain final freeze confirmation;
10. only then consider E1 implementation.

## Freeze barrier

Until the second review is completed and its disposition is accepted:

- `status` remains `DRAFT_NOT_FROZEN`;
- E1 implementation remains forbidden;
- no production-readiness claim is permitted;
- no shared-library release is permitted;
- public ABI layouts must not be described as frozen.
