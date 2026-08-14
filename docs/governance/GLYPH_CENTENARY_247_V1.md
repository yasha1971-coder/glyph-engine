# GLYPH Centenary 247 — Facet Standard V1

Status: **DRAFT STANDARD / NOT A PRODUCTION-READINESS CLAIM**  
Established: **2026-08-14**  
Facet count: **exactly 247**  
Governed by: `GLYPH_PRESERVATION_LAW_V1.md`

## 1. Meaning of the name

De Beers records that the Centenary Diamond was cut with 247 perfectly aligned
facets. GLYPH uses that number as an engineering metaphor: every facet must
contribute a distinct, inspectable property to one coherent system. The analogy
does **not** claim that gemological prestige proves software quality.

The catalog establishes a standard to work toward; its publication does not
close any facet. A facet closes only when its stated evidence exists, can be
replayed, and survives the applicable independent review.

## 2. Non-negotiable interpretation

- GLYPH remains a domain-neutral deterministic exact-byte substrate. A use case
  may add an adapter, never redefine the core truth model.
- `247` is a cardinality invariant. A new facet must supersede or split an
  existing facet through an additive version of this document; it may not be
  silently appended as facet 248.
- Popularity is evidence of exposure, not proof of correctness. Stars, forks,
  issue counts, releases, adopters, standards ballots, and independent
  implementations are recorded as different evidence classes.
- `MUST`, `MUST NOT`, `SHOULD`, and `MAY` follow BCP 14 (RFC 2119 and RFC 8174).
- No item is deleted or destructively replaced without the Owner's explicit
  decision. Better variants coexist until the Owner authorizes disposition.

## 3. Closure rule

Each facet has four parts: an identifier, a normative property, a minimum
closure artifact, and an external exemplar. A closure ledger must later assign
one of `OPEN`, `PARTIAL`, `EVIDENCED`, `INDEPENDENTLY_REPLAYED`, or `OWNER_HELD`.
Only `INDEPENDENTLY_REPLAYED` is final technical closure; owner-governance
facets may close as `OWNER_HELD` after an explicit dated decision.

No aggregate “247/247” claim is permitted unless a clean checkout independently
replays every technical facet and the exact evidence manifest is published.

## 4. Evidence classes

| Code | Evidence class | What it can show | What it cannot show |
|---|---|---|---|
| P | Popularity | Broad attention or user exposure | Correctness, safety, or suitability |
| D | Discussion | Public scrutiny and unresolved trade-offs | Approval or correctness |
| A | Adoption/approval | Use by organizations or standards process | Correct implementation in GLYPH |
| T | Test | Observed behavior for covered cases | General mathematical truth |
| F | Formal | Proof under explicit assumptions | Correct assumptions or deployment |
| R | Replay | Independent reproduction of a claimed result | Fitness for an unstated purpose |

## 5. External exemplar registry

Metrics are a dated snapshot, **2026-08-14**, and are intentionally separated
from normative proof. Dynamic GitHub numbers are approximate because they
change continuously.

| Ref | Exemplar and quantified signal | Property imported into the standard |
|---|---|---|
| E01 | [De Beers: Centenary Diamond](https://www.debeers.co.uk/en-gb/legendary-diamonds.html) — 247 aligned facets | Exact cardinality and coherent alignment metaphor |
| E02 | [ripgrep](https://github.com/BurntSushi/ripgrep) — about 64.4k stars, 2.6k forks, 117 open issues | Fast practical search, honest scope, cross-platform release surface |
| E03 | [RocksDB](https://github.com/facebook/rocksdb) — about 31.7k stars, 6.8k forks, 13,896 commits, 214 releases | Storage-engine maturity, explicit public/internal API boundary, stress testing |
| E04 | [RE2](https://github.com/google/re2) — about 9.7k stars, 1.2k forks; production use since 2006 | Linear-time contract, configurable memory budget, graceful exhaustion |
| E05 | [SQLite testing](https://www.sqlite.org/testing.html) — 4 independent harnesses, 51,445 TCL cases, millions of runs; TH3 100% branch and MC/DC, about 248.5m soak instances | Test the tests, independent oracles, mutation, delivery-build validation |
| E06 | [OSS-Fuzz](https://github.com/google/oss-fuzz) — about 12.3k stars, 2.8k forks; over 13,000 vulnerabilities and 50,000 bugs found across 1,000 projects (May 2025) | Continuous sanitizer-backed fuzzing at ecosystem scale |
| E07 | [seL4](https://github.com/seL4/seL4) — about 5.5k stars, 760 forks, 30 releases; [l4v proofs](https://github.com/seL4/l4v) published separately | Separation of implementation, specification, assumptions, and machine proofs |
| E08 | [mathlib4](https://github.com/leanprover-community/mathlib4) — about 3.3k stars, 1.4k forks and public Zulip discussion | Community-maintained machine-checked mathematics and proof review |
| E09 | [The Update Framework](https://github.com/theupdateframework/python-tuf) — about 1.7k stars, 293 forks, 6,714 commits; CNCF-hosted and used in production | Compromise-resilient metadata, roles, expiry, rollback/freeze protection |
| E10 | [in-toto](https://github.com/in-toto/in-toto) — about 1k stars, 155 forks, 31 releases; CNCF project | Signed step layout and verifiable artifact chain |
| E11 | [Reproducible Builds definition](https://reproducible-builds.org/docs/definition/) | Bit-for-bit independent rebuild with specified inputs and outputs |
| E12 | [SLSA 1.2 Build Track](https://slsa.dev/spec/v1.2/build-track-basics) — status Approved, levels L0–L3 | Provenance, hosted builds, signed provenance, hardened builders |
| E13 | [NIST SP 800-218 SSDF 1.1](https://csrc.nist.gov/pubs/sp/800/218/final) — final publication shaped through public input/workshops | Outcome-based secure development and common assurance vocabulary |
| E14 | [RFC 8785 JCS](https://www.rfc-editor.org/rfc/rfc8785.html) | Invariant hashable JSON representation and duplicate-name prohibition |
| E15 | [RFC 2119](https://www.rfc-editor.org/info/rfc2119/) + [RFC 8174](https://www.rfc-editor.org/info/rfc8174/) — IETF Best Current Practice 14 | Unambiguous normative requirement language |
| E16 | [Semantic Versioning 2.0.0](https://semver.org/) | Declared public API, compatibility signaling, immutable published versions |
| E17 | [SPDX / ISO/IEC 5962:2021](https://www.iso.org/standard/81870.html) — published international standard after committee and ballot stages | Standard SBOM and license/component metadata |
| E18 | [OpenSSF Scorecard](https://scorecard.dev/) | Repeatable repository-security checks and visible risk signals |
| E19 | [CMake Presets](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html#presets) | Named, reviewable configure/build environments |
| E20 | [Prometheus](https://prometheus.io/docs/practices/naming/) and [OpenTelemetry](https://opentelemetry.io/docs/specs/otel/) | Stable observability semantics and vendor-neutral telemetry |
| E21 | [Jepsen](https://jepsen.io/analyses) | Adversarial validation of distributed-system claims and failure histories |
| E22 | [Git object model](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects) | Content addressing, immutable object identity, inspectable history |
| E23 | [CycloneDX](https://cyclonedx.org/specification/overview/) | Machine-readable component, service, dependency and vulnerability inventory |
| E24 | [Sigstore](https://docs.sigstore.dev/) | Keyless signing, transparency log, verifiable release identity |

## 6. The 247 facets

Legend: the final column cites one or more exemplar registry entries; `GLYPH`
means the property is derived from the project's own exact-byte contract.

### A. Identity, purpose, and preservation — 15 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F001 | The core purpose MUST be stated as deterministic exact-byte retrieval over immutable corpus identity. | Normative scope text plus contradiction test | GLYPH |
| F002 | The core MUST remain domain-neutral; domain behavior belongs in adapters. | Architecture boundary and adapter conformance test | GLYPH |
| F003 | Exact search, semantic search, ranking, and fuzzy matching MUST be named as distinct contracts. | Non-goals specification and API separation tests | E02,GLYPH |
| F004 | Every result MUST bind corpus identity, query bytes, algorithm/profile identity, and result semantics. | Canonical result schema and identity mutation tests | E14,E22 |
| F005 | Corpus identity MUST cover byte order, document order, empty documents, duplicates, and boundaries. | Root manifest spec and M21–M24-class mutations | GLYPH |
| F006 | Published immutable artifacts MUST NOT be rewritten in place. | Release/tag immutability check | E16,E22 |
| F007 | Deletion of durable project record MUST require the Owner's explicit dated decision. | Preservation law and decision record | GLYPH |
| F008 | A semantic replacement that destroys prior meaning MUST be treated as deletion. | Change-control test and review checklist | GLYPH |
| F009 | Important movement MUST preserve history and create a dated movement-ledger entry. | Ledger entry linked to commit | E22,GLYPH |
| F010 | Superseded material MUST remain addressable and point to its successor. | Archive/supersession manifest | E16,E22 |
| F011 | Every major claim MUST declare its evidence class P, D, A, T, F, or R. | Claim registry lint | GLYPH |
| F012 | Popularity MUST NOT be presented as correctness proof. | Documentation lint and review rule | E02,E03 |
| F013 | Production readiness MUST remain a prohibited claim until its named closure graph passes. | Claim gate in release verification | E13,E18 |
| F014 | Experimental, draft, frozen, superseded, and production statuses MUST have explicit meanings. | Lifecycle-state specification | E09,E16 |
| F015 | The facet catalog MUST contain exactly F001–F247 with no gaps or duplicates. | `tools/check_centenary_247_v1.py` | E01 |

### B. Mathematical and semantic foundation — 20 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F016 | Byte strings, offsets, intervals, documents, blocks, roots, and results MUST have typed definitions. | Foundations specification accepted by independent review | E07,E08 |
| F017 | All theorem domains and preconditions MUST be explicit, including non-empty-query rules. | Machine-checkable definitions or reviewed formal notation | E08 |
| F018 | Mathematical naturals and implementation integer domains MUST be mapped explicitly. | Bounds lemma plus conversion tests | E07 |
| F019 | Offset coordinates MUST be zero-based or otherwise uniquely declared at every boundary. | Coordinate contract and boundary fixtures | GLYPH |
| F020 | Half-open interval semantics MUST be used consistently or a proved conversion supplied. | Interval algebra tests | E04,GLYPH |
| F021 | Exact-match count MUST equal the cardinality of the complete logical occurrence set. | Count theorem and exhaustive small-model oracle | E05,GLYPH |
| F022 | Locate output MUST be a deterministic ordered subset of complete occurrences. | Prefix theorem and property tests | E08,GLYPH |
| F023 | `max_offsets` MUST constrain returned coordinates without altering total match count. | Algebraic law and mutation test | GLYPH |
| F024 | `returned_count` MUST equal the serialized offset-array length. | Schema invariant and negative fixture | GLYPH |
| F025 | `offsets_complete` MUST be true iff the returned offsets equal the complete ordered set. | Bidirectional theorem and tests | GLYPH |
| F026 | `bounded` MUST have one normative meaning independent of implementation strategy. | Flag algebra and conformance cases | E04,GLYPH |
| F027 | Empty query behavior MUST be explicitly chosen and tested, never inferred. | Normative decision and binary fixture | GLYPH |
| F028 | Empty corpus behavior MUST be explicitly chosen and tested. | Normative decision and fixture | GLYPH |
| F029 | Duplicate documents MUST retain distinct logical identities and coordinates. | Duplicate-document theorem and fixture | GLYPH |
| F030 | Empty documents MUST retain their ordered identity in composition. | Empty-document theorem and mutation | GLYPH |
| F031 | No physical concatenation artifact may create a logical cross-document match. | Boundary theorem and adversarial fixture | GLYPH |
| F032 | No physical block boundary may remove or invent a logical match. | Composition theorem and partition property test | GLYPH |
| F033 | Complete root results MUST equal the stable ordered union of complete block results. | Composition equality theorem | E08,GLYPH |
| F034 | Bounded root results MUST equal the prefix of that complete stable union. | Prefix theorem for all valid bounds | E08,GLYPH |
| F035 | Formal claims MUST enumerate trusted assumptions and excluded failure modes. | Assumption manifest reviewed beside every proof | E07 |

### C. Corpus, index, and format integrity — 20 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F036 | Every corpus MUST have a cryptographic content identity over normative bytes. | Hash manifest and tamper mutation | E22,GLYPH |
| F037 | Source manifests MUST canonicalize path/order/length metadata deterministically. | Canonical schema and cross-platform replay | E14 |
| F038 | All 256 byte values MUST be supported in corpus and query transport. | Exhaustive byte-alphabet fixture | GLYPH |
| F039 | Sentinel representation MUST be outside or rigorously separated from source-byte semantics. | Sentinel theorem and collision tests | GLYPH |
| F040 | Exactly one logical sentinel MUST be validated wherever the format requires one. | Malformed zero/multiple-sentinel fixtures | GLYPH |
| F041 | Suffix-array width and maximum corpus size MUST be encoded and checked before allocation. | Format field, checked arithmetic, limit tests | E13 |
| F042 | BWT length MUST agree exactly with suffix-array and corpus contracts. | Cross-file invariant validator | GLYPH |
| F043 | FM cumulative counts MUST be monotone and equal the BWT histogram. | Structural validator and corrupt fixtures | GLYPH |
| F044 | Rank checkpoints MUST cover the declared BWT domain without wraparound. | Checked-size proof and sanitizer fixtures | E04,E06 |
| F045 | Locate samples MUST bind their step, count, coordinate width, and source identity. | Sample schema and mismatch tests | GLYPH |
| F046 | All size computations MUST use checked add/multiply/conversion operations. | Checked arithmetic library and overflow corpus | E04,E13 |
| F047 | Parsers MUST reject trailing, truncated, duplicated, and reordered normative sections as specified. | Malformed-format corpus | E05,E06 |
| F048 | Unknown mandatory fields or sections MUST fail closed. | Forward-compatibility mutation tests | E09 |
| F049 | Optional extensions MUST be length-delimited and safely skippable. | Extension grammar and unknown-extension test | E09 |
| F050 | Endianness and integer encoding MUST be explicit and portable. | Golden vectors on little/big-endian models | E14 |
| F051 | Format versions MUST be independent from API and implementation versions. | Version matrix and compatibility tests | E16 |
| F052 | Accidental-corruption checksums MUST be distinguished from cryptographic identity. | Threat-model statement and tamper tests | E09,E24 |
| F053 | Index manifests MUST bind every file by role, length, digest, and format version. | Complete manifest validator | E10,E23 |
| F054 | Index creation MUST be transactional or publish only after full validation. | Crash-injection test and atomic-publication protocol | E03,E09 |
| F055 | An index MUST be independently rebuildable from its declared source and toolchain profile. | Two-builder bit comparison or explained variance | E11 |

### D. Query correctness and deterministic execution — 20 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F056 | Count MUST be derived from exact backward-search interval semantics. | Differential oracle against direct byte scan | GLYPH |
| F057 | Locate MUST return only offsets whose source bytes equal the query. | Byte-recomputation gate | GLYPH |
| F058 | Stored byte-check success MUST never substitute for replay recomputation. | M25-class mutation rejection | GLYPH |
| F059 | Count and locate MUST agree on interval and total match count. | Shared conformance vector | GLYPH |
| F060 | Returned offsets MUST be unique unless multiplicity is explicitly part of the contract. | Duplicate-offset rejection test | GLYPH |
| F061 | Returned offsets MUST be in the single normative order. | Order mutation tests | GLYPH |
| F062 | Query bytes MUST be transported without shell, locale, Unicode, or NUL reinterpretation. | Binary API/CLI fixtures for all byte classes | E02,GLYPH |
| F063 | Query length limits MUST be checked before search-state allocation. | Limit boundary and over-limit tests | E04 |
| F064 | The same input identity and profile MUST produce byte-identical result artifacts. | Repeated-run digest gate | E11 |
| F065 | Concurrency MUST NOT change the normative result or its serialization. | Thread-count differential replay | E03 |
| F066 | Hardware acceleration MUST preserve reference semantics exactly. | Scalar/SIMD differential corpus | E02 |
| F067 | Optimization flags MUST NOT change results. | Compiler/optimization matrix | E05 |
| F068 | Unsupported profiles MUST fail with a stable typed error. | Profile mutation and error-schema test | GLYPH |
| F069 | Out-of-range locate coordinates MUST fail before source access. | Sanitizer-backed corrupt sample tests | E06 |
| F070 | Search loops MUST have proved or enforced progress and termination bounds. | Loop invariant or bounded-step instrumentation | E04,E07 |
| F071 | Integer overflow MUST not change interval, count, offset, or error behavior. | Boundary proof plus UBSan corpus | E06 |
| F072 | Multi-document mapping MUST return document identity and document-relative coordinate deterministically. | Mapping schema and fixtures | GLYPH |
| F073 | Root composition MUST verify exact ordered block identity before querying. | Coverage/identity mutation tests | GLYPH |
| F074 | Partial publication or partial query coverage MUST fail closed. | Missing-block mutation tests | E09,GLYPH |
| F075 | A different-root replay MUST fail even when physical bytes happen to match. | M20-class identity mutation rejection | E10,GLYPH |

### E. Evidence, replay, and provenance — 24 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F076 | Every evidence object MUST declare a versioned schema and profile. | Schema validator | E10 |
| F077 | Evidence MUST bind source corpus, index, query, result, and verifier identities. | Complete evidence manifest | E10,E22 |
| F078 | Evidence serialization MUST be canonical before hashing or signing. | RFC 8785 vectors or stricter documented canonicalizer | E14 |
| F079 | Duplicate JSON object names MUST be rejected. | Parser negative fixture | E14 |
| F080 | Evidence bundles MUST enumerate every required file by digest and length. | Bundle manifest replay | E10,E23 |
| F081 | Missing, extra, substituted, truncated, or reordered normative bundle content MUST be detected. | Bundle mutation suite | E05,E10 |
| F082 | Replay MUST operate from a clean extracted bundle without original absolute paths. | Relocation replay fixture | E11 |
| F083 | Replay MUST not trust stored success flags. | Flag-forgery mutation suite | GLYPH |
| F084 | Replay MUST recompute exact source bytes for every returned coordinate. | Independent byte oracle | GLYPH |
| F085 | Replay MUST recompute result identity from canonical fields. | Identity mutation suite | E14,E22 |
| F086 | Replay MUST reject a bundle bound to a different root or query. | Root/query substitution tests | E10 |
| F087 | Replay MUST report typed, stable, machine-readable failure classes. | Error schema and golden vectors | E09 |
| F088 | The verifier MUST be independently implementable from a public specification. | Second implementation with no code sharing | E07,E09 |
| F089 | Reference generator and independent verifier MUST not share the matching oracle. | Independence declaration and code audit | E05 |
| F090 | Evidence MUST include verifier version and executable digest. | Tool manifest | E10,E24 |
| F091 | Evidence SHOULD include build provenance for generator and verifier. | SLSA provenance | E12 |
| F092 | Published evidence SHOULD be signed and transparency-log discoverable. | Sigstore signature and Rekor entry | E24 |
| F093 | Signature verification MUST be separate from semantic replay success. | Two-axis verification report | E09,E24 |
| F094 | Expiry, revocation, and key compromise semantics MUST be defined for signed evidence. | Trust policy and compromise drill | E09 |
| F095 | Evidence timestamps MUST be treated as metadata unless backed by trusted time. | Threat model and timestamp mutation | E09 |
| F096 | A replay report MUST enumerate assumptions, checks performed, and checks skipped. | Structured report schema | E07 |
| F097 | Evidence generation MUST be deterministic or enumerate every nondeterministic field. | Two-run digest comparison | E11 |
| F098 | Long-term evidence MUST retain schema, verifier source, toolchain, and fixtures. | Preservation package | E11,E22 |
| F099 | A final closure marker MUST remain disabled until an independent closure audit verifies all named prerequisites. | Marker gate and external audit report | GLYPH |

### F. Security and hostile-input behavior — 24 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F100 | A threat model MUST identify trusted and attacker-controlled bytes, paths, metadata, and processes. | Reviewed threat-model document | E13 |
| F101 | Direct runtime consumption of untrusted indexes MUST be forbidden until hostile-format gates close. | API guard and documentation claim test | E13 |
| F102 | Every parser MUST have explicit maximum file, section, count, and allocation limits. | Limit profile and boundary tests | E04 |
| F103 | Memory exhaustion MUST produce a controlled error without partial success. | Fault-injection test | E04 |
| F104 | CPU work MUST be bounded by declared input/profile functions or explicitly classified unbounded. | Complexity contract and adversarial benchmark | E04 |
| F105 | Recursion over attacker-controlled structure MUST be eliminated or depth-bounded. | Static check and deep-input test | E04 |
| F106 | Path traversal, symlink escape, and special-file substitution MUST be rejected in bundle handling. | Filesystem attack fixtures | E09,E13 |
| F107 | Temporary files MUST use private unpredictable locations and safe permissions. | Security integration test | E13 |
| F108 | Bundle extraction MUST cap file count, expanded bytes, nesting, and compression ratio. | Archive-bomb fixtures | E13 |
| F109 | Cryptographic comparisons SHOULD be constant-time when secret material is involved. | Crypto API audit | E24 |
| F110 | The project MUST not invent cryptographic primitives for authenticity. | Dependency/policy audit | E24 |
| F111 | Security-sensitive dependencies MUST be pinned, inventoried, and update-monitored. | Lockfile, SBOM, Dependabot-equivalent | E17,E18,E23 |
| F112 | CI tokens MUST use least privilege and workflows MUST pin third-party actions by immutable digest. | Workflow policy check | E18 |
| F113 | Secrets MUST never be printed in logs, evidence, crash reports, or fixtures. | Secret-scanning and redaction tests | E18 |
| F114 | A private vulnerability-reporting channel and response policy MUST exist. | `SECURITY.md` with SLA | E13,E18 |
| F115 | Supported versions and security-fix policy MUST be explicit. | Support table | E16 |
| F116 | C/C++ builds MUST run ASan and UBSan on representative gates. | CI sanitizer jobs | E06 |
| F117 | Concurrency code MUST run TSan or an equivalent race detector. | CI race job | E06 |
| F118 | Parsers and query entry points MUST be continuously fuzzed. | OSS-Fuzz/ClusterFuzzLite dashboard | E06 |
| F119 | Every security defect MUST gain a minimized regression fixture. | Vulnerability-to-test traceability | E05,E13 |
| F120 | Fuzz corpora and dictionaries MUST be versioned and replayable. | Corpus manifest | E06,E22 |
| F121 | Malformed input MUST never yield undefined behavior, crash, hang, or silent truncation. | Sanitizer/fuzzer closure report | E04,E06 |
| F122 | Security claims MUST distinguish accidental corruption, malicious tampering, and authenticity. | Claim taxonomy lint | E09,E24 |
| F123 | Security exceptions MUST be explicit, time-bounded, owned, and visible. | Exception ledger and expiry check | E13 |

### G. Runtime resources, performance, and scale — 22 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F124 | Count, locate, build, load, and replay MUST each have a declared complexity model. | Complexity specification | E04 |
| F125 | Peak resident memory MUST be measured separately for build, load, count, locate, and replay. | Reproducible benchmark report | E03 |
| F126 | Bounded locate MUST bound internal work/memory or disclose why only output is bounded. | Resource proof or explicit limitation | E04 |
| F127 | `max_offsets` MUST be applied without materializing the complete offset set when a streaming algorithm exists. | Memory-scaling test | E03,E04 |
| F128 | The runtime SHOULD support index access without requiring all structures in RAM. | mmap/external-memory profile and benchmarks | E03 |
| F129 | Compression ratios MUST include every runtime-required file, not only BWT payload. | Complete-footprint report | E03 |
| F130 | Benchmark setup MUST report hardware, OS, compiler, flags, corpus identity, cache state, and repetitions. | Benchmark manifest | E02,E11 |
| F131 | Latency MUST report distributions, not only the fastest or mean result. | p50/p95/p99 plus raw samples | E03 |
| F132 | Throughput MUST state concurrency, queueing, corpus, and result-size limits. | Load-test protocol | E03 |
| F133 | Cold-start, warm-cache, CLI, embedded, and server paths MUST be measured separately. | Scenario matrix | E02 |
| F134 | Comparisons with grep or other tools MUST normalize semantics and included integrity work. | Fairness checklist and commands | E02 |
| F135 | Performance regressions MUST have versioned budgets and statistical gates. | Benchmark CI baseline | E03 |
| F136 | Performance wins MUST not bypass byte verification or identity checks. | Optimized/reference differential tests | GLYPH |
| F137 | Large-corpus limits MUST derive from actual integer widths and allocation profiles. | Boundary table and near-limit tests | E03 |
| F138 | Build interruption MUST not publish a valid-looking partial index. | Crash/fault-injection test | E03,E09 |
| F139 | Query cancellation and deadlines SHOULD release resources deterministically. | Cancellation tests | E04 |
| F140 | Memory budgets MUST fail before allocation overflow or system destabilization. | Budget-enforcement tests | E04 |
| F141 | Parallelism MUST be configurable and documented. | Thread-budget API and test | E03 |
| F142 | NUMA, mmap, and page-cache assumptions MUST be recorded for large deployments. | Deployment profile | E03 |
| F143 | Benchmark corpora MUST include compressible, incompressible, repetitive, binary, and adversarial data. | Public corpus manifest | E02,E03 |
| F144 | Benchmark result artifacts MUST be canonical, hashed, and independently replayable. | Evidence bundle | E10,E11 |
| F145 | Scale claims beyond tested limits MUST be labeled projections. | Claim lint and extrapolation model | GLYPH |

### H. Testing, differential validation, and proof — 24 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F146 | A direct byte-scan oracle MUST validate count and locate on tractable corpora. | Differential test harness | E05 |
| F147 | Exhaustive small-corpus testing MUST cover all short byte strings for a reduced alphabet. | Exhaustive run manifest | E05 |
| F148 | Property-based tests MUST generate corpora, partitions, queries, and bounds. | Seeded property suite | E05 |
| F149 | Every normative mutation MUST name the requirement and expected error class. | Mutation traceability table | E05,GLYPH |
| F150 | Mutation coverage MUST be measured; surviving mutations require disposition. | Mutation report | E05 |
| F151 | Branch and condition coverage MUST be measured on safety-critical parser/query code. | Coverage artifact | E05 |
| F152 | The coverage toolchain MUST itself be validated and delivery builds rerun. | Meta-test matching delivered configuration | E05 |
| F153 | Unit, integration, conformance, system, fuzz, soak, and replay suites MUST be distinct. | Test taxonomy and CI jobs | E05,E06 |
| F154 | One required command MUST exercise the complete required gate graph. | Top-level verifier | E05 |
| F155 | CI green MUST mean the same required graph as local release verification. | CI/local parity test | E11 |
| F156 | Platform matrices MUST include supported Linux compilers and at least one non-Linux platform before portability claims. | CI matrix | E02 |
| F157 | Debug, release, sanitizer, and hardened builds MUST be tested separately. | Build-profile matrix | E05,E06 |
| F158 | Golden fixtures MUST be versioned and their generator independently checked. | Fixture provenance | E10,E11 |
| F159 | Cross-implementation tests MUST compare at least two independently written implementations. | Shared conformance corpus | E07,E09 |
| F160 | Cross-version tests MUST verify declared backward/forward compatibility. | Compatibility matrix | E16 |
| F161 | Fault injection MUST cover allocation, short read/write, disk full, interruption, and checksum failure. | Failure suite | E03,E05 |
| F162 | Soak tests MUST target long-duration state, leaks, and rare corruption. | Soak report and resource trend | E05 |
| F163 | Every fixed defect MUST preserve a minimal failing case permanently. | Regression index | E05 |
| F164 | Nondeterministic test failures MUST be treated as defects, not retried into green. | Flake ledger and retry policy | E11 |
| F165 | Test seeds, schedules, and environment MUST be captured for replay. | Failure evidence bundle | E10,E11 |
| F166 | Formalized theorems MUST compile in CI against pinned prover/library versions. | Lean/Isabelle/Coq job | E07,E08 |
| F167 | Executable reference behavior MUST be linked bidirectionally to theorem statements. | Requirement-proof-test map | E07 |
| F168 | Independent review MUST record findings, resolutions, and remaining disagreements. | Public review report | E08 |
| F169 | No self-review by the author/agent may be labeled independent approval. | Reviewer identity rule | E07,E08 |

### I. Build, dependencies, supply chain, and release — 22 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F170 | Dependencies and build tools MUST be version-pinned in required CI/release paths. | Lockfiles and pin audit | E12,E18 |
| F171 | The build MUST expose named reviewed presets for supported configurations. | `CMakePresets.json` | E19 |
| F172 | Portable releases MUST NOT depend on `-march=native` or host-only defaults. | Portable preset and CPU compatibility test | E19 |
| F173 | Performance-native builds MUST be explicitly labeled non-portable. | Build metadata and docs | E19 |
| F174 | Required builds MUST start from declared clean inputs without undeclared network fetches. | Hermetic build test | E11,E12 |
| F175 | Release artifacts MUST be reproducible or publish a precise variance report. | Independent rebuild comparison | E11 |
| F176 | Build provenance MUST identify source, builder, commands, dependencies, and outputs. | SLSA provenance | E12 |
| F177 | Release provenance SHOULD reach SLSA Build L2 before production claims. | Signed hosted provenance verification | E12 |
| F178 | A hardened-release target SHOULD document the path to SLSA Build L3. | Threat analysis and builder controls | E12 |
| F179 | Every release MUST publish an SPDX or CycloneDX SBOM. | Validated SBOM | E17,E23 |
| F180 | Every bundled dependency MUST have license, source, version, and modification notice. | Third-party notices audit | E17 |
| F181 | Dependency vulnerabilities MUST be scanned with a documented triage policy. | Scan report and issue linkage | E13,E18 |
| F182 | Releases MUST be signed with a verifiable identity and immutable digest. | Sigstore or equivalent verification | E24 |
| F183 | Release signatures SHOULD be recorded in a transparency log. | Log inclusion proof | E24 |
| F184 | Update metadata MUST resist rollback, freeze, mix-and-match, and key compromise. | TUF repository test | E09 |
| F185 | Release versions MUST follow a declared compatibility policy. | SemVer policy and API diff gate | E16 |
| F186 | Published version contents MUST never be modified in place. | Re-release prevention gate | E16 |
| F187 | Changelogs MUST distinguish semantics, formats, API, performance, and security changes. | Structured release notes | E16 |
| F188 | Release candidates MUST replay the same full gate as final artifacts. | RC/final digest and gate report | E05 |
| F189 | Install and uninstall paths MUST be tested from packaged artifacts. | Package integration tests | E02 |
| F190 | CMake install/export metadata MUST support downstream `find_package` use. | Consumer-project test | E04,E19 |
| F191 | A release MUST include source, license, notices, checksums, SBOM, provenance, signatures, and verification instructions. | Release completeness checker | E09,E12,E17,E24 |

### J. API, compatibility, portability, and integration — 18 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F192 | Every public API MUST be explicitly enumerated; all other symbols are internal. | Export map and public-header audit | E03,E16 |
| F193 | C ABI ownership, lifetime, alignment, nullability, and thread-safety MUST be normative. | ABI contract and misuse tests | E04 |
| F194 | ABI structs MUST carry size/version fields or equivalent forward-compatible negotiation. | Compatibility fixtures | E09 |
| F195 | Error codes MUST be stable, typed, documented, and independent of log text. | Error registry and ABI tests | E09 |
| F196 | APIs MUST distinguish invalid input, corrupt index, resource limit, unsupported profile, and internal defect. | Failure taxonomy tests | E04,E09 |
| F197 | Buffer APIs MUST report required size without partial ambiguous success. | Two-call contract tests | E04 |
| F198 | No API may expose a pointer whose lifetime depends on undocumented internal storage. | Static/API audit | E04 |
| F199 | Reentrant and thread-safe functions MUST be identified separately. | Concurrency contract and TSan test | E03,E06 |
| F200 | API cancellation, timeout, and resource-budget semantics SHOULD be consistent across bindings. | Binding conformance suite | E04 |
| F201 | Language bindings MUST preserve raw bytes and unsigned coordinate ranges exactly. | Cross-language binary vectors | GLYPH |
| F202 | CLI output MUST have a stable machine-readable mode separate from human diagnostics. | JSON mode schema and golden tests | E02 |
| F203 | Exit codes MUST be documented and stable. | CLI conformance tests | E02 |
| F204 | Paths, locales, time zones, and environment variables MUST not alter normative identities. | Environment variance replay | E11 |
| F205 | Supported OS, architecture, compiler, standard library, and filesystem assumptions MUST be published. | Support matrix | E02 |
| F206 | API and file-format compatibility MUST be versioned independently. | Compatibility policy | E16 |
| F207 | Deprecated APIs MUST remain available for the stated window and point to migration guidance. | Deprecation ledger | E16 |
| F208 | Examples MUST compile and run in CI against installed artifacts. | Example consumer job | E04 |
| F209 | Integration adapters MUST not weaken core identity, exactness, or replay requirements. | Adapter conformance gate | GLYPH |

### K. Operations, observability, and recovery — 14 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F210 | Operational profiles MUST declare startup, readiness, liveness, and shutdown semantics. | Operator contract and tests | E20 |
| F211 | Metrics MUST have stable names, units, types, labels, and cardinality budgets. | Metrics specification | E20 |
| F212 | Logs MUST separate human text from structured stable fields. | Log schema and golden tests | E20 |
| F213 | Traces SHOULD propagate query/evidence correlation without exposing query contents by default. | OpenTelemetry integration and privacy test | E20 |
| F214 | Sensitive corpus/query bytes MUST be opt-in in diagnostics. | Redaction policy and tests | E13,E20 |
| F215 | Health endpoints MUST not report ready before required index validation completes. | Startup fault tests | E20 |
| F216 | Runtime configuration MUST be inspectable with secrets redacted. | Effective-config endpoint/artifact | E20 |
| F217 | Resource-limit failures MUST be observable distinctly from corrupt-input failures. | Metric/error correlation test | E04,E20 |
| F218 | Index rollouts MUST support validation before atomic activation. | Blue/green index switch test | E03,E09 |
| F219 | Rollback MUST restore a previously verified index/runtime pair without metadata ambiguity. | Recovery drill | E09 |
| F220 | Backup scope MUST include source identity, index/evidence manifests, keys/policies, and restore instructions. | Restore test | E09,E10 |
| F221 | Disaster recovery MUST be tested from durable artifacts on a clean environment. | Dated recovery report | E11 |
| F222 | Operational SLOs MUST state corpus size, query class, concurrency, and evidence mode. | SLO document and measurement | E03 |
| F223 | Failure drills MUST preserve a timeline and evidence sufficient for independent reconstruction. | Incident bundle | E10,E21 |

### L. Governance, documentation, and standards process — 14 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F224 | Governance MUST name decision authority, reviewer roles, and escalation paths. | `GOVERNANCE.md` | E09,E10 |
| F225 | The Owner's exclusive deletion authority MUST be visible in contributor workflow. | PR template/checklist link | GLYPH |
| F226 | Architecture decisions MUST be dated, immutable records with status and consequences. | ADR index | E22 |
| F227 | Normative specs MUST identify version, status, scope, dependencies, and superseded documents. | Spec linter | E15,E16 |
| F228 | Normative language MUST use BCP 14 consistently. | Terminology lint | E15 |
| F229 | Every requirement MUST map to implementation, test, evidence, or explicit open status. | Traceability graph | E05,E13 |
| F230 | Documentation MUST distinguish current mainline, research branches, frozen tags, and future plans. | Branch/status map | E16,E22 |
| F231 | Known limitations MUST be version-scoped and not silently describe obsolete releases as current. | Limitation freshness check | E16 |
| F232 | Contributor guidance MUST include build, tests, review, security, preservation, and evidence rules. | Complete `CONTRIBUTING.md` | E18 |
| F233 | Code ownership MUST identify reviewers for security, formats, math, runtime, and documentation. | `CODEOWNERS` and fallback policy | E18 |
| F234 | Change proposals affecting semantics MUST include compatibility and migration analysis. | Proposal template | E09,E16 |
| F235 | Standard changes MUST have a public discussion window and recorded disposition of material objections. | Discussion record and decision log | E08,E09 |
| F236 | An approval MUST identify reviewer, reviewed commit, scope, method, and unresolved reservations. | Signed review record | E07,E08 |
| F237 | Project claims MUST be auditable from public artifacts without relying on private conversation. | Public-claim audit | E10,E11 |

### M. Adoption, independent validation, and standard candidacy — 10 facets

| ID | Normative facet | Minimum closure artifact | Origin |
|---|---|---|---|
| F238 | At least two independent reviewers MUST assess the foundations before they are frozen. | Commit-bound review reports | E07,E08 |
| F239 | At least two independent implementations MUST pass the same conformance corpus before “standard” is claimed. | Cross-implementation report | E09 |
| F240 | At least three external organizations or teams SHOULD run reproducible pilots in different domains. | Public pilot reports | E09,E13 |
| F241 | At least one pilot MUST exercise binary/non-text data and one MUST exercise multi-document composition. | Domain-diverse evidence bundles | GLYPH |
| F242 | Every pilot MUST publish failures and limitations, not only successful latency numbers. | Complete pilot report template | E21 |
| F243 | Public discussions MUST remain open for objections, alternatives, and independent reproduction attempts. | Discussion index with counts and dispositions | E08,E09 |
| F244 | Adoption counts MUST distinguish download, test, pilot, production, and independent implementation. | Adoption evidence taxonomy | E02,E03 |
| F245 | A standards-candidacy proposal MUST include IP/license, governance, conformance, security, and versioning policies. | Candidate submission package | E13,E15,E17 |
| F246 | “Industrial” MUST require supported releases, security response, reproducible delivery, recovery drills, and external users. | Industrial-readiness closure graph | E05,E09,E11,E13 |
| F247 | “GLYPH Centenary 247 complete” MUST require all 247 facets closed under the closure rule and an independent final audit. | Signed facet manifest, clean replay, and audit report | E01,E07,E10 |

## 7. Section cardinality invariant

| Section | Range | Count |
|---|---:|---:|
| A | F001–F015 | 15 |
| B | F016–F035 | 20 |
| C | F036–F055 | 20 |
| D | F056–F075 | 20 |
| E | F076–F099 | 24 |
| F | F100–F123 | 24 |
| G | F124–F145 | 22 |
| H | F146–F169 | 24 |
| I | F170–F191 | 22 |
| J | F192–F209 | 18 |
| K | F210–F223 | 14 |
| L | F224–F237 | 14 |
| M | F238–F247 | 10 |
| **Total** | **F001–F247** | **247** |

## 8. Current baseline declaration

This V1 catalog is the measuring instrument, not a retroactive certification.
Existing GLYPH proof, runtime, operator, embedded, composition, evidence, and
replay artifacts may satisfy individual facets, but each mapping must be audited
against an immutable commit. Until that audit exists, the aggregate state is:

> **247 DEFINED; 0 CLAIMED CLOSED BY THIS DOCUMENT.**

The next additive artifact should be a machine-readable facet closure manifest
bound to a single repository commit. It must preserve open and failed states;
it must not turn absence of evidence into success.

## 9. External validation rule

A GitHub star, fork, issue, discussion, or reaction is never counted as an
approval. Valid approval requires the record defined by F236. Valid independent
replay requires the record defined by F088–F090. The project may publish dynamic
counts for visibility, but must show the measurement date and source.
