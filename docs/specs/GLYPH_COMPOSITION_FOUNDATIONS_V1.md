# GLYPH_COMPOSITION_FOUNDATIONS_V1

Status: research draft  
Version: 1  
Date: 2026-08-14  
Base semantic checkpoint: `c4138812f60704083f099fe0daeb6461abc9d0c2`

## 1. Purpose

This document isolates the mathematical foundation beneath
`GLYPH_COMPOSITION_SEMANTICS_V1`.

It defines an implementation-independent model of:

- ordered byte documents;
- exact document-local substring matches;
- valid document-preserving partitions;
- local-to-global coordinate lifting;
- complete composition;
- canonical ordering and bounded prefixes;
- partition and schedule invariance;
- the boundary between mathematical claims, cryptographic assumptions and
  executable evidence.

The purpose is not to make a broad novelty claim. The core set equality is
elementary. The research question is whether this foundation, combined with
versioned identities, fail-closed coverage and independently replayable
evidence, yields a useful and standardizable exact-retrieval protocol.

## 2. Authority and compatibility

This document is additive.

It does not replace or modify:

- `GLYPH_COMPOSITION_SEMANTICS_V1`;
- the Composition Reference V1 executable checker;
- any existing identity preimage;
- any existing runtime or operator format;
- the P/R/O/I0 verification chain;
- the meaning of the current normal-path or final reference markers.

The frozen executable reference checkpoint remains:

    branch: composition-reference-v1
    commit: c4138812f60704083f099fe0daeb6461abc9d0c2
    mutation matrix: 25/25 EXACT
    final marker: DISABLED

If this document conflicts with the committed Composition Semantics V1 wire
contract, the wire contract governs V1 implementations until a separately
reviewed version explicitly changes it. A mathematical definition may clarify
the contract but must not silently rename or redefine a serialized field.

## 3. Claim classes

Every claim derived from this document belongs to one of four classes.

### 3.1 Unconditional mathematical claim

A statement derived from explicit definitions in ordinary finite mathematics,
without assumptions about SHA-256, filesystems, processes or implementations.

### 3.2 Conditional mathematical claim

A statement true only under named assumptions, such as complete coverage,
checked arithmetic or collision resistance.

### 3.3 Executable conformance claim

A statement that one implementation passed named fixtures and mutation tests.
This is evidence about an implementation, not a universal proof.

### 3.4 Industrial or novelty claim

A statement about usefulness, standardization, originality, scalability or
adoption. It requires measurements, prior-art review or external validation and
does not follow from the composition theorem alone.

These classes must never be collapsed into one another.

## 4. Mathematical universe

### 4.1 Bytes and byte strings

Define the byte alphabet:

    Byte = {0, 1, ..., 255}

Define `ByteString` as the set of all finite sequences over `Byte`, including
the empty sequence.

For `x` in `ByteString`:

    |x|

denotes its byte length. Indexing is zero-based. The half-open slice:

    x[a:b]

is defined when:

    0 <= a <= b <= |x|

### 4.2 Documents

A mathematical document is one byte string occupying one position in an
ordered corpus.

Two byte-identical documents at different positions are distinct documents.
An empty byte string is a valid document.

Paths, names, timestamps, MIME types and application meanings are deliberately
absent from the retrieval model. They may participate in external manifests and
identities, but not in exact-match truth.

### 4.3 Ordered corpus

An ordered corpus is a finite sequence:

    C = [D0, D1, ..., D(n-1)]

where every `Di` is a `ByteString` and `n >= 1` for Composition V1.

Corpus position is semantic. Therefore:

- reordering documents produces a different corpus;
- removing an empty document produces a different corpus;
- deduplicating equal documents produces a different corpus;
- renaming an external source may change source-manifest identity without
  changing this mathematical corpus.

### 4.4 Query domain

The Composition V1 theorem domain contains non-empty byte queries:

    q in ByteString
    |q| > 0

Empty-query behavior is outside this foundation. A later version may define it,
but V1 must not infer such semantics implicitly.

### 4.5 Canonical coordinates

A global coordinate is a pair:

    (doc_id, doc_offset)

with lexicographic order:

    (i, a) < (j, b)

if `i < j`, or if `i = j` and `a < b`.

Block identity, block ordinal, runtime identity and execution schedule are not
parts of the canonical coordinate.

## 5. Direct exact-match semantics

For document `Di` and non-empty query `q`, define:

    Match(Di, q) =
        { o |
          0 <= o,
          o + |q| <= |Di|,
          Di[o:o+|q|] = q }

Overlapping occurrences are included.

Define the complete global match set:

    Matches(C, q) =
        { (i, o) |
          0 <= i < n,
          o in Match(Di, q) }

Define:

    OrderedMatches(C, q)

as `Matches(C, q)` sorted in canonical coordinate order.

Define the complete count:

    Count(C, q) = |Matches(C, q)|

These definitions are the direct document oracle. They do not concatenate
document byte strings and do not permit a match to cross a document boundary.

## 6. Valid partitions

### 6.1 Partition boundaries

A V1 partition of corpus `C` is a strictly increasing sequence of document
boundaries:

    A = [a0, a1, ..., am]

such that:

    m >= 1
    a0 = 0
    am = n
    aj < a(j+1) for every 0 <= j < m

Block `Bj` is the contiguous document subsequence:

    Bj = [D(aj), D(aj+1), ..., D(a(j+1)-1)]

Thus every block is non-empty as a document sequence, while individual
documents may be empty.

### 6.2 Validity consequences

A valid partition guarantees:

- every global document position appears in exactly one block;
- no document is split between blocks;
- no document is omitted;
- no document is duplicated by the partition;
- block order preserves global document order;
- flattening all blocks in order yields exactly `C`.

Arbitrary byte sharding is not a valid Composition V1 partition.

### 6.3 Local corpus and local coordinates

Inside `Bj`, local document index `l` satisfies:

    0 <= l < a(j+1) - aj

and denotes global document:

    D(aj + l)

Define the lift function:

    Lift(j, (l, o)) = (aj + l, o)

The base `aj` is derived from the partition prefix. It is not an independent
authority.

## 7. Local and composed results

Define local exact matches:

    LocalMatches(Bj, q) =
        { (l, o) |
          o in Match(D(aj+l), q) }

Define lifted local matches:

    LiftedMatches(Bj, q) =
        { Lift(j, x) | x in LocalMatches(Bj, q) }

For valid partition `A`, define the composed match set:

    ComposedMatches(C, A, q) =
        union over j = 0 .. m-1 of LiftedMatches(Bj, q)

Define:

    OrderedComposedMatches(C, A, q)

as that set sorted in canonical coordinate order, independent of block
completion order.

Define composed count:

    ComposedCount(C, A, q) =
        sum over j = 0 .. m-1 of |LocalMatches(Bj, q)|

The mathematical sum is unbounded. A concrete V1 implementation must perform
checked unsigned 64-bit arithmetic and fail rather than wrap.

## 8. Coverage and result existence

Let the expected block ordinal set be:

    E = {0, 1, ..., m-1}

Let `V` be the blocks independently verified against the committed root and
let `Q` be the blocks successfully queried with the exact same query bytes.

Complete coverage is:

    E = V = Q

A successful Composition V1 result exists only under complete coverage.

If coverage is incomplete, the mathematical result for the committed
composition is not represented by any partial sum. In particular:

    incomplete coverage != zero matches

Partial block observations may exist as diagnostics, but they are not a
successful composition result.

## 9. Fundamental lemmas

### L1 — Local soundness

For every local coordinate `x` in `LocalMatches(Bj, q)`, the lifted coordinate
`Lift(j, x)` belongs to `Matches(C, q)`.

Reason: `Bj[l]` is definitionally `D(aj+l)`, and lifting changes only the
document index, not the byte offset or source document.

### L2 — Local completeness within a block

For every global match `(i, o)` in `Matches(C, q)` with:

    aj <= i < a(j+1)

there is exactly one local coordinate:

    (i - aj, o)

whose lift equals `(i, o)`.

### L3 — Lift injectivity inside one block

Within fixed block `j`:

    Lift(j, x) = Lift(j, y) implies x = y

### L4 — Lifted block disjointness

For distinct valid blocks `j != k`:

    LiftedMatches(Bj, q)
        intersection
    LiftedMatches(Bk, q)
        = empty set

Reason: valid block document ranges are disjoint.

### L5 — Unique block ownership

For every global document index `i` with `0 <= i < n`, exactly one `j` satisfies:

    aj <= i < a(j+1)

This follows from strict partition boundaries spanning `[0, n)`.

## 10. Core theorems

### T1 — Exact composition equivalence

For every ordered corpus `C`, every valid V1 partition `A` of `C`, and every
non-empty byte query `q`:

    ComposedMatches(C, A, q) = Matches(C, q)

Proof outline:

1. By L1, every composed match is a direct global match.
2. For every direct global match, L5 selects its unique block.
3. L2 constructs the unique local preimage in that block.
4. Therefore every direct global match is composed.
5. L3 and L4 prevent duplication introduced by lifting.

Claim class: unconditional mathematical claim under the definitions of a valid
partition and non-empty query.

### T2 — Count composition

Under the assumptions of T1:

    ComposedCount(C, A, q) = Count(C, q)

Proof outline: T1 gives set equality; L3 and L4 show that the per-block
cardinalities add without overlap.

For an implementation using bounded integers, the executable claim is
conditional on successful checked arithmetic.

### T3 — Partition invariance

For any two valid partitions `A` and `A'` of the same ordered corpus `C`:

    ComposedMatches(C, A, q)
        =
    ComposedMatches(C, A', q)

and:

    ComposedCount(C, A, q)
        =
    ComposedCount(C, A', q)

Proof outline: apply T1 to both partitions and use equality through
`Matches(C, q)`.

This theorem concerns semantic retrieval results. It does not imply equal
physical root identities or equal execution provenance.

### T4 — Canonical order invariance

For every valid partition:

    OrderedComposedMatches(C, A, q)
        =
    OrderedMatches(C, q)

This equality is independent of the order in which block computations finish.
Execution completion order is observational only; canonical coordinate order is
authoritative.

### T5 — Canonical bounded-prefix correctness

For non-negative integer `k`, define:

    Prefix(k, S) = first min(k, |S|) elements of ordered sequence S

Then:

    Prefix(k, OrderedComposedMatches(C, A, q))
        =
    Prefix(k, OrderedMatches(C, q))

The complete match count remains `Count(C, q)` regardless of `k`.

Applying `k` independently as a fresh allowance to every block is not this
operation and may return a non-canonical or oversized sequence.

### T6 — Repartition coordinate stability

For valid partitions of the same ordered corpus, every match retains the same:

    (doc_id, doc_offset)

Block ordinal and runtime index identity may change and therefore belong only
to execution and evidence provenance.

## 11. Identity and cryptographic boundary

The pure theorems compare mathematical values. Real artifacts use finite
serialized preimages and cryptographic digests.

### 11.1 Logical corpus commitment

The architecture uses the semantic name:

    composed_corpus_id

for a partition-independent commitment to the ordered logical document
sequence.

The current Composition V1 serialized field remains:

    runtime_corpus_id

under `GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1`.

Foundations V1 does not rename that field. It records that the current field
realizes the partition-independent logical role. Any future public rename
requires a new version and compatibility statement.

### 11.2 Source-manifest commitment

`source_manifest_id` may bind path bytes and other source-manifest fields not
present in the mathematical retrieval corpus. Therefore equal exact-match
semantics do not imply equal source-manifest identities after a source rename.

### 11.3 Physical composition commitment

`composition_root_id` commits to one physical layout: ordered block records,
runtime index identities, manifest commitments and partition data.

Repartitioning may preserve the logical corpus and every query result while
changing the root preimage.

### 11.4 Conditional digest claims

The following implication is unconditional:

    equal canonical preimages -> equal deterministic digest outputs

The reverse implication is not unconditional:

    equal digest outputs -> equal preimages

It is accepted operationally only under the stated collision-resistance
assumption for the selected digest profile.

Consequently, root binding and tamper rejection are conditional cryptographic
claims, not the same theorem as partition invariance.

## 12. Replay and evidence boundary

T1 through T6 say what a correct result is. They do not establish that a stored
artifact was produced correctly.

Independent replay must separately establish that:

- committed artifacts decode to the claimed mathematical corpus and partition;
- every expected block is present and verifies;
- the query bytes are exactly the committed query;
- every required block is queried;
- local results satisfy the exact-match relation;
- lifting and canonical merge are correct;
- counts and bounded prefixes are recomputed;
- every returned source span equals the query bytes;
- result and evidence identities match their canonical preimages;
- stored success flags are consequences of recomputation, not inputs to trust.

Thus the proof chain has three distinct relations:

    artifact interpretation
        -> mathematical objects
        -> theorem-governed composition result
        -> evidence comparison

An implementation can satisfy the middle theorem while failing artifact
interpretation or evidence comparison. The standalone verifier is required to
test the complete chain.

## 13. Required countermodels and mutation meanings

Each case below violates a named assumption rather than refuting T1.

### C1 — Missing block

Removing a block destroys complete coverage and usually changes the flattened
corpus. A partial sum is not the committed result.

### C2 — Reordered documents

Reordering changes `C`, document IDs and logical identity. T3 applies only to
two partitions of the same ordered corpus.

### C3 — Removed empty document

Although the empty document contributes no matches, removing it changes corpus
length, later document IDs and logical identity.

### C4 — Deduplicated equal document

Equal byte content does not erase positional multiplicity. Deduplication changes
the corpus and may change both count and coordinates.

### C5 — Physical concatenation oracle

Searching `D0 ++ D1` as one byte string introduces candidates spanning document
boundaries. That oracle computes a different relation from `Matches(C, q)`.

### C6 — Split document

Partitioning inside a document violates the V1 partition definition and may
lose or invent boundary matches.

### C7 — Per-block bounded locate

Giving every block the global allowance `k` does not necessarily produce the
first `k` global canonical coordinates.

### C8 — Completion-order merge

Appending results in thread completion order violates canonical order even when
the underlying set and count are correct.

### C9 — Integer wraparound

Machine arithmetic that wraps does not implement mathematical cardinality.
The implementation must fail with a limit error.

### C10 — Wrong-root replay

Equal logical results under repartition do not authorize evidence bound to one
physical root to replay against another root without explicit new verification.

### C11 — Trusted stored byte-check

A stored `ok=true` is not a premise of correctness. If source bytes have changed,
direct span recomputation must reject the artifact.

## 14. Formalization plan

### F1 — Pure finite model

Formalize `Byte`, `ByteString`, ordered corpus, non-empty query, exact matches,
valid partitions, lifting, canonical order and prefixes in a proof assistant.

Preferred first target: Lean 4. Isabelle/HOL or Coq are acceptable if the proof
is independently reviewable.

### F2 — Prove structural lemmas

Mechanize L1 through L5, then T1 through T6.

### F3 — Finite implementation domain

Add unsigned 64-bit representability predicates and prove that checked
arithmetic either agrees with the unbounded model or returns a typed failure.

### F4 — Artifact refinement model

Define a separate refinement relation from parsed V1 artifacts to mathematical
objects. Do not model SHA-256 as an injective function. State cryptographic
assumptions explicitly.

### F5 — Evidence refinement

After Composition Evidence V1 is specified, define when verified evidence
faithfully represents the recomputed mathematical result.

### F6 — Publication state machine

Use a state-machine formalism such as TLA+ for temporary construction, complete
markers, immutable roots, atomic publication, crash interruption, retention and
concurrent readers. These are protocol-safety questions, not consequences of
the set theorem.

## 15. Conformance mapping to Composition V1

| Foundation object | Composition V1 realization |
|---|---|
| Ordered corpus `C` | flattened ordered block-local document records |
| Corpus position | global `doc_id` |
| Document bytes | committed source snapshot bytes |
| Query `q` | exact binary query bytes under existing query identity rules |
| Valid partition `A` | ordered contiguous non-empty runtime units |
| Boundary `aj` | recomputed prefix sum of preceding block document counts |
| Lift | local `doc_id` plus recomputed global document base |
| Complete coverage | expected blocks = verified blocks = queried blocks |
| Direct oracle | document-local byte equality, never physical concatenation |
| Canonical order | `(global_doc_id, doc_offset)` ascending |
| Count | checked sum of complete per-block counts |
| Bounded result | global canonical prefix after complete count coverage |
| Logical commitment | current global `runtime_corpus_id` role |
| Source commitment | global `source_manifest_id` |
| Layout commitment | `composition_root_id` |

This table is a refinement guide. It does not independently prove that any
parser, checker or runtime implements the mapping correctly.

## 16. Acceptance criteria for Foundations V1

The foundations document may be called internally complete only when:

1. all mathematical objects have explicit domains;
2. empty documents, duplicate documents and document order are represented;
3. the query domain is explicit;
4. valid partition assumptions are explicit;
5. direct and composed match relations are separately defined;
6. coverage is a precondition of successful composition;
7. soundness, completeness, count, partition invariance, schedule invariance and
   bounded-prefix correctness are separately stated;
8. mathematical and cryptographic claims are separated;
9. current V1 field names and identity preimages are not silently changed;
10. mutation cases are mapped to violated assumptions;
11. exact non-claims are present;
12. an independent technical review finds no contradiction with
    `GLYPH_COMPOSITION_SEMANTICS_V1`.

Creating this Markdown file satisfies only the specification-draft milestone.
It does not constitute a machine-checked proof.

## 17. Exact non-claims

This document does not establish:

- a newly discovered mathematical law;
- novelty or patentability;
- a production Index Forest;
- independent replay;
- a Composition Evidence V1 encoding;
- cryptographic completeness or non-membership proof;
- correctness of SHA-256 as an injective function;
- performance, scalability or fault tolerance;
- distributed consensus or atomic publication;
- a stable public API or C ABI;
- field routing, ranking, semantic search or approximate matching;
- SIMD, GPU, VRAM or ACEAPEX integration;
- replacement of general search, databases, SIEM or vector systems;
- an industry standard.

The safe current claim is narrower:

    Foundations V1 defines a precise mathematical model and proof obligations
    for partition-invariant composition of document-local exact-byte matches.

## 18. Next boundary

After review of this document, the next additive artifact is:

    GLYPH_COMPOSITION_EVIDENCE_V1

That specification must define the evidence data model, canonical encoding,
domain-separated identity preimage, error semantics, conformance vectors and
security considerations before implementation of the standalone independent
verifier.

The final marker remains disabled, and Index Forest runtime remains blocked,
until the evidence and independent replay boundary is accepted.
