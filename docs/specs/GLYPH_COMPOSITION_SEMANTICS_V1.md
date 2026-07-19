# GLYPH_COMPOSITION_SEMANTICS_V1

Status: normative draft  
Version: 1  
Phase: composition precondition  
Date: 2026-07-20

## Purpose

Define the reference semantics for composing multiple already verified
GLYPH operator runtime units into one logical ordered corpus.

This specification defines:

- composition identity;
- block membership;
- global document identity;
- complete coverage;
- exact count aggregation;
- global bounded locate;
- deterministic merge;
- document and block boundary behavior;
- fail-closed error semantics;
- repartition stability;
- minimum evidence obligations.

This specification does not define an optimized Index Forest implementation.

It defines the semantic contract that any future Index Forest, multi-block
runtime, field-routing layer or distributed composition mechanism must
preserve.

## Status boundary

The composition layer is not currently part of the required GLYPH
verification closure.

The current required closure remains:

    GLYPH PROOF GRAPH OK
    GLYPH RUNTIME CONFORMANCE OK
    GLYPH OPERATOR CONFORMANCE OK
    GLYPH EMBEDDED I0 CONTRACT VERIFY PASS
    VERIFY OK

This specification must not modify or weaken:

- P1 through P12 reference semantics;
- R0 through R6 runtime conformance;
- O1 through O6 operator conformance;
- the Embedded I0 pre-freeze contract;
- the existing binary runtime formats;
- canonical document-local coordinates.

## Normative dependencies

Composition V1 depends on the semantics defined by:

- `GLYPH_CORPUS_IDENTITY_V1`;
- `GLYPH_BINARY_RUNTIME_MULTIDOC_V1`;
- `GLYPH_BINARY_RUNTIME_EVIDENCE_V1`;
- `GLYPH_DOCUMENT_BOUNDARY_SEMANTICS_V1`;
- `GLYPH_OPERATOR_CORPUS_MANIFEST_V1`;
- `GLYPH_OPERATOR_RUNTIME_INDEX_V1`;
- `GLYPH_OPERATOR_QUERY_V1`;
- `GLYPH_OPERATOR_PATH_V1`;
- `GLYPH_OPERATOR_EVIDENCE_BUNDLE_V1`.

Composition V1 references existing identities.

It does not redefine them.

## Terminology

### Global corpus

The logical ordered document sequence searched by the complete composition.

### Runtime unit

One independently valid GLYPH operator runtime index containing an ordered
non-empty sequence of logical documents.

A runtime unit is also called a block in this specification.

### Block order

The canonical order of runtime units in the composition root.

### Global document order

The ordered concatenation of all block-local document sequences in canonical
block order.

### Coverage

The condition that every runtime unit committed by the composition root was
verified and queried successfully.

### Repartition

A change in the grouping of one unchanged global ordered document sequence
into different runtime units.

## Identity profiles

GLYPH currently has multiple identity profiles.

They serve different purposes and must remain distinct.

### P4 proof-layer corpus identity

`GLYPH_CORPUS_IDENTITY_V1` defines the P4 proof-layer `corpus_id`.

Its identity model includes canonical document names.

Composition V1 does not redefine or replace P4.

### Runtime corpus identity

Operator and binary-runtime artifacts use:

    GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1

This identity binds:

- document count;
- canonical document order;
- document IDs;
- byte lengths;
- document SHA-256 values.

It does not bind source paths.

In this specification the term:

    runtime_corpus_id

means the existing operator field named:

    corpus_id

under the `GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1` profile.

### Source manifest identity

The existing:

    source_manifest_id

binds the canonical ordered source manifest, including relative path bytes.

It is defined by:

    GLYPH_OPERATOR_CORPUS_MANIFEST_V1

### Runtime index identity

Each runtime unit has an existing:

    runtime_index_id

defined by:

    GLYPH_OPERATOR_RUNTIME_INDEX_V1

It binds the concrete runtime topology, manifests, binaries and ordered
SA/BWT/FM artifact commitments for that runtime unit.

### Composition root identity

Composition V1 introduces one new identity:

    composition_root_id

It identifies one exact physical composition layout:

- one global runtime corpus;
- one global source manifest;
- one ordered sequence of runtime units;
- one exact runtime-unit partition;
- one exact set of runtime index commitments.

`composition_root_id` is layout-sensitive.

Repartitioning an unchanged global corpus changes `composition_root_id`.

## Canonical composition model

A Composition V1 root represents:

    B = [B0, B1, ..., Bm-1]

where each `Bi` is one independently valid operator runtime unit.

Each block contains an ordered document sequence:

    Bi = [Di,0, Di,1, ..., Di,ni-1]

The global logical corpus is:

    C =
        B0.documents
        ++ B1.documents
        ++ ...
        ++ Bm-1.documents

The `++` operator means ordered sequence concatenation.

It does not mean physical byte concatenation for substring matching.

## V1 block constraints

Composition V1 requires:

    block_count >= 1

Each runtime unit must contain:

    block_document_count >= 1

Empty documents are valid members of a block.

An empty runtime unit is forbidden in V1.

Every block must be a complete, independently verified
`GLYPH_OPERATOR_RUNTIME_INDEX_V1` unit.

Composition V1 must not import or depend on the legacy segmented v0.x runtime.

## Global document identifiers

Canonical match coordinates remain:

    (doc_id, doc_offset)

The global `doc_id` is the position of the document in the complete global
ordered document sequence.

For block `Bi`, define:

    global_doc_base(B0) = 0

    global_doc_base(Bi) =
        sum(block_document_count(Bj))
        for all j < i

For a block-local document ID:

    local_doc_id

the canonical global document ID is:

    global_doc_id =
        global_doc_base(Bi) + local_doc_id

`global_doc_base` is derived from canonical block order and preceding block
document counts.

It is not an independent authority.

If a serialized artifact includes `global_doc_base`, replay must recompute it
and require exact equality.

## Coordinate stability

`block_id`, block ordinal and `runtime_index_id` must not become part of the
canonical match coordinate.

The canonical coordinate remains:

    (global_doc_id, doc_offset)

Therefore repartitioning an unchanged global document sequence preserves:

- every global `doc_id`;
- every `doc_offset`;
- exact match count;
- canonical coordinate order.

Physical block provenance is recorded separately.

## Global runtime corpus identity

The composition must recompute the global `runtime_corpus_id` from the
flattened ordered document sequence using exactly:

    GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1

The global runtime identity preimage is:

    ASCII("GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1")
    BYTE(0x00)
    U64_BE(global_document_count)

For every document in global `doc_id` order:

    U64_BE(global_doc_id)
    U64_BE(document_byte_length)
    document_sha256_raw_32_bytes

The committed global `runtime_corpus_id` must equal SHA-256 of this preimage.

Block-local corpus identities do not replace this global recomputation.

## Global source manifest identity

The composition must also reconstruct one valid global operator source
manifest over the flattened document sequence.

The global `source_manifest_id` is computed using exactly:

    GLYPH_OPERATOR_CORPUS_MANIFEST_V1

For every document in global `doc_id` order it binds:

- global `doc_id`;
- canonical relative path bytes;
- byte length;
- document SHA-256.

The global source manifest must satisfy all existing operator path,
ordering, file-domain and identity rules.

A composition must not introduce duplicate canonical source paths that would
make the global operator source manifest invalid.

## Block-local identities

For every block, the composition verifier must independently verify:

- block source manifest;
- block-local `corpus_id`;
- block-local `source_manifest_id`;
- block runtime manifest;
- block `runtime_index_id`;
- runtime artifact commitments;
- runtime complete marker;
- source snapshot commitments.

The block-local ordered document records must exactly match the corresponding
contiguous range in the reconstructed global document sequence.

## Contiguous range rule

Each block represents one contiguous interval of global document IDs.

For block `Bi`:

    range_start =
        global_doc_base(Bi)

    range_end =
        range_start + block_document_count(Bi)

The interval is:

    [range_start, range_end)

The complete ordered block sequence must cover:

    [0, global_document_count)

exactly once.

Because the ranges are derived by prefix sums, V1 permits:

- no gaps;
- no overlaps;
- no reordered internal block document sequence;
- no independently assigned global document bases.

## Duplicate documents

Byte-identical documents remain distinct when they occupy distinct positions
in the global source manifest.

They retain distinct global document IDs.

Content deduplication is forbidden.

Two documents with equal byte length and SHA-256 may coexist when their
source-manifest entries are independently valid.

## Duplicate blocks

The same `runtime_index_id` must not appear more than once in one composition
root.

Repeating one block would repeat its complete source-manifest range and
silently double-count its results.

Different runtime layouts over equal source bytes are not globally forbidden.

They may exist in different composition roots.

Within one root, the flattened source-manifest sequence must remain valid and
each global source document position must occur exactly once.

## Empty documents

Empty documents:

- remain present in global identity;
- preserve their global document IDs;
- contribute no match coordinates;
- affect all later global document bases;
- must not be removed during composition.

## Composition root format

The Composition V1 root format label is:

    GLYPH_COMPOSITION_ROOT_V1

A root contains at least:

- format;
- global document count;
- block count;
- global runtime corpus ID;
- global source manifest ID;
- ordered block records;
- composition root ID;
- publication-complete status.

Each ordered block record contains at least:

- block ordinal;
- block document count;
- runtime index ID;
- runtime manifest SHA-256.

The block ordinal must equal its zero-based position in the root sequence.

## Composition root identity preimage

The normative `composition_root_id` preimage is:

    ASCII("GLYPH_COMPOSITION_ROOT_V1")
    BYTE(0x00)

    global_runtime_corpus_id_raw_32_bytes
    global_source_manifest_id_raw_32_bytes
    U64_BE(global_document_count)
    U64_BE(block_count)

For each block in canonical root order:

    U64_BE(block_ordinal)
    U64_BE(block_document_count)
    block_runtime_index_id_raw_32_bytes
    block_runtime_manifest_sha256_raw_32_bytes

The resulting identity is:

    composition_root_id =
        SHA256(composition_root_id_preimage)

Display form is lowercase hexadecimal with 64 characters.

JSON whitespace, absolute filesystem paths, temporary directories,
wall-clock time, build duration and host names do not participate in this
identity.

## Root validity

A Composition V1 root is valid only if:

1. the format is exactly `GLYPH_COMPOSITION_ROOT_V1`;
2. all integer values fit unsigned 64-bit representation;
3. checked arithmetic is used for every prefix sum and total;
4. block count is at least one;
5. every block document count is at least one;
6. block ordinals are dense and ordered from zero;
7. every `runtime_index_id` is well-formed and unique within the root;
8. every runtime manifest SHA-256 is well-formed;
9. every referenced block exists and verifies;
10. every block-local manifest is internally valid;
11. flattened block document records form one valid global operator manifest;
12. recomputed global `runtime_corpus_id` matches the committed value;
13. recomputed global `source_manifest_id` matches the committed value;
14. recomputed `composition_root_id` matches the committed value;
15. publication-complete status is valid;
16. no unsupported runtime profile is present.

A root failing any condition is not queryable.

## Publication rule

A Composition V1 root must be constructed in a temporary sibling location.

It becomes visible only after:

- every referenced runtime unit exists;
- every block identity verifies;
- all global identities are recomputed;
- all root validity rules pass;
- the complete root manifest is written;
- a composition-complete marker is written.

An interrupted build must not appear as a complete composition.

Publishing a changed layout creates a new root and a new
`composition_root_id`.

An existing root is immutable.

## Query fan-out

Composition V1 uses:

    ALL_ROOT_BLOCKS_REQUIRED_V1

Every non-empty query must be executed against every block committed by the
root.

V1 has no attribute routing, pruning, field mask or optional-block semantics.

A block returning zero matches is still a successfully queried and covered
block.

A block that was not queried is missing coverage.

## Query identity

Composition V1 uses the exact query-byte identity rules already defined by
the operator and runtime layers.

The exact query bytes and their commitments must be identical for all blocks.

Different blocks must not receive different query transformations.

## Count semantics

For query `q`, let:

    count_i(q)

be the complete exact count returned by block `Bi`.

The global count is:

    global_match_count(q) =
        checked_sum(count_i(q))
        for i = 0 .. block_count - 1

Every block count must be obtained successfully before a successful global
result may exist.

Count overflow must produce a typed limit failure.

Count must never be reduced because locate is bounded.

## Global locate order

Every block-local coordinate:

    (local_doc_id, doc_offset)

is converted to:

    (
        global_doc_base(block) + local_doc_id,
        doc_offset
    )

The complete global coordinate sequence is ordered by:

    global_doc_id ascending
    then doc_offset ascending

Because blocks contain contiguous global document ranges and appear in global
document order, canonical merge is equivalent to:

1. visit blocks in root order;
2. preserve each block's canonical local coordinate order;
3. remap local document IDs to global document IDs;
4. append the remapped sequences.

Thread or process completion order is not authoritative.

## Global bounded locate

`max_offsets` applies to the complete global coordinate sequence.

It must not be applied independently as a full allowance to every block.

A limit of zero is valid.

The required semantic result is the canonical prefix of length:

    min(global_match_count, max_offsets)

A correct V1 execution may use two phases.

### Phase 1 — complete count and coverage

Every expected block is verified and queried for exact count.

Failure of any block aborts the composition query.

### Phase 2 — bounded canonical fill

Set:

    remaining = max_offsets

Visit blocks in root order.

For each block:

- if `remaining == 0`, no locate output is required from that block;
- otherwise request at most `remaining` canonical local coordinates;
- remap them to global coordinates;
- append them;
- subtract the returned count from `remaining`.

The total count remains the complete Phase 1 count.

The returned coordinate sequence must equal the global canonical prefix.

## Result flags

Composition results must preserve the existing operator meaning of:

- `match_count`;
- `returned_count`;
- `bounded`;
- `offsets_complete`;
- `max_offsets`.

Required:

    match_count =
        complete global count

    returned_count =
        length(returned coordinates)

    offsets_complete =
        (returned_count == match_count)

A successful count-only query with `max_offsets = 0` is valid.

## Boundary semantics

Composition does not change document-boundary semantics.

The authoritative boundary policy remains:

    DOCUMENT_LOCAL_MATCHES_ONLY_V1

No valid match may cross:

- two documents inside one block;
- the final document of one block and the first document of another block;
- any virtual sentinel;
- any runtime artifact boundary.

A physical concatenation of block source bytes is not the matching oracle.

Cross-block matching is not filtered after counting.

It is structurally absent because every logical document remains an
independent matching domain.

## Determinism

For one valid root, one exact query and one `max_offsets` policy, the following
must be deterministic:

- global runtime corpus identity;
- global source manifest identity;
- composition root identity;
- expected block sequence;
- complete match count;
- returned coordinate sequence;
- bounded/completeness flags;
- coverage result.

Sequential, reversed, parallel or randomized completion schedules must
produce the same canonical result.

## Complete coverage

A successful Composition V1 result requires:

    expected_blocks == verified_blocks == queried_blocks

under exact ordered block identity comparison.

Coverage is not inferred from a count.

Coverage must be established independently.

The following states are different:

    complete coverage + zero matches

and:

    incomplete coverage + unknown result

The second state must never be represented as `match_count = 0`.

## Partial success

Partial success is forbidden.

If any required block is:

- missing;
- unreadable;
- incomplete;
- mutated;
- incompatible;
- unverifiable;
- not queried;
- changed during query;
- inconsistent with the root;

the complete composition query fails.

Previously computed block results must not be emitted as a successful
composition result.

## Reference error classes

The following names define composition-level reference error classes.

They are not public C ABI status codes.

### COMPOSITION_E_ROOT_INVALID

The root structure, field domain, ordering, duplication rule or publication
state is invalid.

### COMPOSITION_E_ROOT_MISMATCH

The recomputed `composition_root_id` differs from the committed value.

### COMPOSITION_E_IDENTITY

A global or block-local corpus, source-manifest or runtime identity does not
match the committed records.

### COMPOSITION_E_COVERAGE

One or more expected blocks were not successfully verified and queried.

### COMPOSITION_E_VERSION

A required block or root format/profile is unsupported or incompatible with
Composition V1.

### COMPOSITION_E_LIMIT

An integer conversion, document-base calculation or count summation exceeds
the permitted numeric domain.

### COMPOSITION_E_RUNTIME

A required block runtime query fails.

### COMPOSITION_E_VERIFY

Replay, source-byte checking, manifest verification or artifact verification
fails.

### COMPOSITION_E_INTERNAL

An implementation violates a composition invariant not attributable to valid
external input.

## Failure-output rule

A failed composition query must not return a successful result object.

A failure object must identify at least:

- reference error class;
- failed phase;
- composition root ID when available;
- affected block ordinal when applicable;
- affected runtime index ID when available;
- human-readable diagnostic text.

Diagnostic text is not authoritative identity material.

## Repartition semantics

Consider two valid composition roots `R1` and `R2`.

If they flatten to exactly the same global ordered operator document
sequence, then they must have identical:

- global runtime `corpus_id`;
- global `source_manifest_id`;
- global document count;
- canonical query counts;
- canonical query coordinates.

If their block boundaries or runtime artifacts differ, they must have
different:

- ordered block records;
- block runtime index identities when rebuilt;
- composition root identities.

Therefore:

    semantic retrieval result may remain equal

while:

    execution-layout provenance changes

Evidence committed to `R1` must not replay as evidence committed to `R2`
without explicitly verifying the new root.

## Reorder semantics

Changing global document order changes:

- global document IDs;
- runtime corpus identity;
- source manifest identity;
- composition root identity;
- potentially canonical coordinate order.

A reordered composition is a different logical corpus.

It is not repartitioning.

## Source rename semantics

Renaming one source document while preserving document bytes and global order:

- preserves runtime `corpus_id`;
- changes `source_manifest_id`;
- changes `composition_root_id`;
- preserves numeric match coordinates;
- changes path-bearing evidence.

## Runtime rebuild semantics

Rebuilding one block with a different but semantically equivalent supported
runtime layout:

- preserves global runtime `corpus_id`;
- preserves global `source_manifest_id`;
- may change block `runtime_index_id`;
- changes `composition_root_id`;
- must preserve exact count and canonical coordinates.

## Minimum evidence requirements

A future Composition V1 evidence artifact must bind at least:

- evidence format version;
- global runtime `corpus_id`;
- global `source_manifest_id`;
- `composition_root_id`;
- complete ordered expected block records;
- complete ordered verified block records;
- exact query identity;
- query binary commitments;
- `max_offsets` policy;
- complete per-block count commitments;
- complete global match count;
- returned canonical global coordinates;
- bounded and completeness flags;
- document-boundary policy identifier;
- composition policy identifier;
- coverage policy identifier.

Required identifiers are:

    composition_policy =
        "ORDERED_CONTIGUOUS_RUNTIME_UNITS_V1"

    coverage_policy =
        "ALL_ROOT_BLOCKS_REQUIRED_V1"

    document_boundary_policy =
        "DOCUMENT_LOCAL_MATCHES_ONLY_V1"

The exact evidence serialization and evidence identity preimage belong to a
separate Composition Evidence V1 specification.

## Replay requirements

Independent composition replay must:

1. verify the root format and root identity;
2. verify every expected runtime unit;
3. reconstruct the flattened global source manifest;
4. recompute global runtime corpus identity;
5. recompute global source manifest identity;
6. verify contiguous global document ranges;
7. verify the exact query bytes;
8. query every expected block;
9. recompute the complete count;
10. recompute the canonical global coordinate prefix;
11. byte-check every returned source span;
12. verify complete coverage;
13. verify all policy identifiers;
14. reject any stored success flag not supported by recomputation.

Replay must fail if any expected block is unavailable.

## Required reference fixtures

The Composition V1 reference gate must include at least three runtime units.

The complete fixture must contain:

- an empty document;
- two byte-identical documents at distinct positions;
- a document containing `0x00`;
- a document containing `0xFF`;
- a query present in multiple blocks;
- a query absent from every block;
- a query crossing a document boundary only in physical concatenation;
- a query crossing a block boundary only in physical concatenation;
- bounded locate cases;
- `max_offsets = 0`;
- a query longer than every document.

## Required repartition fixture

The same global ordered document sequence must be represented by at least two
different valid block partitions.

The gate must prove:

- equal global runtime corpus IDs;
- equal global source manifest IDs;
- unequal composition root IDs;
- equal complete counts;
- equal canonical coordinates;
- equal bounded canonical prefixes.

## Required scheduling fixture

The same query must be evaluated under simulated:

- canonical block order;
- reverse execution order;
- randomized completion order.

Canonical output must be bytewise identical after normalization.

## Required mutation failures

The executable reference gate must reject or fail correctly for at least:

1. one required block removed;
2. one valid block substituted;
3. block order changed without updating the root;
4. one block entry duplicated;
5. one runtime manifest byte changed;
6. one runtime manifest hash changed;
7. one `runtime_index_id` changed;
8. global runtime corpus ID changed;
9. global source manifest ID changed;
10. composition root ID changed;
11. one source document byte changed;
12. one block result omitted while coverage is claimed;
13. incomplete coverage represented as zero matches;
14. merged coordinates reordered;
15. local-to-global document base changed;
16. integer overflow attempted;
17. `max_offsets` applied independently per block;
18. unsupported root version;
19. unsupported runtime profile;
20. replay attempted against a different root;
21. document order changed;
22. empty document removed;
23. duplicate document deduplicated;
24. physical concatenation used as the matching oracle;
25. stored byte-check success trusted without recomputation.

## Reference success marker

The future executable composition reference gate must print:

    GLYPH COMPOSITION REFERENCE OK

only after every required semantic obligation and mutation test passes.

This marker must not be added to the required top-level `VERIFY OK` closure
until a separate project decision explicitly promotes composition from
research-tier reference semantics.

## Claim boundary

After the reference gate exists, the safe claim is limited to:

GLYPH defines and executable-tests reference semantics for composing multiple
verified operator runtime units with:

- stable global document coordinates;
- layout-sensitive composition identity;
- complete all-block coverage;
- deterministic count and bounded locate;
- fail-closed missing-block behavior;
- independent reference replay.

The reference gate alone does not establish:

- production Index Forest implementation;
- performance or scalability;
- distributed execution;
- field-aware routing;
- incremental update;
- compaction;
- arbitrary block pruning;
- multi-machine fault tolerance;
- C ABI support;
- Python or Go bindings;
- SIMD or GPU acceleration;
- ACEAPEX integration;
- replacement of Elasticsearch, Redis, Snowflake or vector databases.

## Explicit non-goals

Composition V1 does not define:

- a network protocol;
- a stable public API;
- a shared-library implementation;
- block compaction;
- block mutation;
- mutable roots;
- field predicates;
- access control;
- ranking;
- semantic search;
- approximate matching;
- cross-document matching;
- cross-block byte-stream matching;
- performance requirements.

## Completion condition

`GLYPH_COMPOSITION_SEMANTICS_V1` is complete only when:

1. this normative specification is committed;
2. an independent reference oracle is committed;
3. the required three-block fixture exists;
4. all required mutation cases are executable;
5. repartition stability is demonstrated;
6. scheduling independence is demonstrated;
7. complete coverage is fail-closed;
8. global bounded locate matches the independent oracle;
9. independent replay exists;
10. the existing P/R/O/I0 verification chain remains unchanged and green;
11. the reference checker prints:

        GLYPH COMPOSITION REFERENCE OK

Until those conditions are satisfied, Composition V1 remains a normative
draft and no implementation claim is permitted.
