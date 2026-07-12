# GLYPH_BUNDLE_COMPLETENESS_V1

Status: normative draft  
Version: 1  
Proof obligation: P11  
Date: 2026-07-13

## Purpose

Define when a GLYPH evidence bundle is complete, internally consistent,
portable, and independently replayable.

P11 establishes the bridge between deterministic evidence artifacts from P10
and the end-to-end verification closure required by P12.

## Dependencies

P11 depends on P1 through P10.

In particular:

- P4 defines canonical corpus identity;
- P7 defines canonical locate coordinates;
- P8 defines binary-safe query transport;
- P9 defines document-local boundary semantics;
- P10 defines deterministic authoritative artifact identity.

## Core invariant

A valid bundle contains every byte, schema, manifest entry, and replay component
required to independently verify its authoritative evidence.

For a valid bundle `B`:

    verify_bundle(B) == PASS

without requiring:

- the GLYPH source repository;
- an existing GLYPH build directory;
- the original absolute source path;
- network access;
- environment-specific configuration;
- hidden files outside the bundle;
- undeclared data dependencies.

## Bundle model

A P11 bundle contains exactly one manifest and a declared set of payload files.

Required logical roles:

1. evidence artifact;
2. source corpus bytes;
3. artifact schema;
4. independent replay program.

The minimum canonical layout is:

    bundle/
        bundle_manifest_v1.json
        artifact.json
        corpus.bin
        schema.json
        replay.py

Additional files are allowed only when explicitly listed in the manifest.

Unlisted files are forbidden.

## Manifest

The manifest is:

    GLYPH_BUNDLE_MANIFEST_V1

It must bind:

- bundle version;
- artifact version;
- required runtime;
- replay entrypoint;
- explicit external dependency list;
- every payload path;
- every payload role;
- every payload byte length;
- every payload SHA256;
- canonical bundle-root SHA256.

## Required runtime

The runtime requirement must be explicit.

For the P11 reference fixture:

    Python 3
    Python standard library only

This is an explicit runtime dependency, not a hidden data dependency.

The bundle must not import GLYPH repository modules.

## Canonical paths

All manifest paths must be:

- relative;
- slash-separated;
- normalized;
- non-empty;
- unique;
- free of `..`;
- free of `.` path components;
- free of absolute roots;
- free of NUL bytes.

Symlinks are forbidden.

Every declared path must resolve inside the bundle root.

## File coverage

Manifest coverage is exact:

    declared payload paths
    ==
    actual payload paths excluding the manifest

Therefore:

- a missing declared file is invalid;
- an extra undeclared file is invalid;
- duplicate entries are invalid;
- role duplication for singleton roles is invalid.

Required singleton roles:

- `artifact`;
- `corpus`;
- `schema`;
- `replay`.

## File integrity

For every declared file:

    actual_size == manifest.size_bytes

and:

    SHA256(actual_bytes) == manifest.sha256

The verifier must read and hash the file itself.

It must not trust hashes copied from the artifact.

## Bundle-root digest

The bundle-root digest is:

    SHA256(canonical_json(manifest.files))

where `manifest.files` is canonically sorted by path.

The manifest itself is excluded from the digest preimage to avoid
self-reference.

The digest proves internal manifest consistency, not publisher identity or
digital signature authenticity.

## Artifact-schema consistency

The schema must explicitly bind the artifact version.

The artifact must satisfy all required fields and declared primitive types.

At minimum, schema validation must check:

- artifact version;
- corpus object;
- corpus path;
- corpus SHA256;
- corpus byte length;
- query hex;
- query byte length;
- query SHA256;
- document-boundary policy;
- exact match count;
- canonical coordinates;
- returned count;
- bounded flag;
- offsets-complete flag;
- byte-check flag.

Unknown schema versions must be rejected.

## Corpus binding

The artifact must bind the bundled corpus by:

- relative path;
- SHA256;
- byte length.

The verifier must recompute all three.

An artifact referring to an external or absolute corpus path is invalid.

## Query binding

`query_hex` is authoritative.

The verifier must:

1. decode query bytes from lowercase hexadecimal;
2. reject malformed or noncanonical hex;
3. recompute query length;
4. recompute query SHA256.

Display text is never authoritative.

## Replay

Replay must execute from a directory outside the original repository and must
use only bundle-local payload files plus the explicitly declared runtime.

The replay program must independently verify:

1. manifest structure;
2. exact manifest file coverage;
3. path safety;
4. file sizes;
5. file SHA256 values;
6. bundle-root digest;
7. schema version;
8. artifact-schema consistency;
9. corpus identity;
10. query identity;
11. exact byte matches;
12. canonical coordinates;
13. match count;
14. bounded-locate semantics;
15. document-boundary policy;
16. byte-check result.

## No external data

A valid P11 bundle must contain no authoritative dependency on:

- source paths outside the bundle;
- URLs;
- repository-relative paths;
- temporary files;
- environment variables;
- current working directory;
- hostname;
- time;
- random state.

## Network independence

The replay path must not require network access.

A URL or network dependency declared inside authoritative bundle metadata is
invalid.

## Self-contained definition

For P11, `self-contained` means:

    all authoritative data and replay logic are inside the bundle

except for the explicitly declared general-purpose runtime:

    Python 3 standard library

It does not mean that the Python interpreter itself is embedded in the bundle.

## P12 handoff contract

P11 must produce a machine-readable result containing:

    p12_ready = true

only if all of the following hold:

- manifest coverage is exact;
- all file hashes match;
- schema matches artifact;
- corpus identity matches;
- query identity matches;
- independent replay succeeds;
- no undeclared external data dependency exists;
- the bundle can be copied to another directory and replayed there;
- all required mutation tests fail.

P12 may consume this result as the bundle node of the end-to-end proof graph.

P11 does not by itself establish the complete P1–P12 chain.

## Required fixtures

P11 fixtures must include:

1. ASCII corpus and query;
2. embedded `0x00`;
3. `0xFF`;
4. invalid UTF-8;
5. zero matches;
6. repeated matches;
7. bounded locate;
8. complete `00..ff` byte alphabet;
9. bundle copied to a different directory;
10. replay launched with isolated Python mode;
11. replay launched outside the repository;
12. manifest entries presented in noncanonical source order and normalized.

## Required mutation failures

The checker must reject:

1. missing artifact;
2. missing corpus;
3. missing schema;
4. missing replay program;
5. altered artifact bytes;
6. altered corpus bytes;
7. altered schema bytes;
8. altered replay bytes;
9. incorrect file size;
10. incorrect file SHA256;
11. incorrect bundle-root digest;
12. unlisted extra file;
13. duplicate manifest path;
14. duplicate singleton role;
15. absolute path;
16. parent traversal path;
17. symlink payload;
18. schema/artifact-version mismatch;
19. missing required artifact field;
20. corpus SHA256 mismatch;
21. corpus size mismatch;
22. malformed query hex;
23. query SHA256 mismatch;
24. incorrect match count;
25. incorrect coordinate;
26. unsorted coordinates;
27. duplicate coordinates;
28. false byte-check;
29. wrong boundary policy;
30. external corpus path;
31. URL dependency;
32. undeclared external dependency;
33. replay entrypoint outside bundle;
34. bundled replay importing repository code;
35. bundle copied without one required file;
36. manifest role mismatch.

## P11 invariant

For any valid bundle `B`:

    declared_files(B)
    ==
    actual_payload_files(B)

and for every payload file `f`:

    SHA256(bytes(f))
    ==
    manifest_sha256(f)

and:

    independent_replay(B)
    ==
    artifact_authoritative_result(B)

## Non-claims

P11 does not establish:

- publisher identity;
- digital signature authenticity;
- legal admissibility;
- semantic truth of source content;
- completeness of an external real-world collection;
- end-to-end closure of every P1–P11 implementation path.

The final proof-chain closure belongs to P12.

## Completion condition

P11 is complete only when:

1. this specification exists;
2. a canonical manifest format exists;
3. exact file coverage is verified;
4. all payload hashes are independently verified;
5. schema and artifact match;
6. corpus and query identity match;
7. replay succeeds outside the repository;
8. copied-bundle replay succeeds;
9. no hidden data dependency is required;
10. all mutation fixtures are rejected;
11. existing `./verify.sh` remains green;
12. the checker emits `p12_ready = true`.
