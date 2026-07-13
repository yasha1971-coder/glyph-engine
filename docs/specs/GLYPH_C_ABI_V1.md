# GLYPH_C_ABI_V1

Status: normative draft; implementation blocked
Version: 1
Date: 2026-07-14
Phase: I0 contract freeze

## Purpose

Define the first stable public C ABI for embedding the verified GLYPH
binary-safe runtime.

The authoritative public declaration is:

    include/glyph/glyph.h

No implementation exists during I0.

## Scope

C ABI V1 provides:

- library ABI version discovery;
- opening one immutable operator runtime index;
- immutable index metadata;
- document identity and raw relative-path lookup;
- exact binary count;
- bounded exact locate;
- explicit close.

C ABI V1 does not provide:

- index construction;
- source-byte extraction;
- dynamically allocated result arrays;
- pointers into mmap regions;
- digital signing;
- network or IPC transport;
- asynchronous callbacks.

## Language boundary

The header must compile as strict C99/C11 and as C++.

The ABI must use:

- opaque handles;
- fixed-width integer types;
- `glyph_` symbol names;
- versioned exported function names;
- caller-owned memory.

The ABI must not expose:

- STL types;
- C++ classes;
- exceptions;
- references;
- templates;
- C++ language booleans;
- platform-width integer types;
- internal file-format structures.

## ABI version

The ABI version is independent of:

- index format version;
- bundle format version;
- signed-statement version;
- library marketing version.

`glyph_abi_version_v1()` returns `GLYPH_ABI_VERSION_V1`.

A new incompatible ABI requires new symbol names.

## Handle ownership

`glyph_index_open_v1()` creates one opaque handle.

On success:

- `*out_index` is non-null;
- the caller owns the handle;
- the handle may be shared for concurrent read-only operations;
- the handle remains valid until successful close.

On failure:

- `*out_index` is null.

The caller must not copy ownership into multiple independent close paths.

## Close behavior

`glyph_index_close_v1()` receives a pointer to the caller's handle variable.

On success:

- all mappings and owned resources are released;
- `*inout_index` is set to null;
- the old handle value is invalid.

If active operations exist:

- close returns `GLYPH_E_BUSY`;
- the handle remains valid;
- `*inout_index` remains unchanged.

A null handle variable is rejected with `GLYPH_E_ARG`.

## Query input lifetime

Query data is borrowed.

The runtime may read query bytes only during the call.

The runtime must not:

- retain the query pointer;
- write to query memory;
- require a terminator;
- interpret bytes as UTF-8 or another text encoding.

`query_size` is authoritative.

An empty query is rejected.

## Locate output model

Locate coordinates are written into caller-owned storage.

The caller supplies:

- `max_results`;
- `coordinate_capacity`;
- a coordinate buffer large enough for `max_results`.

Required relation:

    max_results <= coordinate_capacity

When `max_results` is zero:

- `coordinates` may be null;
- `coordinate_capacity` must be zero;
- the total match count is still returned.

A locate operation must never allocate output proportional to total match
count.

## Locate result semantics

Successful locate returns:

- `total_matches`;
- `returned_matches`;
- `complete`.

Required invariants:

    returned_matches <= max_results
    returned_matches <= coordinate_capacity

If:

    total_matches <= max_results

then:

    returned_matches == total_matches
    complete == 1

Otherwise:

    returned_matches == max_results
    complete == 0

Coordinates use canonical global ordering:

    doc_id ascending
    then doc_offset ascending

## Failure output semantics

For valid output pointers, the runtime initializes scalar result structures
before performing fallible work.

On failure:

- scalar counts are zero;
- completeness is zero;
- coordinate-buffer contents are unspecified;
- coordinate-buffer contents must be ignored;
- no evidence artifact may treat the call as successful.

Timeout, OOM, cancellation, verification error, or internal error does not
produce a successful partial locate result.

## Document path semantics

Source paths are returned as raw relative path bytes.

They are not required to be valid UTF-8.

They are not null-terminated.

`glyph_document_path_v1()` first reports the exact required byte size.

If the caller buffer is too small:

- the function returns `GLYPH_E_LIMIT`;
- `out_required_size` is still authoritative;
- no truncated path is reported as success.

## Structure versioning

Public extensible structures begin with:

    uint32_t struct_size

The caller initializes `struct_size`.

The implementation must:

- reject structures smaller than the V1 prefix;
- read only fields defined by the V1 prefix;
- reject unknown non-zero V1 flags;
- ignore bytes beyond the known V1 prefix.

Reserved V1 fields must be zero when supplied by the caller.

## Status codes

Status values are stable and append-only.

V1 defines:

- `GLYPH_OK`
- `GLYPH_E_ARG`
- `GLYPH_E_FORMAT`
- `GLYPH_E_VERIFY`
- `GLYPH_E_VERSION`
- `GLYPH_E_IO`
- `GLYPH_E_NOMEM`
- `GLYPH_E_LIMIT`
- `GLYPH_E_TIMEOUT`
- `GLYPH_E_BUSY`
- `GLYPH_E_CLOSED`
- `GLYPH_E_INTERNAL`
- `GLYPH_E_UNSUPPORTED`

A C++ exception must never cross an exported function boundary.

Unknown exceptions map to `GLYPH_E_INTERNAL`.

Human-readable error text is not part of the stable ABI contract.

## Open options

Open options may limit:

- total mapped runtime bytes;
- document count;
- maximum query bytes.

Zero means implementation-defined safe default, not unlimited by implication.

Index opening always performs the integrity and structural verification required
by the mmap trust model.

C ABI V1 has no flag that disables required verification.

## Query options

Query options reserve a relative timeout field.

A zero timeout means no ABI-level deadline.

Before timeout enforcement is implemented, a non-zero timeout must return
`GLYPH_E_UNSUPPORTED` before query work begins.

Once timeout support is declared conformant, expiry returns
`GLYPH_E_TIMEOUT` and never successful partial evidence.

## Thread-safety contract

After successful open, these operations are intended to be concurrently
read-safe on one handle:

- index metadata;
- document metadata;
- document path;
- count;
- locate.

Close is not concurrent-read-safe as a successful destructive operation.

Concurrent close must return `GLYPH_E_BUSY` while any operation is active.

## CLI parity requirement

The future CLI must use the same reusable runtime core as the C ABI.

A separate CLI-only search implementation is forbidden.

The full semantic baseline must be re-executed through a dynamically loaded
public shared library and a pure-C harness.

## Acceptance boundary

The header is only a contract during I0.

No production or compatibility claim is permitted until:

- the shared library exists;
- exported symbols are audited;
- a pure-C consumer passes;
- count and locate match the verified oracle;
- error and lifetime gates pass;
- clean-checkout verification passes.
