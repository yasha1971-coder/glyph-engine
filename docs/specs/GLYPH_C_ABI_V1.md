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

## Pre-freeze exact semantics

### Initial supported host profile

The first conformant implementation target is:

    64-bit little-endian Linux

The public header may compile on another host.

A runtime build that cannot safely implement the V1 address-space or on-disk
format requirements must return `GLYPH_E_UNSUPPORTED` or be excluded from the
supported build matrix.

A 32-bit runtime is not conformant during E1.

### Nullable option structures

For `glyph_index_open_v1()`:

- `options == NULL` selects the library V1 default open policy;
- a non-null options structure must contain a valid `struct_size`;
- all unknown V1 flags and non-zero reserved fields are rejected.

For count and locate:

- `options == NULL` means zero flags and no ABI-level timeout;
- a non-null query-options structure must contain a valid `struct_size`;
- a non-zero timeout returns `GLYPH_E_UNSUPPORTED` until timeout support has
  passed its executable gate.

### Exact output initialization

Before fallible work begins, valid scalar outputs are initialized.

For `glyph_index_open_v1()`:

    *out_index = NULL

For count:

    *out_count = 0

For locate, after validating the output structure size:

- `complete = 0`;
- `total_matches = 0`;
- `returned_matches = 0`;
- all known reserved output fields are zeroed;
- caller-supplied `struct_size` remains unchanged.

For metadata output structures, after validating `struct_size`:

- all known output fields except `struct_size` are zeroed before fallible work;
- all known reserved fields are zero on successful return;
- `struct_size` remains the caller-supplied value.

Coordinate-buffer contents are unspecified after failure and must be ignored.

### Exact argument rules

A null required output pointer returns `GLYPH_E_ARG`.

A non-zero query size with a null query pointer returns `GLYPH_E_ARG`.

A zero query size returns `GLYPH_E_ARG`.

For locate:

- `max_results` must not exceed `coordinate_capacity`;
- when `max_results == 0`, `coordinates` may be null and
  `coordinate_capacity` must be zero;
- when `max_results > 0`, `coordinates` must be non-null;
- no coordinate beyond `returned_matches` is part of the result.

### Document-path size probe

`glyph_document_path_v1()` supports an exact size probe.

When:

    buffer == NULL
    buffer_capacity == 0

the function returns `GLYPH_OK` and writes the exact raw path-byte length to
`out_required_size`.

When capacity is insufficient:

- the function returns `GLYPH_E_LIMIT`;
- `out_required_size` contains the exact required size;
- no truncated path is reported as success.

When capacity is sufficient:

- exactly the raw path bytes are copied;
- no terminator is appended;
- the function returns `GLYPH_OK`.

A null buffer with non-zero capacity returns `GLYPH_E_ARG`.

### Repeated close

For `glyph_index_close_v1()`:

- `inout_index == NULL` returns `GLYPH_E_ARG`;
- `*inout_index == NULL` returns `GLYPH_E_CLOSED`;
- successful close sets `*inout_index` to null;
- a repeated close through the same handle variable therefore returns
  `GLYPH_E_CLOSED`.

### Close concurrency barrier

Read-only operations may execute concurrently on one handle.

The caller must establish a close barrier:

- after a thread begins `glyph_index_close_v1()`, no new operation may begin
  on that handle;
- operations that started before the close attempt may still be active;
- if such operations are active, close returns `GLYPH_E_BUSY`;
- on `GLYPH_E_BUSY`, the handle remains valid and unchanged;
- the caller may retry only after those operations complete;
- successful close occurs only when there are no active operations.

The raw C handle does not make stale-pointer use safe after successful close.

### Frozen V1 structure layout

For the initial supported host profile, the ABI requires these natural-layout
sizes:

| Type | Size |
|---|---:|
| `glyph_open_options_v1` | 72 bytes |
| `glyph_query_options_v1` | 56 bytes |
| `glyph_coordinate_v1` | 16 bytes |
| `glyph_locate_result_v1` | 56 bytes |
| `glyph_index_info_v1` | 120 bytes |
| `glyph_document_info_v1` | 96 bytes |

Packing pragmas are forbidden.

The executable ABI gate must validate both sizes and field offsets.

## Acceptance boundary

The header is only a contract during I0.

No production or compatibility claim is permitted until:

- the shared library exists;
- exported symbols are audited;
- a pure-C consumer passes;
- count and locate match the verified oracle;
- error and lifetime gates pass;
- clean-checkout verification passes.
