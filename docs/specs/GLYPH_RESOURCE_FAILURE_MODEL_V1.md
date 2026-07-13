# GLYPH_RESOURCE_FAILURE_MODEL_V1

Status: normative draft; implementation blocked
Version: 1
Date: 2026-07-14
Phase: I0 contract freeze

## Purpose

Define deterministic failure behavior for memory pressure, limits, timeouts,
interrupted builds, and bounded outputs.

## Architectural separation

GLYPH separates:

1. query plane;
2. build plane.

The query plane is a long-lived in-process library over immutable indexes.

The build plane performs heavy index construction in an isolated worker
process.

## Query-plane limits

The future open handle must enforce configured limits for at least:

- total mapped bytes;
- document count;
- maximum query bytes;
- caller result capacity.

A declared value exceeding a configured limit must fail before a dangerous
allocation or mapping.

The status is `GLYPH_E_LIMIT`.

## Checked arithmetic

Every untrusted arithmetic operation involving:

- file size;
- section count;
- section offset;
- section length;
- element width;
- document count;
- coordinate capacity;
- query length

must use checked addition and multiplication.

Overflow must return `GLYPH_E_FORMAT` or `GLYPH_E_LIMIT`.

It must not wrap.

## Allocation behavior

The runtime must not allocate memory proportional to:

- total corpus size during an ordinary query;
- total match count during bounded locate;
- a hostile declared count before bounds validation.

Catchable allocation failure returns:

    GLYPH_E_NOMEM

No C++ allocation exception may cross the C ABI.

## Operating-system OOM boundary

An in-process library cannot guarantee recovery from every host OOM policy.

Linux overcommit or the OOM killer may terminate the process before a normal
allocation error is returned.

Therefore GLYPH must not claim:

    in-process OOM proof

The production build plane must use external enforcement such as:

- cgroups;
- systemd limits;
- containers;
- dedicated worker processes;
- process deadlines.

## Bounded locate

`max_results` is mandatory input to locate.

The result must distinguish:

- total matches;
- returned matches;
- completeness.

Output memory is owned and bounded by the caller.

No hidden unbounded coordinate collection is allowed before truncation.

## Timeout contract

The ABI reserves a relative timeout in nanoseconds.

During a phase where timeout support is not implemented:

- zero is accepted;
- non-zero returns `GLYPH_E_UNSUPPORTED`;
- the function performs no query work.

After timeout support becomes conformant:

- the timer begins at function entry;
- expiry returns `GLYPH_E_TIMEOUT`;
- scalar result fields are zero;
- coordinate contents are ignored;
- no evidence artifact may treat the call as success.

The implementation must define and test polling points.

## Partial-result rule

The following failures never return successful partial evidence:

- timeout;
- memory failure;
- I/O failure;
- verification failure;
- unsupported feature;
- internal failure;
- close race.

The only successful incomplete locate is an explicitly bounded locate with:

    complete == 0

and otherwise valid exact count and canonical prefix coordinates.

## Build-plane isolation

Index construction must not execute inside the long-lived query server process
for the industrial deployment profile.

The builder worker must:

- read committed source inputs;
- write into a private temporary directory;
- operate under memory and time limits;
- publish nothing before complete verification;
- return structured failure;
- leave the previous published index intact after failure.

## Interrupted build

A killed, timed-out, OOM-terminated, or crashed builder must not create a valid
published completion marker.

A partially written temporary directory must not be accepted as a runtime
index.

## Disk-full behavior

Disk-full and short-write conditions must:

- return structured failure;
- prevent completion-marker publication;
- preserve the previously published index;
- leave recoverable temporary state or remove it safely.

## Publication durability

Logical atomic rename is not identical to power-loss durability.

A production claim of crash-consistent publication requires executable tests
for the exact synchronization sequence, including:

- payload synchronization;
- temporary-directory synchronization;
- rename;
- parent-directory synchronization.

Until those gates pass, GLYPH may claim logical interrupted-build rejection,
not proven power-loss durability.

## Required testing

Future executable gates must include:

- allocation-failure injection;
- huge declared size with tiny real file;
- arithmetic overflow fixtures;
- timeout before work;
- timeout during count;
- timeout during locate;
- zero-result capacity;
- insufficient result capacity;
- builder OOM kill;
- builder timeout kill;
- disk-full simulation;
- short-write simulation;
- interrupted publication;
- previous-index preservation.
