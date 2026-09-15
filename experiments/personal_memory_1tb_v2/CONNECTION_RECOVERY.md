# Connection and preview isolation fix

Reported symptom: browser disconnected during a response, and the old handler
attempted a second error response on the broken socket. The old HTTPServer also
processed one connection at a time, allowing preconnects or slow preview operations
to block listing/confirmation. The trace does not prove stored archive corruption
or completion of any deletion. No private trace/token/path is reproduced here.

A bounded threaded server (12 connections, socket idle/write timeout 10 seconds)
now separates network I/O from state operations. Responses are buffered before
network write; BrokenPipe/connection reset/timeout closes that request without a
second response. POST bodies are bounded and read before acquiring the state lock.
Mutation and catalog operations remain serialized by an RLock; existing writer and
browser lifetime locks remain. Read/decode operations use two separate slots and
run outside the state lock, with snapshot membership checked before and after.
Concurrent deletion may cancel an in-flight read, which then fails closed. Data
already transmitted to a browser cannot be recalled. Native PDF rendering may
still reject a structurally invalid PDF even if its saved byte hash is correct.

Preview restore worker deadline is 30 seconds; download retains its 330-second
budget. Existing 1 GiB address-space, 128 MiB file-output and 300 CPU-second limits
are applied inside the worker before opening the archive, rather than preexec_fn
in a threaded parent. Socket delivery owns no mutation lock. Expensive writes are
still serialized; this is not an industrial multiwriter service. No PDF structure
repair or graphical-rendering validation is claimed.

113 synthetic tests pass in 36.802 seconds (CONNECTION_RECOVERY_TEST_OUTPUT.txt).
New cases: idle TCP preconnection does not block listing; reset during preview has
no server error callback; delayed restore allows listing and confirmed deletion;
a corrupt payload can be deleted without preview; simultaneous updates from the
same parent have exactly one winner and one stale rejection. Previous deletion
crash/recovery, shared block preservation and integrity tests remain green.
Laptop rerun remains required. Existing source files/archive are not rewritten.
