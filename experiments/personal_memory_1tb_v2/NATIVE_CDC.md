# Exact-boundary native CDC experiment

Opt-in only: build gear_native.cpp with g++ -O3 -std=c++17 -shared -fPIC,
then set GLYPH_CDC_NATIVE to the absolute trusted .so path. No runtime downloads,
auto compilation, or automatic enablement. Default remains split_python.
The native function uses the Python-generated Gear table and current parameters;
it retains unsigned 64-bit wraparound and resets after every boundary. Chunk bytes
are still sliced in Python. There is no format, Gear table, recipe or codec change.
Library code is trusted executable code; use only a library built from this source.

Tests compare every boundary AND chunk payload on empty, minimum/maximum edges,
zero, 0xff, random, insertion and deletion inputs and alternative parameters.
The full suite is run with GLYPH_CDC_NATIVE enabled. This does not prove all inputs;
the identical scalar recurrence supplies the implementation equivalence argument.
Native tests skip if g++ unavailable, so inspect skips when reproducing.

NATIVE_CDC_SYNTHETIC_RESULT.json: two synthetic 4MiB inputs, three repeats, alternating
Python/native order. Input one random.Random(241).randbytes(4194304), input two
(b'GLYPH public synthetic structured data 123456789\n'*100000)[:4194304].
Observed Python medians 0.7330/0.7438s, native 0.004588/0.003754s: ~160x/~198x.
Native includes ctypes/table construction/boundary buffer/Python bytes slicing;
external comparison hashes excluded from timings. Small cache-friendly inputs in
this execution environment, NOT laptop speed or end-to-end Vault acceleration.
A separate codec trial showed ~2.546s one worker vs ~2.012s two workers for these
same two inputs, including read/compress/roundtrip; not representative Silesia.

encode_microscope.py runs the same measurements on an explicit single directory
(no recursion, 1..500 files <=64MiB, exact excluded password basename), with optional
--codec-study. It records every candidate's size/time and three alternating serial/
two-process trials with exact verification. Existing output refused. Payloads are
not published; only hashes/timing/counts. Codec profiling does NOT invoke CDC.

Important corrections to the optimization hypothesis:
* Silesia 101.56s archive build did not invoke chunk_versions.split at all.
* Its actual manifest selected bzip2 for six of twelve files, not zero or one.
* xz-only density and 8-worker scaling cannot be assumed. This experiment measures
  xz candidate sizes and uses two workers first to avoid uncontrolled RAM growth.
* No zstd, sample router or dictionary implementation was added.

Falsifiers: any different CDC boundary/payload rejects this implementation; any
end-to-end unchanged cost rejects CDC as the bottleneck of THAT path. Removing
codecs is rejected if the size penalty exceeds the agreed workload-specific budget.
Run laptop profiler only after pausing competing endurance work to avoid resource
contention. The old unchanged Python decoding path remains the fallback.
