# Independent prior art and next RAM gate

External [Move-r](https://github.com/LukasNalbach/Move-r) commit `ca760d8a56ac659413c9e3505e7800839963d1b0` has exact count/locate and multi-thread full text reconstruction. Functional search+restore is not new to GLYPH.

[Orbit](https://github.com/drnatebrown/orbit) commit `034d2ea150f1a88b6b18602aa2938426022343df` is a GPL-3.0 research library for `O(r)` move structures. [SEA 2026 paper](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.SEA.2026.9) proves BWT inversion in `O(n)` time and `O(r)` working space under its model. Measure on enwik8 **and** highly repetitive chr1/version corpora. Do not import GPL code into the runtime without explicit owner/legal review; independent benchmarking is permitted.

Compare same pinned raw SHA, exact source restore, query counts/coordinates, serialized index and proof bytes, peak RSS, 1/2/4 thread wall, cold/warm latency and hostile rejection. Fail if bytes or coordinates differ. GLYPH's differentiation remains proof-carrying canonical evidence, not mere searchable compression. No merge/production changes.
