# Known Limitations — GLYPH v0.x

## Sentinel limitation

Current FM-index builds require:

    corpus + appended real 0x00 sentinel

Therefore:

- input corpora must not contain 0x00 bytes
- arbitrary raw-byte corpora are not yet fully supported

Future solution:

- 257-symbol alphabet
or
- explicit out-of-band sentinel representation

## Static corpus assumption

GLYPH v0.x assumes immutable corpora.

Incremental index mutation is not yet implemented.

## Exact retrieval only

GLYPH currently supports deterministic exact byte retrieval.

Not implemented:

- fuzzy matching
- ranking
- semantic retrieval
- regex engine
- approximate nearest neighbor search

## Build-time cost

FM-index construction is still expensive for large corpora.

Large datasets require substantial:

- RAM
- disk bandwidth
- build time

## API stability

Index formats and manifest semantics may still evolve during v0.x development.

Backward compatibility is not yet guaranteed.
