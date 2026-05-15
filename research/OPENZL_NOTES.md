# OpenZL notes

Source: encode.su thread, OpenZL discussion.

## Signal

OpenZL is interesting not because GLYPH should become a compressor, but because it validates a broader idea:

    structure matters before interpretation

OpenZL uses data descriptions, profiles, transforms, and chunking to exploit structured data.

GLYPH should remain a deterministic byte-exact retrieval engine, but this direction is relevant for future structural retrieval.

## Useful ideas to watch

- schema-aware processing
- deterministic structure descriptions
- chunking for very large inputs
- zero-copy field access
- transform graphs
- profile-driven pipelines
- structural preprocessing before heavier interpretation

## What not to copy

- LLM-generated schemas as a required path
- file-specific overfitting
- non-reproducible training tricks
- compression-first product direction

## GLYPH relevance

Possible future direction:

    deterministic structural retrieval

Examples:

- exact byte spans
- repeated layouts
- schema-aligned regions
- binary motifs
- protocol fragments
- structured log/event fields

This is not part of v0.2.

For now, GLYPH keeps its core invariant:

    exact bytes in -> deterministic matches out
