# Succinct data structure limits notes

Status: research notes.

## Focus

GLYPH uses FM-index style exact retrieval.

Document limits honestly:
- symbol distribution effects
- binary alphabet behavior
- UTF-8 / mixed byte patterns
- entropy effects
- index size growth
- shard scaling
- construction cost

## Rule

Do not claim "works best for everything".

Preferred framing:

    deterministic exact retrieval over static corpora,
    with measurable limits.
