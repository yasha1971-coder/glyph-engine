# GLYPH_AI_DISCOVERY_FILES_V1

Status: deployed discovery files  
Date: 2026-06-28

## Purpose

Record the AI/search discovery files added for glyph.rs.

The goal is to make the public GLYPH reviewer portal easier to crawl, cite, and understand by classic search engines and AI retrieval systems, while avoiding overclaiming.

## Files added

- `robots.txt`
- `llms.txt`
- `sitemap.xml`

## Files updated

- `index.html`
- `site/index.html`

## JSON-LD

The homepage now includes JSON-LD structured data for:

- `WebSite`
- `SoftwareSourceCode`
- `SoftwareApplication`

## Policy

The current policy allows classic search crawlers, AI search/retrieval crawlers, AI user-request fetchers, and AI training crawlers.

This is intentional because GLYPH is currently an open public research/engineering project seeking discoverability and external review.

## Verification

After deployment, check:

    curl -A "OAI-SearchBot" https://glyph.rs/
    curl -A "OAI-SearchBot" https://glyph.rs/robots.txt
    curl -A "OAI-SearchBot" https://glyph.rs/llms.txt
    curl -A "OAI-SearchBot" https://glyph.rs/sitemap.xml

Expected result:

- homepage returns full static HTML
- `robots.txt` returns crawler directives
- `llms.txt` returns clean Markdown
- `sitemap.xml` returns XML sitemap
