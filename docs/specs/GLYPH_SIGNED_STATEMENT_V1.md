# GLYPH_SIGNED_STATEMENT_V1

Status: normative draft for future E5 implementation
Version: 1
Date: 2026-07-14
Phase: I0 semantic freeze

## Purpose

Define the minimum semantics for authenticating an existing GLYPH evidence
bundle root.

This specification does not implement signing.

## Integrity versus authenticity

SHA-256 commitments establish byte identity relative to a known expected hash.

A digital signature establishes that a holder of a private key signed an exact
statement.

Neither property proves that source claims are true.

## Verifier-side policy

A verifier must select one explicit policy:

- `ALLOW_UNSIGNED`
- `REQUIRE_TRUSTED_SIGNATURE`

The following policy is forbidden:

    verify a signature only when one is present

That behavior permits signature stripping.

Under `REQUIRE_TRUSTED_SIGNATURE`, absence of a required signature is failure.

## Signed statement fields

The signed statement must bind at least:

- `statement_version`;
- `bundle_format_version`;
- `index_format_version`;
- `corpus_id`;
- `bundle_root_sha256`;
- `engine_commit`;
- `runtime_profile`.

Every field is authoritative.

Unknown or unsupported versions must be rejected before signature acceptance is
used as evidence.

## Statement version

The statement version is:

    GLYPH_SIGNED_STATEMENT_V1

## Signing preimage

The signature preimage is domain-separated:

    ASCII("GLYPH_SIGNED_STATEMENT_V1")
    ||
    0x00
    ||
    canonical_statement_bytes

The domain separator is part of the signed bytes.

Signing only a bare root digest is forbidden.

## Canonical statement encoding

The future implementation must define one deterministic canonical byte
encoding.

It must specify:

- field names;
- field order or canonical object ordering;
- integer encoding;
- string encoding;
- hex letter case;
- whitespace;
- duplicate-key rejection;
- unknown-field policy.

The canonical encoding must be independently reproducible.

Defining and freezing this encoding is a mandatory pre-implementation gate for
the signature phase.

No GLYPH implementation may produce a signature before canonicalization is
frozen and covered by executable positive, mutation, and independent-replay
fixtures.

## Target algorithm

The target V1 signature algorithm is:

    Ed25519

Algorithm identifiers are explicit and must not be inferred from key length.

## Signature envelope

A detached signature envelope must contain at least:

- envelope version;
- algorithm;
- signed statement;
- signer key identifier;
- public-key fingerprint;
- signature bytes.

The envelope itself is not trusted before signature verification.

## Trust store

Cryptographic validity and trust are separate.

A verifier must evaluate:

1. whether the signature is mathematically valid;
2. whether the key is in the configured trusted-key set;
3. whether policy requires that key or permits it;
4. whether statement versions and identities match the bundle being verified.

## Required rejection cases

Future fixtures must reject:

- missing required signature;
- stripped signature;
- malformed signature;
- wrong public key;
- untrusted but cryptographically valid key;
- changed bundle root;
- changed corpus identity;
- changed format version;
- changed runtime profile;
- changed engine commit;
- replay of a signed root under another format interpretation;
- duplicate statement fields;
- non-canonical statement encoding.

## What a valid signature proves

A valid trusted signature proves:

> The holder of the corresponding private key signed the exact canonical
> GLYPH statement.

## What a valid signature does not prove

It does not prove:

- source truth;
- legal admissibility;
- authorization of a human or company without external key policy;
- builder honesty;
- creation time;
- freshness;
- absence of replay;
- secrecy;
- absence of compromise of the private key.

## Freshness and timestamps

Trusted timestamps and replay prevention are separate future layers.

They must not be implied by an Ed25519 signature alone.

## Compatibility boundary

Introducing signatures must not change the existing bundle root silently.

The unsigned bundle format and signed envelope remain separately versioned.
