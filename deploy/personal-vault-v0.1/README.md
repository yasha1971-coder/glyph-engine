# GLYPH Personal Vault V0.1 — Pilot Deployment

Status: release-candidate deployment layer for the already-green product MVP core. This does **not** freeze the lifetime on-disk format and does **not** enable deletion of originals.

## Deployment invariant

The deployment sequence is deliberately one-way and fail-closed:

```
Git commit
  -> product CI green
  -> product-only release archive + SHA256SUMS
  -> optional GitHub build-provenance attestation for an approved SHA
  -> install into a versioned release directory
  -> laptop doctor
  -> laptop real-folder pilot
  -> full laptop verify
  -> exact search + restore + SHA equality
  -> first replica transfer to OVH staging
  -> full remote verify
  -> root SHA equality
  -> atomic publish of OVH replica
```

A failure at any step leaves the previous installation and the original user files untouched.

## Why laptop comes first

The laptop is the first product environment because the primary V0.1 question is whether a user can safely ingest, find and restore real files. OVH is the second independent node, not the place where untested research is developed.

## Product/research boundary

The V0.1 release archive intentionally contains only the product path needed for:

- `doctor`
- `version`
- `init`
- `add`
- `verify`
- `status`
- `list`
- exact byte `search`
- `restore`
- `free-space --dry-run`

Rejected LLM planners, embedding experiments, semantic routers and model weights are not shipped in the V0.1 release.

## Safety policy

V0.1 has these hard safety rules:

1. source deletion is disabled;
2. `free-space` is report-only;
3. the first real-data pilot always uses a copy of user files;
4. OVH transfer verifies the laptop Vault before transfer;
5. the laptop manifest root must not change during transfer;
6. the same GLYPH release commit must be active on both nodes;
7. the remote copy is transferred to an `.incoming-*` staging path;
8. the remote staging Vault must pass a full restore/hash verification;
9. its root SHA-256 must equal the laptop root SHA-256;
10. only then is the staging path atomically renamed to the requested OVH Vault path.

No script in this directory performs source deletion.

## Build release candidate

From a clean checkout at the commit you intend to pilot:

```bash
bash deploy/personal-vault-v0.1/make-release.sh dist
(cd dist && sha256sum -c *.tar.gz.sha256)
```

The archive records the exact Git SHA and contains an internal `SHA256SUMS` manifest. The release packager builds only the two native construction helpers used by the product path. Archive metadata is normalized; GitHub build provenance is a separate release-level proof of where the artifact was built.

## Attested release for an approved commit

The manual `publish-release-v0.1` workflow requires the exact 40-character commit SHA. It checks out that SHA directly, re-runs the product MVP gates, builds the product-only archive, verifies its checksum and generates a GitHub artifact attestation before uploading it.

When consuming an attested artifact with GitHub CLI available, verify provenance as an additional supply-chain check:

```bash
gh attestation verify glyph-personal-vault-v0.1-*.tar.gz \
  --repo yasha1971-coder/glyph-engine
```

Attestation supplements, rather than replaces, the archive SHA-256 and GLYPH's own runtime verification.

## Install on the laptop

The current V0.1 packaged pilot targets Linux. Extract the archive, then from its top-level directory:

```bash
(cd /path/to/download && sha256sum -c glyph-personal-vault-v0.1-*.tar.gz.sha256)
tar -xzf glyph-personal-vault-v0.1-*.tar.gz
cd glyph-personal-vault-v0.1-*
bash deploy/personal-vault-v0.1/install.sh
~/.local/bin/glyph version
~/.local/bin/glyph doctor
```

Installation is versioned under:

```text
~/.local/share/glyph/releases/<40-char-git-sha>/
```

`~/.local/bin/glyph` is only switched after the staged installation passes its internal hashes and `doctor` check. Older releases remain present for rollback.

## First laptop pilot

Use a COPY of a modest real folder first. Do not point V0.1 at the only copy of irreplaceable data.

```bash
mkdir -p ~/GlyphPilot
# Put a copied real test folder at ~/GlyphPilot/source-copy

glyph init ~/GlyphPilot/vault
glyph add ~/GlyphPilot/vault ~/GlyphPilot/source-copy
glyph verify ~/GlyphPilot/vault
glyph status ~/GlyphPilot/vault
glyph list ~/GlyphPilot/vault
```

Exact search:

```bash
glyph search ~/GlyphPilot/vault 'some exact phrase you know exists'
```

Restore one object by the logical path reported by `glyph list`:

```bash
glyph restore ~/GlyphPilot/vault 'relative/path/file.ext' ~/GlyphPilot/restored-file.ext
sha256sum ~/GlyphPilot/source-copy/relative/path/file.ext ~/GlyphPilot/restored-file.ext
```

The two SHA-256 values must match.

Check reclaimable source bytes without deleting anything:

```bash
glyph free-space ~/GlyphPilot/vault --dry-run
```

## Promote the identical release to OVH

Install the **same release archive** on the OVH account and run:

```bash
glyph version
glyph doctor
```

The `git_sha` reported by `glyph version` must equal the laptop value.

For the pilot, use an unprivileged dedicated account such as `glyph`. Keep code and Vault data separate conceptually:

```text
~glyph/.local/share/glyph/releases/<sha>/   # immutable/versioned software release
/srv/glyph/vaults/                          # Vault replicas
```

A later production hardening pass may move software releases to a root-owned `/opt/glyph/releases/` tree; that is intentionally not required to prove the V0.1 user loop.

Do not overwrite an existing Vault during the first replica transfer.

## Verified first replica to OVH

From the laptop:

```bash
bash deploy/personal-vault-v0.1/transfer-vault.sh \
  ~/GlyphPilot/vault \
  glyph@YOUR_OVH_HOST \
  /srv/glyph/vaults/pilot-001
```

The transfer script refuses to publish if the software release differs, the source Vault changes during transfer, remote verification fails, or the remote root hash differs. Disposable cache, journal and derived state are not treated as canonical replica payload.

## Rollback

Software rollback does not mutate Vault data. Point the command symlink back to a previously installed release directory:

```bash
ln -sfn ~/.local/share/glyph/releases/<previous-git-sha>/deploy/personal-vault-v0.1/glyph.py ~/.local/bin/glyph
glyph version
glyph doctor
```

Do not use an older binary against a Vault after a future format migration unless that release is explicitly declared compatible. V0.1 does not yet implement format migration.

## What is intentionally postponed

Not part of the V0.1 product deployment contract:

- permanent deletion of originals;
- automatic quarantine expiry;
- online multi-writer sync;
- live incremental laptop-to-OVH replication;
- erasure coding;
- FUSE/File Provider/Cloud Files integration;
- semantic/LLM search;
- background daemon/service.

These are later layers. The first pilot exists to prove the smallest trustworthy user loop before adding them.
