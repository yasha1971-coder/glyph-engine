# STRUCTURAL_FINGERPRINT_REPLAY_V0_EXAMPLES

Status: measured
Date: 2026-07-01

## Purpose

Replay deterministic structural fingerprint artifacts and verify that they reproduce from source bytes.

## Results

| artifact | ok | errors |
|---|---:|---|
| `mini_structural_fingerprint_v0.json` | True | `` |
| `reymont_structural_fingerprint_v0.json` | True | `` |
| `webster_structural_fingerprint_v0.json` | True | `` |

## Decision

All replay checks passed: `True`

## Non-claims

- This does not predict best codec.
- This only verifies deterministic reproduction of the structural fingerprint artifact.

