# Personal Vault V0 rollback contract

This experiment stays on one branch: `personal-vault-v0-runtime-v2`.

## Immutable baseline

The exact pre-experiment Runtime V2 state is:

`990d07b773197f747681754c171677934d4ee586`

This is the rollback target and must never be redefined by later Personal Vault work.

## Rule

Every Personal Vault change is additive and committed separately. Do not rewrite Runtime V2 history and do not modify `rlbwt-binary-safe-v2`.

If the experiment must be abandoned, restore the *tree state* of the experiment branch to the baseline while preserving history. Preferred rollback is a normal revert of Personal Vault commits, not a force-reset.

After rollback, this must be true:

`git diff --exit-code 990d07b773197f747681754c171677934d4ee586..HEAD -- . ':(exclude)experiments/personal_vault_v0/ROLLBACK.md'`

If an exact byte-for-byte repository tree rollback is required, remove this contract file too and require:

`git diff --exit-code 990d07b773197f747681754c171677934d4ee586..HEAD`

## Current first experiment checkpoint

The first closed-loop harness head before debugging is:

`47f4cd63bf3e8144a3d88fec5648c38eeb8e7214`

This SHA is an audit checkpoint, not a new baseline.

## Safety boundary

No Personal Vault rollback may mutate:
- `main`
- `rlbwt-binary-safe-v2`
- OVH
- demo deployment
- `yasha-context`

Rollback is never automatic. It is performed only after explicit user instruction.
