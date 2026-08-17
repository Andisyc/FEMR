# FRS-TRAIN-v025 Bounded-1 Startup Repair Plan

## Accepted Boundary

- Admit fresh HSL-v2 initialization for the active relational Stage-3 identity:
  `segment_replay_relational + pairwise_edge + Actor-only`.
- Preserve the scalar compatibility identity:
  `segment_replay_hrl + grouped_scale_only + non-relational owner`.
- Restore only the residual Actor, policy distribution state, and exact 158D
  prefix normalizer. Do not mutate Critic, optimizer, Replay, sampler, GMT, or
  transaction state.
- Do not run IsaacLab, server operations, training, Git actions, or destructive
  operations.

## Root Cause And Pinch Point

The official bounded run constructed the correct relational algorithm, then
failed in `_validate_v015_stage3_hsl_initializer_runtime` because that owner
still admitted only the scalar objective and reduction. The checkpointing
validator is the Pinch Point: it can reject an invalid identity before any HSL
state restoration.

## Repair And Proof

1. Add a RED relational HSL-v2 cold-start case plus mixed-identity negative
   cases to the existing HSL checkpoint contract.
2. Replace the scalar-only predicate with the two closed objective/reduction/
   ownership combinations; keep all validation before restore.
3. Include the HSL-v2 contract in the aggregate preflight suite.
4. Isolate bounded and recursive-preflight variables in the contract child
   process while retaining them for the parent training command.
5. Run HSL, checkpoint-v20, relation transaction, official entrypoint,
   launcher, formal offline audit, aggregate suite, combined official preflight,
   syntax, JSON, and diff checks. Stop before simulator execution.

## Stop Condition

Stop if the repair requires new method semantics, checkpoint schema changes,
MOSAIC host changes, or any server/live execution to resolve an offline fact.
