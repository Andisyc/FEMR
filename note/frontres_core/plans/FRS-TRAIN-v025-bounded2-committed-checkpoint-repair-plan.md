# FRS-TRAIN-v025 Bounded-2 Committed Checkpoint Repair Plan

## Accepted Boundary

- Save and reload checkpoint-v20 after one exact committed FRS-TRAIN-v025
  relational Actor update.
- Preserve checkpoint-v19 scalar receipt validation and idle checkpoint-v20
  save/reload behavior unchanged.
- Reject mixed, legacy, partial, or in-flight receipts before filesystem commit.
- Do not alter Gain, relational comparison, Actor credit, optimizer, Replay,
  simulator, launcher, or checkpoint schema.
- Do not run IsaacLab, training, server operations, Git actions, or destructive
  operations.

## Root Cause And Pinch Point

The bounded-2 run completed one relational transaction and its final telemetry,
then checkpoint-v20 save called `_v015_transaction_checkpoint_payload` without
the v025 identity. Its receipt validator therefore used the legacy v024 scalar
default and rejected the valid v025 receipt. The checkpoint transaction payload
is the persistence Pinch Point shared by committed save and resume validation.

## Effect Sketch

`save_runner` selects checkpoint-v20 -> `_build_v025_relational_checkpoint_identity`
-> `_v015_transaction_checkpoint_payload` ->
`_v015_committed_transaction_receipt` -> atomic checkpoint write -> strict
checkpoint-v20 reload. The scalar checkpoint-v19 builder reaches the same
Pinch Point with a different exact identity.

## Repair And Proof

1. Extend the existing checkpoint-v20 test with a committed relational receipt;
   observe the real `legacy contract identity` RED before production edits.
2. Make each checkpoint identity builder pass its exact expected receipt
   identity into the existing validator. Keep the validator fail-closed and do
   not infer, translate, or rewrite receipt fields.
3. Prove committed checkpoint-v20 save/reload, idle checkpoint-v20, scalar
   checkpoint-v19 characterization, mixed-identity rejection, aggregate
   contracts, official offline preflight, compilation, diff check, and final
   review.

## Stop Condition

Stop if the repair requires a schema/version change, receipt conversion,
training semantics, MOSAIC host changes, or live execution to establish an
offline persistence fact.
