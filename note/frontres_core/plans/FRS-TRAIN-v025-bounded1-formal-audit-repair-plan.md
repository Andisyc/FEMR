# FRS-TRAIN-v025 Bounded-1 Formal Audit Repair Plan

## Accepted Boundary

- Admit the active relational Stage-3 identity at the existing one-action-K
  formal audit boundary.
- Preserve the scalar compatibility audit unchanged.
- Keep the audit read-only: it may validate and print facts, but may not alter
  Actor, Critic, optimizer, Replay, transaction, simulator, or checkpoint state.
- Do not run IsaacLab, training, server operations, Git actions, or destructive
  operations.

## Root Cause And Pinch Point

The second bounded run reached the real one-action-K collector and frozen GMT,
then `_print_one_action_k_audit_facts` unconditionally required the legacy
scalar Critic identity. The algorithm correctly publishes
`inert-legacy-compat + target none` for relational Actor-only training. The
formal audit helper is the Pinch Point because both active and compatibility
collectors pass through it before any optimizer update or transaction commit.

## Repair And Proof

1. Add a RED characterization that sends the complete relational identity
   through the actual one-action-K audit helper and rejects a mixed identity.
2. Make the helper select one of two closed identity sets: relational
   Actor-only or legacy scalar Actor/Critic. Do not introduce fallback values.
3. Exhaustively retain the observation, role, action, horizon, frozen-GMT, and
   finite-value assertions shared by both routes.
4. Run the focused formal-audit contract, relational transaction and
   persistence contracts, official entrypoint/launcher checks, aggregate suite,
   combined official offline preflight, syntax, JSON, and diff checks.
5. Save the attempt history and final review. Stop before simulator execution.

## Stop Condition

Stop if the repair requires changing relational method semantics, Gain/PPO,
checkpoint schema, training state, MOSAIC host behavior, or any live execution
to establish an offline fact.
