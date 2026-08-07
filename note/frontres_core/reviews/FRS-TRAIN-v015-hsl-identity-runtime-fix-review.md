# FRS-TRAIN-v015 HSL Identity Runtime Fix Review

Date: 2026-08-07
Review mode: `final_gate_review`
Verdict: `APPROVE`
Repository discipline: `active`

## Accepted Boundary

The frozen HSL-v2 proposal artifact retains its original
`FRS-TRAIN-v014` identity. A fresh Stage-3 campaign owns
`FRS-TRAIN-v015` and checkpoint-v10. Stage-3 may restore only the accepted
HSL actor, distribution and prefix-normalizer state; it must reject an HSL
artifact carrying the Stage-3 training identity before mutation.

No MOSAIC host behavior, HSL weights, Stage-3 network, Gain/PPO semantics,
optimizer construction, schedule, action layout or GMT identity is changed.

## Runtime Root Cause

`FRS_TRAIN_V015_SPLIT_LR_SENTINEL.log` reached the split Adam construction
with Actor LR `3e-6` and Critic LR `1e-5`, then failed before rollout/update in
`_validate_v015_hsl_checkpoint_resume`. The inspected server artifact reported
the exact HSL-v2 format, method, observation/action layout and GMT identity,
but its training identity was `FRS-TRAIN-v014`. The newly activated Stage-3
code incorrectly required `FRS-TRAIN-v015` from that independent artifact.

## Responsibility And Dependency Delta

- `frontres_checkpoint_quality.py` remains the single owner of the serialized
  HSL identity and now exports its frozen training-contract projection.
- `frontres_checkpointing.py` consumes that projection for HSL save and strict
  pre-mutation restore. Stage-3 checkpoint-v10 identity remains unchanged.
- `frontres_policy_quality_eval.py` compares the HSL scaffold with the frozen
  HSL identity while comparing the tested policy with the active v015 manifest.
- No wrapper, fallback, migration path, second identity resolver or mutable
  owner was introduced. Caller knowledge is capped at the existing checkpoint
  identity inspection boundary.

The Characterization Test is the pulled sentinel plus the inspected HSL
identity. The Effect Sketch is HSL save -> restricted load/inspection ->
Stage-3 initializer/quality request. The Pinch Points are the existing strict
HSL validator and active request builder. Their test fixtures are the Enabling
Points; no production seam was added.

## Findings

The follow-up final review found and repaired three small in-scope issues:

1. P2: the public constant was named `V015_HSL...` although it identifies the
   frozen v014 artifact, and it was missing from `__all__`. It is now the
   generation-neutral `FRONTRES_HSL_ARTIFACT_TRAINING_CONTRACT_ID`.
2. P2: the active request test fake treated every unknown route as `policy`, a
   silent fallback that weakened its oracle. It now accepts only exact `hsl`
   and `policy` paths and rejects every other route.
3. P3: the TRAIN-v015 warmup evidence comment had drifted one indentation
   level. Its block ownership is now visually correct.

No open P0, P1, P2 or P3 findings remain. The patch restores the active contract boundary,
keeps failure closed before mutation and does not broaden checkpoint acceptance.
Security, external IO, concurrency and resource-bound dimensions are
not applicable because the change is deterministic identity validation only.

FRS-ENG-v001 is the active discipline. The change keeps checkpoint identity in
one semantic owner, preserves dependency direction through the existing
Gateway, adds no cross-layer private access or fallback, and retains atomic
pre-mutation rejection. The obsolete quality fixtures are explicitly
historical and excluded rather than revived as a second evaluator owner.

Removal review did not authorize deleting the historical v015 quality scripts:
they still reference each other and characterize a retired evaluator surface.
They remain excluded from the active aggregate until that whole legacy surface
is retired, rather than being partially rewritten in this v017 fix.

## Annotation And Atlas Delta

- the HSL quality inspector, HSL payload builder and pre-mutation validator now
  expose reviewed Chinese B1/B2/B3 blocks;
- the v017 request builder now includes its B3 immutable-request handoff;
- the active identity regression has explicit fixture/accept/reject blocks and
  a fail-closed fake route;
- Code Quality Evidence Atlas was regenerated from current source. Raw scan
  facts remain separate from this review verdict.

## Evidence Consumed

- `frontres_hsl_v007_s1_contract.py`: accepted v014 save/inspect/Stage-3 init;
  v015 HSL identity and other incompatible payloads reject before mutation.
- `frontres_v017_policy_quality_eval_contract.py`: v014 HSL plus v015 policy is
  accepted; v015 HSL is rejected by the active evaluator.
- `frontres_segment_all_contract_suite.py`: 50/50 contracts passed.
- `python -m py_compile`, `git diff --check`, launcher syntax, Module Test,
  Repository Reading, Code Quality Evidence and Design Inspector checks passed.

The local evidence closes owner, consumer, negative and persistence behavior.
The server sentinel has not been rerun with this patch, so live transaction and
parameter-delta evidence remain unconfirmed. Long training remains blocked
until the same bounded sentinel passes beyond HSL initialization.
