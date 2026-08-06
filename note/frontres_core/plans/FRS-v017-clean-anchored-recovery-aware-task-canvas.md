# FRS-v017 / TRAIN-v014 Task Canvas

## Objective

Keep the TRAIN-v013 nested K-DR curriculum while replacing the retired bounded
task-action interface with one direct full-6D HSL-to-Stage3 coordinate and
strict HSL-v2/checkpoint-v9 identity. METHOD-v017, GAIN-v007, PPO-v005,
EVAL-v004, actor/GMT authority and the Concept Figure remain unchanged.

## Semantic Authority

```text
METHOD-v017 -> GAIN-v007 -> PPO-v005 -> TRAIN-v014 -> EVAL-v004
```

Engineering discipline: `FRS-ENG-v001`.

## Current Cursor

```text
Design Inspector human review       complete
TRAIN-v013 activation               complete (E-FI-108)
Engineering Plan rebase/review      complete; READY (E-FI-108)
Affected Module Test Cards          updated and passed (E-FI-109)
Step 1 / 2 offline migration        complete (E-FI-109)
Direct full-6D action closure       completed offline (E-FI-114)
Formal Runtime Audit Phase A        DP-01 through DP-10 reviewed offline
Formal Runtime Audit Phase B        runtime-complete; B01-B08 emitted once (E-FI-128)
Step 2 / 2 bounded live sentinel    complete; checkpoint-v9 iteration 1 saved
Policy Quality support-frame Q0     completed offline (E-FI-129)
Policy Quality evaluator            ready offline (E-FI-130)
Policy Quality efficacy             unconfirmed; real evaluation pending
```

## Current Contract Flow

```text
explicit per-K DRStageSpec
-> use configured frozen-GMT boundary (currently measured as 2.381)
-> resolve K/M/phase and current d_cap
-> sample one of Easy/Medium/Hard/Broken at 20/30/40/10
-> seal two Segment scenarios and exact M attempts
-> grouped exact-one update
-> committed-only DR/K progress
-> direct finite [B,6] HSL/Stage3 action
-> strict checkpoint-v9
```

## Owner Map

- identity: `frontres_interfaces.py`;
- K/M/DR phase: existing `frontres_segment_warmup.py`;
- composition: `scripts/rsl_rl/train.py::main()` and existing launcher;
- transaction seal: existing formal transaction and Aggregate owners;
- materialization: existing sampler/perturbation owner;
- update: existing grouped PPO owner;
- persistence: existing `frontres_checkpointing.py` Gateway;
- diagnostics: existing read-only diagnostics/telemetry owners.

No new runner, evaluator, service, wrapper, registry, second Critic or online
controller is admitted.

## Verified Evidence

- E-FI-105: previous unchanged module surface passed 18/18 under v012;
- E-FI-106: legacy `single_update/update_loop` selection fails before runner
  construction;
- E-FI-107: old pre-training `g_K` assumption was identified as the wrong
  unresolved object;
- E-FI-108: human-confirmed nested K-DR semantics activated as TRAIN-v013 and
  the pre-code plan passed FRS-ENG-v001 review.

- E-FI-109: TRAIN-v013/checkpoint-v8 implementation, 18-card/49-contract
  regression, Phase A and final code review complete.
- E-FI-112: Segment Replay human wording, current scalar transaction audit
  projection and v017/v007/v005/v013 Architecture/Registry identity agree.
- E-FI-113: K-step Curriculum is human-confirmed; the sealed formal transaction
  consumes exact active K/M without consulting the retired state-driven budget,
  while one-action-K and checkpoint identities remain unchanged.
- E-FI-114: HSL/Stage3 now use one direct finite `[B,6]` coordinate; legacy
  12D slicing, action tanh/scale and HSL-v1/checkpoint-v8 identities reject.
- E-FI-115 Phase A: DP05 found that recursive policy `train()` can reopen GMT
  training mode and checkpoint-v9 does not bind the configured GMT SHA256.
- E-FI-116: DP07 local report and final telemetry carry the complete
  owner-produced v007 decomposition, scale/beta identity and semantic ZMP N/A
  without recomputation, row drift, zero-fill or training feedback.
- E-FI-117: DP08 removes the retired Stage-1 `dz` target clamp; the real target
  producer and independent validator preserve positive and negative anti-DR
  translation while HSL remains 158D actor-only and Stage-3-isolated.
- E-FI-118: DP09 uses the formal `critic_only -> actor_ramp -> joint` identity;
K16/M3 and K32/M4 reuse one Critic, while critic-only preserves seeded
Actor/std parameters and Adam state and checkpoint-v9 rejects the old label.
- E-FI-119: DP10 fixes the only legal future-intent offsets to `(1,2)` in the
layout/config owner, rejects `(1,3)` before mutation, and proves the unmocked
offline `870D + 58D -> 928D -> FrontRES 158D / GMT 770D` route with one actor
action. TEST-04/05/10/16 and the 49/49 deterministic aggregate pass.
- E-FI-120 Phase B control surface: `note/architecture/06_frontres_runtime_audit_atlas.html`
  projects AUDIT-B01 through AUDIT-B08 in official Stage3 order.
- E-FI-121 installs those eight read-only, fail-closed checks in the existing
  formal owners and passes the focused offline instrumentation regressions.
  The HSL-v2 artifact is runtime-confirmed; the single live transaction remains
  unexecuted and separately authorized.
- Before E-FI-128, E-FI-122 records the first live attempt: B01 passed, then the command correctly
  rejected execution-mode selection before sealed-scenario installation. The
  existing reset seam now owns `install -> mode -> refresh` and focused offline
  regressions passed; B02-B08 were still pending at that point.
- E-FI-123 records that the rerun crossed reset, observation, one-action and
  frozen-GMT execution, then exposed `[K,4]` versus `[K,8]` Repair evidence.
  The one-action-K owner now applies `repair_rows` exactly once to every
  trajectory field; focused offline regressions passed before the next rerun.
- E-FI-124 records that the next rerun crossed those repairs and reached grouped
  candidate conversion, where the formal request owner lacked its explicit PPO
  batch-type import. The dependency is now direct and contract-checked; the
  active 49-contract suite passed before the next rerun.
- E-FI-125 records that the following run reached the final observation trace
  check with correct live tensors but missing v017 command/Critic trace fields.
  The Repair collector now publishes measured 58D/289D identities with strict
  role-row validation; the active 49-contract suite passed before the next
  rerun.
- E-FI-126 records that the next run live-confirmed B02/B05 and stopped at an
  audit-only float32-vs-Python return mean mismatch. B06 now reproduces the
  formal float32 reduction, rejects a corrupted mean, and passed the active
  49-contract suite before the final rerun.
- E-FI-127 records that B03/B04 were still called only by the legacy evaluator
  collector. The active v017 Repair collector now sends its existing sealed
  snapshot, observation trace, policy action and eight frozen-GMT step facts to
  the shared audit owner; focused and 49/49 aggregate contracts passed before
  E-FI-128 supplied the required live evidence.
- E-FI-128 records the final official K8/M2 transaction. B01-B08 each emit
  exactly once; four policy rows receive one grouped optimizer update;
  critic-only changes the Critic but not Actor/std; committed checkpoint-v9 is
  saved with matching GMT SHA, 928/158/770 layout and transaction receipt.
- The same log is not policy-efficacy evidence. It exposes the first Q-mechanism
  contradiction: attempts on the second environment row of each Segment show
  `48.675 m` and `64.481 m` support-foot drift because world-frame Repair feet
  are compared with the once-per-Segment Clean baseline from another env row.
- E-FI-129 closes that first-invalid owner offline: execution capture subtracts
  each selected `scene.env_origins` before sealing `foot_pos`; cross-origin,
  row-permutation, malformed-origin and unchanged-v007-ordering contracts pass.
E-FI-109 proves offline code and persistence semantics. It does not prove
simulator/live class populations, policy quality or deployment behavior.

## Active Files

- active contract: `contracts/active/training/FRS-TRAIN-v014-direct-full6-action-curriculum.md`;
- engineering plan: `FRS-v017-clean-anchored-recovery-aware-engineering-plan.md`;
- checklist: `../checklists/FRS-v017-clean-anchored-recovery-aware-checklist.md`;
- Design Inspector: `../../architecture/runtime/04_frontres_design_inspector.data.json`;
- Test Atlas: `../../architecture/testing/05_frontres_module_test_atlas.data.json`.

## Open Risks

- E-FI-128's extreme Physics rows remain invalid historical evidence; E-FI-129
  repairs the coordinate owner, but actual policy quality has not been rerun;
- the old `frontres_dr_curriculum.py` episode-length/frontier controller remains
  code but must stay unreachable from the active route;
- real IsaacLab class/strength distribution and policy effect remain S4 facts.

## Stop Conditions

Stop on new method semantics, duplicate schedule owner, online outcome feedback,
hidden DR default, mixed/partial curriculum progress, Actor/std drift during
critic-only, Critic reset at K transition, v7 mutation, Clean actor leakage,
MOSAIC host change or an unresolvable P0/P1.

## Next Action

Formal Runtime Audit Phase B is complete at E-FI-128 and the support-foot Q0
coordinate blocker is closed offline at E-FI-129. E-FI-130 migrates the
retained evaluator to strict v017/checkpoint-v9 and fixes the K16/M3 manifest.
E-FI-131 closes the composition-root mistake that classified the read-only
evaluator as a legacy live runner. The next bounded action is to rerun that
evaluator on model_3500 and judge Intent,
Contact, phase-ZMP, survival, sustained lean and unplanned steps using the
corrected evidence. No training or long run is implied.
