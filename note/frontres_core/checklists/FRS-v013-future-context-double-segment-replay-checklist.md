# FRS-v013 Acceptance Checklist, superseded fixed-65D-tape route

Status: superseded by FRS-v015-future-intent-single-action-k-checklist.md; retained as v013 S1/S2 evidence history only
Updated: 2026-07-19
Plan: ../plans/FRS-v013-future-context-double-segment-replay-engineering-plan.md

The active acceptance surface is FRS-v015-future-intent-single-action-k-checklist.md.
The older policy-quality checklist remains evidence history and is not an
authorization to run the new route.

## Documentation Closure

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Active semantic owner chain | contracts registry/history | S0 T-doc/T-version | completed | FRS-METHOD-v014, FRS-TRAIN-v005, FRS-PPO-v003 |
| Human method map | Concept Figure plus Method-to-Code atlas | S0 T-map/T-contract | partial: method figure and JSON pass; viewer-wide checker stops at unrelated QUALITY-ID-01 | 2026-07-19 check_method_figure.mjs; M-11, SR-01, runtime/02_frontres_flow.data.json |
| Bounded implementation plan | plan and this checklist | S0 T-plan/T-matrix | completed | FRS-v013 engineering plan |

## Step 1: Fixed Noisy Future Context

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Current joint-reference provenance | MultiMotionCommand.command plus MotionPerturber | S1 T-provenance/T-differential | completed: legacy mismatch isolated; 65D command tape replaces the invalid legacy future route | evidence_ledger_v013_future_context_2026-07-19.md E-FC-1 through E-FC-10 |
| Ordered H layout and actor dimensions | command/runtime actor bridge | S1 T-layout/T-shape; S2 T-legacy-reject | completed offline: required H, 65D carrier, actor augmentation, and legacy-layout rejection; checkpoint migration remains S3 | evidence_ledger_v013_future_context_2026-07-19.md E-FC-8 through E-FC-10 |
| Fixed Noisy segment provenance | command tape plus live probe | S1 T-hash/T-provenance/T-clean-isolation; S2 T-connect | completed offline: one source hash reaches M rows, reset, current/H actor reference, and K command execution | evidence_ledger_v013_future_context_2026-07-19.md E-FC-8 through E-FC-10 |
| Selection-time Noisy lifecycle across M attempts | sampler-domain scenario binder | S1 T-lifecycle/T-immutability/T-hash/T-metamorphic/T-coverage | completed: deterministic local lifecycle contract | evidence_ledger_v013_future_context_2026-07-19.md E-FC-6 |
| Formal current/future/K route | selection -> reset -> command -> actor -> GMT | S2 T-connect/T-no-mixed-reference/T-H-not-K | completed offline; full suite 59/59 | evidence_ledger_v013_future_context_2026-07-19.md E-FC-9/E-FC-10 |
| Legacy layout rejection | checkpoint and resume owner | S1/S3 T-schema/T-missing/T-resume | partial: runtime actor/normalizer rejects legacy layout; checkpoint/resume migration is not started | FRS-TRAIN-v005 HSL Interface Continuity |

## Step 2: Double Segment Transaction

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Declarative all-policy attempt plan | sampler plan | S1 T-role/T-order/T-ID/T-min-M | completed offline | evidence_ledger_v013_future_context_2026-07-19.md E-TX-1 through E-TX-3; pure layout only, not a frozen parameter snapshot |
| All M policy samples retain old-policy credit identity | sampler plan, live sampler, reset, and storage | S1 T-role/T-snapshot/T-meta/T-mutation | completed offline: S1a all-policy layout is bound to a real state-dict fingerprint; foreign IDs and parameter mutation fail closed | evidence_ledger_v013_future_context_2026-07-19.md E-TX-4/E-TX-5 |
| Frozen snapshot metadata reaches reset, storage, and candidate PPO batch | live sampler -> live probe -> storage -> PPO adapter | S1 T-snapshot/T-reset/T-storage/T-no-mixed-reference | completed offline: one sealed carrier reaches batch, Clean reset, storage, and row-aligned PPO adapter; mixed Noisy hash fails before reset | evidence_ledger_v013_future_context_2026-07-19.md E-TX-5/E-TX-6/E-PPO-4 |
| Fixed x_t and fixed Noisy scenario across attempts | reset/window/live sampler | S1 T-state/T-hash/T-metamorphic | not started | FRS-METHOD-v014 Method And Recovery Boundary |
| Multi-Segment accumulator then one step | live probe transaction owner | S2 T-transaction-complete/T-no-early-step/T-exact-one-update | completed offline | E-TX-7/E-TX-8: complete S1b carrier, zero collection steps, one real fixture optimizer step, repeated/double update rejection; legacy immediate-update route remains isolated |
| Search/manual credit isolation | transaction accumulator/storage boundary | S1/S2 T-role/T-no-credit | completed offline at transaction gate | E-TX-8: non-policy role and source-local mixed Noisy identity fail before finalization; PPO loss ownership remains Step 3 |

## Step 3: Grouped PPO, K-A Rebase Required

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Single policy row per attempt | storage and PPO candidate owner | S1 T-row/T-schema/T-value | pending: K-A rebase | Must prove one action/statistics/return/advantage tuple per eligible attempt independent of K |
| Motion/Segment/attempt formula | frontres_segment_ppo.py | S1 T-value/T-permute | pending: v002 valid-step-row fixture is superseded | Rebuild E-PPO-1 with one row per attempt and unequal motion/Segment/M/K evidence |
| Sign-preserving non-amplifying scale | frontres_segment_ppo.py | S1 T-sign/T-scale/T-metamorphic | pending reproof on K-A rows | Rebase E-PPO-2 after removing valid-step row reduction |
| No priority/Gain/focal/M/K/evidence-count multiplier | PPO owner plus static source test | S1 T-static/T-source/T-isolation | pending reproof on v003 | Rebase E-PPO-3; K affects return only, never policy-row mass |
| Storage-to-loss metadata route | storage -> grouped candidate -> grouped loss | S1 T-connect/T-missing | pending: candidate-only K-A rebase; legacy adapter remains isolated | Rebuild E-PPO-4 without a K-row interpretation |

## Step 4: Formal Route And Resume

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| S0 formal owner and legacy isolation | train entry, live loop, storage, legacy update | S0 T-config/T-route/T-retirement | completed read-only | Step 4-S0 audit; legacy bypass remains isolated |
| K-A storage-row semantic | live storage plus candidate loss owner | S1 T-row/T-K-evidence | code-confirmed for live storage; candidate rebase pending | one first-step tuple + K return evidence; FRS-PPO-v003 |
| v014-only Stage 3 config route | train entry and runner | S2 T-route/T-retirement | blocked by Step 3-KA | FRS-TRAIN-v005 Formal Transaction Route |
| Future-layout checkpoint identity | HSL/warmup/checkpoint owner | S3 T-schema/T-version/T-resume | blocked by S1/S2 | FRS-TRAIN-v005 HSL Interface Continuity |
| Required transaction diagnostics | runner/diagnostics owner | S2 T-diagnostic/T-unconfirmed | blocked by Step 3-KA | FRS-METHOD-v014 Required Diagnostics And Evidence |

## Step 5: Live Sentinel

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| One real fixed-reference transaction | formal Stage 3 owner | S4 T-live/T-state/T-provenance/T-order | not started; user review required before launch | Must show reset hash, noisy hash, one snapshot, M attempts, one optimizer step |
| Grouped loss mass in real runtime | PPO/diagnostics owner | S4 T-live/T-mass/T-sign | not started; user review required before launch | Must show motion/Segment/attempt/step mass shares |
| Frozen GMT and full-6D action identity | actor/GMT owner | S4 T-live/T-frozen/T-full6 | not started; user review required before launch | Existing E70 is prior-interface evidence only |

## Active Isolation Rules

- No Clean reference, perturbation truth, or noise timing reaches the actor.
- No Noisy physical prefix before x_t is introduced under this method version.
- No first-policy-only PPO eligibility and no optimizer call during M-attempt
  collection.
- Exactly one PPO policy row per eligible policy attempt; K execution evidence
  must never become a second row or loss-mass multiplier.
- No priority, raw Gain, focal advantage power, M, K, or evidence-step-count
  loss multiplier.
- `grouped_scale_only` rejects missing, mismatched, or partial transaction
  metadata; it never silently reduces a minibatch as a full transaction.
- No long training or policy-quality claim before the S4 sentinel is reviewed.
