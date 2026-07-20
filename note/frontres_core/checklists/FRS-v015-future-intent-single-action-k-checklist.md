# FRS-v015 Future-Intent / Single-Action K Acceptance Checklist

Status: current, volatile acceptance surface. R0--R5 are complete at
`E-FI-18`--`E-FI-23`; R6-S0 telemetry/preflight is complete at `E-FI-24`.
The S4 transaction has not run because the security boundary requires explicit
informed authorization before local owner files are transferred to SUST_Main_2.
Updated: 2026-07-21.

Plan: `../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md`

This checklist uses repository evidence tiers (`S0` static, `S1` deterministic
module semantic, `S2` offline connection, `S3` persistence, `S4` live
sentinel). The plan's migration gates are `G0`--`G5`; they are not a second
meaning of the `S` column.

`E-FI-0` is confirmed method semantics. `E-FI-1` is a read-only owner audit.
`E-FI-2` and `E-FI-3` are bounded deterministic implementation evidence only.
`E-FI-4` is a bounded H0 read-only source audit; it is not HSL migration or
runtime evidence.
`E-FI-5` records the user-confirmed H0-A semantic closure and v007 contract
activation; it is not source implementation evidence.
Historical v013/v002 evidence is not acceptable evidence for a v015 item
unless a new test explicitly rebases it.

## G0 / Step 0: Documentation and Owner Audit

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Active semantic owner chain | contract registry / active contracts | S0 T-doc/T-version | completed | `E-FI-0`; registry v015/v006/v003/v003/v003 |
| Human method map | Concept Figure / Method-to-Code | S0 T-map/T-contract | completed | `M-02`, `SR-01`, `M-11`, `M-06`, `Q-PAIR`, `Q-01` mappings |
| White-box owner baseline | plan / evidence ledger | S0 T-source/T-owner | completed | `E-FI-1` |
| Replanned step/checklist/canvas | plan/checklist/task canvas | S0 T-plan/T-matrix | completed | this plan, this checklist, task canvas |

## G1 / Step 1A: Immutable Local Scenario

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Local scenario schema | stage1 hooks / live sampler | S1 T-schema | completed | `E-FI-2`; `x_t`, artifact, q29 intent, Clean continuation, K, identity |
| q29 preservation invariant | materializer / command carrier | S1 T-invariant/T-differential | completed | `E-FI-2`; fixture proves `Pi_internal(Noisy) == Pi_internal(Clean)` |
| Immutable identity | sampler / scenario carrier | S1 T-hash/T-metamorphic | completed | `E-FI-2`; hash covers all five scenario elements; no M-trial mutation |
| Provenance split | materializer / command | S1 T-provenance | completed | `E-FI-2`; q29 is deployment/Noisy carrier; continuation is GMT-only Clean carrier |
| No ambiguous 65D tape | materializer | S1 T-legacy-reject | completed | `E-FI-2`; H and K cannot read the same full-tape field |

## G1 / Step 1B: Future-Intent Actor Bridge

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| q29 H layout | `frontres_runtime.py` / observation layout | S1 T-shape/T-offset | completed | `E-FI-3`; ordered `[B, |H|*29]` positive future offsets |
| No future root/global field | actor bridge | S1 T-differential/T-provenance | completed | `E-FI-3`; changing future root/global does not change actor tail |
| No Clean actor leak | actor bridge / normalizer | S1 T-clean-isolation/T-source | completed | `E-FI-3`; same numeric q29 retains deployment/Noisy provenance |
| Legacy and stats rejection | actor bridge / normalizer | S1 T-legacy-reject/T-layout | completed | `E-FI-3`; `[H,65]` or incompatible stats fail closed |
| Role-aligned formal carrier | command snapshot / `frontres_runtime.py` | S1 T-role-expand/T-shape/T-permute | completed | `E-FI-20`; command-owned `[8,3,29]` produces ordered `[8,58]`, ignores the poisoned B=4 policy batch, and preserves role permutation/identity |
| Exact FEMR/GMT visibility | config / runner / actor | S1 T-158-actor/T-770-GMT/T-zero-reject | completed | `E-FI-21`; config `100D` plus q29 tail `58D` resolves to FEMR `158D`, frozen GMT retains the final `770D`, and zero-prefix fallback rejects |
| Unmocked formal observation connection | command / observation / normalizer / actor | S2 T-connect/T-consumer | pending R5 | `E-FI-18`; `E-FI-16` replaced `_read_live_observations()` and did not cover this boundary |

## Gate H0: HSL Audit (Read Only)

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Warmup observation route | `frontres_warmup.py` / HSL target | S0 T-source/T-layout | completed | `E-FI-4`; raw obs -> normalizer -> direct residual actor bypasses q29 bridge |
| HSL target provenance | target builder / MDP observation owner | S0 T-source/T-target | completed | `E-FI-4`; current anti-DR oracle label is distinct from enabled quartet/Clean rollout label |
| Human decision before migration | user + active training contract | S0 T-decision | completed | `E-FI-5`; H0-A permits proposal-only current-frame HSL and forbids Clean rollout labels |

## HSL Migration Step H1: Proposal-Only Future-Intent Initialization

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Stage-1 q29 actor route | `frontres_warmup.py` / v015 bridge | S1 T-HSL-layout/T-HSL-provenance | completed | `E-FI-6`; sealed q29 tail before normalizer and residual actor |
| Current-frame target only | warmup / `observations.py` | S1 T-HSL-target/T-source | completed | `E-FI-6`; anti-DR Delta SE(3) only, no Clean future or rollout target |
| Stage-3 label isolation | `frontres_hsl_rollout_target.py` / storage/loss | S1 T-HSL-stage3-reject | completed | `E-FI-6`; no target/weight/harm write or loss consumer |
| Direct Stage-3 anti-DR write isolation | `frontres_rollout_step.py` | S1 T-HSL-direct-write-reject | completed | `E-FI-6`; v015 rejects a nonzero online HSL writer before transition storage |
| v015 Stage-3 loss isolation | `frontres_unified.py` / G1 config | S1 T-HSL-loss-reject | completed | `E-FI-6`; nonzero supervised loss or floor is rejected and config is zero |
| Legacy checkpoint rejection | checkpoint loader / normalizer | S1 T-HSL-legacy-checkpoint-reject | completed | `E-FI-6`; old warmup payload is rejected before restoration |
| Offline connector | fake Stage-1 + fake Stage-3 owners | S2 T-HSL-connect | completed | `E-FI-7`; q29 -> normalizer -> actor -> current target; v015 zero-write/storage/batch/loss proof |

## G2 / Step 2A: Two-Role Reset Layout

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Two scored roles only | training setup / command | S1 T-role/T-count | completed | `E-FI-8`; v015 layout is Repair + Noisy only, candidate/Clean=0 |
| Shared reset scenario | reset hooks / command | S1 T-state/T-identity | completed | `E-FI-8`; fake retry shares `x_t`, artifact, q29 intent, C, K, hash without perturbation sampling |
| Quartet isolation | active config/layout | S1 T-legacy-reject | completed | `E-FI-8`; projected/candidate/search/Clean names and fixed-tape/reference-window mixing fail closed |

## G2 / Step 2B: One-Action Frozen-FEMR K Collector

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| One policy tuple per attempt | rollout/probe/storage | S1 T-row/T-action-count | completed | `E-FI-9`; exactly one Repair action/log-prob/value/mean/sigma tuple regardless of K; no return/advantage yet |
| FEMR freeze after t | rollout step | S1 T-frozen/T-metamorphic | completed | `E-FI-9`; guarded candidate route rejects a second actor sample and later command repair writes are zero |
| Full Clean GMT continuation | command route | S1 T-continuation/T-cursor | completed | `E-FI-9`; explicit command C cursor serves q29/dq29/root after t, never H intent |
| Offline lifecycle connection | reset -> actor -> GMT -> storage | S2 T-connect/T-no-mixed-reference | completed | `E-FI-9`; fake Clean reset -> t actor -> frozen GMT C capture -> immutable one-tuple carrier; formal route remains blocked |

## G3 / Step 3A: Intent Gain Core

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| q29 intent fidelity | `frontres_gain.py` | S1 T-value/T-sign | completed | `E-FI-10`; both branches compare to typed deployment/Noisy I, with the expected signed q29 error reduction |
| Root exclusion | `frontres_gain.py` | S1 T-root-exclusion/T-provenance | completed | `E-FI-10`; v003 input has no Clean/root/global fields and rejects non-deployment q29 provenance |
| No-op protection | `frontres_gain.py` | S1 T-noop/T-invariant | completed | `E-FI-10`; equal Noisy/Repair execution gives zero intent gain, while changing fixed I changes the result |
| Physics/cost continuity | Gain owner | S1 T-pair/T-full6/T-unconfirmed | completed | `E-FI-10`; K-normalized paired Physics and all-6D cost are explicit; absent qvel/qacc/one-action temporal values remain `NaN` |

## G3 / Step 3B: Gain Return and Priority Consumers

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| q29 capture provenance | live probe / storage | S1 T-provenance/T-consumer | completed | `E-FI-11`; post-`t` robot q29 and deployment/Noisy `I[t]` reach the sole v003 Gain owner; H/C/K retain separate roles |
| Return migration | probe / storage | S1 T-consumer/T-no-v002-fallback | completed | `E-FI-11`; candidate-only one-row `return=gain_total` and `advantage=return-old_value` reject the v002 Clean-global owner |
| Priority migration | sampler | S1 T-priority-isolation/T-no-v002-fallback | completed | `E-FI-11`; immutable scenario-keyed evidence copies raw v003 decomposition but cannot mutate sampler state or actor-loss mass |
| Single active consumer chain | Gain + all above | S1 T-single-owner | completed | `E-FI-11`; return and priority preserve the same v003 decomposition, provenance, hash, and invalid-row mask |

## G3 / Step 3C: Diagnostics and Evaluation Isolation

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Local diagnostic/evaluator v003 | `frontres_segment_diagnostics.py` + legacy evaluator gate | S1 T-diagnostic/T-evaluator/T-no-v002-fallback/T-no-zero-fill | completed | `E-FI-12`; sealed carrier projects q29 intent/provenance, physics, cost, total, identity, and K; periodic/offline/sequence v002 evaluators reject v015 before capture |
| Composition isolation | `frontres_segment_diagnostics.py` protocol owner | S1 T-composition-isolation | completed | `E-FI-12`; separate deployment-composition protocol has explicit false return/PPO/priority feedback and rejects injected local evidence |

## G4 / Step 4A: Metadata and Grouped Candidate Adapter

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| One-row v015 schema | storage / PPO batch | S1 T-schema/T-row | completed | `E-FI-13`; one ordinary-valid Repair attempt row retains sealed scenario/hash/x_t/q29/K/evidence-step identity |
| Grouped mass reproof | `frontres_segment_ppo.py` | S1 T-value/T-permute/T-scale | completed | `E-FI-13`; equal motion -> Segment -> attempt mass survives permutation and K/evidence metadata changes |
| Candidate metadata preservation | storage adapter | S1 T-metadata/T-legacy-reject | completed | `E-FI-13`; only the v015 candidate adapter retains metadata, while `to_ppo_batch()` and old fixed-tape metadata reject |
| Mixed/partial transaction rejection | accumulator / candidate adapter | S1 T-fail-closed | completed | `E-FI-13`; mixed local identity, duplicate source/trial, or partial complete-transaction batch reject |

## G4 / Step 4B: Formal Route and One Update

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Formal grouped connection | transaction planner / runner / dedicated update loop / probe / PPO | S2 T-connect/T-order/T-q29-route | completed | `E-FI-14`; injected CPU fake request seals `2 Segment x 2 attempts` into the unchanged grouped loss while generic train/live loop remains isolated |
| Exact one update | sealed-plan validator / optimizer boundary | S2 T-exact-one-update/T-partial-reject | completed | `E-FI-14`; zero during provider/collection, exactly one explicit optimizer counter increment only after all rows validate |
| Legacy and HSL isolation | config / runner / probe | S2 T-no-legacy-route/T-warmup-isolation/T-fail-closed | completed | `E-FI-14`; legacy `to_ppo_batch()`, non-grouped normalizer, HSL/warmup, mixed metadata, and 65D actor tail fail before step |

## G4 / Step 4C: Persistence

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Future-intent layout identity | checkpoint/runtime normalizer | S3 T-layout/T-checkpoint | completed | `E-FI-15`; CPU fake checkpoint declares exact q29 H offsets/layout, grouped-loss identity, and prefix-stat fingerprint |
| Resume compatibility | checkpoint/runner | S3 T-resume/T-legacy-reject | completed | `E-FI-15`; old/unversioned `[H,65]`, mismatched H offsets, and tampered prefix stats reject before mutable restore |
| Transaction atomicity on resume | checkpoint/transaction owner | S3 T-atomicity | completed | `E-FI-15`; collecting/sealed/failed work cannot persist or resume, while a committed exact-one receipt resumes only as idle history |

## G5 / Step 5A: Local Live Identity Sentinel

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Pre-live v015 sentinel config and entrypoint isolation | config / runner boundary / `train.py` | S2 T-config/T-entrypoint/T-legacy-isolation | completed | `E-FI-16`; explicit H offsets and v015-only dispatch reject legacy modes before any live command |
| Pre-live local scenario to transaction connector | live sampler / reset request / probe / grouped adapter | S2 T-state/T-order/T-mass | completed, observation excluded | `E-FI-16`; fake proves sealed grouped exact-one route, but stubs `_read_live_observations()` |
| Formal command/observation connector | command / observation / runtime / actor | S2 T-command-connect/T-history-layout/T-role-tail/T-consumer | completed offline | `E-FI-23`; actual `_read_live_observations`, semantic `58/290/870 + 58 -> 928 -> 158/770`, role-aligned deployment q29 |
| Bounded local runtime trace | formal Stage-3 route | S4 T-live/T-state/T-provenance/T-frozen | failed before actor; preserved | `v015_step5a_s1.log`; reset reaches 4 Repair + 4 Noisy, then command construction raises before action/update |
| Runtime grouped update trace | PPO/diagnostics | S4 T-live/T-order/T-mass | pending R6 authorization | R5 proves offline grouped update delta one; R6 may run one bounded live transaction |

## R0--R6 Formal Observation Remediation

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| R0 observation contract freeze | plan / checklist / evidence | S0 T-source/T-log/T-plan | completed | `E-FI-18`; `870 + 58 = 928`, FEMR `158`, GMT `770`, current-command provenance frozen |
| R1 current GMT command | `commands.py::MultiMotionCommand` | S1 T-current-command/T-shape/T-provenance/T-current-only/T-continuation-isolation | completed | `E-FI-19`; role-aligned deployment q29/dq29 at t produces `[B,58]`, while future horizon/q-only/mixed routes reject and Clean C remains K-only |
| R2 role-aligned q29 H tail | command snapshot / `frontres_runtime.py` | S1 T-role-expand/T-offset/T-permute/T-no-root/T-no-Clean | completed | `E-FI-20`; read-only command snapshot exposes only intent/identity/provenance and routes offsets `(1,2)` as `[B,58]` |
| R3 FEMR/GMT authority split | config / runner / actor | S1 T-928-layout/T-158-actor/T-770-GMT/T-zero-reject | completed | `E-FI-21`; full `[B,928]`, FEMR prefix `[B,158]`, frozen GMT suffix `[B,770]`, fail-closed zero prefix |
| R4 persistence revalidation | checkpoint / normalizer | S3 T-layout/T-prefix-stats/T-legacy-reject/T-atomicity | completed | `E-FI-22`; v2 binds `(1,2)`, `928/158/770`, full 158D prefix fingerprint, committed receipt, and rejects v1/full/zero/65D/unversioned/partial identities before mutation |
| R5 unmocked formal connection | command / observation / runtime / actor / transaction | S2 T-connect/T-history-layout/T-consumer/T-one-action/T-exact-one-update | completed | `E-FI-23`; real `_read_live_observations`, semantic 58/290/870 + 58 -> 928 -> 158/770, post-advance C K, 2 Segment x 2 attempts, update delta one |
| R6 bounded live sentinel | formal Stage-3 route | S4 T-live/T-identity/T-order/T-mass | partial: S0 ready, remote transfer blocked | `E-FI-24`; structured snapshot/fail-closed telemetry and remote assets pass, but no source transfer or transaction occurred |

## G5 / Step 5B: Deployment Composition Evaluation

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| Separate sequence protocol | sequence evaluator | S4 T-composition/T-protocol | user-gated | persistent-artifact deployment reference explicitly named |
| No feedback into local training | evaluator / storage | S4 T-isolation | blocked by G3 | no local return/PPO/priority mutation |

## Active Isolation Rules

- Clean `x_t` restores dynamics only and never becomes actor reference.
- Future actor context is deployment/Noisy q29 intent, never raw future root,
  global pose, Clean provenance, noise label, or perturbation timing.
- The Clean continuation is GMT-only and does not share the actor H carrier.
- The local experiment has Noisy and Repair scored roles only.
- K measures one first action; it never adds later noise, later FEMR action, or
  extra PPO rows.
- Intent fidelity compares both executions to I; no direct Repair-vs-Noisy
  similarity or Clean-global Style fallback is active.
- K, M, best-of-M, priority, and evidence-step count never multiply actor-loss
  mass.
- HSL is limited to the H1-S1a Stage-1 q29/current anti-DR initializer; all
  v015 Stage-3 HSL label/direct-write/loss/checkpoint entries reject. H1-S2
  connectivity remains user-gated.
- No formal route, persistence migration, or live run begins before its listed
  preceding evidence tier passes.
- The pre-action GMT command is current deployment-carrier q29/dq29 at `t`;
  Clean C remains inaccessible until the explicit post-action K executor opens.
