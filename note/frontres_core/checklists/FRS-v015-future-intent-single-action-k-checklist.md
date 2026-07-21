# FRS-v015 Future-Intent / Single-Action K Acceptance Checklist

Status: current, volatile acceptance surface. R0--R6 and Step 5A are complete
at `E-FI-18`--`E-FI-27`, including the final bounded S4 identity/formal-update
sentinel. `E-FI-32` rebases the remaining path: no compatible trained v015
checkpoint or defined external Noisy `.npz` exists, the CLI is
implemented-not-runnable, and Step 5B-S4 is blocked behind G2--G6. `E-FI-33`
closes G1 as a stopped S0 audit. `E-FI-34` blocks G2-S0 on the Stage-1 q29
carrier decision and proposal-only critic isolation.
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
| Unmocked formal observation connection | command / observation / normalizer / actor | S2 T-connect/T-consumer | completed | `E-FI-23`; production `_read_live_observations()` connects `870+58=928` to FEMR `158D` and GMT `770D` without replacement |

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
| Offline lifecycle connection | reset -> actor -> GMT -> storage | S2 T-connect/T-no-mixed-reference | completed | `E-FI-9` local carrier, `E-FI-23` unmocked formal connection, and `E-FI-27` bounded S4 route; no mixed reference or later actor action |

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
| Bounded local runtime trace | formal Stage-3 route | S4 T-live/T-state/T-provenance/T-frozen | completed | `E-FI-27`; `4 Repair + 4 Noisy`, paired scenario/hash/x_t, deployment q29, one action, K=8, no later FEMR action |
| Runtime grouped update trace | PPO/diagnostics | S4 T-live/T-order/T-mass | completed | `E-FI-27`; actor `928D`, critic `289D`, four valid attempts, equal group mass, `optimizer_step_delta=1`, `exact_one_update=true` |

## R0--R6 Formal Observation Remediation

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| R0 observation contract freeze | plan / checklist / evidence | S0 T-source/T-log/T-plan | completed | `E-FI-18`; `870 + 58 = 928`, FEMR `158`, GMT `770`, current-command provenance frozen |
| R1 current GMT command | `commands.py::MultiMotionCommand` | S1 T-current-command/T-shape/T-provenance/T-current-only/T-continuation-isolation | completed | `E-FI-19`; role-aligned deployment q29/dq29 at t produces `[B,58]`, while future horizon/q-only/mixed routes reject and Clean C remains K-only |
| R2 role-aligned q29 H tail | command snapshot / `frontres_runtime.py` | S1 T-role-expand/T-offset/T-permute/T-no-root/T-no-Clean | completed | `E-FI-20`; read-only command snapshot exposes only intent/identity/provenance and routes offsets `(1,2)` as `[B,58]` |
| R3 FEMR/GMT authority split | config / runner / actor | S1 T-928-layout/T-158-actor/T-770-GMT/T-zero-reject | completed | `E-FI-21`; full `[B,928]`, FEMR prefix `[B,158]`, frozen GMT suffix `[B,770]`, fail-closed zero prefix |
| R4 persistence revalidation | checkpoint / normalizer | S3 T-layout/T-prefix-stats/T-legacy-reject/T-atomicity | completed | `E-FI-22`; v2 binds `(1,2)`, `928/158/770`, full 158D prefix fingerprint, committed receipt, and rejects v1/full/zero/65D/unversioned/partial identities before mutation |
| R5 unmocked formal connection | command / observation / runtime / actor / transaction | S2 T-connect/T-history-layout/T-consumer/T-one-action/T-exact-one-update | completed | `E-FI-23`; real `_read_live_observations`, semantic 58/290/870 + 58 -> 928 -> 158/770, post-advance C K, 2 Segment x 2 attempts, update delta one |
| R6-F1 command-clock isolation | `commands.py::MultiMotionCommand` | S1 T-t-clock-hold/T-K-clock-hold/T-legacy-clock/T-duplicate-refresh-reject | completed | `E-FI-25`; local current/C clocks hold across IsaacLab command compute, legacy rows retain ordered advance, duplicate direct refresh still rejects |
| R6-F2 critic-observation route | one-action evidence / candidate storage / formal evaluator | S1/S2/S4 T-critic-route/T-role-order/T-missing-reject/T-shape-reject/T-exact-one-update | completed and live-confirmed | `E-FI-26` deterministic route plus `E-FI-27` actor `928D` / critic `289D` bounded-live trace |
| R6 bounded live sentinel | formal Stage-3 route | S4 T-live/T-identity/T-order/T-mass | completed | `E-FI-27`; final hashed log closes observation, identity, K, grouped mass, and exact-one-update stop conditions |

## G5 / Step 5B: Deployment Composition Evaluation

| Item | Owner | Required S/T | Status | Evidence pointer |
| --- | --- | --- | --- | --- |
| 5B-S0 formal route audit | sequence evaluator / runner / config | S0 T-owner/T-shape/T-legacy/T-write-audit | completed | 2026-07-21 read-only audit: no v015 end-to-end owner exists; legacy owner mutates sampler state and is fail-closed for v015 |
| 5B-S1 immutable request/report kernel | `frontres_segment_sequence_eval.py` | S1 T-npz-schema/T-identity/T-corruption-protocol/T-report/T-no-feedback/T-config-fail-closed/T-legacy-reject | completed | `E-FI-28`; explicit structured deployment `.npz`, file/protocol hashes, immutable per-frame report, no training-state fields, and legacy mixed-mode rejection |
| 5B-S2A deployment carrier and H snapshot | command / runtime bridge | S1/S2 T-install/T-current/T-H/T-frame-order/T-cursor/T-boundary/T-row-alignment/T-provenance/T-identity/T-mixed-reference/T-no-execution/T-no-training-state | completed | `E-FI-29`; sealed request -> immutable q29/dq29 sequence -> current `[B,58]` and dense H `[B,H+1,29]`; no clamp, mixed carrier, actor/GMT/runner, or training-state path |
| 5B-S2B formal composition executor | sequence evaluator / config / runner | S2 T-connect/T-per-frame/T-frozen-GMT/T-report/T-zero-write/T-formal-entry/T-legacy-isolation | completed | `E-FI-30`; pre-materialized deployment `.npz`, `T-max(H)` unclamped rows, `870+58=928`, one 6D action + frozen-GMT read per frame, atomic JSON, unchanged optimizer/sampler/storage/transition fingerprints |
| 5B-S4-S0 dedicated live CLI | v015 CLI / formal runner / checkpoint owner | S2 T-path/T-gpu/T-protocol/T-config/T-dispatch/T-zero-update/T-owner/T-no-training | implemented-not-runnable | `E-FI-31` proves config/dispatch only; `E-FI-32` confirms no compatible trained checkpoint and retires the external Noisy-file prerequisite |
| 5B-S4 bounded composition evidence | paired formal deployment evaluator | S4 T-composition/T-pair/T-isolation/T-protocol/T-checkpoint | blocked behind G2--G6 | requires trained/reloaded v015 checkpoint, selection-time fixed carrier, and same-carrier No-FEMR/GMT versus FEMR/GMT comparison |

## Post-Observation-Change Test Path: G0--G7

| Gate | Owner | Required S/T | Status | Evidence / stop condition |
| --- | --- | --- | --- | --- |
| G0 document/test-path rebase | contract / plan / checklist / Architecture | S0 T-doc/T-dependency/T-status | completed | `E-FI-32`; stop if missing checkpoint or external Noisy `.npz` is still treated as available input, or S4 is runnable |
| G1 training readiness audit | config / HSL / runner / PPO / checkpoint / train entry | S0 T-owner/T-layout/T-checkpoint/T-train-dispatch/T-stop | completed, stopped on confirmed gaps | `E-FI-33`; old `870D` Stage-1 route, undefined HSL identity, legacy Stage-3 dispatch, and absent exact v015 checkpoint producer confirmed |
| G2-S0 HSL persistence contract freeze | Stage-1 carrier / preset / runner layout / warmup / checkpoint owner | S0 T-owner/T-carrier/T-layout/T-target/T-payload/T-stop | completed by user decision | `E-FI-34` plus user confirmation; use a minimal command-owned proposal carrier in existing modules, not the full local scenario |
| G2-S1a HSL proposal carrier | existing `commands.py` / runtime-layout bridge / existing HSL S1 test | S1 T-carrier/T-shape/T-provenance/T-immutability/T-no-C-K/T-local-isolation | completed | `E-FI-35`; current artifact identity and deployment q29 only; no x_t/C/K/Segment/attempt state; no new source or test module |
| G2-S1b formal proposal-only HSL route | Stage-1 preset / runner / warmup | S1 T-config/T-formal-layout/T-HSL-input/T-current-target/T-actor-only/T-critic-unchanged/T-legacy-reject | completed | `E-FI-36`; `928D -> FEMR 158D / GMT 770D`, actor changes, critic grad/state unchanged, energy route absent |
| G2-S2 HSL identity/save/reload | `frontres_checkpointing.py` / warmup save connector | S3 T-schema/T-save/T-reload/T-tamper/T-GMT-identity/T-normalizer/T-forbidden-payload/T-unmutated-reject | completed | `E-FI-37`; strict actor/distribution/158D-prefix payload, GMT artifact/normalizer binding, five rejected tamper/legacy cases remain unmutated |
| G2-S3 fresh-runner HSL connectivity | runner / q29 bridge / normalizer / actor / checkpoint loader | S2/S3 T-fresh-runner/T-output/T-layout/T-zero-state-leak | completed | `E-FI-38`; same artifact/q29/raw input reproduces exact combined 928D, normalized 158D and bounded 6D proposal after strict reload |
| G2-S4-S0 bounded HSL connector | existing Stage-1 config/runtime/warmup/checkpoint owners | S1/S3 T-bounds/T-telemetry/T-shadow-reload/T-legacy-reject | completed | `E-FI-39`; explicit bounded flag, real-input sentinel schema, actor-only gradient, zero critic delta, strict identity, exact 158D/6D shadow reload; no new source module |
| G2-S4-S0a full-6D diagnostic repair | `frontres_warmup.py` / existing HSL S1 contract | S1 T-full-6D/T-no-mask/T-regression | completed | `E-FI-40`; removed undefined legacy `_sup_mask`; source contract rejects `_sup_mask` and `frontres_active_task_dims`; five deterministic regression classes pass |
| G2-S4-S0b cross-device reload verifier | `frontres_checkpointing.py` / existing HSL S1 contract | S1/S3 T-exact-state/T-exact-input/T-device-tolerance/T-large-drift-reject | completed | `E-FI-41`; strict state/input unchanged; 6D CUDA/CPU proposal uses `rtol=1e-5, atol=1e-6`, logs max error and bitwise status; `5e-7` passes and `1e-3` rejects |
| G2-S4-S1 bounded HSL smoke | formal Stage-1 route | S4 T-live-input/T-current-target/T-save/T-fresh-reload | completed | `E-FI-42`; 8 envs, one step, actor grad norm `9.7264719`, critic grad/delta zero, strict HSL-v1, reload max error `2.79396772e-09`, no PPO entry |
| G3-S0 Stage-3 migration/save audit | config / train entry / runner / HSL load / sealed transaction / checkpoint | S0 T-owner/T-load-boundary/T-formal-dispatch/T-legacy-isolation/T-save-producer/T-fresh-reload/T-stop | completed | owner audit froze actor-only migration, formal dispatch, committed save, and fresh-reload gaps before `E-FI-43` |
| G3-S1A actor-only HSL migration | checkpoint owner / runner connector / Stage-3 preset | S1/S3 T-explicit/T-layout/T-actor-only/T-prefix/T-zero-state-leak/T-legacy-reject/T-pre-mutation/T-dispatch-stop | completed | `E-FI-43`; explicit HSL-v1 restores actor/std/158D prefix only; formal q29/grouped config; legacy train blocked before G3-S1B |
| G3-S1B formal transaction dispatch/save | formal Stage-3 training owner / transaction provider / committed checkpoint trigger | S2/S3 T-provider/T-complete-transaction/T-grouped/T-exact-one-update/T-legacy-isolation/T-commit/T-save | completed | `E-FI-44`; whole M budgets fill Repair rows, ordinary formal branch performs one update and saves once only after matching commit |
| G3-S2 exact save/fresh reload | checkpoint owner / fresh inference runner | S3 T-save-producer/T-v015-identity/T-commit-receipt/T-fresh-runner/T-prefix-normalizer/T-proposal-equality/T-legacy-reject | completed | `E-FI-45`; same semantic 158D policy receives the exact-one Adam update, actual `save_runner`, and strict fresh reload with exact q29/normalizer/6D output |
| G3 Stage-3 engineering readiness | actor migration / formal transaction / v015 checkpoint | S2/S3 T-actor-migration/T-formal-dispatch/T-one-transaction/T-exact-one-update/T-save/T-fresh-reload | completed offline | `E-FI-43`--`E-FI-45`; no S4/trained-policy claim; bounded training and quality remain G5 |
| G4 controlled carrier materializer | reference corruption preparation owner | S1/S2 T-materialize/T-hash/T-determinism/T-no-label/T-no-resample | completed | `E-FI-46`; ordinary `.npz` + fixed protocol/root identity -> deterministic atomic carrier; q29 unchanged; current `[B,58]` and H `[B,H+1,29]` consume the same hash |
| G5-S0 formal training/quality preflight | config / formal transaction / checkpoint / quality evaluator | S0 T-owner/T-shape/T-HSL-artifact/T-transaction/T-save/T-reload/T-quality-route/T-stop | completed, stopped on confirmed gaps | `E-FI-47`; train/exact-one/committed-save are reachable, but post-save fresh reload and v015 quality report are absent; quartet/v011/v002 quality route is incompatible |
| G5-S1 transaction quality telemetry | v015 diagnostics owner / formal probe connector | S1/S2 T-action-shape/T-finite/T-v003-source/T-component/T-positive-negative/T-row-mask/T-identity/T-no-feedback/T-legacy-reject | completed | `E-FI-48`; immutable sealed v003 rows are projected after collection and published only as post-update diagnostics; missing fields and mixed identity fail closed |
| G5-S2A strict quality checkpoint/manifest | checkpoint validator / quality request and manifest owners | S1/S3 T-HSL-v1/T-Stage3-v015/T-manifest/T-layout/T-prefix/T-pre-mutation/T-tamper/T-legacy-reject | completed | `E-FI-49`; v015-only immutable request binds strict manifest/file fingerprints to separate HSL-v1 and Stage3-v015-v2 receipts; v1/v002, padding, tamper, route swap and partial transaction reject |
| G5-S2B two-role held-out quality evaluator | quality evaluator / active v015 local scenario and v003 owners | S1/S2 T-two-role/T-same-scenario/T-one-action/T-frozen-K/T-v003/T-zero-HSL-policy/T-isolation/T-report/T-legacy-reject | completed | `E-FI-50`; zero/HSL/policy route evidence binds manifest item and checkpoint SHA, preserves Repair/Noisy scenario/K identity, computes v003 only, rejects state/identity drift, and atomically reports |
| G5-S3 save/fresh-reload/quality connectivity | checkpoint owner / fresh runner / held-out evaluator | S2/S3 T-commit/T-save/T-fresh-runner/T-identity/T-normalizer/T-proposal-equality/T-report/T-isolation | completed | `E-FI-51`; real exact-one transaction -> actual save -> independent strict fresh reload preserves q29, 928/158/770, prefix stats and 6D proposal; strict HSL/policy identities reach one atomic report with zero training-state mutation |
| G5-S4-S1A explicit launch/live telemetry | Stage3 launchers / formal live summary / existing transaction diagnostics | S1/S2 T-explicit-HSL/T-offsets/T-no-resume/T-no-periodic-legacy/T-one-iteration/T-action/T-v003/T-identity/T-exact-one/T-no-feedback | completed | `E-FI-53`; strict bounded launcher and immutable post-update transaction telemetry pass focused deterministic contracts; no simulator/training/live |
| G5-S4-S1B formal held-out/fresh-report route | formal runner / v015 quality owner / checkpoint / fixed manifest | S1/S2/S3 T-owner-install/T-manifest/T-two-role/T-one-action-K/T-save/T-fresh/T-layout/T-proposal/T-hash/T-report/T-isolation/T-legacy-reject | completed | `E-FI-54`; fixed 16-item manifest, formal auto-install, immutable 4+4 scenario, strict route actor swap/restore, deterministic proposal, actual-save/fresh equality and atomic v003 report pass offline |
| G5-S4-S1C held-out index/K resolver | formal held-out sampler owner | S1/S2 T-K4-index/T-K8-budget/T-K8-continuation/T-unique-identity/T-heldout/T-save-fresh/T-report | completed | `E-FI-55`; `(motion,start)` uniquely owns x_t while manifest K8 owns materialization/execution; duplicate identity rejects |
| G5-S4-S1D quality inference-mode isolation | held-out evaluator / policy and normalizer modes | S1/S2 T-train-mode-write/T-zero-write/T-mixed-mode-restore/T-exception-restore/T-heldout/T-save-fresh/T-observation | completed | `E-FI-56`; zero/HSL/policy freeze policy plus prefix/GMT/privileged/teacher normalizers before observation read, signatures cover every normalizer, and exact modes restore on success/error |
| G5-S4-S1E manifest-item lifecycle isolation | held-out evaluator / command and immutable batch lifecycle | S1/S2 T-route-order/T-item-close/T-next-item/T-exception-close/T-command-close/T-batch-close/T-no-feedback/T-save-fresh | completed | `E-FI-57`; each item keeps one sealed scenario for zero/HSL/policy, then closes command and batch before the next identity; failure paths close and close-state writes reject |
| G5-S4-S2 command/artifact/threshold preflight | formal train and quality routes | S0 T-artifact/T-command/T-telemetry/T-threshold/T-stop | partial, user-gated | `E-FI-55`; training artifacts and exact-one save are runtime-confirmed; corrected quality command and numeric thresholds still require confirmation |
| G5-S4-S4 bounded training/policy-quality | formal v015 train and held-out quality routes | S4 T-train/T-action/T-gain/T-harm/T-checkpoint/T-fresh-reload/T-report | partial | `E-FI-55`; bounded train/update/save completed, but fresh held-out quality did not start because the first shell command selected a legacy default HSL checkpoint |
| G6 paired composition connectivity | baseline/repair sequence evaluator | S1/S2 T-pair/T-identity/T-baseline/T-repair/T-no-feedback | blocked by G5 | same carrier/reset/GMT; no metrics enter training state |
| G7 bounded live composition | paired formal evaluator | S4 T-composition/T-pair/T-protocol/T-isolation/T-checkpoint | blocked by G6 | one trained checkpoint and fixed carrier; stop on resampling, absent baseline, or state mutation |

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
