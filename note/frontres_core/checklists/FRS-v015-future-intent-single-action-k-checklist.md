# FRS-v015 Physics-Constrained Intent Migration Checklist

Status: active, volatile acceptance surface. Updated: 2026-07-28.

| Step | Owner / tier | Acceptance assertion | Status | Evidence / stop |
| --- | --- | --- | --- | --- |
| P0 | governance | Noisy stays the shared zero-action baseline; Physics constraints and scalar Intent objective have separate authority | completed | E-FI-72; v005 proposal |
| P0 | owner audit | v004 scalar path is traced through Gain, storage return, scalar Critic/PPO, formal update, diagnostics and checkpoint | code-confirmed | CodeGraph 2026-07-23 |
| P0 | mismatch | Concept Figure Q-01 expressed constrained Intent while source still used v004/v003/v009 | resolved | E-FI-74 replaces the active formal source route; old identities reject |
| P1 | semantic | exact Contact/ZMP/survival residuals retain physical meaning and do not saturate severe states | completed | E-FI-73; GAIN-v005 physical-unit residuals |
| P1 | optimization | grouped joint constraint-gradient projection/recovery is exact, fail-closed and needs no second learned network | completed | E-FI-73; PPO-v004 KKT/recovery/no-common-descent rules |
| P1 | replay priority | constraint/frontier evidence may affect scenario selection but cannot become scalar Physics reward or actor-loss mass | completed | E-FI-73; selection-only authority |
| P1 | Critic curriculum | v004 Critic rejects; first v010 target entry is fresh critic-only; every global K increase recalibrates the same Critic before actor ramp/joint | completed | E-FI-73; TRAIN-v010 checkpoint-v5 identity |
| P1 | contracts | METHOD-v016, GAIN-v006, PPO-v004 and TRAIN-v010 agree on objective, constraints, Critic, optimizer and persistence identities | completed | E-FI-73 plus E-FI-82 supersession; coordinated active contracts |
| P2 | S1 evidence | raw K-step Contact/phase-ZMP/survival, N/A semantics, constraint residuals and projection pass value/collision/permutation/missing tests | completed | E-FI-74; no scalar Physics fallback |
| P2 | S2 connectivity | same sealed Noisy baseline and M Repair rows reach scalar objective plus vector constraints, grouped equal mass and exact-one update | completed | E-FI-74; v004/v003 formal consumers reject |
| P2 | S2 gradients | scalar Critic receives only Intent objective; actor receives Intent direction constrained by Physics; frozen GMT and actor inputs unchanged | completed | E-FI-74; KKT and parameter-authority contracts pass |
| P2 | S3 persistence | checkpoint-v5/receipt bind solver and contracts; v004/v009, tampered, or partial resume rejects before mutation | completed | E-FI-74; no persistent dual state or legacy scalar Gain payload |
| P3 | S4 official | one 8-env transaction logs raw evidence, constraints, projection, objective/value/advantage, parameter deltas and exactly one committed update | completed | E-FI-75; live log plus read-only checkpoint-v5 payload identity |
| P4-S0 | S0 admission audit | checkpoint-v5 phase, Critic budget, formal continuation route and held-out evidence coverage are read-only resolved before actor-ramp | completed | E-FI-76: K8 critic-only 1/200; formal resume contradiction and quality-report gaps confirmed |
| P4-S1 | S1 launcher/config | strict checkpoint-v5 resume is mutually exclusive with HSL initialization and preserves exact v010 schedule/iteration | completed | E-FI-77; explicit resume replaces HSL flag, missing/legacy/partial/mixed identity rejects |
| P4-S1 | S2 resume connectivity | actor/Critic/optimizer/sampler/normalizer/receipt restore exactly and one fake transaction advances exactly one update | completed | E-FI-77; strict semantic checkpoint and transaction contracts pass |
| P4-S1 | S1/S2 quality evidence | atomic report exposes raw expected/actual Contact, phase-ZMP trajectories/N-A masks, survival and evaluator-only sustained lean | completed | E-FI-77; missing evidence fails closed and no Clean actor route is added |
| P4-S1 | S3 persistence/isolation | resumed save remains coordinated checkpoint-v5 and evaluation mutates no optimizer/sampler/transaction/normalizer state | completed | E-FI-77; committed receipt survives idle re-save; save/reload isolation passes |
| P4-S2 | S4 critic-only schedule | strict v5 resume executes exactly 199 K8 critic-only updates from absolute iteration 1 to 200 | completed | E-FI-78; 199/199 telemetry rows are `critic_only`, actor weight 0, actor/std delta 0, Critic delta nonzero |
| P4-S2 | S4 transaction/persistence | every accepted transaction has 4 valid rows, equal attempt mass and exact-one update; rejected scenarios do not step; final save is coordinated v5 at iteration 200 | completed | E-FI-78; `v015_p4_critic_k8_to_200_gpu3.log`; `model_200.pt` persistence sentinel |
| P4-S3 | S4 actor-ramp lineage | the strict actor-ramp lineage advances beyond iteration 200 and is the parent of the `model_251.pt` full-resume input | runtime-complete, prior evidence | E-FI-79 directly confirms the descendant checkpoint is loaded; this row is no longer the active boundary |
| P4-LT | S4 long training | `model_251.pt` reaches iteration 2000 through complete 2-Segment x 2-attempt exact-one transactions | runtime-complete with defect | E-FI-79; 1749/1749 committed, but four `CONSTRAINT_RECOVERY` rows violate KKT after norm rescale |
| P4-KKT | S1 projection regression | norm-rescaled recovery is reprojected and remains within every active halfspace | completed | E-FI-80; recorded near-opposing float32 fixture fails before and passes after repair |
| P4-KKT | S2 formal consumer | serialized KKT above checkpoint-v5 tolerance or inconsistent with directional derivatives rejects before further training state use | completed | E-FI-80; transaction-route negative contracts pass |
| P4-ZMP | S1 estimator/carrier | raw per-contact force/point/normal produces a permutation-stable world ZMP and sealed Clean foot pose produces immutable `[K,6]` expected support envelopes | completed offline | E-FI-81; golden/missing/flight/hash/provenance contracts pass |
| P4-ZMP | S2 formal connectivity | one-action-K Physics capture uses only contact-wrench ZMP against expected phase/envelope; root/capture proxy and actor-visible Clean geometry are unreachable | completed offline | E-FI-81; reset/K/transaction/observation contracts pass |
| P4-ZMP | S3 persistence | checkpoint-v5 binds the new estimator/support/Contact/phase identities and old v5 payloads reject before state mutation | completed offline | E-FI-81; strict checkpoint/save/reload contracts pass |
| P4-ZMP-v006 | S1 evidence semantics | valid expected-support/actual-unloaded rows remain scored Contact violations with role-specific ZMP N/A; malformed payload and loaded-without-resultant fail closed | completed offline | E-FI-82; estimator/Gain/no-load contracts pass |
| P4-ZMP-v006 | S2 formal connectivity | explicit Repair/Noisy applicability reaches one-action-K, GainResult, ReturnEvidence, transaction telemetry and atomic quality report without finite/NaN inference, zero fill or row loss | completed offline | E-FI-85; role-specific masks survive formal transaction, final snapshot and held-out report |
| P4-ZMP-v006 | S3 persistence | checkpoint-v5 accepts only GAIN-v006/schema-v2 and rejects v005/schema-v1 before mutation | completed offline | E-FI-82; strict resume contract passes |
| P4-ZMP-v006 | final consumer | sentinel final JSON preserves sealed Repair/Noisy Contact and role-specific phase-ZMP applicability/N/A; adjacent exact-one checkpoint-v5 is strict and fail-closed | completed offline | E-FI-83; final serializer and temporary persistence contracts pass |
| P4-ZMP | S4 sensor authority | official IsaacLab emits finite raw filtered contacts and supported-phase ZMP with correct role/hash identity | completed live | E-FI-84; final snapshot, exact-one update and strict model_1.pt checkpoint sentinel pass |
| P4-ZMP-v006 | applicability carrier closure | all four Repair/Noisy applicability combinations preserve explicit masks; paired ZMP gain is finite iff both are applicable; PPO consumes Repair only; final serializers retain both | completed offline | E-FI-85; S1/S2/S3 focused regressions and final-consumer evidence pass |
| P4-ZMP-v006 | raw-contact capacity continuation | 8-env raw filtered views provision 2048 contacts per foot sensor, retain exact-saturation fail-closed, and complete the official continuation without capacity loss or semantic drift | completed live | E-FI-86; 1999/1999 committed transactions reach absolute iteration 2000, KKT max 0, no capacity/applicability error, formal model_2000.pt save emitted |
| P4-EVAL | S1/S2 final consumer | policy-only deployment report preserves post-fix checkpoint, Clean-reset/Noisy-carrier separation, real ContactSensor/phase-ZMP, Intent, survival, lean and unplanned-step trajectories with zero training feedback | completed offline, live pending | E-FI-87; five focused contracts plus 63/63 aggregate suite; dedicated shell driver ready |
| P4 | quality | post-fix model_2000 directly preserves Intent, Contact, phase-ZMP and survival without sustained lean/unplanned stepping | pending one deployment run | E-FI-87 closes the report/CLI offline; actual JSON/log remains S4 unconfirmed |
| P5-A | contract/governance | TRAIN-v011 freezes K8/M2 -> K16/M3 -> K32/M4, two Segment sources, review boundaries 2000/3500/4825/6500/8000, max 8000 and checkpoint-v6 without changing METHOD/GAIN/PPO/HSL | completed document-only | E-FI-88; registry, Concept Figure, plan/canvas and Architecture synchronized |
| P5-B | S1 schedule/layout | parser resolves active K/M/phase and exact two-Segment x M rows for M=2/3/4; env width is exactly 4M and sampler state cannot change formal M | completed offline | E-FI-89; exact schedule/parser, M2/M3/M4 sampler and launcher/config contracts |
| P5-B | S2 formal connectivity | sealed exact-M transaction reaches grouped equal-mass PPO-v004, final telemetry and exactly one committed update with no formula change | completed offline | E-FI-89; transaction/unmocked/local-sentinel contracts and 63/63 suite |
| P5-B | S3 persistence | checkpoint-v6 binds full K x M schedule, max/review boundaries, stage/rows, optimizer/sampler/normalizers and receipt; checkpoint-v5 rejects pre-mutation | completed offline | E-FI-89; strict round-trip/tamper/v5 pre-mutation rejection and fresh reload |
| P5-C | first-live stage evidence | first K16/M3 and K32/M4 formal transactions prove active stage, required env width, actor freeze, same-Critic delta, KKT, exact-one and unchanged fingerprint inside the real block | ready; runtime authority required | no separate sentinel when existing telemetry is complete |
| P5-C | checkpointed quality | boundaries 2000/3500/4825/6500/8000 receive committed save plus rolling Intent/repair/Contact/ZMP/survival/lean/step/action/value review and CONTINUE/PAUSE-REPAIR/STOP-DESIGN | ready; runtime authority required | systematic no-op/lean/unplanned support/Physics regression stops design; one bad transaction does not |

## Pass Rule

Engineering passes only when the formal route contains one scalar Intent
objective/Critic and explicit non-compensatory Physics constraints through
storage, actor update, diagnostics, and persistence. Retaining Noisy is required;
using it as the Physics threshold is forbidden.

## Fail Rule

Stop on a single scalar Physics utility, severe-state saturation, adverse-row
masking, Clean/Contact/ZMP leakage to actor input, second actor/Critic/optimizer,
mixed scenario/K/M identity, state-driven formal M, schedule mutation,
checkpoint-v5 migration into v011, v004/v003 fallback, partial transaction update, or
any live optimizer count other than one.
