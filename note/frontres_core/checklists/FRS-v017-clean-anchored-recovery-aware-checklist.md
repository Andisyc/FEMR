# FRS-v017 / TRAIN-v015 Engineering Checklist

This checklist tracks only the active
`v017/v007/v005/v015/v004` route. TRAIN-v014/checkpoint-v9 Stage-3 evidence and
HSL-v1 remain historical only. The current route cold-starts from the frozen
HSL-v2 artifact whose proposal identity remains TRAIN-v014, then persists only
TRAIN-v015/checkpoint-v10.

| Gate | Owner / tier | Acceptance assertion | Status | Evidence / stop |
| --- | --- | --- | --- | --- |
| GOV-AUTH | governance | TRAIN-v015/checkpoint-v10 is active; frozen HSL-v2 alone retains its TRAIN-v014 proposal identity | completed | E-FI-135/E-FI-136 |
| GOV-PLAN | engineering review | Boundary Record, owner/interface/lifecycle/proof route pass FRS-ENG-v001 | completed | v013 plan review READY; E-FI-108 |
| CARD-CONFIG | S1 | Training Config card covers K/M plus explicit inner-DR restart/advance and no hidden defaults | completed, human-confirmed rerun | TEST-02; E-FI-110 |
| CARD-PERTURB | S1 | Perturbation card covers 20/30/40/10 boundaries, no-resample and no feedback | completed, human-confirmed rerun | TEST-06; E-FI-110 |
| CARD-CHECKPOINT | S3 | Checkpoint card covers v9 direct-action state and v8/v7/g_K pre-mutation rejection | completed, contract rerun | TEST-16; E-FI-114 |
| V013-IDENTITY | S0/S1 | TRAIN-v013/checkpoint-v8/DR schema identities replace v012/v7 once | completed | focused identity and import negatives; E-FI-109 |
| V013-SCHEDULE | S1 | K8/M2, K16/M3, K32/M4 resolve exact phases and explicit per-K DR specs | completed | warmup contract; E-FI-109 |
| V013-FOUR-CLASS | S1 | Easy/Medium/Hard/Broken use exact 20/30/40/10 weights and d_cap boundaries | completed | distribution contract; E-FI-109 |
| V013-PROBING | governance | 2.381 is the current measured frozen-GMT boundary; direct config and optional offline Probing have identical downstream authority | documented; optional Probing not required for current campaign | E-FI-111; no source/runtime claim |
| V013-TRANSITION | S1/S2 | committed K change restarts DR, freezes Actor/std and retains same Critic | completed offline | warmup + transaction contracts; E-FI-109 |
| V013-TRANSACTION | S2 | one transaction is homogeneous in K/M/DR identity; abort cannot advance progress | completed offline | formal transaction contract; E-FI-109 |
| V013-NO-FEEDBACK | S1/S2 | Gain/PPO/eval/diagnostics cannot mutate DR schedule or class selection | completed offline | distribution + telemetry contracts; E-FI-109 |
| V013-PERSIST | S3 | v8 roundtrip restores full schedule/cursor/RNG/receipt; v7/g_K rejects pre-mutation | completed | checkpoint contract; E-FI-109 |
| V013-DIAG | S1/S2 | telemetry carries stage/class/strength/d_cap/progress without recomputation | completed offline | transaction aggregate; E-FI-109 |
| V013-MODULE-ALL | S0/S1/S3 | all 18 human-confirmed cards pass without changing their independent answers | completed | 18/18 focused cards; E-FI-110; aggregate not used as substitute |
| V013-CONSTRUCTION | code discipline | first coherent owner boundary has no open P0/P1 | completed | FRS-ENG-v001 construction review; E-FI-109 |
| V014-DIRECT-ACTION | S1/S2/S3 | HSL and Stage 3 share one finite direct `[B,6]` action; no action mask/scale/tanh/clip/clamp or 12D slicing survives | completed offline; Phase A DP-04 confirmed | focused HSL/action/log-prob/storage/checkpoint contracts and 49/49 regression; E-FI-114 |
| V013-FORMAL-A | formal S2/S3 | official config -> transaction -> update -> v9 save/telemetry has no old action transform, curriculum bypass or future-context layout drift | completed offline; DP-01 through DP-10 reviewed | E-FI-111 through E-FI-119; Phase B/live pending |
| V014-FORMAL-A-K | formal S1/S2/S3 | sealed TRAIN-v014 K/M bypasses the retired state-driven budget; one direct action produces one PPO row and K evidence frames; checkpoint-v9 preserves the same stage/action identity | completed offline; live pending | sampler/audit isolation, one-action-K and checkpoint contracts; E-FI-113/E-FI-114 |
| V014-DP05-FROZEN-MODE | S1 | policy train mode leaves Actor/Critic trainable but forces GMT policy/normalizer/estimator to eval, no-grad and optimizer-excluded | completed offline | focused frozen-GMT contract + 49/49 regression; E-FI-115 |
| V014-DP05-GMT-IDENTITY | S3 | checkpoint-v9 saves GMT SHA256 + 770D layout + normalizer identity and rejects a different GMT before mutation | completed offline | strict save/resume and same-shape different-GMT pre-mutation rejection; E-FI-115 |
| V014-DP07-PROJECTION | S1 | local report carries every owner-produced v007 component and fixed scale/beta identity without recomputation or zero-fill | completed offline | TEST-18 + Gain/step1 focused regression; E-FI-116 |
| V014-DP07-CONSUMER | formal S2 | official fake Stage3 transaction preserves the same row-aligned decomposition through final telemetry serialization | completed offline | formal exact-one transaction connectivity; E-FI-116 |
| V014-DP07-LEGACY | S0/S2 | v006 report and projection/KKT fields cannot enter the v017 formal report or serializer | completed offline | strict report type/contract identity and absent projection fields; E-FI-116 |
| V014-DP08-TARGET | S1 | Stage-1 target equals exact finite current anti-DR `[B,6]`, including positive anti-DR `dz`; no per-axis mask/scale/clip/clamp | completed offline | TEST-10 HSL target regression; E-FI-117 |
| V014-DP08-ISOLATION | S1/S2/S3 | HSL remains 158D actor-only, HSL-v2-only and Stage-3 supervised-target-free | completed offline | HSL S1/S2, observation authority and checkpoint regressions; E-FI-117 |
| V014-DP08-FORMAL-A | formal S2 | official Stage-1 producer reaches the same validator/loss and cold-start initializer without legacy target authority | integrated offline | Stage-1 entrypoint + HSL S2 connectivity; E-FI-117 |
| V014-DP10-LAYOUT | S1/S2 | the only legal deployment/Noisy q29 future offsets are exactly `(1,2)`; `(1,3)` and other layouts reject before mutation | completed offline | TEST-04/05 and config/layout rejection contracts; E-FI-119 |
| V014-DP10-AUTHORITY | formal S2 | the real observation reader produces `870D + 58D = 928D`; FrontRES sees 158D, frozen GMT sees 770D, with one actor action and no Clean/root/global future leakage | completed offline; live pending | unmocked observation connectivity + TEST-10; E-FI-119 |
| V014-DP10-PERSIST | S3 | checkpoint-v9 preserves exact future-layout identity and rejects offset drift before mutation | completed offline | TEST-16 strict checkpoint contract; E-FI-119 |
| V015-GOV-AUTH | governance | TRAIN-v015/checkpoint-v10 is active; v014 is historical; registry and Inspector agree | completed | active contract/registry/Inspector; E-FI-135 |
| V015-SPLIT-GROUPS | S1 | one Adam has exactly named, disjoint Actor `3e-6` and Critic `1e-5` groups; fixed std is excluded | completed offline | real optimizer owner contract; E-FI-135 |
| V015-PHASE-COMMIT | S1/S2 | critic-only preserves Actor parameters and Adam state while Critic changes; exact-one count is shared | completed offline | optimizer and formal transaction contracts; E-FI-135 |
| V015-CONFIG | S1/S2 | official Stage-3 composition installs fixed split LR; shared, adaptive and partial inputs reject | completed offline | entrypoint/preflight contracts; E-FI-135 |
| V015-TELEMETRY | S2 | final committed serializer emits exact Actor/Critic LR facts without training feedback | completed offline | transaction/typed telemetry contracts; E-FI-135 |
| V015-PERSIST | S3 | checkpoint-v10 round-trips groups/LRs/moments/count and v9/missing/duplicate/overlap/nonfinite/malformed identity rejects pre-mutation | completed offline | strict checkpoint contract; E-FI-135 |
| V015-REGRESSION | S0-S3 | complete affected Stage-3 suite retains all prior method behavior | completed offline | 50/50 contract suite; E-FI-135 |
| V015-LIVE | live S4/S3 | one official K8/M2 critic-only transaction emits split groups/LRs, Actor zero delta, Critic nonzero delta, step delta 1 and checkpoint-v10 | authorized, execution pending | pulled first attempt failed pre-rollout on the repaired HSL identity boundary; no quality claim |
| PHASE-B-01 | live S4 | official train entry emits active contract/HSL-v2/K8-M2/offset `(1,2)` identity with no legacy route | runtime-confirmed | AUDIT-B01 / Runtime Audit Atlas R01 / E-FI-122 |
| PHASE-B-02 | live S4 | sampler emits two sealed Segments x exact M=2 with four immutable policy rows and no reset resampling | runtime-confirmed | AUDIT-B02 / R02 / E-FI-123 / E-FI-124 / E-FI-126 |
| PHASE-B-03 | live S4 | real reset/command/observation route proves B=8 roles and `870+58 -> 928 -> 158/770` with Noisy provenance | runtime-confirmed | AUDIT-B03 / R03 / E-FI-128 |
| PHASE-B-04 | live S4 | each attempt has one finite `[6]` action, then FEMR freezes while eval/no-grad GMT executes K8 | runtime-confirmed | AUDIT-B04 / R04 / E-FI-128 |
| PHASE-B-05 | live S4 | Clean=2, Noisy=2 and Repair=4 feed complete v007 `G_I/G_P/P_N/P_R/lambda/cost/G_total` evidence | runtime-confirmed | AUDIT-B05 / R05 / E-FI-121 / E-FI-126 |
| PHASE-B-06 | live S4 | storage writes four policy rows, not K-expanded rows; return equals G_total and identity remains sealed | runtime-confirmed | AUDIT-B06 / R06 / E-FI-128 |
| PHASE-B-07 | live S4 | grouped loss gives two Segments equal influence and exactly one optimizer update; K8 critic-only freezes Actor/std | stale rerun required for v015 | prior AUDIT-B07/E-FI-128 used the v014 optimizer identity; v015 split-LR evidence pending |
| PHASE-B-08 | live S4/S3 | one committed receipt advances iteration/curriculum and saves checkpoint-v10 with split-LR/GMT/layout identity | stale rerun required for v015 | prior AUDIT-B08/E-FI-128 saved checkpoint-v9; v10 evidence pending |
| QUALITY-Q0-SUPPORT-FRAME | Q-mechanism | support-foot drift compares Clean and Repair in one shared environment-local coordinate before Physics aggregation | completed offline; policy efficacy pending | cross-origin zero-drift, row-permutation and malformed-origin contracts; E-FI-129 |
| PHASE-B-ARTIFACT | authority | a real HSL-v2/TRAIN-v014 proposal artifact is verified before TRAIN-v015 cold start | runtime-confirmed | `/hdd0/yuxuancheng/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt`; direct identity inspection E-FI-136 |
| V014-DP09-PHASE | S1/S2 | resolver, typed request, formal transaction and telemetry use only `critic_only`, `actor_ramp`, `joint`; old `actor_warmup` rejects | completed offline | warmup/interface/formal transaction contracts; E-FI-118 |
| V014-DP09-CONTINUITY | S2 | K8->K16 and K16->K32 retain the same Critic identity and learned state while critic-only updates it | completed offline | K16/M3 and K32/M4 formal transaction fixture; E-FI-118 |
| V014-DP09-FREEZE | S2 | critic-only preserves Actor/std parameters and their existing optimizer state exactly while Critic changes | completed offline | seeded-Adam rollback regression; E-FI-118 |
| V014-DP09-PERSIST | S3 | checkpoint-v9 stores `actor_ramp`, restores it exactly and rejects `actor_warmup` before mutation | completed offline | strict checkpoint-v9 roundtrip/tamper contract; E-FI-118 |
| V013-FINAL | code discipline | complete diff has no open P0/P1 or added wrapper/owner/hotspot reason | completed for module closure | E-FI-110; P0=0/P1=0, three explicit existing P2 risks |
| V013-DOC | governance | Test Atlas/inventory/evidence/checklist match observed module facts and do not claim connectivity | completed for module closure | E-FI-110 |
| S4-LIFECYCLE | live | one bounded K8/M2 transaction proves real class/strength/no-resample/exact-M | runtime-confirmed | E-FI-128 |
| S4-PERSIST | live/S3 | one committed checkpoint-v10 and independent fresh reload are exact | v015 live save pending; v9 evidence historical | E-FI-135 offline strict roundtrip; no fallback or second update |
| S4-QUALITY | live | real Gain/Contact/ZMP/survival/action facts are finite or semantic N/A | evaluator ready offline after dataset/reset and read-only collection-lifecycle closures; real evaluation pending | E-FI-129 closes the coordinate blocker; E-FI-130 through E-FI-134 establish the read-only route; E-FI-136 updates it to HSL-v2/TRAIN-v014 plus strict checkpoint-v10/TRAIN-v015 identity; prior E-FI-128 quality rows remain invalid |

## Preserved Completed Surface

The following unchanged behaviors retain E-FI-105/E-FI-106 evidence and are
rerun only as regression during Step 1: 928/158/770 visibility, full-6D action,
Clean/Noisy once plus M Repair, one-action-K, exact two-Segment x M, grouped
scalar PPO-v005, exact-one commit, HSL-v2 actor-only initialization, v007 Gain,
read-only v004 evaluation, and legacy updater pre-construction rejection.

## Pass Rule

Module Test Closure passes only when all 18 human-confirmed cards retain their
contract-derived independent answers and pass S0/S1 plus module-owned S3, with
no open code-review P0/P1. It admits a separate Formal Runtime Audit Phase A
human review; it does not prove that review's official-path claims. Step 2
requires separate explicit
authorization and cannot be inferred from offline closure.

## Fail Rule

Stop on duplicate curriculum ownership, hidden/default DR schedule, active
episode-length/frontier controller, Gain/PPO sampler feedback, resampling,
mixed/partial transaction progress, Actor/std drift in critic-only, Critic
reinitialization at K transition, checkpoint-v7 mutation, Clean actor leakage,
MOSAIC host change, or a weakened Test Card.
