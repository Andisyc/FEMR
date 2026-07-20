# FrontRES Policy-Quality Evidence Checklist

Status: retained policy-quality evidence; not the current v015 implementation checklist
Updated: 2026-07-19
Historical plan: `note/frontres_core/plans/engineering_plan.md`
Current v015 checklist: `note/frontres_core/checklists/FRS-v015-future-intent-single-action-k-checklist.md`

## Scope

This file preserves the Q1/Q2 policy-quality and transaction-audit evidence.
The current implementation acceptance surface is the K-A checklist linked
above. Existing formal training, periodic online eval, offline eval, and
sequence eval remain protected unless a later K-A plan step explicitly names
their owner boundary.

## Step Acceptance

| Step | Owner | Required S/T | Status | Evidence / blocker |
| --- | --- | --- | --- | --- |
| Q1-0 Governance | formal audit/evidence/Atlas/register | S0 `T-doc/T-identity` | completed | E70: joint actor_weight=1.0, valid accepted updates, frozen GMT, complete model_701; no quality claim |
| Q1-A Manifest | `frontres_policy_quality_manifest.py` | S1 `T-schema/T-hash/T-permute/T-missing/T-immutable` | completed | Q-E1 focused and aggregate contracts pass |
| Q1-B State restore | `frontres_policy_quality_eval.py` state helpers | S1/S2 `T-state/T-role/T-frame/T-cache/T-RNG/T-restore` | completed | Q-E2 focused pass; aggregate 46/46 |
| Q1-C Counterfactual routes | quality evaluator + frozen HSL adapter | S2 `T-counterfactual/T-frozen/T-source/T-shape/T-forward/T-isolation/T-metamorphic` | completed | Q-E3 focused pass; aggregate 47/47; formal wiring remains Q1-D |
| Q1-D Entry/isolation | CLI, `train.py`, `on_policy_runner.py` | S0/S2 `T-route/T-import/T-mode/T-no-call/T-state` | completed | Q-E4 focused pass; aggregate 48/48 |
| Q1-E Atlas/preflight | Quality Atlas + focused/aggregate suites | S0-S2 `T-connect/T-link/T-schema/T-isolation` | completed | Q-E6: official entry installs six real owner adapters; aggregate 51/51 |
| Q1-F Live identity | quality evaluator real simulator boundary | S4 `T-live/T-state/T-identity/T-frozen/T-isolation` | completed | Q-E11: comparison signature and three state hashes match; policy/noisy local dynamics and cached perturbation match; local_rp corruption is present |
| Q2 Counterfactual bank | zero/HSL/policy matched quality evaluator | S1/S2 then S4 `T-schema/T-matched/T-oracle/T-bucket/T-seed` | completed; quality gate failed | Q-E13: technical identity passes 16/16, but HSL-Zero is positive on only 1/8 motions and negative on 4/8; first divergence is HSL/Gain before PPO |
| Q2-A Gain learnability decomposition | independent Q2 reporter + canonical component artifact | S1/S2 `T-value/T-diff/T-oracle/T-meta` | completed | Q-E14: inferred Repair weight 0.15; among 16 HSL items, 5 degrade Style+Physics before cost, 1 is cost-dominated, 3 have insufficient pre-cost margin, 4 are noise-floor unresolved, 3 improve |
| Q2-B HSL output-target alignment | HSL rollout target owner + dedicated policy-quality evaluator | S1/S2 then S4 `T-transform/T-oracle/T-live` | completed; magnitude failure localized | Q-E17: 16/16 x K=8 target evidence passes; cosine median 0.910 but action/target norm ratio median 10.65x, max 23.29x |
| Q2-C1 Checkpoint lineage | checkpoint schema + offline lineage audit | S1 `T-schema/T-lineage/T-missing` | partial | Q-E18: saver omits effective supervised config and source-checkpoint identity; local model_200/model_warmup artifact is absent |
| Q2-C2 HSL loss-gradient scale | supervised loss owner + Q2-B persisted tensors | S1 `T-value/T-gradient/T-scale/T-metamorphic` | completed; formula contradiction localized | Q-E18: direction_pos grad L2 1.157e3 versus magnitude 0.00619 and over 0.00494; held-out Q2-B evidence only |
| Q2-D Stage 3 over-amplitude correction | canonical Gain -> returns/advantages -> controlled PPO mean update on fixed Q2 evidence | S1/S2 then S4 `T-sweep/T-sign/T-credit/T-gradient/T-direction/T-identity` | partial | Q-E20/Q-E23: offline evaluator, transaction identity, and official pre-update credit capture are implemented; live tuple evidence remains |
| Q2-D1 Scale-sweep executor | dedicated Q2-D module + canonical lower-level quality owners | S1/S2 `T-scale/T-order/T-state/T-schema/T-isolation` | completed offline | Q-E20: six sorted scales restore identical state, reuse canonical owners, and remain isolated from old eval control flow |
| Q2-D2 Credit/mean oracle | storage-derived advantage + Gaussian score-gradient + cloned update delta | S1/S2 `T-sign/T-source/T-gradient/T-direction/T-no-mutation` | completed offline | Q-E20: score-gradient sign and canonical Segment PPO clone update move mean toward preferred sampled action without source mutation |
| Q2-D3 Real failed-sample causality | scaled-HSL physical Gain + real Stage 3 batch credit/update | S4 `T-live/T-gain/T-credit/T-update/T-identity` | partial | Scale ordering observed but old artifact identity was UNCONFIRMED; Q-E23 closes offline wiring, one transaction-complete scale route and one official credit artifact remain |
| Q2-E Double Segment Replay transaction | fixed old policy -> repeated on-policy attempts per Segment -> cross-Segment batch -> one PPO step | S0/S2 `T-order/T-role/T-state/T-connect` | retained mismatch baseline | Q-E24: current path has one policy row per Segment and calls optimizer inside each sampler step; the K-A implementation plan supersedes this audit step |
| Method v015 active semantics | future q29 intent + single first action + frozen-FEMR Clean-continuation K evidence | S0 `T-design/T-identity/T-boundary` | active semantic closure; implementation not started | `contracts/active/method/FRS-METHOD-v015-future-intent-single-action-k-replay.md` and v015 checklist |
| Concept Figure, v015 intent | `FRS-DP-10 / M-11` Future Motion Context and `FRS-DP-06/07 / Q-PAIR/Q-01` two-role intent Gain | S0 `T-map/T-contract` | Concept Figure/runtime map synchronized; code is contract-mismatch | `architecture/concept/03_frontres_concept_tabs.data.json` and runtime/02_frontres_flow.data.json |

## Phase B Runtime Closure Index

E70 closes the formal runtime prerequisite without making a quality claim.
The permanent Runtime Atlas retains these synchronized owner IDs:

`AUDIT-ROUTE-01`, `AUDIT-PERTURB-01`, `AUDIT-PERTURB-02`,
`AUDIT-SEGDATA-01`, `AUDIT-SAMPLER-01`, `AUDIT-KPLAN-01`,
`AUDIT-RESET-LIFECYCLE-01`, `AUDIT-ANCHOR-Z-01`, `AUDIT-KROLLOUT-01`,
`AUDIT-OBS-01`, `AUDIT-ACTION-01`, `AUDIT-APPLY-01`, `AUDIT-GMT-01`,
`AUDIT-PAIR-01`, `AUDIT-PAIR-EVIDENCE-01`, `AUDIT-GAIN-01`,
`AUDIT-RETURN-01`, `AUDIT-HSL-LOAD-01`, `AUDIT-WARMUP-01`,
`AUDIT-PPO-01`, `AUDIT-PERSIST-01`, and `AUDIT-DIAG-01`.

Current acceptance: E67-E70 establish transaction identity, mixed-K and
warmup behavior, resume continuity, full-weight joint PPO, frozen GMT, and
complete persistence. Policy efficacy remains unconfirmed and begins at Q1.

## Hard Isolation Gates

- [x] Existing periodic eval source behavior is untouched through Q1-E.
- [x] Existing offline eval source behavior is untouched through Q1-E.
- [x] Existing sequence eval source behavior is untouched through Q1-E.
- [x] Existing `train`, `sequence_eval`, and offline modes never call the new
  quality owner through Q1-E.
- [x] Quality mode never calls Segment sampler sampling or PPO optimizer step
  through Q1-E; it calls only the dedicated named-owner executor.
- [x] Quality evaluator does not restore checkpoint sampler/warmup/optimizer
  state through Q1-E.
- [x] No old evaluator imports the quality evaluator through Q1-E.
- [x] Stage 3 defaults and checkpoint schema are unchanged through Q1-E.

## Q1-A Manifest Matrix

| Invariant | Test kind | Status |
| --- | --- | --- |
| same semantic item -> same signature | S1 `T-hash` | completed, Q-E1 |
| motion/frame/perturbation/K/seed change -> signature changes | S1 `T-metamorphic` | completed, Q-E1 |
| row permutation preserves keyed result identity | S1 `T-permute` | completed, Q-E1 |
| manifest is immutable after creation/load | S1 `T-immutable` | completed, Q-E1 |
| checkpoint/sampler state cannot alter manifest selection | S2 `T-isolation` | completed, Q-E1 |
| missing or duplicate identity fails closed | S1 `T-missing/T-schema` | completed, Q-E1 |

## Q1-B State Matrix

| Invariant | Test kind | Status |
| --- | --- | --- |
| preroll uses zero FrontRES, not tested policy | S2 `T-route` | completed, Q-E2 |
| scoring state includes root/joint pose and velocity | S1/S2 `T-state` | completed, Q-E2 |
| command/reference/correction caches are captured | S2 `T-cache` | completed, Q-E2 |
| episode lifecycle, origin, frame/K/perturbation are captured | S2 `T-role/T-frame` | completed, Q-E2 |
| restore reproduces all fields and state hash | S2 `T-restore` | completed, Q-E2 |
| relevant RNG state is restored or explicitly isolated | S2 `T-RNG` | completed, Q-E2 |

## Q1-C Counterfactual Matrix

| Invariant | Test kind | Status |
| --- | --- | --- |
| zero route executes exact 6D zero action | S1/S2 `T-value/T-shape` | completed, Q-E3 |
| HSL route uses frozen model_200 actor, not supervised target | S2 `T-source/T-frozen` | completed, Q-E3 payload contract |
| policy route uses tested checkpoint actor | S2 `T-source/T-persist` | completed, Q-E3 payload contract |
| observation layout/normalizer identity is explicit | S2 `T-layout/T-persist` | completed, Q-E3 |
| all routes start from identical state/signature | S2 `T-identity/T-restore` | completed, Q-E3 |
| canonical action application and Gain owners are reused | S2 `T-forward/T-connect` | callback contract complete Q-E3; formal wiring Q1-D |
| optimizer/sampler/warmup state is unchanged | S2 `T-isolation/T-state` | completed, Q-E3 |

## Q1-D/E Integration Matrix

| Invariant | Test kind | Status |
| --- | --- | --- |
| dedicated `policy_quality_eval` dispatch only | S0/S2 `T-route/T-mode` | completed, Q-E4 |
| old modes have zero quality-owner calls | S2 `T-no-call` | completed, Q-E4 |
| quality mode has zero old-eval/training calls | S2 `T-no-call/T-isolation` | completed, Q-E4 |
| thin connectors contain no evaluation logic | S0 `T-static` | completed, Q-E4 |
| 8 Quality Atlas cards map to source/checklist IDs | S0/S2 `T-link/T-schema` | completed, Q-E5 |
| focused contracts and aggregate suite pass | S2 `T-regression` | completed, Q-E6; 51/51 includes formal real-owner installation |

## Live Gate

No live command may be issued until Q1-A through Q1-E are complete and
user-reviewed. The first live run proves only state/signature equality and
route isolation. It does not prove policy superiority, Gain correctness,
generalization, or long-training readiness.

## Current Decision

Q1/Q2 remain evidence history. Q-E24 is the code-confirmed baseline for the
superseded v013 transaction migration. The active implementation gate is now
the v015 plan/checklist pair; no code, test, live run, or long-training
decision is authorized by this historical evidence checklist.
