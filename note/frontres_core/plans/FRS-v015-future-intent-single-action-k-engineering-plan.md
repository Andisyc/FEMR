# FRS-v015 K-Stage Critic Curriculum Engineering Plan

Status: active, volatile engineering plan. Updated: 2026-07-22.

## Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Method: `../contracts/active/method/FRS-METHOD-v015-future-intent-single-action-k-replay.md`
- Training: `../contracts/active/training/FRS-TRAIN-v009-k-stage-critic-curriculum.md`
- Gain/PPO: FRS-GAIN-v004 / FRS-PPO-v003, unchanged
- Checklist: `../checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- Evidence: `../../testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`

## Preserved Foundation

G0-G5-P1 and `E-FI-0--E-FI-68` remain valid for local scenarios, q29 future
intent, 928/158/770 authority, two roles, one action, K-step frozen-GMT
evidence, multi-Segment x M atomicity, grouped PPO, v004 Physics Gain,
proposal-only HSL, and one bounded critic-only update. No prior evidence proves
K-stage Critic Curriculum.

## Original Contract Mismatch And Closure

The confirmed method treats K as a global curriculum stage. Before E-FI-70,
the source let `frontres_segment_sampler.py::plan_rollout_budget()` assign
`8/16/32/64` from each Segment's replay state, so different horizons may reach
one scalar Critic without K input. `frontres_segment_warmup_phase()` indexes one
global v008 warmup only and does not re-enter critic-only when K changes.
Checkpoint v3 binds v008/global iteration but not a K schedule, stage, or local
phase. E-FI-70 closes this mismatch on the v009 formal route through an
explicit schedule, homogeneous-K override, stage-local recalibration and
checkpoint v4. C4 live transition and long training remain blocked.

## Source Of Truth

| Semantic object | Active owner after implementation | Legacy path | Isolation rule |
| --- | --- | --- | --- |
| K curriculum schedule | existing `frontres_segment_warmup.py` pure schedule kernel | per-Segment K in `plan_rollout_budget()` | formal v009 ignores/rejects segment-owned K |
| active K stage | formal runner before transaction selection | global iteration-only v008 phase | one immutable stage identity per transaction |
| Critic target | existing v004 return/storage/PPO path | mixed `return_K` targets | every row in a transaction has one `active_k` |
| transition | committed-update cursor | state change during collection | advance only after committed receipt |
| persistence | `frontres_checkpointing.py` | checkpoint v3/v008 | checkpoint v4/v009 exact schedule fingerprint |

## Step Map

### C0 (Preparatory): Contract And White-Box Rebase

Objective: freeze the global K-stage/single-Critic semantics and locate the
current mixed-K mismatch.

Scope: Training v009, Method v015 clarification, Concept Figure M-06/M-05
interaction, registry, plan, checklist, canvas, Architecture, and evidence.

Non-scope: source code, tests, checkpoint I/O, simulator, training, live run.

Evidence: E-FI-69 plus exact owner/shape/identity audit.

Stop: any unresolved choice about global versus per-Segment K, Multi-Critic,
K actor input, or final objective horizon.

Why separate: this is the human semantic and contract-version boundary.

### C1 / 4: Pure Curriculum Kernel And Config Identity

Objective: implement immutable schedule validation and absolute iteration ->
`(stage, active_k, stage_iteration, phase, actor_weight)` mapping.

Scope: existing `frontres_segment_warmup.py`, config dataclasses, Stage3 CLI
parsing, and focused pure contracts. The schedule explicitly carries ordered
`(K,N_c,N_a,N_joint)` stages; the final stage remains joint.

Non-scope: sampler/reset/storage/PPO/checkpoint mutation, simulator, live run.

Evidence: S1 boundary, invalid-order, final-stage, permutation, and deterministic
fingerprint tests; v008 global scheduler remains historical only.

Stop: implicit schedule defaults, non-monotonic K, zero recalibration, or a
need for multiple Critics.

### C2 / 4: Formal Homogeneous-K Transaction And Phase Update

Objective: connect the C1 stage identity to official selection, sealed
transaction metadata, v004 return, and the existing exact-one grouped update.

Scope: existing live sampler/formal request/update/diagnostic owners. Every
selected Segment and M attempt receives `active_k`; per-Segment adaptive K is
rejected on v009 formal training. Stage/phase cannot change while open.

Non-scope: Gain/PPO formula, actor/Critic architecture, M semantics, HSL,
checkpoint format, simulator/live run.

Evidence: S2 fake official-entry connectivity across one K transition;
old-stage commit then new-stage critic-only, equal group mass, actor/std zero
delta, Critic nonzero delta, exact-one update, mixed-K fail-closed.

Stop: K cannot be made transaction-homogeneous, stage change precedes commit,
or grouped PPO needs cross-K weighting.

### C3 / 4: v009 Persistence And Fresh Resume

Objective: create checkpoint v4 identity for exact curriculum resume.

Scope: existing checkpoint owner and save connector; bind schedule fingerprint,
stage index, active K, local iteration, phase, absolute committed update, and
v004/v003/layout identities.

Non-scope: checkpoint payload expansion beyond existing training state,
actor-only Stage3 migration, simulator/live run.

Evidence: S3 save/reload/pre-mutation tests; exact resume equality; v008,
different schedule, mixed-K, and partial transaction rejection.

Stop: resume can restart a K stage, cross phase, or mutate before identity
validation.

### C4 / 4: Bounded Official K-Transition Sentinel

Objective: prove the official Stage3 route crosses exactly one K boundary.

Scope: one explicit small engineering schedule, 8 envs, enough transactions to
commit the old stage and enter new-K critic-only, one update per transaction,
committed v009 checkpoint and synchronized evidence/Architecture closeout.

Frozen bounded schedule: `8:1:1:1,16:1:1:0`, four formal iterations and
checkpoint interval one. Expected order is K8 critic-only, K8 actor-ramp, K8
joint, then K16 critic-only. `model_3.pt` must expose the next K16
critic-only identity; the fourth receipt must prove K16 actor/std zero delta
and Critic nonzero delta before `model_4.pt` advances to actor-ramp.

Non-scope: policy-quality acceptance, long training, multiple seeds,
deployment composition, paper experiments.

Evidence: S4 stage/K/phase identity, expected/actual K, actor/std zero delta at
new stage, Critic nonzero delta, exact-one counts, checkpoint v4/v009 receipt.

Stop: mixed K, missing stage identity, actor drift, no Critic update, transition
before commit, resume mismatch, or any legacy v008 path.

Why separate: this is the only simulator/material-cost boundary.

## Post-C4 Boundary

After C4, `formal-runtime-audit` decides runtime closure and
`policy-quality-audit` owns final-K label learnability, checkpoint trajectory,
no-op/harm detection, and long-training admission. Short-K success cannot
authorize long-sequence or deployment claims.

## Cursor

Current cursor: `C1-C3 complete with E-FI-70 deterministic evidence; stop at
C4 / 4 bounded official K-transition live gate`.
