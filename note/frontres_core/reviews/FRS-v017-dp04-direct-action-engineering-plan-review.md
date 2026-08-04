# FRS-v017 DP04 Direct-Action Engineering Plan Review

Engineering Plan Review

Mode: engineering_plan_review

Verdict: READY

Discipline: active (`FRS-ENG-v001`)

Accepted behavior/non-scope: one direct finite world-frame `[B,6]` action is
shared by HSL, Stage 3, PPO and deployment. The change does not alter 158D
observation authority, GMT composition, Gain/PPO objective mathematics,
K/M/DR curriculum, simulator behavior or policy quality.

Boundary reviewed: `FrontRESActorCritic` action owner -> HSL/rollout/PPO
consumers -> task-space writer, plus HSL-v2/checkpoint-v9 persistence identity.

Findings:

- P0: none.
- P1: none.
- P2: `front_residual_actor_critic.py` is a hotspot, but this change removes the
  bounded-coordinate responsibility rather than adding another owner.
- P3: none.

Flow/lifecycle gaps: none in the plan. Old HSL-v1 and checkpoint-v8 are required
to reject before mutable restoration. Live behavior remains a later evidence
boundary rather than an implementation dependency.

Pattern admissions accepted/rejected: no new pattern is admitted. The existing
Actor remains the semantic owner; checkpointing remains a persistence Gateway.
A wrapper, Service Layer, second action class or compatibility adapter would
remove no dependency and is rejected as Speculative Generality/Middle Man.

Proof-route gaps: none before coding. Owner, consumer, negative and persistence
tests are named. Formal Phase A must be repeated for DP04 after implementation.

External method/formal/runtime blockers: none for offline implementation.
Simulator, training and live execution remain unauthorized.

Required plan revisions: none.

Residual facts that remain unconfirmed until implementation: direct action
forwarding through all active consumers; HSL-v2/checkpoint-v9 roundtrip; old
identity pre-mutation rejection; absence of active bounded-action fallback.
