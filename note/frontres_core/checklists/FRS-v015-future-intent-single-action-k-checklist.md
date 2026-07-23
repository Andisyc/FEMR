# FRS-v015 K-Stage Critic Curriculum Acceptance Checklist

Status: active, volatile acceptance surface. Updated: 2026-07-23.

| Step | Owner / tier | Acceptance assertion | Status | Evidence / stop |
| --- | --- | --- | --- | --- |
| C0 | governance | Training v009, Method v015, M-06/M-05, registry, plan, Architecture and evidence agree on global K stages and one Critic | completed | E-FI-69; JSON/reference/diff validation passed |
| C0 | source audit | current per-Segment `8/16/32/64` planner, global v008 warmup, and checkpoint v3 mismatch are named and isolated | code-confirmed | owner audit 2026-07-22 |
| C1 | S1 schedule | strictly increasing explicit K schedule maps every boundary to exact stage/K/local iteration/phase/actor weight | completed | E-FI-70 pure curriculum contract |
| C1 | S1 failure | invalid K/order/duration/final-stage/schedule fingerprint rejects deterministically | completed | explicit CLI schedule; no formal default |
| C2 | S2 K identity | every Segment x M row in one formal transaction has the same active K and schedule fingerprint | completed | E-FI-70 mixed-K fail-closed |
| C2 | S2 transition | stage advances only after committed exact-one update; next transaction is new-K critic-only | completed | actor/std delta 0, Critic delta nonzero |
| C2 | S2 isolation | per-Segment adaptive K cannot reach v009 formal storage/return/PPO/diagnostics | completed | formal planner overrides K; legacy remains ablation-only |
| C3 | S3 persistence | checkpoint v4 binds v009 schedule/stage/K/local iteration/phase and resumes exactly | completed | strict save/fresh reload |
| C3 | S3 rejection | v008, unversioned, different schedule, mixed-K, collecting/failed transaction rejects before mutation | completed | pre-mutation S3 contracts |
| C4 | S4 official | bounded 8-env official route crosses one K boundary with homogeneous transactions, new-K critic-only isolation, exact-one updates and committed v009 save | completed | E-FI-71; exact four-phase order and model_1--model_4 v4/v009 identities |
| Quality | Q | final-K label learnability, unsafe-tier target distinguishability, Critic calibration, checkpoint trajectory and long-training admission | blocked by E-FI-71 quality evidence | 0/16 admissible Repairs, 0/16 positive Gain, 3/16 positive raw advantages |

## Pass Rule

Critic Curriculum passes engineering only when C1-C4 agree on the same schedule
fingerprint and formal transaction identity. A deterministic scheduler or one
K=8 run cannot prove K transition behavior.

Engineering passage does not authorize long training: the live return target
must first provide meaningful paired ranking, and Critic calibration must be
separated from target saturation.

## Fail Rule

Stop on mixed-K actor credit, K-dependent actor input, Multi-Critic expansion,
transition before commit, actor update during new-K calibration, checkpoint
identity drift, or any need to change Gain/PPO/HSL/one-action-K/M semantics.
