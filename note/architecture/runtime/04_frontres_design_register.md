# FrontRES Design Inspector

Status: DP07/DP09 symmetric-log utility was human-confirmed on 2026-08-10 and
activated as METHOD-v020 / GAIN-v008 / PPO-v008 / TRAIN-v019. Actor stays 158D,
Critic stays 449D and GMT stays 770D. Raw `G_total` and hard Physics evidence
remain unchanged; each attempt maps through fixed `sign(G)*log1p(abs(G))`
before Actor advantage and before the M4 Critic mean. Networks, split LR, M,
K/DR and simulator are unchanged. TRAIN-v018/checkpoint-v13 is historical;
checkpoint-v14 requires fresh runtime evidence.

Interactive page: `../02_frontres_design_inspector.html`

## Purpose

Atlas 04 is the human method-inspection surface. It answers only one question:
how does each accepted design point participate in the same Stage-3 training
Transaction?

It is not a repository reader, evidence ledger, risk register, contract browser,
or second Concept Figure.

## Interaction Model

The page contains three visual levels:

1. ten compact parent design-point names in causal order;
2. one shared Transaction spine below the index;
3. one minimal detail reading card below the spine.

The Concept Figure supporting block `M-12 Perturbation Probing` is mapped to
`FRS-DP-01P` in TRAIN-v013 and explained inside the Perturbation Data parent
card. It does not create an eleventh parent tab: it is the optional way to
acquire the frozen GMT boundary consumed by that design point.

Selecting a design point never replaces the shared spine. It only:

- highlights the steps owned or constrained by that design point;
- replaces the single bottom reading card with that design point's ordered
  atomic decisions.

All unselected steps remain visible as quiet gray context. Their explanations do
not expand. The bottom card contains no owner, code, evidence, risk, status, or
field-category headings; every numbered line is itself a method decision.

## Shared Transaction Spine

The canonical visual order is:

```text
pre-Transaction initialization
-> resolve K/M and training phase
-> select exactly two Segments
-> seal one scenario per Segment
-> restore all roles to the same Clean replay state x_t
-> sample exact M Repair actions per Segment from frozen pi_old
-> FrontRES emits one full-6D Delta SE(3) action at t
-> FrontRES remains frozen while frozen GMT executes K steps
-> construct Clean/Noisy/Repair rollout evidence
-> use Clean direction, Noisy zero point, and every Repair consequence to form
one raw Recovery-Aware Gain per attempt, then map each attempt to fixed utility
-> seal 2 x M PPO policy rows
-> form one shared 449D support-conditioned state value and mean-M utility target per Segment
-> use every attempt's utility advantage, scale only the Critic loss, clip
Actor/Critic separately, and execute exactly one grouped optimizer update
-> atomically commit checkpoint, curriculum and Critic target moments
```

The main page renders these as short Chinese action statements rather than the
English outline above.

## Design-Point Highlight Map

| Parent design point | Highlighted Transaction responsibility |
| --- | --- |
| Perturbation Data | seal one first-frame root artifact, inject no new corruption during K, and retain random Segment coverage with a soft preference only for genuinely cheaper prefix-preroll resets |
| Segment Replay | restore the same `x_t`, collect exact-M attempts for both Segments, and pass every valid attempt to grouped PPO so their different Gain values provide the current one-action reachable-frontier ordering |
| K-step Curriculum | resolve active K/M, execute K-step evidence, and advance only at committed boundaries |
| FrontRES 6D Repair | consume the deployable actor prefix and emit one full-6D `Delta SE(3)` action at `t` |
| Frozen GMT | freeze FrontRES and let frozen GMT execute the common continuation |
| Paired Rollouts | execute one Clean anchor and one fixed Noisy zero point once per sealed Segment, then read-only reuse both while evaluating M Repair rollouts |
| Repair Gain | retain raw `G_total`; transform each attempt with fixed symlog, subtract one shared state value for Actor credit, and average utilities for the Critic target |
| HSL Warmup | initialize the proposal Actor before the first Stage-3 Transaction and never use HSL as its target |
| Actor & Critic Warmup | calibrate the 449D state-value Critic on M4 utility targets, then release Actor on the same utility; retain non-amplifying loss scale and separate clipping |
| Future Motion Context | seal q29 at `t+1,t+2` plus action-pre current/planned support context for the 449D Critic while keeping Actor at 158D and GMT at 770D |

## Atomic Decisions Kept In The Primary View

Numbers and information boundaries appear only when they are part of the method
decision itself. They are not rendered as separate metadata chips:

- `exactly two Segments`;
- Segment selection remains stochastic. When reaching `x_t` requires prefix
  preroll, estimated lower-preroll-cost Segments receive a soft preference;
  every valid Segment keeps nonzero probability, and direct cached-state reset
  does not penalize a late start frame;
- TRAIN-v013 treats 2.381 as the already measured maximum reliable perturbation
  boundary for the current frozen GMT, robot and perturbation definition. The
  current campaign configures it directly; a changed setup may obtain a new
  value through an optional offline frozen-GMT Noisy-only survival probe. That
  probe only acquires and freezes the outer boundary: it is not an online
  controller, a per-K `g_K`, or a Gain/PPO feedback path. For the current
  explicit stage-local
 ceiling `d_cap`, the restored strength mixture is
 `Easy 20%: [0, 0.25d_cap)`, `Medium 30%: [0.25d_cap, 0.70d_cap)`,
 `Hard 40%: [0.70d_cap, d_cap]`, and
 `Broken tail 10%: (d_cap, min(1.10d_cap, 2.381)]`. These are perturbation
  strength classes, not the separate Safe/Repairable/Broken/Harmful execution
  outcomes. The four classes preserve basic restoration, ordinary learning,
  frontier learning, and limited failure-boundary exposure respectively;
 `d_cap` comes from explicit committed stage progress and approaches the frozen
  boundary, not online survival or Gain feedback; 2.381 is neither a mastery
  condition nor a per-K `g_K`;
- the active campaign uses `K8/M4 -> K16/M4 -> K32/M4` as the outer
  curriculum. Each K owns an inner lower-to-higher DR curriculum;
- at each K transition, the committed state is preserved, DR returns to a
  lower but still informative distribution, Actor/std freeze, and the same
  Critic recalibrates before Actor-ramp and Joint Optimize resume;
- same-scenario Repair ordering across K remains diagnostic-only. It does not
  force the next K to inherit the previous stage's high DR distribution;
- one sealed Segment samples strength once. Its Noisy and M Repair rollouts
  share that strength and artifact; Clean remains uncorrupted, and Gain/PPO do
  not control the curriculum;
- active schedule `K8/M4 -> K16/M4 -> K32/M4`;
- `K64` is not active under `FRS-TRAIN-v013`;
- each sealed Segment owns one Clean rollout, one zero-action Noisy rollout,
  and M Repair rollouts; only the M Repair rows enter PPO. Runtime env packing
  is engineering orchestration rather than a method-level `4 x M` identity;
- one action at `t`, then FrontRES frozen through K;
- deployment uses no-feedback composition: every per-frame residual is applied
  to the current frame of the sealed Noisy/deployment stream. Physical state
  continues across steps, but a repaired reference is never written back as
  the next frame's actor input or reference base;
- `Delta SE(3)` translation and rotation are world-frame residuals. Quaternion
  composition therefore uses
  `q_repaired = Exp(Delta theta_world) * q_noisy` (left multiplication). This
  remains an implementation convention derived from the world-frame meaning,
  not another primary Atlas variable;
- the full-6D Actor uses no perturbation-family action mask, per-axis scale,
  `tanh`, `clip`, or `clamp`. Upward `dz` is softly discouraged by the HSL
  initialization, Clean-anchored K-step consequence, and full-6D repair cost;
  without a one-sided projection this is deliberately not claimed as a hard
  prohibition;
- full-6D repair cost converts translation and rotation into fixed semantic
  units before combination:
  `sqrt((||Delta t||/0.10m)^2 + (||Delta theta||/5deg)^2)`; it prefers the
  smaller intervention when rollout outcomes are comparable and does not
  decide whether Physics or Intent improved;
- zero optimizer steps during collection;
- exactly one grouped optimizer update after complete sealing;
- every valid Repair attempt remains one equal-structure-mass PPO row; ordering
  comes from its own `G_total` and advantage, not winner-only selection,
  argmax, best-of-M weighting, or replay priority;
- full observation `928D`, FrontRES Actor prefix `158D`, FrontRES Critic state
 `449D`, frozen GMT suffix `770D`;
- the actor reads two future 29D internal-Intent frames, `q29[t+1]` and
  `q29[t+2]`, from the same sealed Noisy/deployment reference;
- those two frames contribute `58D` to the `158D` Actor input and to the
 `289D + 58D + 102D = 449D` state-value Critic input. The 102D block contains
 current actual Contact/load/ZMP plus 32 planned-support pairs and their valid
 mask; Repair-after evidence, `G_total` and the 6D Repair action remain excluded;
- H supplies the two-frame Actor/Critic context; K remains the executable-evidence
horizon.
- Clean Rollout is evaluator-only phase and demo-quality evidence; it does not
  become an actor input or PPO row.
- Held-out Policy Quality loads the tested checkpoint-v14 Actor, 449D Critic
  and 449D privileged-observation normalizer inside one reversible inference
  scope. It reports every raw Gain and compares the shared Segment value with
  the exact-M4 utility mean; the value-loss normalizer remains output-preserving
  and is not applied to `V(s)`.
- each sealed Segment executes one Clean Rollout and one fixed zero-action Noisy
  Rollout exactly once; both observed K-step outcomes are then sealed and
  read-only reused across all M Repair comparisons;
- Noisy artifact, corruption protocol, application point and hash do not change
  across M attempts. Shared baselines remove avoidable resampling noise but do
  not claim complete cancellation of simulator dynamics noise;
- Gain is divided into Intent Gain and Physics Gain; their interaction inside
  `G_total` defines Recovery-Aware. Segment Replay preserves all valid
  same-scenario attempts and lets their different `G_total` advantages carry
  the ordering.
- Clean gives the desired motion direction, Noisy defines zero improvement, and
  M Repair outcomes expose the current policy's empirical one-action reachable
  frontier. Clean equality is not a positive-Gain threshold.
- Gain evaluates each attempt; Segment Replay exposes the currently reachable
  direction through all candidate rows rather than selecting only the winner.
- each evidence term uses `r_j(X|Clean) = D_j(X,Clean) / S_j`; `D_j` is a
  channel-specific remaining problem and `S_j` is a fixed semantic unit;
- continuous `D_j` accumulates the complete K-step consequence with fixed
 `tau_k=k/K` position weights and divides by weighted applicable exposure; this
 distinguishes recovery from deterioration without adding a tunable time
 parameter;
- Contact and survival retain event/exposure semantics, so early illegal
 support changes or failure cannot be discounted by the time-position weight;
- weighted-effective-time normalization prevents Intent from scaling
 approximately with `K` and pressure-weighted Physics from scaling
 approximately with `K^2` while one-action repair cost remains unchanged;
- Intent retains root orientation, joint pose, drift-removed key-body pose,
  root-local linear/angular velocity and root height; acceleration is initially
  diagnostic because finite-difference noise is high;
- Physics retains expected Contact-phase mismatch, loaded support-foot drift,
  loaded-support phase-ZMP envelope violation and survival lost-horizon
  fraction;
- sustained lean without extra support compensation belongs to Intent; extra
  stepping, dragging and changed support belong to Physics;
- fixed-scale normalization aligns unlike units, preserves severity across
  Segments, avoids near-zero recovery-ratio denominators and candidate-derived
  moving standards, and does not clip severe states to `[0,1]`;
- Intent sensitivity scales approximately with `1/S_I`, while the
  pressure-weighted Physics contribution scales approximately with `1/S_P^2`;
  each final fixed scale must remain physically interpretable and constant
  across Segments, attempts and K stages;
- normalized channels are aggregated separately inside Intent and Physics with
 `M(z)=log(mean(exp(z)))`; one severe item increasingly controls its family,
 while every retained item remains visible and the `1/n` baseline is invariant
 to family size;
- the older raw difference was refined first into two Clean-conditioned
 semantic evaluations and then from a per-scenario recovery ratio into the
 fixed-scale, family-level difference `G_I=I_N-I_R` and `G_P=P_N-P_R`;
- Clean defines the correct motion semantics, Noisy is the no-action zero point,
  and Repair supplies the candidate consequence, so positive Gain requires only
  improvement over Noisy rather than equality with Clean;
- aggregate Physics pressure uses `lambda_RA = (P_N + P_R) / 2`, giving the
 same Physics change more weight in an imbalanced state and less weight near
 stability;
- `lambda_RA * G_P = (P_N^2 - P_R^2) / 2`; the two endpoints both reward
 severe recovery and penalize a Repair that creates a new severe imbalance;
- the fixed semantic units are: root orientation `0.087 rad`, joint pose
  `0.087 rad`, key-body pose `0.10 m`, linear velocity `0.75 m/s`, angular
  velocity `2.0 rad/s`, root height `0.05 m`, Contact `0.10 exposure`,
  support-foot drift `0.03 m`, phase-ZMP `0.02 m`, and survival
  `0.10 horizon fraction`; they remain constant across Segments, attempts, and K;
- `C_repair = sqrt((||Delta t||/0.10m)^2 + (||Delta theta||/5deg)^2)` is the
  magnitude of the single full-6D action in fixed semantic repair units;
  `beta` says how much extra recovery must justify one extra repair unit, and
  the first bounded live calibration uses provisional global `beta_init=0.02`;
- bounded live telemetry retains cost-free recovery `R=G_I+lambda_RA G_P`,
  `C_repair`, `beta*C_repair`, `G_total`, and within-Segment rank changes;
- only same-Segment trade-off pairs where greater recovery also costs a larger
  action define positive break-even values `beta* = Delta R / Delta C`; a
  recovery-superior, no-more-expensive attempt is dominant rather than a beta
  calibration pair;
- live evidence never mutates beta inside a run. Human review may revise the
  single global value between bounded calibration runs; once accepted, it is
  frozen across Segments and K rather than becoming a per-stage controller;
- `G_total = G_I + lambda_RA G_P - beta C_repair` is the complete
  Recovery-Aware candidate score consumed by Segment Replay ranking;
- raw `return_K=G_total` remains diagnostic; training uses
 `U(G)=sign(G)*log1p(abs(G))` per attempt, Critic target `mean_m U(G_m)`, and
 Actor advantage `U(G_m)-V_old(s)`;
- Actor and Critic gradients are clipped independently at 0.5, then the two
 named LR groups still execute exactly one Adam step and persist as
 checkpoint-v14;
- Contact phase, support-foot drift, phase-ZMP and survival remain fail-closed
  Physics evidence, but their learning route is `P_X -> G_P -> G_total`; the
  old independent constraint projection and KKT actor gate retire rather than
  operating in parallel;
- actual no-load with expected support is handled by the evaluator as a Contact
  violation while phase-ZMP is `N/A`; this applicability branch is engineering
  logic, not another primary Atlas decision;
- exact extra RNG-restoration and persistence mechanics remain in the
  contract/engineering layer. The Atlas states the direct `HSL -> HRL` role
  transition rather than inventing a separate Actor-migration concept.

Everything else belongs in active contracts, method-to-code maps, runtime audit,
or evidence ledgers.

## Language Contract

Chinese owns the sentence. English remains only for established method objects,
identifiers, and variables.

Preferred forms:

- `冻结的 GMT 执行同一条 Clean continuation`;
- `从同一 pi_old 采样 exact M 个修复动作`;
- `完整封存后执行一次 grouped update`;
- `FrontRES 在 t 仅输出一次 Delta SE(3)`.

Forbidden style:

- a Chinese half-sentence joined to an English predicate;
- an English evidence paragraph inside the primary Transaction view;
- repeated qualifiers such as identity, provenance, owner, evidence level, and
  open risk when they do not change the visible Transaction;
- translating stable symbols in a way that makes them harder to match to the
  contracts.

## Content Removed From The Primary Card

- Implementation Evidence;
- Open Risk;
- source and contract footers;
- code-owner links;
- separate Facts, Matrix, Authority, Formula, Dependencies, and Review panels;
- repeated scientific-problem and owned-object paragraphs;
- global mapping-gap banners.

These facts remain in their authoritative documents. Removing them from Atlas 04
does not delete or supersede them.

## Acceptance

- the ten index buttons remain visually compact and each names one canonical
  parent design point;
- exactly one shared Transaction spine is visible;
- selecting any button preserves the same step order;
- every index button remains a short parent design-point name;
- selecting it changes only Transaction highlights and the one bottom detail
  reading card;
- the affected cards explicitly show Clean/Noisy/Repair's distinct evaluator
  roles and present only the human-confirmed active Gain formula;
- the Paired Rollouts card shows one Clean and one fixed Noisy execution per
  Segment, immutable reuse across M, and the accepted residual-noise boundary;
- the affected cards distinguish Clean direction, Noisy zero point, per-attempt
 Gain evidence, and M-attempt reachable-frontier search;
- the Repair Gain card distinguishes recovering from deteriorating K-step
 trajectories without discounting early Contact/survival failures;
- the Repair Gain card shows separate smooth worst-item aggregation for Intent
 and Physics before group-level improvement and Recovery pressure;
- the Repair Gain and Actor & Critic Warmup cards show per-attempt `G_total` as
 Actor evidence, the exact-M mean as the shared state-value target, and the old
 independent Physics projection as retired;
- the bottom card contains four to eight numbered atomic decisions and no
  implementation/evidence panels;
- Segment Replay visibly covers same-`x_t` reset, exact-M collection, zero updates
  during collection, and exact-one grouped update. Its human-facing explanation
  says that motions, Segments and attempts receive equal voting weight, so a
  group cannot dominate merely because it contains more rows;
- K-step Curriculum visibly shows K8/M4, K16/M4, K32/M4 and concise K64 inactive
  status;
- K-step Curriculum shows one Clean, one Noisy, and M Repair evaluations per
  Segment without turning runtime env packing into a method identity;
- Perturbation Data preserves random Segment coverage while expressing only a
  soft prefix-preroll efficiency preference, and it shows the preserved
  frozen-GMT frontier-envelope strength curriculum;
- FrontRES 6D Repair exposes world-frame full-6D semantics, no-feedback
  per-frame deployment composition, and the soft upward `dz` treatment without
  hiding a hard clip, mask, or scale;
- HSL is visibly pre-Transaction rather than a per-Transaction operation;
- Actor & Critic Warmup states the direct `HSL -> HRL` transition, the 449D
 state-value input, M4 symlog-mean target, separate gradient clipping and
 checkpoint-v14 cold-start boundary;
- the `Future Motion Context` detail card explicitly states `t+1,t+2`,
  `29D x 2 = 58D`,
 extraction from one fixed deployment Noisy reference, Actor/Critic reuse, and
 exclusion of future root/global, Clean/evaluator and 6D action information;
- primary prose follows the Chinese-sentence language contract;
- the full page is readable at default browser zoom without vertical scanning
  through multiple long panels.
