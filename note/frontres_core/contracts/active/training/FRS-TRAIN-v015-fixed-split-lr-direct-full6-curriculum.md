---
contract_id: FRS-TRAIN-v015
status: active
effective_date: 2026-08-07
updated_date: 2026-08-07
supersedes: FRS-TRAIN-v014
scope: Direct full-6D HSL-to-HRL Recovery-Aware scalar training with one fixed split-LR Adam, coordinated K x exact-M schedule, per-K lower-to-higher DR curriculum, Critic recalibration, actor ramp, grouped scalar PPO, and strict checkpoint-v10 persistence
---

# Fixed Split-LR Direct Full-6D Nested K-DR Training Curriculum

## Design Delta

FRS-TRAIN-v013 named the full-6D world-frame action but retained an older
implementation identity in which HSL and Stage 3 interpreted Actor outputs
through `tanh` and separate position/rotation scales. That transform changes
the meaning of the Actor output and contradicts FRS-METHOD-v017. It is retired.
FRS-TRAIN-v015 preserves the v014 direct action coordinate throughout:

```text
158D actor prefix -> finite raw [B,6] Delta SE(3)
-> HSL target/loss or Stage-3 Normal distribution
-> PPO storage/log-prob -> command representation conversion -> frozen GMT
```

No action mask, per-axis action scale, `tanh`, `clip`, or `clamp` may transform
the active full-6D proposal. Numerical domain protection inside unrelated
quaternion mathematics and PPO's own clipped objective does not define or
bound the action and remains outside this rule.

FRS-TRAIN-v012 required a separately calibrated and frozen `g_K` before each K
stage, but did not define its statistic, observation budget, or freeze rule.
That requirement is retired. Frozen-GMT survival measures baseline difficulty;
it cannot determine in advance when a learned Repair needs a longer K to expose
its consequence.

FRS-TRAIN-v015 preserves the proven K8/M2 -> K16/M3 -> K32/M4 transaction
schedule and gives each K an inner, deterministic lower-to-higher DR curriculum.
At every K transition, DR returns to the configured lower informative
distribution while the same Critic recalibrates. In the current GMT, robot and
perturbation setting, the maximum reliable frozen-GMT perturbation boundary was
measured experimentally at `dr_scale=2.381`. The current campaign may configure
that known boundary directly. It is an outer training-support boundary, not a
mastery threshold, graduation test or precomputed per-K frontier.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `M-02` | Per-K Inner DR Curriculum |
| `FRS-DP-01P` | Perturbation Probing | `M-12` | Optional GMT Boundary Acquisition |
| `FRS-DP-02` | Segment Replay | `SR-01` | Exact-M Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Nested K x M x DR Schedule |
| `FRS-DP-08` | HSL Warmup | `M-03` | First Entry From HSL |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Critic Recalibration And Actor Ramp |

## Preserved Method Authority

- actor input remains the deployable 158D prefix;
- actor output remains one world-frame full-6D `Delta SE(3)` at `t`;
- one attempt remains one PPO policy row regardless of K;
- one Segment executes one Clean, one fixed zero-action Noisy, and exact M
  Repair rollouts;
- each transaction selects exactly two Segments and performs exactly one grouped
  optimizer update after complete sealing;
- H=2 remains actor context and K remains executable-evidence horizon;
- Clean continuation remains frozen-GMT/evaluator-only evidence;
- FRS-GAIN-v007, FRS-PPO-v005, HSL-v2 direct-action identity, beta, full-6D cost, single-`local_rp`,
  grouped equal mass, and no-feedback deployment remain unchanged.

No rho, second actor/Critic/optimizer, Clean actor future, K actor input, Noisy
physical prefix, noise label, perturbation timing, online adaptive curriculum,
Gain/PPO-to-sampler feedback, independent Physics projection, or KKT actor gate
is introduced.

## First Entry From HSL

A fresh campaign may initialize only from strict
`frontres-v017-hsl-proposal-v2`:

```text
restore proposal actor parameters, full-6D distribution/std and 158D normalizer
fresh-initialize Recovery-Aware scalar Critic, optimizer and sampler
resolve and seal TRAIN-v015 curriculum identity
enter K8/M2 critic_only with actor_loss_weight=0
```

The accepted HSL-v2 file retains its frozen `FRS-TRAIN-v014` artifact identity.
`FRS-TRAIN-v015` identifies the fresh Stage-3 campaign and checkpoint-v10; it
must not be written into, required from, or used to migrate the HSL artifact.

HSL initialization and strict Stage-3 resume are mutually exclusive. Old
HSL-v1 artifacts and Stage-3 checkpoint-v8 artifacts reject before mutation.
Old Stage-3 Critic, optimizer, transaction or curriculum state cannot initialize
a fresh v015 campaign.

## Scalar Critic Authority

At active K stage `j`:

\[
V^{RA}_j(o_{critic,t})
\approx \mathbb E[G_{\mathrm{total},K_j}\mid o_{critic,t}],
\qquad
A^{RA}_{K_j}=G_{\mathrm{total},K_j}-V^{RA}_{old,j}(o_{critic,t}).
\]

There is one scalar Critic and no K-specific head. It predicts the complete
FRS-GAIN-v007 return. K changes the consequence horizon and therefore triggers
recalibration; M only changes the number of same-scenario candidates.

## Coordinated K x M Schedule

```text
stage 0 = (K=8,  M=2, critic_only=200, actor_ramp=500, joint_review=1300)
stage 1 = (K=16, M=3, critic_only=300, actor_ramp=300, joint_review=900)
stage 2 = (K=32, M=4, critic_only=400, actor_ramp=300, joint_review=625)
```

K is strictly increasing; M is at least two and non-decreasing. K64 is not
active. Each transaction contains exactly two Segments, `2*M` policy rows and
`4*M` role rows, requiring 8, 12 and 16 env respectively. Replay priority may
select Segment identity only; it cannot change K, M, DR state, group mass or an
open transaction.

## Per-K Inner DR Curriculum

### Optional GMT Boundary Acquisition

The curriculum requires one frozen upper support boundary for the current GMT,
robot and perturbation definition. For the current campaign that boundary is
the already measured value `2.381`, so no additional Probing run is required.
When the GMT checkpoint, robot or perturbation scale changes, an optional
offline Perturbation Probing pass may estimate the replacement boundary from a
frozen-GMT Noisy-only survival sweep. The resulting scalar is frozen before
Stage 3 and supplied as the same `reference_ceiling` configuration value.

Probing only acquires the boundary. It does not train FrontRES, decide when K
changes, create a per-K `g_K`, adapt DR online, or receive Gain/PPO/evaluation
feedback. Direct configuration and offline Probing therefore lead to the same
downstream curriculum.

Each K stage owns one explicit immutable DR schedule identity:

```text
DRStageSpec = {
  K,
  start_distribution_id,
  advance_rule_id,
  advance_updates,
  reference_ceiling = 2.381,
  class_boundaries = (0.25, 0.70, 1.00, 1.10),
  class_weights = (0.20, 0.30, 0.40, 0.10),
}
```

The schedule owner resolves a current stage-local Hard-class ceiling `d_cap`
from that frozen spec and committed local progress. `d_cap` is not an estimated
`g_K`, an online outcome statistic, or a learned controller. The curriculum
raises `d_cap` toward the configured frozen-GMT boundary. With the current
boundary `2.381`, the Broken-tail outer ceiling never exceeds 2.381;
consequently the terminal Hard ceiling is `2.381 / 1.10`, so the 10% Broken-tail
support never collapses to an empty interval. Exact starting distributions,
advance-rule ID and advance update counts are explicit engineering/configuration
parameters; no hidden default, episode-length controller or Gain feedback may
choose them.

For the current `d_cap`, the four strength classes are sampled as:

\[
\begin{aligned}
\mathcal D_{easy}   &: d\in[0,0.25d_{cap}),          & w&=0.20,\\
\mathcal D_{medium} &: d\in[0.25d_{cap},0.70d_{cap}),& w&=0.30,\\
\mathcal D_{hard}   &: d\in[0.70d_{cap},d_{cap}],    & w&=0.40,\\
\mathcal D_{broken} &: d\in(d_{cap},\min(1.10d_{cap},2.381)], & w&=0.10.
\end{aligned}
\]

Easy preserves low-disturbance restoration and Demo quality. Medium supplies
ordinary repairable signal. Hard supplies the main difficult recovery
evidence. Broken tail only exposes the nearby failure boundary and cannot
dominate the mixture. These are sampled-strength classes, not the observed
Safe/Repairable/Broken/Harmful outcome taxonomy.

One sealed Segment samples one class, strength and single `local_rp` artifact.
Its Noisy baseline and all M Repair attempts reuse them; Clean is uncorrupted.
The sampler cannot resample on reset or change the mixture from Gain, PPO,
evaluation or diagnostics.

## K Transition And Critic Recalibration

The transition order is:

```text
commit the complete current transaction
-> advance outer K/M stage
-> install the new K stage's lower informative DR distribution
-> freeze actor/std; retain the same Critic and compatible optimizer state
-> critic-only recalibration
-> actor ramp
-> joint optimization while the inner DR curriculum advances
```

The lower distribution must still leave a clear Noisy-to-Clean repair signal;
it is neither zero nor nearly Clean. The transition resets training difficulty,
not Actor, Critic, optimizer identity, sampler history, committed receipt or
previously learned policy.

Cross-horizon ordering of the same sealed scenario and attempts is recorded for
diagnosis only during the first campaign. It cannot adapt K or DR online and is
not a graduation condition.

## Exact-M Frozen-Policy Transaction

Before opening a transaction, resolve and seal contract/checkpoint/curriculum
identity, active stage/K/M/DR spec and progress, beta, two Segment scenarios and
fixed Clean/Noisy lifecycle, and frozen `pi_old`.

Collection performs zero optimizer steps. Every Segment produces exact M valid
Repair rows under one homogeneous K/M/DR identity. Only complete sealing may
call FRS-PPO-v005 once. Failure or partial collection cannot advance optimizer,
sampler, inner DR progress, K/M stage, phase, absolute iteration, receipt or
checkpoint state. Stage and DR advancement happen only after a committed
receipt.

## Critic Recalibration And Actor Ramp

For stage-local committed iteration `l_j`:

```text
0 <= l_j < N_c_j:
  phase = critic_only; actor_loss_weight = 0
N_c_j <= l_j < N_c_j + N_a_j:
  phase = actor_ramp
  actor_loss_weight = (l_j - N_c_j + 1) / N_a_j
l_j >= N_c_j + N_a_j:
  phase = joint; actor_loss_weight = 1
```

Actor/std are immutable during critic-only. The same Critic continues updating
during actor ramp and joint training.

## Fixed Split-LR Optimizer Identity

Stage 3 owns exactly one `FrontRESTrackedAdam` with two named, disjoint and
exhaustive parameter groups:

```text
actor:  residual_actor parameters, learning_rate = 3e-6
critic: scalar Critic parameters, learning_rate = 1e-5
schedule = fixed
```

The task-space exploration std remains a fixed buffer and is absent from both
groups. There is no second optimizer or scheduler. Adaptive or group-wide LR
writes and the retired shared Stage-3 LR option fail closed. The phase owner
still controls gradients: critic-only installs no Actor gradient and preserves
Actor parameters plus Adam state; actor-ramp and joint retain their existing
loss weights. Both groups participate in the same exact-one optimizer call and
carry the same persisted optimizer step count.

## Beta Calibration Boundary

The first bounded calibration uses `beta_init=0.02`. Telemetry may support
human revision between bounded runs, but training cannot mutate beta. Once
accepted, beta is fixed across Segments, attempts and K stages.

## Checkpoint-v10 Identity

```text
checkpoint_schema = frontres-v017-checkpoint-v10
method_contract_id = FRS-METHOD-v017
gain_contract_id = FRS-GAIN-v007
optimization_contract_id = FRS-PPO-v005
training_contract_id = FRS-TRAIN-v015
scalar_target_id = clean-anchored-recovery-aware-gain-v1
dr_curriculum_schema_id = nested-k-dr-four-class-v1
```

The payload binds the 928/158/770 layout; actor/std/Critic/normalizers/optimizer;
complete K x M schedule; every `DRStageSpec`; active stage/K/M/phase/actor
weight; current committed inner-DR progress and `d_cap`; fixed scales, beta and
Gain identity; sampler/RNG state; exact row/Segment counts; absolute committed
update; and adjacent committed transaction receipt.

Checkpoint-v9 and earlier, HSL-v1, v012 `g_8/g_16/g_32`, unversioned payloads, mismatched DR specs,
hidden episode-length-controller state, partial receipts, HSL-as-resume or
identity drift reject before any mutable restoration.

The optimizer payload must contain exactly the `actor` and `critic` groups,
their exact configured LRs, disjoint role-correct membership, optimizer moments
and one shared nonnegative step count. Missing, duplicated, overlapping or
non-finite group identity rejects before actor, Critic, optimizer, sampler,
curriculum or receipt mutation. Checkpoint-v9 is retained only as historical
evidence and cannot initialize or resume this campaign.

## Required Telemetry

- stage/K/M, active DR class, strength, `d_cap`, DR progress/spec identity,
  phase, actor weight and absolute/local iteration;
- transaction/Segment/attempt/row counts, exact-one update and committed receipt;
- `G_I`, `G_P`, `lambda_RA`, repair cost, `G_total`, value/advantage and Critic
  calibration;
- Contact, support drift, phase-ZMP, survival, sustained lean and unplanned
  support changes;
- action magnitude and actor/std/Critic parameter deltas;
- direct-full6 action identity, both group LRs, checkpoint-v10 and all active contract identities;
- diagnostic-only cross-horizon ordering, never sampler feedback.

## Required Evidence And Stop Conditions

Deterministic evidence must cover K/M/phase resolution, all four DR class
boundaries and weights, stage-local DR restart/advance, exact-M and environment
widths, critic-only isolation, same-Critic K transition, no-resample, no
Gain/PPO feedback, committed-only progress, checkpoint-v10 roundtrip and v9/
`g_K` pre-mutation rejection. Formal and live evidence remain separate gates.

Stop if any active HSL, Stage-3, PPO, checkpoint, evaluation or deployment
consumer reintroduces bounded/squashed action semantics; starting/advance
parameters are hidden defaults; an episode-length or
online outcome controller regains authority; K and DR advance simultaneously;
high DR carries across a K transition; actor/std drift during critic-only; the
Critic is reinitialized at K transition; a transaction mixes K/M/DR identity;
failure advances curriculum; exact-one update fails; Clean leaks to the actor;
or a second sampler/Critic/optimizer/Gain authority is introduced.
