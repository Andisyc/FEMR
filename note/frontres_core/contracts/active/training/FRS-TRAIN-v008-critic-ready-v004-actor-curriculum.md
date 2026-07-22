---
contract_id: FRS-TRAIN-v008
status: active
effective_date: 2026-07-22
updated_date: 2026-07-22
supersedes: FRS-TRAIN-v007
scope: proposal-only HSL actor initialization followed by v004 critic-only, actor-ramp, and joint grouped-PPO training over sealed one-action-K transactions
---

# Critic-Ready v004 Actor Curriculum Training Contract

## Design Delta

`FRS-TRAIN-v007` correctly separated proposal-only HSL from Stage-3 executable
optimization, but mapped `M-05 Actor & Critic Warmup` only to a generic formal
transaction route. It did not define the gradient schedule that protects the
HSL actor while a fresh scalar Critic learns the paired return.

`FRS-TRAIN-v008` closes that design point:

```text
proposal-only HSL actor initialization
-> v004 critic-only calibration
-> linear actor takeover while Critic keeps learning
-> joint grouped PPO
```

This is not a second Critic, a second optimizer, or an HSL continuation loss.
It controls which gradients from the existing scalar PPO objective may update
the existing full-6D actor at each persisted Stage-3 iteration.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-08` | HSL Warmup | `M-03` | Proposal-Only HSL Initialization |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Critic-Ready Actor Curriculum |

The Concept Figure already states `M-05 = Actor & Critic Warmup`; no new block
or human-facing method semantic is required.

## Proposal-Only HSL Initialization

Stage 1 remains governed by the strict `frontres-v015-hsl-proposal-v1`
identity:

```text
current Noisy root artifact
+ deployment/Noisy q29 I[t:t+H]
-> full-6D actor distribution initialization
```

The only HSL target is the current-frame anti-DR `Delta SE(3)` proposal. HSL
may initialize the residual actor, action distribution parameters, and 158D
prefix normalizer only. It must not carry a Critic, critic normalizer,
optimizer, sampler, transaction, Gain, return, advantage, priority, or
Stage-3 supervised target.

HSL authority ends before the first Stage-3 transaction. Stage 3 rejects the
legacy quartet/Clean-global rollout label and every nonzero supervised loss.

## Stage-3 Observation And Evidence Boundary

The existing v015 boundary remains unchanged:

- actor observation: deployable 158D prefix;
- scalar Critic observation: sealed row-aligned 289D critic observation;
- frozen GMT observation: 770D suffix;
- one full-6D `Delta SE(3)` policy action and one PPO row per attempt;
- K-step frozen-FEMR executable evidence;
- one complete multi-Segment x M transaction before exactly one update.

Expected support mode, actual ContactSensor evidence, phase-conditioned ZMP,
Clean continuation, and Physics admissibility remain Gain/evaluator evidence.
They do not become actor inputs. This contract does not add or change critic
features; it changes the Critic's v004 return target and actor-gradient timing.

## v004 Critic Target

For every ordinary-valid attempt, the existing scalar Critic predicts one
value `V(o_critic_t)`. Its Stage-3 target is the one-action-K return produced
from the active `FRS-GAIN-v004` paired utility:

```text
return_K = v004 paired support-mode/Physics/Intent repair return
advantage_K = return_K - V_old(o_critic_t)
```

The exact return/GAE convention remains owned by the existing PPO/storage
contract. Critic Warmup does not redefine Gain, introduce a second target, or
use HSL labels. Missing or mixed v003/v004 identity fails before loss.

## Critic-Ready Actor Curriculum

Let `i` be the persisted absolute Stage-3 iteration, `N_c` the explicit
critic-only duration, and `N_a` the explicit actor-ramp duration. Formal
training requires `N_c > 0` and `N_a > 0`. Current engineering defaults are
`N_c=200` and `N_a=500`; they are versioned run configuration, not universal
method constants.

### Critic-only

For `0 <= i < N_c`:

```text
phase = critic_only
actor_loss_weight = 0
critic_update = enabled
actor parameters = frozen
action std/log_std = frozen
```

The frozen HSL actor still samples the M policy attempts under the sealed old
policy. Only the scalar Critic learns the v004 return. Actor, distribution, GMT,
normalizers, sampler, and scenario evidence must not change through the
optimizer step except for explicitly train-mode normalizer behavior already
owned outside this contract.

### Actor warmup

For `N_c <= i < N_c + N_a`, let `j=i-N_c`:

```text
phase = actor_warmup
actor_loss_weight = (j + 1) / N_a
critic_update = enabled
```

The same weight scales the actor-side PPO contribution consistently without
changing grouped reduction, advantage sign, or Critic loss. The Critic keeps
learning throughout takeover.

### Joint PPO

For `i >= N_c + N_a`:

```text
phase = joint
actor_loss_weight = 1
critic_update = enabled
```

Actor and Critic then train jointly from the same v004 transaction evidence.
HSL remains disabled.

## Transaction Atomicity

The phase is resolved once from the persisted iteration before a transaction
opens and is immutable until that transaction commits. A multi-Segment x M
transaction may not cross phases. Collection performs no optimizer step.
After all attempts seal, the formal owner performs exactly one phase-governed
optimizer update.

`critic_only` must clear or exclude every non-Critic gradient before clipping
and stepping. `actor_warmup` and `joint` must retain the same grouped
motion -> Segment -> attempt reduction and one-row K semantics.

## Cold Start, Resume, And Identity

Cold Stage-3 initialization from `frontres-v015-hsl-proposal-v1` restores only
actor/distribution/158D prefix state. The scalar Critic starts fresh and the
persisted Stage-3 iteration starts at zero, therefore entering `critic_only`.

An exact Stage3-v015/v008 full resume must bind and restore:

- `training_contract_id=FRS-TRAIN-v008`;
- `gain_contract_id=FRS-GAIN-v004`;
- `N_c`, `N_a`, persisted absolute iteration, and resolved phase identity;
- actor, scalar Critic, optimizer, compatible normalizers, sealed transaction
  receipt, and existing v015 layout/checkpoint identities.

Resume must continue the same phase; it may not restart Warmup. A v003 Gain or
v007/unversioned Stage-3 checkpoint cannot full-resume into v008 because its
Critic and optimizer were trained under a different target/gradient contract.
Actor-only reuse of such a Stage-3 checkpoint is not authorized by this
contract. The strict HSL-v1 initializer remains the only accepted actor-only
cold-start artifact.

## Formal Owners

| Object | Owner | Required v008 behavior |
| --- | --- | --- |
| phase schedule | `frontres_segment_warmup.py::frontres_segment_warmup_phase()` | persisted iteration -> critic_only / actor_warmup / joint and actor weight |
| formal update | `frontres_segment_live_probe.py::run_frontres_v015_formal_transaction_update()` | consume the phase, apply actor weight, clear non-Critic gradients in critic_only, exact-one step |
| PPO loss | `frontres_segment_ppo.py::compute_frontres_segment_ppo_loss()` | unchanged grouped formula with phase-owned actor weight and continuously enabled value loss |
| launch/config | `train.py` plus Stage3 launchers | explicit nonzero `N_c/N_a` for formal training; zero allowed only for a named plumbing-only sentinel that cannot claim Warmup evidence |
| persistence | `frontres_checkpointing.py` | bind v008/v004/schedule/iteration/phase and reject incompatible resume before mutation |

## Current Contract Mismatch

The current formal v015 route does not implement v008:

- `_require_v015_formal_transaction_config()` rejects every nonzero
  `frontres_segment_critic_warmup_iterations` or actor-warmup value;
- `_v015_formal_ppo_config()` hardcodes `actor_loss_weight=1.0`;
- `run_frontres_v015_formal_transaction_update()` does not resolve a Warmup
  phase and does not clear actor/std gradients in critic-only;
- the latest v015 S4 log explicitly ran with `critic_warmup=0` and
  `actor_warmup=0`;
- deterministic and E68-E70 live Warmup evidence belongs to the older update /
  v002 Gain route and cannot prove v008 formal connectivity.

These paths are `contract-mismatch`, not accepted alternatives.

## Required Diagnostics

Every committed transaction must report:

- training and Gain contract identities;
- persisted absolute iteration, `N_c`, `N_a`, phase, phase iteration, and actor
  loss weight;
- actor/std/Critic gradient and parameter deltas separately;
- v004 return, old value, raw/scaled advantage, value loss, actor loss, and
  total loss;
- complete transaction identity, grouped mass, exact-one step, and committed
  checkpoint receipt.

During critic-only, actor/std delta must be exactly zero and Critic delta must
be finite and nonzero on valid non-tie evidence. During actor warmup, actor
weight must be monotonic and Critic updates must remain enabled.

## Acceptance Evidence

| Gate | Required proof |
| --- | --- |
| S1 schedule/gradient | phase boundaries, actor-weight formula, critic-only actor/std invariance, Critic update, actor-ramp monotonicity, joint weight, missing/mixed identity rejection |
| S2 formal connectivity | complete v004 transaction -> phase-aware grouped loss -> exact-one update; critic-only and actor-ramp fixtures use the formal v015 owner, not the legacy update connector |
| S3 persistence | cold HSL start enters iteration 0 critic-only; exact v008/v004 save/reload preserves schedule and phase; v003/v007/unversioned resume rejects before mutation |
| S4 bounded live | one 8-env v004 transaction at critic-only records actor/std zero delta, nonzero Critic delta, exact-one update, and committed v008/v004 checkpoint |

Deterministic fixtures may cross all three phase boundaries without simulator
cost. Actual long progression through actor warmup and joint belongs to X1 and
does not require a 700-iteration pre-training sentinel.

## Stop Conditions

Stop G5-P1 and return to design review if:

- v004 does not produce a finite scalar target for the Critic;
- the formal v015 owner cannot apply Warmup without falling back to the legacy
  update route or changing grouped PPO;
- actor or std changes during critic-only;
- Critic receives no gradient on valid unequal v004 returns;
- phase changes during an open transaction or resume restarts the schedule;
- a v003/v007 checkpoint can mutate v008 state;
- implementing Warmup requires a second Critic, second optimizer, HSL target,
  new actor input, or changes to H, K, one-action-K, transaction, or grouped
  reduction semantics.

Until S1/S2/S3/S4 pass, v008 is accepted training semantics and the current
formal v015 route is an explicit implementation mismatch. Additional training,
X1, multi-seed, deployment composition, and paper experiments remain blocked.
