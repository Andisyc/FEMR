---
contract_id: FRS-TRAIN-v009
status: active
effective_date: 2026-07-22
updated_date: 2026-07-22
supersedes: FRS-TRAIN-v008
scope: single-Critic global K-stage curriculum with per-stage critic recalibration, actor ramp, joint grouped PPO, homogeneous-K transactions, and exact curriculum persistence
---

# K-Stage Critic Curriculum Training Contract

## Design Delta

FRS-TRAIN-v008 correctly introduced one fresh scalar Critic, critic-only
calibration, actor ramp, and joint PPO. It indexed that schedule only by the
global Stage-3 iteration and therefore assumed one stationary K-step return
target.

The active code also contains a different K policy: Segment state may assign
`8/16/32/64` independently to different Segments in the same training period.
That makes one Critic fit several horizon-dependent targets without a K input
and contradicts the confirmed curriculum semantics.

FRS-TRAIN-v009 defines one global ordered K curriculum:

```text
K_0 stage: critic-only -> actor-ramp -> joint
-> transaction boundary
K_1 stage: critic-only -> actor-ramp -> joint
-> ...
K_final stage: critic-only -> actor-ramp -> joint until the run ends
```

There is still one full-6D actor, one scalar Critic, and one optimizer. The
Critic keeps its parameters when K advances, but the actor and action std are
frozen while that Critic recalibrates to the new return horizon.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Global K-Stage Identity |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Per-Stage Critic-Ready Schedule |

`M-06` controls the evidence horizon. `M-05` controls the gradient schedule
whenever that horizon changes. No new Concept Figure block is introduced.

## Global K-Stage Identity

One run provides an explicit immutable schedule:

```text
C_K = [(K_j, N_c_j, N_a_j, N_joint_j)] for j=0...J-1
```

The contract requires:

- `K_j` is positive, strictly increasing, and no larger than `K_max`;
- `N_c_j > 0` and `N_a_j > 0` for every stage;
- every non-final stage has `N_joint_j > 0`;
- the final stage remains joint after its critic-only and actor-ramp periods;
- the schedule and its fingerprint are frozen before the first transaction.

These durations are run configuration, not universal method constants. A
formal run without the complete schedule fails before sampling.

At any persisted iteration there is exactly one active tuple:

```text
(k_stage_index, active_k, stage_iteration, phase, actor_loss_weight)
```

Every Segment and every M attempt in one transaction uses `active_k`. Segment
state may affect selection or replay priority under their existing contracts,
but it may not choose K on the v009 formal route. The existing per-Segment
adaptive K planner is legacy/incompatible with v009 formal training.

## Horizon-Indexed Critic Target

Within stage `j`, the scalar Critic learns only:

```text
V_j(o_critic_t) ~= E[return_{K_j} | o_critic_t]
advantage_{K_j} = return_{K_j} - V_old_j(o_critic_t)
```

Because one stage exposes one K, K is not added to the actor or Critic
observation. The Critic does not retain a separate head per K. Earlier K values
are curriculum scaffolding; only the final K has final method authority.

K must measure the first action long enough to expose the current stage's
intended Contact/ZMP/survival consequences. No Critic can repair consequences
that are absent from `return_K`; long-sequence evaluation remains separate.

## Per-Stage Critic-Ready Schedule

For local stage iteration `l_j`:

```text
0 <= l_j < N_c_j:
  phase = critic_only
  actor_loss_weight = 0

N_c_j <= l_j < N_c_j + N_a_j:
  phase = actor_warmup
  actor_loss_weight = (l_j - N_c_j + 1) / N_a_j

l_j >= N_c_j + N_a_j:
  phase = joint
  actor_loss_weight = 1
```

The Critic remains trainable in all phases. At a K transition, Critic weights
and compatible optimizer state continue as initialization; the Critic is not
reinitialized. Actor and std freeze again until the new stage's critic-only
period completes.

## Transaction And Transition Atomicity

The curriculum identity is resolved before a transaction opens and sealed
until it commits. A transaction may not mix K stages, active K values, phases,
or schedule fingerprints. Collection performs no optimizer update; one complete
multi-Segment x M transaction still produces exactly one grouped update.

Stage advancement occurs only after a committed transaction. The next
transaction observes `stage_iteration=0`, the new `active_k`, and
`phase=critic_only`. A failed or partial transaction cannot advance curriculum
state.

## Persistence And Compatibility

The next Stage3 checkpoint identity must bind:

- `training_contract_id=FRS-TRAIN-v009`;
- ordered curriculum schedule and fingerprint;
- `k_stage_index`, `active_k`, `stage_iteration`, phase, and actor weight;
- absolute committed-update iteration;
- FRS-GAIN-v004, FRS-PPO-v003, v015 observation layout, normalizers, sampler,
  optimizer, and committed transaction receipt.

Exact resume requires the same schedule fingerprint and restores the same
stage/phase before mutable state. FRS-TRAIN-v008, unversioned, mixed-K, or
partial-transaction checkpoints cannot mutate v009 state. The strict HSL-v1
initializer remains the only actor-only cold start.

## Formal Owners

| Object | Required owner | v009 responsibility |
| --- | --- | --- |
| schedule kernel | existing `frontres_segment_warmup.py` | immutable schedule validation and iteration -> K stage/phase mapping |
| config/entry | existing config and Stage3 launcher | require complete ordered schedule; no hidden defaults on formal training |
| transaction K | existing live sampler/formal request owners | override legacy per-Segment K with one sealed active K |
| formal update | `run_frontres_v015_formal_transaction_update()` | consume sealed stage/phase, exact-one update, critic-only actor/std isolation |
| persistence | `frontres_checkpointing.py` | bind schedule fingerprint and exact stage resume; reject v008 before mutation |
| diagnostics | existing formal transaction telemetry | report stage, K, local iteration, phase, target/value/advantage and parameter deltas |

No new production module is required.

## Forbidden Behavior

- multiple Critic networks or K-specific Critic heads;
- K in the actor observation or future-intent interface;
- mixed K values inside one v009 transaction;
- per-Segment state choosing K on the formal v009 route;
- a K transition during collection or before a committed update;
- continuing actor updates immediately after K changes;
- reinitializing the Critic or restoring HSL into Stage 3;
- treating short-K improvement as long-sequence stability;
- changing one-action-K, M-attempt identity, grouped PPO, Gain v004, H, GMT,
  or the actor/Critic architecture.

## Acceptance Evidence

| Gate | Required proof |
| --- | --- |
| C1 / S1 | schedule ordering, boundary mapping, repeated critic-only at every K, monotonic actor ramp, final-stage joint persistence, invalid schedule rejection |
| C2 / S2 | formal transaction uses one active K for all Segments/attempts, phase is sealed, legacy adaptive K cannot enter, exact-one grouped update remains unchanged |
| C3 / S3 | checkpoint binds schedule/stage/K/phase, exact resume restores them, v008/mixed schedule/partial transaction reject before mutation |
| C4 / S4 | bounded official route crosses one K transition and records old-stage commit -> new-K critic-only, actor/std zero delta, Critic nonzero delta, and committed v009 checkpoint |

## Stop Conditions

Stop before implementation or training if:

- the official route cannot make a transaction K-homogeneous without changing
  one-action-K or fixed M semantics;
- the final K or stage schedule is treated as an implicit code default rather
  than explicit run identity;
- the actor updates before the new-K Critic calibration period completes;
- schedule transition or resume can occur without a committed receipt;
- v008 or a different schedule can mutate v009 state;
- implementation requires Multi-Critic, a second optimizer, K actor input,
  mixed-horizon actor credit, or a change to Gain/PPO/HSL semantics.

Long training remains blocked until C1-C4 close and the policy-quality audit
defines the final-K admission boundary.

## Implementation Status

E-FI-70 closes deterministic C1-C3:

- the existing warmup owner validates and fingerprints the explicit schedule;
- the formal sampler/request/update route seals one active K for every
  Segment x M row and re-enters critic-only at a K transition;
- committed receipts bind the consumed stage identity;
- checkpoint `frontres-v015-checkpoint-v4` binds the next schedule/stage/K
  identity and rejects v008, v3, partial, tampered, or different-schedule
  resume before mutable restoration.

C4 bounded official live transition evidence is still absent. This contract
does not authorize long training or policy-quality claims before C4.
