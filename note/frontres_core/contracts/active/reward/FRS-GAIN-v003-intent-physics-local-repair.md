---
contract_id: FRS-GAIN-v003
status: active
effective_date: 2026-07-19
updated_date: 2026-07-20
supersedes: FRS-GAIN-v002
scope: Stage 3 two-role local-repair paired Gain with root-invariant articulated-intent retention, physical executability, and Delta SE(3) cost
---

# Intent-Physics Local Repair Gain Contract

## Design Delta

`FRS-GAIN-v002` used immutable full Clean rollout motion as the Style target.
That teaches an unavailable deployment-time target and does not directly state
what FEMR preserves.

The active target is the trusted articulated motion intent carried by the
deployment Noisy reference:

```
I_s = Pi_internal(R^N) = Pi_internal(R^C)
```

FEMR learns whether its current root repair lets frozen GMT realize this same
internal intent more effectively than doing nothing.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Two-Role Pairing And Time |
| `FRS-DP-07` | Repair Gain | `Q-01` | Core Decision |

## Core Decision

The canonical decomposition is:

```
intent_gain = fidelity_internal(Repaired, I_s)
            - fidelity_internal(Noisy, I_s)

physics_gain = physics_quality(Repaired)
             - physics_quality(Noisy)

repair_cost = magnitude_and_temporal_cost(Delta SE(3)_t)

gain_total = w_intent * intent_gain
           + w_physics * physics_gain
           - w_repair * repair_cost
```

The implementation may retain the diagnostic/storage alias `style_gain` only
when it is explicitly documented as this root-invariant
`intent_realization_gain`. It must no longer mean full Clean global-motion
similarity.

## Internal-Intent Fidelity

`I_s` is the root-invariant 29DoF articulated-motion track taken from the
Noisy/deployment reference. The minimal fidelity comparison is between the
executed robot articulated state and this joint-space intent, excluding absolute
root position and root orientation.

The active raw components are:

- 29DoF joint-angle / relative articulated-pose fidelity;
- explicitly accepted joint velocity and acceleration fidelity, if available
  with identical units in Noisy and Repaired branches;
- optional local relative-link metrics only when they are root-invariant and
  separately evidenced.

The following are not internal-intent fidelity components:

- global body MPJPE;
- root translation;
- root-orientation geodesic error;
- direct Repair-vs-Noisy rollout similarity.

Direct Repair-vs-Noisy similarity rewards the no-op because the two executions
become identical when `Delta SE(3)=0`. Both branches must instead be compared
to the same fixed `I_s`.

## Physics Gain

Physics asks whether frozen GMT executes the reference more reliably.

Required raw components remain:

- success/fall;
- raw survival steps, report-only;
- survival quality = `survival_steps / effective_horizon_K`;
- paired survival gain;
- ZMP/support margin;
- contact consistency.

Physics remains a paired Noisy-versus-Repaired comparison under the same local
scenario. It is not a Clean target and is not a generic environment reward.

## Repair Regularizer

Repair cost remains an ordinary regularizer:

- full-6D `Delta SE(3)_t` magnitude;
- temporal change of the executed correction;
- zero/near-zero intervention diagnostics on unperturbed local scenarios, when
  those rows are explicitly available.

It is not a third definition of task success and may not conceal a global Clean
motion objective.

## Two-Role Pairing And Time

For one local scenario, Noisy and Repair must share:

```
same motion and Segment identity
same Clean dynamic reset x_t
same current root artifact
same future internal intent I_s
same full Clean continuation C_s
same horizon K
```

At `t`, Noisy uses the uncorrected root artifact and Repair uses the same
artifact plus the policy action. At `t+1 ... t+K`, FEMR is frozen in both
branches and GMT consumes the common Clean continuation.

Clean is not a third scored branch. Its roles are limited to the shared
continuation and offline calibration that the Noisy/deployment q29 track is a
trusted internal-motion intent.

## Training, Deployment, And Evaluation Alignment

The actor sees only current Noisy root/anchor error plus future `I_s` from the
Noisy/deployment reference. It never sees Clean future provenance or a full
Clean rollout target.

The local K experiment measures the isolated causal effect of the first repair
action. A separate full-sequence composition evaluation may test repeated
deployment repairs under persistent artifacts; it is not allowed to redefine
this main Gain or inject later noise into the first-action credit signal.

## Single Active Gain Owner

`frontres_gain.py` is the only active owner of the intent/physics/cost
calculation. PPO return, Segment Replay priority evidence, diagnostics,
periodic evaluation, and sequence evaluation must consume the same
`gain_total` and decomposition once v003 is implemented.

The current Clean-global Style owner and legacy family-specific score are both
contract-mismatch / legacy paths. Neither may remain active through a fallback,
difficulty heuristic, diagnostic replacement, or evaluation metric.

## Forbidden Inputs And Semantics

- full Clean rollout as an actor reward target;
- Clean future provenance in actor input;
- future raw root/global Noisy reference as an intent target;
- generic environment, teleoperation, or velocity-command reward;
- a score against the modified Repaired reference;
- direct Repair-vs-Noisy similarity as intent retention;
- perturbation-family action masks;
- online or batch-dependent epsilon/style tolerance.

## Required Diagnostics

```
intent.q29_noisy / repaired / gain
intent.qvel_noisy / repaired / gain, when enabled
intent.qacc_noisy / repaired / gain, when enabled
intent_invariant.noisy_vs_clean
physics.success/fall/survival noisy / repaired
physics.survival_steps_raw noisy / repaired
physics.survival_quality noisy / repaired
physics.zmp_margin_noisy / repaired / gain
physics.contact_noisy / repaired / gain
repair.delta_se_norm
repair.delta_se_temporal_change
gain.intent / gain.physics / cost.repair / gain.total
```

Every component must print its source/provenance and be reported as
`UNCONFIRMED` rather than silently zeroed.

## Bounded Implementation Evidence

`E-FI-10` (2026-07-20) implements only the pure S1 owner:

- `frontres_gain.py::compute_intent_physics_local_repair_gain()` accepts typed
  deployment/Noisy q29 intent, paired execution/physics facts, and executed
  full-6D correction evidence;
- its input surface has no Clean/root/global fidelity field, rejects invalid
  q29 provenance, and exposes `style_gain` only as an explicit alias for
  root-invariant `intent_gain`;
- absent optional qvel/qacc or one-action temporal evidence remains `NaN`.

`E-FI-11` (2026-07-20) adds only candidate-only deterministic S1 consumer
connectivity: post-`t` robot q29 and the sealed deployment/Noisy `I[t]`
produce v003 Gain, one return/advantage carrier, and immutable scenario-keyed
priority evidence. The carrier rejects v002/Clean-global fallback and cannot
mutate sampler state or actor-loss mass.

`E-FI-10` and `E-FI-11` alone do not establish formal storage insertion,
sampler-state update, diagnostics, evaluation, PPO, checkpoint, formal-route,
simulator, training, or live connectivity.

`E-FI-12` (2026-07-20) adds only candidate-only diagnostic consumption: the
sealed v003 decomposition is formatted as q29 intent/provenance, Physics,
Repair Cost, and Total Gain. Legacy v002 evaluators reject v015 before capture,
and the distinct composition protocol cannot feed back into local return,
priority, or PPO. Formal/local live evaluation remains unproven.

`E-FI-13` (2026-07-20) adds only the next candidate-only consumer boundary:
the same v003 one-row return evidence becomes sealed local-scenario metadata
and a grouped candidate batch. q29 provenance, scenario/hash, `x_t`, K, and
actual evidence-step count remain aligned; priority is not read as an actor-loss
weight, and legacy fixed-tape storage is rejected. Formal PPO/update, sampler
mutation, checkpointing, simulator, training, and live evidence remain outside
this proof.

## Acceptance

1. S1: deterministic q29/root-invariance, intent sign, no-op, and root-
   exclusion fixtures.
2. S2: two-role local K connection proves shared `x_t/I_s/C_s`, one action,
   frozen later FEMR, and one active Gain owner.
3. S3: return, priority, periodic evaluation, sequence evaluation, and
   diagnostics consume v003 with no Clean-global or legacy fallback.
4. S4: full-sequence deployment composition evaluation is reported separately
   from local-Gain evidence.

Until these gates pass, v003 is accepted semantics and current code is an
explicit implementation mismatch, not an alternative active reward design.
