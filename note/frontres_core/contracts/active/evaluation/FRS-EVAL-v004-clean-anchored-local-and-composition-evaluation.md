---
contract_id: FRS-EVAL-v004
status: active
effective_date: 2026-08-01
updated_date: 2026-08-10
supersedes: FRS-EVAL-v003
scope: Clean-anchored one-action-K local evaluation and isolated full-sequence deployment-composition evaluation
---

# Clean-Anchored Local And Composition Evaluation

## Design Delta

FRS-EVAL-v003 treated Clean as continuation data rather than an executed local
baseline and reported the v006 scalar-Intent/Physics-projection decomposition.
The active method now requires one observed Clean anchor, one observed fixed
zero-action Noisy baseline, and M observed Repair outcomes per sealed Segment.
This contract aligns evaluation with METHOD-v020, TRAIN-v019, PPO-v008 and
GAIN-v008 without changing the separate full-sequence
deployment-composition question.

The 2026-08-10 revision does not change the held-out scientific question.
checkpoint-v13 remains a strict legacy K16/M4 compatibility route.
checkpoint-v14 uses the same held-out Segment identity but reports both raw
FRS-GAIN-v008 attempts and `U(G)=sign(G) log1p(abs(G))` utility. The tested
policy route temporarily installs the checkpoint's 158D Actor, 449D
state-value Critic, Actor-prefix statistics and 449D privileged-observation
normalizer, then restores every prior state. The Critic loss normalizer remains
output-preserving and is reported only as checkpoint identity; it is never
applied to `V(s)` during evaluation.

The optional TRAIN-v019 Critic learnability probe reuses this same evaluator
with an explicit `repeat_count`. It fixes two K8/M4 Segment states and repeats
only the M4 Actor sampling and Repair rollout. The first repeat seals the full
normalized 928D Actor and 449D Critic inputs; later repeats reuse those tensors
while still resetting and executing the same physical Segment. Fresh live
observations may be retained as drift diagnostics but cannot condition the
later action samples. The default remains one pass, so ordinary held-out
evaluation is unchanged. This diagnostic repeat dimension is not a new
held-out benchmark, a condition-alignment requirement, or a training feedback
path.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Local Clean/Noisy/Repair Evaluation |
| `FRS-DP-07` | Repair Gain | `Q-01` | Local Report |

## Two Evaluation Questions

Local one-action-K evaluation asks whether each Repair improves the current
sealed Segment relative to Noisy while moving toward the Clean motion semantics.
It supplies training-quality evidence but performs no update itself.

Full-sequence composition evaluation asks whether per-frame FEMR corrections
remain useful on an immutable deployment/reference carrier. It is deployment
quality evidence and never becomes a local PPO return, replay priority, or
curriculum controller.

The two questions have different clocks and must not share mutable lifecycle
state or silently substitute one report for the other.

Evaluation exposes exactly three independent capabilities: Held-out Policy
Quality for the local question, Deployment Composition for the full-sequence
question, and DR Sweep for the separate robustness-budget validation. Training
may trigger Held-out Policy Quality only through an external scheduler reading
a committed checkpoint. The training loop does not own or execute an embedded
periodic evaluator.

## Local Clean/Noisy/Repair Evaluation

For each sealed Segment:

```text
select one scenario and Clean dynamic x_t
-> execute Clean once through the common K-step continuation
-> execute fixed zero-action Noisy once from the same x_t
-> execute M Repair attempts from the same x_t and frozen pi_old
-> reuse the sealed Clean and Noisy evidence for every Repair comparison
-> produce one raw GAIN-v008 decomposition per valid Repair attempt
-> transform each valid raw Gain independently for utility-space calibration
```

Clean and Noisy are evaluator baselines, not PPO policy rows. Only Repair owns
one sampled full-6D action and one policy row. Reset may restore dynamics but
must not resample or mutate scenario, artifact, q29 Intent, continuation, K,
Clean/Noisy evidence, or identity.

Clean defines intended pose, expected Contact phase, support/recovery envelope,
planned dynamic lean, and survival semantics. Noisy defines the do-nothing zero
point. Clean evidence is forbidden from actor observation, future context, HSL
target, deployment input, or policy storage.

## Local Report

The atomic local report must retain:

- motion, Segment/start-frame, scenario/noisy hash, `x_t`, K, H offsets,
  continuation, baseline and attempt identities;
- per-step and aggregated Clean/Noisy/Repair Intent channels;
- expected and actual Contact, support-foot drift, loaded-support phase-ZMP
  applicability/violation/recovery, survival, sustained lean, and unplanned
  support changes;
- fixed `S_j`, within-family aggregates `I_X` and `P_X`, signed `G_I` and
  `G_P`, `lambda_RA`, full-6D `C_repair`, `beta`, and `G_total`;
- METHOD-v020 / TRAIN-v019 / PPO-v008 / checkpoint-v14 identity, the 449D
  state-value/support-context contract and Critic normalizer fingerprints;
- every raw `G_total_m`, every independently transformed `U(G_total_m)`, one
  shared raw `V(s)` per Segment, `mean_m U(G_total_m)` and the utility-space
  `V(s) - target` error, without action conditioning or post-mean transform;
- valid policy-row mask and same-Segment attempt ordering.

When `repeat_count > 1`, the report additionally retains the repeat index,
per-repeat M4 action fingerprint, fixed scenario/noisy-hash/`x_t` identity,
reference fingerprints plus exact used-input equality for the normalized 928D
Actor and 449D Critic inputs, repeated `V(s)` statistics, and per-Segment target
mean, population standard deviation, standard error, minimum and maximum. The
live observation-history drift is reported separately and is not accepted as
the policy condition. Any drift in the inputs actually consumed by Actor or
Critic fails closed. This answers whether repeated realized M4 targets for one
fixed policy state are stable enough to learn; it does not claim policy
quality.

Missing required evidence, identity drift, non-finite applicable values, or a
silent zero/default fails the local item closed. Physics is reported through
the v008 raw scalar ordering; the retired v006 constraint projection/KKT fields may
appear only as explicitly historical diagnostics and cannot determine status.

## Full-Sequence Deployment Composition

One ordinary deployment/reference `.npz` plus one fixed corruption protocol is
materialized once into an immutable deployment carrier. Baseline and Repair
start from the same canonical initial state and consume the same carrier:

```text
Baseline: fixed carrier -> frozen GMT
Repair:   same carrier -> per-frame FEMR -> frozen GMT
```

At frame `t`, FEMR reads the current deployment artifact and sealed Noisy q29
future Intent, applies one full-6D residual only to the current reference frame,
and never writes that repaired frame back into the future reference stream.
Physical state continues normally. This is no-feedback reference composition,
not a sequence of independent one-action-K training rows.

The composition report retains reference/protocol/carrier identity, route-start
state identity, FEMR action sequence, Intent, Contact preservation, phase-ZMP,
survival, sustained lean, unplanned support changes, and atomic completion.

## Isolation

Evaluation is inference-only. It may temporarily install the tested Actor,
Critic, Actor-prefix statistics and Critic observation normalizer only inside
one reversible checkpoint context. It must not mutate their source states,
the Critic value-loss normalizer, optimizer, sampler, transaction, curriculum,
warmup, checkpoint, return, priority, or PPO state. Success and exception paths
restore all module/normalizer state and inference/training mode and close their
carrier/scenario lifecycle.

Repeated evaluation must also fail closed if the fixed discrete Segment
identity changes, normalized Critic input drift exceeds its derived numeric
limit, any target is non-finite, the M4 action groups collapse to the same
fingerprint across repeats, or training state changes.

Legacy quartet/Clean-global Style reports, v002/v006 Gain fallbacks, direct
runner-private access, hidden padding, and mixed Baseline/Repair state are
forbidden on the active route.

The retired periodic/offline/sequence evaluator, its runner modes, and its CLI
configuration are removed rather than retained as a fourth evaluation system.

## Required Evidence And Stop Conditions

Deterministic evidence must cover baseline single-execution/reuse, exact M4
Repair identity, row-role isolation, v008 field completeness, checkpoint-v14
and legacy checkpoint-v13 Critic/normalizer installation and restoration,
per-attempt transform-before-M4 calibration arithmetic, missing-evidence
fail-closed behavior, route
permutation, atomic report production, and zero training-state mutation.
Bounded physical evaluation must later establish the real
Contact/phase-ZMP/survival and demo-quality facts; deterministic connectivity
alone is not efficacy evidence.

Stop if Clean reaches actor-visible data, Clean or Noisy becomes a policy row,
baselines are re-executed per attempt, local and composition identities mix,
v006 projection controls evaluation, or any evaluation path writes training
state.
