---
contract_id: FRS-METHOD-v015
status: active
effective_date: 2026-07-19
updated_date: 2026-07-22
supersedes: FRS-METHOD-v014
scope: FrontRES Stage 3 local root-artifact repair with deployment-provenance future 29DoF intent, fixed-policy multi-attempt Segment Replay, one policy action per attempt, and single-action K-step frozen-GMT evidence
---

# Future-Intent Single-Action Segment Replay

## Design Delta

`FRS-METHOD-v014` treated one complete 65D Noisy tape as the common source of
actor current/H context and K-step GMT execution. That bundles two different
objects and is not the accepted local-repair experiment.

This version separates them:

```
current root artifact          -> what FEMR repairs now
future 29DoF internal intent   -> what disambiguates the current repair
shared Clean continuation      -> fixed K-step environment used to measure its consequence
```

FEMR remains a `Noisy -> Executable` local reference repair policy. It does
not infer a full Clean reference, generate a recovery motion, or learn a
Noisy-to-Clean global root projection.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `M-02` | Local Root-Artifact Scenario |
| `FRS-DP-02` | Segment Replay | `SR-01` | Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Single-Action K-step Evidence |
| `FRS-DP-04` | FrontRES 6D Repair | `M-04` | Actor Observation And Action |
| `FRS-DP-05` | Frozen GMT | `M-10` | Method Boundary |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Two-Role Local Counterfactual |
| `FRS-DP-07` | Repair Gain | `Q-01` | Intent-Preserving Executability Gain |
| `FRS-DP-10` | Future Motion Context | `M-11` | Future Intent Context |

## Method Boundary

For a selected Segment at dynamic start `x_t`, FrontRES owns one full-6D
task-space action:

```
current robot/balance/tracking state
+ current Noisy root/anchor artifact
+ future internal motion intent
-> Delta SE(3)_t
-> frozen GMT execution
```

`x_t` is a replayable Clean dynamic state used to reproduce local dynamics for
every attempt. It is not a Clean actor reference. No Noisy physical prefix is
introduced before `x_t`.

The active deployment assumption remains a non-streaming `.npz` reference-file
path. At deployment, the actor reads the current available reference artifact
and the same deployable internal-motion future window; it never reads a Clean
reference or perturbation truth.

## Local Root-Artifact Scenario

One selected scenario `s` contains exactly three immutable parts:

```
1. x_t                         replayable Clean dynamic reset state
2. root_artifact_t             one root/ground/anchor perturbation at t
3. C_s = R^C[t+1 : t+K]        common full Clean GMT continuation
```

The perturbation acts on the current repairable root-level reference only. It
does not alter the trusted articulated motion intent:

```
I_s[t : t+H]
  = Pi_internal(R^N[t : t+H])
  = Pi_internal(R^C[t : t+H])
```

`Pi_internal` is the root-invariant 29DoF articulated-motion projection. The
equality above is a required perturbation-construction invariant, not a reason
to route Clean data into the actor. The actor must read `I_s` through the
Noisy/deployment reference provenance; Clean may only verify the invariant
offline.

The existing `noisy_segment_hash` field may remain as a compatibility carrier,
but its active meaning is the hash of this immutable local scenario: current
root artifact, intent-window source, Clean-continuation identity, `x_t`
identity, and horizon coverage. It must not imply that every K frame is noisy.

## Future Intent Context

Future context is retained because the same current robot state and current
root artifact can require opposite repairs when the upcoming support phase or
articulated motion differs. This is an observation-aliasing / conflicting-
gradient problem, not a request to anticipate future noise.

The actor observation is:

```
o_t = [current robot/balance/tracking state,
       current Noisy root/anchor error,
       I_s[t : t+H]]
```

`H` is a short ordered future window of deployable 29DoF internal intent (and
its explicitly accepted local derivatives/phase fields, if present). It is not
a future raw 65D reference window.

Forbidden actor inputs:

- future root translation, anchor pose, or global orientation from either
  Noisy or Clean reference;
- Clean future provenance;
- perturbation label, perturbation time, or truth metadata;
- a Noisy physical prefix before `x_t`.

If future q29 does not distinguish a claimed global task intention, the active
method does not silently add raw noisy root. A separately verified deployable
task-command channel would be a future design decision.

## Two-Role Local Counterfactual

The main K-step experiment has only two causal rollout roles:

```
Noisy baseline:
  reset x_t
  -> uncorrected root_artifact_t
  -> FEMR frozen at t+1 ... t+K
  -> GMT executes C_s

Repair attempt m:
  reset the same x_t
  -> root_artifact_t + policy-sampled Delta SE(3)_t^m
  -> FEMR frozen at t+1 ... t+K
  -> GMT executes the same C_s
```

Clean is neither a third scored rollout nor an actor target. It is the common
post-intervention continuation that holds the rest of the physical experiment
fixed. Candidate/search rows are not a third causal role. With `M_s` attempts,
the physical evidence set is one Noisy baseline plus `M_s` Repair attempts.

## Single-Action K-step Evidence

For every motion `g`, Segment `s`, and policy attempt `m`, storage contains
one policy tuple:

```
P_gsm = (o_t, sampled_action_t, old policy statistics,
         return_K, advantage_K, policy_row_valid,
         transaction/motion/segment/trial/scenario identities,
         horizon_k, evidence_valid_step_count)
```

There is one FEMR action at `t`, exactly one PPO policy row, and K-step
frozen-FEMR GMT evidence. `K` does not create K policy actions, K old-policy
statistics, K loss rows, or K actor-loss mass.

The K curriculum changes how long the first action's consequence is measured.
It does not add future root perturbations or future FEMR actions to this
single-action experiment.

## Frozen-Policy Multi-Attempt Transaction

One transaction freezes `pi_old`, selects multiple local scenarios, and for
each selected Segment:

1. seals one `x_t`, current root artifact, intent window, and Clean continuation;
2. restores the same `x_t` before every attempt;
3. collects `M_s >= 2` independently sampled policy actions under `pi_old`;
4. evaluates each action against the same Noisy baseline / common continuation;
5. stores one ordinary-valid policy row per attempt;
6. aggregates evidence only after all attempts finish.

No optimizer step occurs during collection or between selected Segments. All
ordinary-valid policy attempts remain PPO candidates; best-of-M is replay
evidence only and never a PPO selector.

## Intent-Preserving Executability Gain

The return is the paired improvement caused by the first repair action:

```
intent_gain = fidelity_internal(Repair, I_s) - fidelity_internal(Noisy, I_s)
physics_gain = executability(Repair)         - executability(Noisy)
repair_cost  = cost(Delta SE(3)_t)
```

The reward owner is `FRS-GAIN-v004`. Physics admissibility first preserves the
expected support mode and evaluates ZMP under the corresponding Contact phase;
only Physics-admissible repairs may be ordered by Intent improvement. Direct
similarity between Repair and Noisy rollout is forbidden because it rewards a
no-op. Full Clean global
rollout similarity is also forbidden as an actor target. Clean can calibrate
the `I_s` assumption and supply the common continuation only.

## Required Evidence

The implementation route must prove:

- root-only perturbation preserves the 29DoF intent invariant;
- actor current artifact and H intent have deployment provenance, while no
  future raw root or Clean field reaches the actor;
- Noisy and Repair branches share `x_t`, `I_s`, `C_s`, K, and scenario hash;
- FEMR has exactly one nonzero/authorized action at `t`, then is frozen;
- GMT consumes the common full Clean continuation after `t`;
- there are two scored causal roles and one PPO row per policy attempt;
- local intent/physics/cost Gain reaches returns, priority evidence, and
  evaluation through one active Gain owner;
- grouped PPO still gives equal motion -> Segment -> attempt mass and exactly
  one optimizer update per completed transaction.

### Bounded Implementation Evidence

`E-FI-8` completes the deterministic two-role reset/layout portion of the
required evidence: Noisy and Repair rows reuse one sealed `x_t`, current
artifact, q29 intent, Clean continuation, K, and hash; legacy scored roles and
generic future command reads fail closed.

`E-FI-9` completes deterministic candidate-only S1/S2 evidence for the next
local boundary: one Repair policy tuple is sampled at t, later actor samples are
guarded out, command-owned Clean C advances q29/dq29/root for frozen GMT K
steps, K masks remain exact, and the K executor closes without mutating the
sealed scenario so the next M attempt can reset it. It does not establish Gain,
return/advantage, priority, grouped PPO, checkpoint, formal-route, simulator,
training, or live evidence.

`E-FI-11` completes the next candidate-only deterministic S1 boundary: post-`t`
robot q29 and the same sealed deployment/Noisy `I[t]` reached the then-active
FRS-GAIN-v003 carrier. That evidence remains valid for carrier connectivity,
not for the superseding v004 ordering. It does not insert legacy storage,
mutate replay state, add PPO loss
mass, or establish diagnostics, evaluation, grouped PPO, checkpoint, formal
route, simulator, training, or live evidence.

`E-FI-12` completes the following candidate-only deterministic S1 diagnostic
boundary: the sealed v003 carrier projects into q29-intent/physics/cost/total
local-K diagnostics, while v015 actively rejects legacy v002 periodic/offline/
sequence evaluators. A separate deployment-composition protocol has no local
return, replay-priority, or PPO feedback. It does not execute a real evaluator,
composition sequence, simulator, training, checkpoint, or formal route.

`E-FI-13` completes candidate-only Step 4A storage/adapter evidence: every
Repair attempt is bound to exactly one immutable v015 metadata row carrying
transaction/snapshot/motion/start/Segment/source/trial plus scenario/hash/
`x_t`/q29-provenance/K/evidence-step identity. The evidence-step count derives
from actual frozen-GMT survival, does not create a policy row or actor-loss
mass, and legacy fixed-tape metadata or `to_ppo_batch()` fail closed. The
candidate-only grouped v003 loss accepts the complete metadata batch, but no
  formal runner, optimizer, persistence, simulator, training, or live route is
  claimed.

`E-FI-14` completes the bounded Step 4B S2 proof: an immutable v015 formal
transaction plan expects every `(source_index, trial_index)` row for at least
two selected Segments and two attempts each, checks one frozen old-policy
snapshot plus shared motion/start/Segment/scenario/hash/`x_t`/q29/K identity,
and accepts only grouped-candidate adapter shards. A CPU fake provider seals
the complete `2 x 2` transaction, invokes the unchanged grouped v003 loss once,
and records exactly one explicit optimizer step. The actor route selects q29
intent before normalization and excludes the legacy 65D tail. This is not a
generic training-entry, checkpoint/resume, simulator, or live-runtime claim.

`E-FI-15` established the bounded Step 4C S3 save/load and transaction-atomicity
mechanics under the pre-R3 generic-dimension fixture. `E-FI-22` supersedes its
layout-accuracy claim: `frontres_checkpointing.py` now saves and validates the
exact `(1,2)` q29 layout, `870+58=928` combined observation, `100+58=158` FEMR
prefix, `770D` frozen-GMT suffix, grouped-loss identity, and full-prefix
normalizer fingerprint before any mutable restore. A collecting, sealed, or
failed transaction cannot be saved or resumed; a completed transaction crosses
the checkpoint boundary only as a metadata-only exact-one-update receipt,
never as raw scenario references or candidate batches. This does not establish
generic checkpoint cadence/dispatch, real resume, simulator, training, or
live-runtime evidence.

`E-FI-16` completes Step 5A-S0 deterministic pre-live connectivity evidence:
an explicit v015-only entrypoint selects a sealed split local scenario, routes
it through Repair/Noisy reset, q29 actor input, one action/frozen-GMT K evidence,
the unchanged v003 candidate adapter, and the existing exact-one grouped update
owner. It preserves artifact/I/C/hash/`x_t` identity and rejects legacy tape,
HSL, and legacy batch routes. It is not a real-environment or live-runtime
claim.

`E-FI-23` completes R5 offline S2 observation connectivity with a semantic CPU
fixture and the real `_read_live_observations()` owner. The route reads the
actual `MultiMotionCommand.command` into a five-frame `290D` command history
inside raw `870D`, prepends the command-owned `58D` q29 H tail, normalizes the
combined `928D`, and exposes only `158D` to FEMR while frozen GMT consumes the
final `770D`. After the one t action, each Clean-C offset advances before a
fresh GMT observation is read; actor H is not reopened and no later FEMR action
occurs. Two fixed scenarios with two attempts each retain shared identity and
produce four grouped rows followed by exactly one update. This is not
simulator, live timing, training-quality, or deployment evidence.

`E-FI-25` records the first R6 live stop and its bounded implementation repair.
IsaacLab's automatic command callback had re-entered the legacy
`time_steps/reference/cache` clock after the unique t action, conflicting with
the explicit local-scenario current/C clock. `MultiMotionCommand` now owns one
clock dispatcher: sealed local current and Clean-C rows hold through command
compute, while non-local legacy rows keep their original ordered advance. The
direct duplicate-cache-install guard remains active. This is deterministic S1
evidence; the repaired S4 transaction remains unconfirmed.

## Forbidden Active-Path Assumptions

- `Clean x_t` means the actor observes Clean reference.
- H is a complete Noisy 65D future tape or K execution tape.
- K frames must be independently noised to evaluate the first repair.
- FEMR acts after `t` inside a one-row K experiment.
- current and future raw root/global reference are both trusted motion intent.
- Clean full rollout is the Style target for the actor.
- Repair-vs-Noisy rollout similarity is an intent-retention score.
- K, valid evidence-step count, replay priority, or best-of-M changes actor-loss mass.

## Owned Subcontracts

- Formal Stage 3 route: `../training/FRS-TRAIN-v008-critic-ready-v004-actor-curriculum.md`.
- Paired Gain: `../reward/FRS-GAIN-v004-support-mode-physics-admissibility.md`.
- Grouped PPO: `../optimization/FRS-PPO-v003-single-policy-row-k-evidence-grouped-reduction.md`.
- Evaluation: `../evaluation/FRS-EVAL-v003-local-repair-composition-evaluation.md`.
