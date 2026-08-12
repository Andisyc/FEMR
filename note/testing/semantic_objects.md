# FEMR Current Semantic Objects

Updated: 2026-08-12

## Current-Visit Scenario Replay

Outer Replay owns scheduling, not historical training samples. A selected
ScenarioKey is rerun by the current frozen `pi_old` for exact M4 attempts before
the update. The Critic target is the arithmetic mean of those four current
symlog utilities; all four current utilities independently form Actor
advantages. M4 variance/SE/h95 and current Critic excess error are selection
diagnostics only. Persisted Replay state contains identity, latest priority,
lifetime committed visits, staleness, pool/capacity, RNG and last receipt; it
contains no historical utility, policy anchor, compatibility KL or reset.

## Observation Payload

```text
current robot/balance/tracking state
+ current Noisy root/anchor artifact
+ future 29DoF internal-motion intent I[t:t+H]
-> FrontRES actor observation
```

Owners: `frontres_observation_layout.py`, `normalizer.py`,
`front_residual_actor_critic.py`, checkpoint/eval/export loaders.

The legacy 870D layout and full-65D future reference prefix are implementation
mismatches until migrated. Evidence must prove actor provenance, q29 invariant,
root exclusion, checkpoint identity, and all sink dimensions.

## Full-6D Repair Action

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Owners: policy distribution, rollout step, task-space correction, Segment
storage, Segment PPO, checkpoint, eval.

Invariant: sampled action, old log-prob, old mean/sigma, stored action, and
executed repair share one representation. Perturbation family never narrows the
six dimensions.

## Segment Identity And Trial Role

One local scenario owns motion id, start frame, dynamic reset x_t, one current
root artifact, future intent I, Clean continuation C, horizon K, source, trial
index, and rollout evidence.

The two scored roles are Noisy and Repair. Candidate/search rows are not a third
active scored role. Ordinary-valid Repair attempts may reach PPO; non-policy
evidence remains invalid before the batch boundary.

## Effective Horizon K

Sampler states assign `8/16/32` under TRAIN-v012. One FEMR action is authorized at t;
FEMR is frozen for t+1...t+K while GMT consumes C. K must survive local-pair
construction, reset, rollout accumulation, done masks, returns, sampler
evidence, diagnostics, and evaluation.

Implementation and formal-route integration are separate evidence claims.

## Gain Decomposition

```text
one executed Clean baseline + one fixed zero-action Noisy baseline
+ M Repair attempts
-> Clean-conditioned Intent and Physics remaining problems
-> signed Noisy-to-Repair improvement
-> continuous remaining Physics pressure
-> one scalar Recovery-Aware G_total - full-6D repair cost
```

Owner contract:
`frontres_core/contracts/active/reward/FRS-GAIN-v007-clean-anchored-recovery-aware-ranking.md`.

Intent and Physics jointly define the scalar Recovery-Aware ordering. The
retired independent Physics projection/KKT route cannot reach PPO-v005. Clean
is an executed evaluator anchor only; it never enters actor input. Full-65D
tape, quartet scoring, and v006 scalar/constraint semantics are historical.

Required lifecycle:

```text
same x_t/current artifact/I/C/K in Clean/Noisy/M-Repair evidence lifecycle
-> fixed-unit, valid-time-normalized K evidence
-> Clean-conditioned Intent/Physics remaining problems
-> v007 Recovery-Aware scalar return
-> every valid attempt enters grouped equal-mass exact-one PPO-v005
-> active held-out and deployment evaluation remain no-feedback
-> decomposed diagnostics expose missing evidence as UNCONFIRMED
```

## Raw Foot-Ground Contact Evidence

Aliases: raw contact points, normal forces, contact normals, contact counts,
contact starts, raw filtered ContactSensor views, and contact-wrench ZMP input.

Owner path:

```text
foot-to-ground filtered ContactSensor views
-> frontres_segment_live_probe.py raw-row adapter
-> [B, foot, C, 3] points/normals + [B, foot, C] force/mask
-> frontres_balance.py contact-wrench ZMP
-> paired one-action-K Physics evidence
```

`C` is a per-foot PhysX capacity and may differ between left and right feet.
The adapter must right-pad both feet to `C_max`, preserve every original value,
and mark padded slots invalid before concatenating the foot axis. This object is
ephemeral runtime evidence: it is not an actor input or checkpoint payload.

## Role-Specific ZMP Applicability

For each valid policy row, Repair and Noisy independently carry whether at
least one K-step has actual loaded support under the expected Contact phase:

```text
zmp_applicable_repaired [B]
zmp_applicable_noisy    [B]
zmp_constraint_applicable == zmp_applicable_repaired
```

Repair and Noisy aggregate margins are finite exactly under their own masks.
Paired `physics_zmp_gain` is finite exactly when both masks are true. The PPO
constraint consumes only the Repair alias; diagnostics and evaluators retain
both identities. Applicability may not be reconstructed from finite/NaN values.

## Rollout Transaction Identity
`audit_transaction_id` and `audit_batch_signature` are diagnostic evidence
metadata, not reward, action, or PPO features. One capture creates them from
the ordered row tuple `(segment_id, role, motion_id, start_frame, effective_K)`.
Cards 15/16/17 must preserve the same pair; storage rejects a mixed transaction.
Card 22 reports the set/count of transactions after update-loop aggregation and
must distinguish `single` from `aggregate`.

## PPO Tuple

The Segment PPO tuple contains full-6D action, old log-prob, old mean/sigma,
value, return, advantage, and valid mask. Old-policy tensors are detached.
Trial metadata and Gain diagnostics do not become new PPO action dimensions.

## Checkpoint State

Checkpoint identity includes policy architecture, observation stats, action
sigma, optimizer, Segment sampler, current target normalizer and Replay
selection state. TRAIN-v024 uses checkpoint-v19/replay-v5; numerical historical
outcomes are forbidden persistence. Load, resume, evaluation, and export must
agree on ownership and dimensions.

## Evaluation Evidence

Evaluation is independent of policy update and sampler mutation. EVAL-v004
keeps three capabilities: held-out one-action-K policy quality, full-sequence
deployment composition, and the independent DR sweep. Training may schedule a
held-out run, but it does not own a fourth evaluator. Retired offline/sequence
evaluators cannot enter the active route.

Local K evaluation reports the executed Clean/Noisy/Repair evidence lifecycle,
v007 Intent/Physics decomposition, Repair cost, total Gain, scenario identity,
K, reset, continuation, and action evidence. Full-sequence composition remains
separate and cannot enter a local PPO return.

Missing evidence is `UNCONFIRMED`, never zero.

The v015 deployment-composition S1 object is an explicit structured `.npz`
identity plus a canonical `persistent_full_sequence` corruption protocol. Its
immutable report owns per-frame FEMR action use, q29 intent error, physics
success/fall, ZMP margin, contact consistency, and accumulated failures. It has
no local return, priority, sampler, PPO, optimizer, Clean continuation, or
local-scenario field. `E-FI-29` connects only the immutable request to a
command-owned q29/dq29 sequence and read-only current/H snapshot. Per-frame
actor/GMT execution, metrics, atomic report production, and formal dispatch are
CPU-connected at `E-FI-30` for exactly `T-max(H)` unclamped frames. The formal
current code input is a pre-materialized deployment `.npz`; S2B does not infer
or draw corruption. `E-FI-31` adds one v015-only CUDA-visible CLI that resolves
checkpoint/file/report identities without requesting Segment Replay or calling
learn/update. `E-FI-32` corrects the target semantic object: the user supplies
an ordinary reference `.npz`, while a planned selection-time owner creates one
fixed controlled artifact carrier and records source/protocol/carrier hashes.
The trained v015 checkpoint is produced by training, not supplied before it.
Composition requires paired baseline/repair execution. Current CLI/S2B remain
implemented-not-runnable until G1--G6 close; physical metrics and simulator
timing remain the G7 S4 boundary.
