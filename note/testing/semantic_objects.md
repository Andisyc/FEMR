# FEMR Current Semantic Objects

Updated: 2026-07-27

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

Sampler states may assign `8/16/32/64`. One FEMR action is authorized at t;
FEMR is frozen for t+1...t+K while GMT consumes C. K must survive local-pair
construction, reset, rollout accumulation, done masks, returns, sampler
evidence, diagnostics, and evaluation.

Implementation and formal-route integration are separate evidence claims.

## Gain Decomposition

```text
intent_gain  = internal_fidelity(Repaired | I_noisy)
             - internal_fidelity(Noisy | I_noisy)
physics_gain = physics_quality(Repaired)       - physics_quality(Noisy)
repair_cost  = full6 magnitude + temporal change (+ valid Clean no-op cost)
gain_total   = w_intent * intent_gain + w_physics * physics_gain
             - w_repair * repair_cost
```

Owner contract:
`frontres_core/contracts/active/reward/FRS-GAIN-v003-intent-physics-local-repair.md`.

Current code status: Clean-global Style, full-65D tape, and quartet scoring are
contract-mismatch paths and do not implement this semantic object.

Required lifecycle:

```text
same x_t/current artifact/I/C/K in Noisy/Repair local roles
-> root-invariant intent and paired physics components
-> shared normalization/scales
-> per-row K aggregation
-> PPO return + sampler evidence
-> periodic/sequence eval
-> decomposed diagnostics
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
sigma, optimizer, Segment sampler, and any future persisted Gain scales. Load,
resume, evaluation, and export must agree on ownership and dimensions.

## Evaluation Evidence

Evaluation is independent of policy update and sampler mutation. Local K
evaluation reports root-invariant intent metrics, Physics metrics, Repair cost,
separate paired gains, total Gain, scenario identity, K, reset, continuation,
and action evidence. Full-sequence composition evaluation remains separate and
cannot enter a local PPO return.

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
