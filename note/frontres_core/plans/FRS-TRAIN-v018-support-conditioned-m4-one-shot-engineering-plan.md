# FRS-TRAIN-v018 Support-Conditioned M4 One-Shot Plan

```yaml
plan_id: FRS-TRAIN-v018-support-conditioned-m4-one-shot
status: executed-through-k8-m4-training-start
date: 2026-08-09
owners: [workflow-governance, frontres-observation, frontres-stage3]
human_authority: user-confirmed support-conditioned Critic and all-stage M4 through training start
```

## Decision And Change Contract

Requested behavior: keep one scalar state-value Critic, but make its state
sufficient for action-pre executability by adding current actual foot Contact,
per-foot load fractions, current contact-wrench ZMP applicability/margin, and a
masked Kmax=32 planned-support sequence. Use exact M=4 from K8 onward so the
arithmetic Segment-mean target is estimated from four frozen-policy attempts.

Preserved behavior: Actor input remains 158D, GMT input remains 770D, Repair is
direct world-frame full-6D, Gain remains FRS-GAIN-v007, PPO remains
FRS-PPO-v007, the Critic target remains the arithmetic exact-M mean, Actor credit
remains per-attempt, split LR remains 3e-6/1e-5, K/DR phase durations remain
unchanged, and each committed transaction performs exactly one grouped update.

Single owners and consumers:

- `frontres_segment_physics.py` is the IsaacLab Gateway for one immutable 102D
  action-pre support context; it is the only simulator/sensor reader.
- `frontres_observation_layout.py` owns the deterministic 289+58+102=449 layout
  and fail-closed shape/finite/detach contract.
- `frontres_segment_warmup.py` and `frontres_interfaces.py` own the one
  K8/M4 -> K16/M4 -> K32/M4 transaction shape.
- `frontres_segment_one_action_k.py` only orchestrates those owners before the
  policy action; PPO, checkpointing and telemetry consume their projection.

Public input/output: current privileged `[B,289]`, sealed Noisy future Intent
`[B,58]`, and support context `[B,102]` produce detached finite Critic state
`[B,449]`. The support block is ordered as Contact `[B,2]`, load fractions
`[B,2]`, ZMP applicable/margin `[B,2]`, planned support `[B,32,2]`, and valid
mask `[B,32]`. Inapplicable current ZMP uses `(applicable=0, margin=0)`; padded
future steps use `(support=0, mask=0)`. These zeros are explicit masked
encodings, not missing-evidence fallback.

Dependency direction: simulator command/sensors -> physics Gateway -> immutable
support tensor -> observation composer -> policy Critic. Domain layout and PPO
must not import IsaacLab objects. Actor/GMT/Gain/sampler must not consume the
support tensor.

State and transaction boundary: support context is captured after the sealed
scenario reset and before sampling/applying the Repair action, then reused by
all M4 attempts sharing that Segment. It owns no mutable training state. A
complete two-Segment transaction contains 8 policy rows, 16 Repair/Noisy role
rows, and one optimizer/checkpoint commit.

Forbidden dependencies: sampled 6D action, Repair-after Contact/ZMP/survival,
`G_total`, evaluator outputs, noise labels, private simulator fields, silent
fallback, alternate Critic input paths, or MOSAIC host changes.

Legacy boundary: 347D METHOD-v018 / TRAIN-v017 / checkpoint-v12 is archived and
strictly rejected. HSL-v2 remains a valid Actor-only initializer and never
provides a Critic, optimizer or normalizer migration.

Hotspot/effect sketch: the observation composer is the WELC Pinch Point;
`_read_live_observations` is the existing Enabling Point; the physics Gateway is
the Humble Object isolating volatile sensor reads. Responsibilities added are
one 102D capture and one strict layout. No wrapper, service, second optimizer,
second Critic, replay dataset or Gain path is added.

## Coarse Execution

1. Activate METHOD-v019 / TRAIN-v018 / checkpoint-v13 and update the Design
   Inspector plus Module Test Cards.
2. Write failing semantic tests, then implement the 449D action-pre route, M4
   schedule, persistence and telemetry as one narrow change.
3. Run focused and aggregate regressions, construction review, formal runtime
   audit, Git sync and one bounded official transaction.
4. Only if the bounded receipt proves identity, finite values, exact-one update,
   nonzero Critic delta and checkpoint-v13 roundtrip, start a fresh HSL-v2
   cold-start long run in a new output directory.

## Module Test Cards

- `TEST-02 Training Config`: K8/K16/K32 all resolve M4, 16 env rows, unchanged
  phase/DR values and one immutable fingerprint; old M2/M3 schedules reject.
- `TEST-05 Observation Layout`: hand-built current support and plan fixtures
  produce exact 102D ordering and 449D Critic state; permutation preserves rows;
  malformed/gradient/nonfinite/action-dependent payloads reject.
- `TEST-15 Segment PPO`: M4 arithmetic mean remains the Critic target while all
  four realized returns retain their own Actor advantages; no winner filtering.
- `TEST-16 Checkpointing`: checkpoint-v13 round-trips 449D normalizer and exact
  identities; v12 or malformed support identity rejects before mutation.
- `TEST-18 Runtime Diagnostics`: official telemetry reports 158/449/770,
  action-pre support identity, M4 row counts, split LR and committed-only state.

Independent oracles are hand-computed load fractions, contact-wrench ZMP,
planned-mask ordering, exact arithmetic means, parameter snapshots, optimizer
step count and atomic checkpoint sentinels.

## Stop Conditions

Stop before long training on any identity/shape/finite/provenance mismatch,
action-dependent Critic input, Actor/GMT/Gain drift, non-M4 stage, partial
transaction mutation, shared gradient clipping, more than one optimizer step,
zero Critic delta in critic-only, failed checkpoint-v13 fresh reload, or missing
formal runtime receipt. A bounded sentinel is lifecycle evidence only and does
not establish policy quality.

## Execution Receipt

Offline compile, Design Inspector validation, focused contracts, construction
review and the 53-contract aggregate suite passed. Server commit `0d8e412`
then completed one bounded official transaction with exactly one each of
`AUDIT-B01..B08`, zero runtime error matches, nonzero Critic delta, frozen
Actor/std, exact-one optimizer step and checkpoint-v13 readback. Fresh HSL-v2
K8/M4 training subsequently started on GPU 0 as PID `3372457`; its log is
`/hdd0/yuxuancheng/FEMR/log/FRS_TRAIN_V018_K8_M4_FULL_COLDSTART_20260809.log`.
