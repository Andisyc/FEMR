---
contract_id: FRS-TRAIN-v004
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-19
supersedes: FRS-TRAIN-v003
superseded_by: FRS-TRAIN-v005
scope: HSL initialization, Actor/Critic warmup, and formal Stage 3 future-conditioned Double Segment Replay transaction route
---

# Future-Conditioned Double Segment Replay Training Contract

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| FRS-DP-02 | Segment Replay | SR-01 | Transaction Route |
| FRS-DP-08 | HSL Warmup | M-03 | HSL Interface Continuity |
| FRS-DP-09 | Actor & Critic Warmup | M-05 | Actor And Critic Warmup |
| FRS-DP-10 | Future Motion Context | M-11 | Future-Context Observation Route |

## HSL Interface Continuity

Stage 2 HSL initializes the same actor that Stage 3 optimizes. Under this
version, both stages consume the future-conditioned FrontRES observation:

    existing robot, balance, and tracking observation
    + current Noisy command reference
    + sparse future Noisy command references
    -> one full-6D actor distribution

The legacy 870D input is not an exception path for Stage 3. The final dimension
is derived from the active future-offset layout and command representation.
Stage 2 checkpoint metadata must therefore record the observation layout,
future offsets, command feature definition, and normalizer shape. A legacy
checkpoint may not silently load into the v013 actor.

HSL still learns a proposal-only full-6D Delta SE(3) repair. It does not learn
noise labels, perturbation timing, confidence, rho, acceptance, or a
family-specific action head.

For the active G1 route, the command representation is the canonical 65D
Noisy tape `[joint_pos(29), joint_vel(29), anchor_pos(3), anchor_quat(4)]`.
The existing current command/anchor observation is sourced from its cursor
frame; the added future tail is `[B, |H| * 65]`. `frontres_future_offsets` is
required and has no silent default. A loaded actor or normalizer whose declared
input layout cannot consume this tail must reject the v013 route before action
sampling; it may not silently use the legacy 870D artifact.

## Actor And Critic Warmup

The warmup schedule remains:

    HSL initialization
    -> critic warmup with actor held fixed
    -> actor warmup with introduced actor weight
    -> joint Actor/Critic Segment PPO

This protects optimization state, not the old observation interface. Every
phase must use the same v013 future-conditioned actor layout, full-6D action
identity, frozen GMT boundary, and grouped PPO contract.

## Future-Context Observation Route

For every selected Segment beginning at t:

    1. restore Clean dynamic replay state x_t;
    2. materialize one fixed Noisy reference segment R_tilde_s;
    3. install the current, sparse future, and K-step execution reference from
       R_tilde_s;
    4. construct the actor observation without Clean reference or perturbation
       truth;
    5. retain reference provenance and noisy_segment_hash in the rollout
       metadata.

Item 2 is a selection-time operation, not an attempt-time operation. The
selected scenario owns R_tilde_s through all M_s attempts; restoring x_t only
restores dynamics and never resamples reference corruption. Each attempt must
receive the same materialized values, or a deterministic rematerialization
whose noisy_segment_hash proves byte-equivalence. The scenario closes only
after its attempts and replay evidence have been aggregated; an immutable
cache may retain it, but no consumer may mutate or redraw it under that
scenario identity.

The reset at item 1 establishes repeatable dynamics. It must not be confused
with an actor-visible Clean command. No Noisy physics roll-in before x_t is
required or permitted by this route.

`MultiMotionCommand` is the sole runtime tape owner. Its execution cursor is
advanced by GMT K-step execution only; actor H reads are offset lookups and do
not advance it. The sampler lifecycle, reset adapter, and actor route therefore
have consumer-only authority over a sealed scenario tape.

## Transaction Route

The formal Stage 3 path is:

    Stage 3 configuration
    -> verify v013 actor/checkpoint/normalizer layout
    -> freeze one old policy snapshot
    -> select multiple replay Segments
    -> bind each Segment to Clean x_t and fixed R_tilde_s
    -> collect all M_s >= 2 policy-sampled attempts without optimizer mutation
    -> paired K-step Clean/Noisy/Repaired executable evidence
    -> transaction-complete policy storage with group metadata
    -> Critic/Actor warmup phase control
    -> grouped PPO loss across all eligible M_s attempts
    -> exactly one optimizer step
    -> rollout-evidence-only replay-priority update
    -> checkpoint and diagnostics

No helper or offline executor establishes formal-route integration unless it
reaches this exact transaction boundary.

## Trial Roles And Credit

Every action attempt has an explicit role and immutable identities:

    transaction_id, policy_snapshot_id, motion_id, segment_id, trial_index,
    noisy_segment_hash, horizon_k, and valid-step mask

All policy-sampled M_s attempts are policy-credit candidates. Their eligibility
depends on the ordinary reset/finite/valid mask, not on being the first
attempt. Search, manual, oracle, Clean, Noisy baseline, and counterfactual
actions may populate paired evidence or replay statistics but cannot be
relabeled as policy samples.

Each policy attempt may have paired Clean and Noisy execution branches. Those
branches provide Gain evidence; they are not extra PPO actions.

## K-step And Future-Horizon Contract

H and K have separate owners:

    future offsets H: actor information layout;
    horizon K: frozen-GMT execution, returns, and delayed-regret evidence.

The sampler can assign different K values only after the fixed Noisy segment
has enough coverage for both H and K. A row's K, reference hash, and valid-step
mask must survive reset, rollout, Gain, storage, grouped loss, replay update,
checkpoint diagnostics, and live evidence.

## PPO And Sampler Boundary

PPO consumes transaction-complete policy actions and the matching old-policy
statistics, returns, advantages, and group metadata. Its actor reduction and
sign-preserving group scale are owned by:

    ../optimization/FRS-PPO-v002-cross-segment-group-normalization.md

Sampler priority consumes rollout-time executable evidence. It must not read
post-update KL, parameter delta, logger state, or PPO loss weights. Priority
may determine selection but may not multiply the grouped PPO loss.

The paired Gain remains:

    gain_total = w_style * style_gain
               + w_physics * physics_gain
               - w_repair * repair_cost

with the configured component weights owned by:

    ../reward/FRS-GAIN-v002-style-physics-repair.md

Generic environment reward and the retired RP-only score have no fallback
authority.

## Required Diagnostics

- actor/checkpoint/normalizer future-layout identity;
- H offsets and K distribution as separate fields;
- current/future/execution reference provenance and noisy_segment_hash;
- Clean x_t reset fidelity, with no claim of a Noisy prefix roll-in;
- one old policy snapshot for all attempts in a transaction;
- selected, attempted, valid, PPO-eligible, and excluded row-role counts;
- per-motion, per-Segment, per-attempt, and per-valid-step loss mass shares;
- raw/bounded/executed full-6D actions, old mean/sigma, and policy action log
  probability;
- one optimizer step after transaction completion;
- replay-priority evidence after, never inside, the optimizer transaction;
- warmup phase, actor weight, actor parameter delta, and frozen-GMT state.

Missing runtime evidence is UNCONFIRMED. It must never be silently emitted as a
zero count or inferred from an offline helper.

## Acceptance

The future context, transaction, and grouped loss each require:

    implementation gate: deterministic S1 owner-contract test;
    integration gate: S2 formal-route connectivity/order test;
    live gate: S4 minimal real-runtime identity and diagnostic sentinel.

Long training stays blocked until all three mechanisms have passed their S1/S2
tests and the user reviews the S4 sentinel boundary. A live sentinel does not
authorize a policy-quality or recovery claim beyond its observed artifact.
