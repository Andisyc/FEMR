# FRS-TRAIN-v020 Outer Scenario Replay One-Shot Engineering Plan

Status: offline-ready; official bounded transaction pending
Date: 2026-08-10
Contracts: FRS-METHOD-v021 / FRS-TRAIN-v020
Closure: formal long-training command ready, not launched

## Change Contract

Requested behavior: close outer replay with stable seeded Scenario identities,
rank plus staleness selection, current-policy M4 recollection and committed-only
state mutation.

Preserved behavior: Actor 158D, Critic 449D, full-6D action, Gain-v008,
PPO-v008, symmetric-log utility, K/M/DR, split LR, GMT and simulator.

Semantic owner: one `FrontRESOuterScenarioReplay` Aggregate. Its consumers are
the formal transaction preparation, formal commit telemetry and checkpoint
serializer. The runner orchestrates but does not recalculate selection or
priority.

Public boundary:

```text
select(current_k, global_candidates) -> two immutable selections
stage(visit evidence, V_old, U(G), transaction identity) -> validated delta
commit(delta, committed receipt) -> exact-one state transition
state_dict/load_state_dict -> strict checkpoint-v15 replay identity
```

The ScenarioKey is a validated immutable Value Object. The replay state is
FrontRES domain data and has no simulator dependency. A narrow RNG-isolation
adapter supplies the key's perturbation seed to the unchanged MOSAIC command
materializer.

Dependency direction: runner and checkpointing depend on the owner projection;
the owner depends only on tensors/scalars and Scenario identities. Gain/PPO,
simulator objects, old PPO rows and post-update gradients are forbidden inputs.

Transaction boundary: staged pre-update delta plus a matching committed receipt
form one Unit of Work. Assignment after validation is non-throwing. Failed,
partial and duplicate transactions cannot partially mutate replay state.

Legacy: bare-segment priority remains available only to historical/non-formal
routes. The active formal TRAIN-v020 route must not call it.

Hotspot decision: the 1060-line segment sampler already has Divergent Change
pressure. Scenario identity, per-K ranking and commit lifecycle therefore move
to one CCP-coherent module instead of adding another responsibility there.
The formal runner gains orchestration only; no wrapper or class hierarchy is
introduced.

## Effect Sketch And Pinch Points

```text
run_stage3.sh config
 -> runner construction
 -> current-K selection [Pinch: outer replay owner]
 -> seeded local materialization [Seam: command call; Enabling Point: runner adapter]
 -> frozen-policy exact-M collection
 -> PPO pre-update values/utility
 -> staged replay delta [Pinch: owner.stage]
 -> one Adam step and transaction receipt
 -> replay commit [Pinch: owner.commit]
 -> checkpoint-v15 save/load
```

## Implementation Batches

1. Add immutable ScenarioKey/record/selection/staged-delta and deterministic
   outer replay owner with strict persistence.
2. Route formal selection and isolated seeded materialization through the owner.
3. Stage priority from pre-update Actor advantages and commit it with the
   matching transaction receipt; add diagnostics.
4. Bump Contract/config/checkpoint identity to v020/v15 and reject v14 resume.
5. Update Atlas, test registry, evidence ledger and command package.

## Confirmed Module Test Cards

### TEST-21A Scenario Identity

Responsibility: reproduce one sealed local Scenario without changing global
policy RNG.

Input: asymmetric motion/frame/K/family/strength/seed identity and deterministic
materializer fake.

Expected: same key reproduces the same artifact/hash; a seed change changes the
key; caller RNG state is restored.

Falsifier: equal key yields different hash, or replay changes Actor RNG.

Forbidden shortcut: snapshotting current output as the oracle.

### TEST-21B Selection And Learning Value

Responsibility: select two distinct current-K Scenarios from 40/50/10 sources.

Input: tiny labeled records with hand-computed utility errors, ranks and
staleness, including negative raw Gain.

Expected: priority uses mean absolute utility error; rank and staleness control
replay; empty pools fall back to global; K8 score is invisible at K16.

Falsifier: positive-Gain clamp, raw-magnitude weighting, duplicate key or K leak.

### TEST-21C Committed Mutation

Responsibility: mutate replay state once for one matching committed receipt.

Input: valid staged delta plus matching, mismatched, rejected and duplicate
receipt cases.

Expected: matching commit changes seen/score/staleness/RNG exactly once; all
other cases produce zero delta and fail closed where applicable.

Falsifier: any partial or duplicate mutation.

### TEST-21D Persistence

Responsibility: strictly restore the active replay owner.

Input: small state with two ScenarioKeys, different K scores and advanced RNG.

Expected: save/load reproduces selection and complete state; missing/foreign
fields and checkpoint-v14 reject before mutation.

Falsifier: silent default, reconstructed key or first-consumer divergence.

Human status: confirmed by the 2026-08-10 one-shot execution authorization.

## Review And Stop

Plan review must be READY before production edits. Construction and final-gate
reviews use the active engineering discipline and research-ML profile. S0/S1,
S2 official-offline and S3 persistence must pass. S4 simulator evidence is
live-only and is not attempted without separate bounded-live authority.

Stop before publishing a formal command if any P0/P1, module mismatch, formal
edge gap or persistence gap remains.

## Execution Closure

Offline module, formal-transaction, checkpoint-v15, telemetry, historical
Evaluation compatibility, design sentinel and the complete 57-contract suite
pass. Construction review found and closed the missing outer replay telemetry
projection; formal audit now reports exact-one replay state delta, source class,
learning value, EMA and pool sizes. No open offline P0/P1 remains.

The only remaining gate is one official bounded K8/M4 simulator transaction on
the synchronized server. It must show two stable key digests, finite learning
values, `optimizer_step_delta=1`, `outer_replay_state_delta=1`, and atomic
checkpoint-v15 readback. Long training must not start before that log is
reviewed.
