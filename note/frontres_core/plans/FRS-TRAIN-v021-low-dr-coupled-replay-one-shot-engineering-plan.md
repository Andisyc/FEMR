# FRS-TRAIN-v021 Low-DR Coupled Replay One-Shot Engineering Plan

Status: active
Date: 2026-08-11
Contracts: FRS-METHOD-v022 / FRS-GAIN-v008 / FRS-PPO-v009 / FRS-TRAIN-v021
Terminal outcome: verified local training-ready command, not launched

## Engineering Boundary Record

Requested behavior: replace Critic-only with low-DR joint Actor/Critic
adaptation; couple Actor weight and DR growth; make outer replay phase-aware and
current-DR-compatible; persist the new identity strictly.

Preserved behavior: Actor 158D, Critic 449D, full-6D Repair, M4, K8/16/32, raw
Gain-v008, per-attempt symlog utility, exact-M mean target, split LR 3e-6/1e-5,
separate clipping, exact-one Adam step, GMT and simulator.

Owners and public boundaries:

- schedule/DR owner: `frontres_segment_warmup.py` maps committed iteration and
  explicit schedule to phase, Actor weight, `d_cap` and four-class sample;
- replay owner: `frontres_outer_scenario_replay.py` plans two sources, stages
  `E_V/E_A`, commits records/RNG and serializes replay v2;
- orchestration: sampler supplies current curriculum and materializes the
  owner's immutable plan; formal transaction consumes it and commits once;
- persistence: checkpointing owns checkpoint-v16 strict save/load.

Public input/output: committed iteration and ten-field schedule -> immutable
K/M/phase/DR identity; identity + dataset eligibility -> two immutable replay
selections; exact-M advantages -> staged two-score replay candidate; matching
receipt -> one committed state delta.

Dependency direction: runner/simulator adapters depend on deterministic
schedule/replay owners. Owners may consume immutable tensors/records but never
runner/env private state. No new wrapper, service, registry, open dict payload,
silent fallback, second scheduler or MOSAIC change.

State boundary: selection is a preview; records, both score maps, staleness and
RNG mutate only with the matching exact-one receipt. Strict resume rejects v15
before mutation. Evaluation remains read-only.

Legacy characterization: the passing v020 warmup, replay and transaction tests
pin old behavior. The old Critic-only phase, single score map, replay schema v1
and checkpoint-v15 are superseded and must be rejected, not retained as an
active fallback.

Effect Sketch / Pinch Points:

```text
run_stage3.sh config
 -> schedule parser/identity [Pinch: warmup owner]
 -> class-before-source selection [Pinch: replay.plan]
 -> sealed Scenario materialization [Seam: sampler callback]
 -> exact-M utility advantages
 -> E_V/E_A candidate [Pinch: replay.stage]
 -> one Adam receipt
 -> atomic replay commit [Pinch: replay.commit]
 -> checkpoint-v16 save/load [Pinch: checkpoint identity]
```

Hotspot decision: keep deterministic decisions in the two existing deep owners.
The large sampler/checkpoint modules receive orchestration and identity edits
only; no new semantic responsibility or speculative abstraction is admitted.

## TDD Implementation Batches

1. Update contract sentinel and schedule pseudo-samples; verify RED for nonzero
   first Actor weight, new phases and coupled `d_cap`, then implement the
   schedule owner.
2. Add replay pseudo-samples for hand-computed `E_V/E_A`, class-before-source,
   absolute-interval eligibility and empty-pool fallback; verify RED, then
   implement replay schema v2.
3. Add formal transaction cases proving first-transaction Actor/Critic deltas,
   phase-selected score carriage, current-DR identity and atomic failure; verify
   RED, then route sampler/formal consumer changes.
4. Add checkpoint-v16 roundtrip, v15 pre-mutation rejection, optimizer/LR/
   normalizer/curriculum/replay equivalence; verify RED, then update persistence,
   CLI/config and telemetry identities.
5. Run focused cases, compile, affected contract suite, official offline pseudo-
   transaction, construction/final review and formal long-training audit. Update
   Atlas/register/ledger with evidence and produce one cold-start command.

## Confirmed Module Test Cards

### TEST-22A Coupled Schedule

Production boundary: `resolve_frontres_k_stage_identity`.
Fixture/oracle: asymmetric small stage counts with hand-computed phase, weight
and linear DR progress at first, boundary and K-transition iterations.
Expected: first Actor weight is positive; Actor/DR progress are monotonic;
joint begins at weight/progress one; each K resets low without freezing Actor.
Counterexample: any zero Actor weight or DR growth delayed until joint.

### TEST-22B Phase-Aware Replay Scores

Production boundary: `FrontRESOuterScenarioReplay.stage/plan`.
Fixture/oracle: two M4 advantage groups with different mean error and centered
spread. Hand-compute `E_V` and `E_A`; seed records at two K values.
Expected: warmup ranks by `E_V`, joint by `E_A`, and both EMA maps persist.
Counterexample: mean absolute advantage used for both phases or K leakage.

### TEST-22C DR-Compatible Selection

Production boundary: `FrontRESOuterScenarioReplay.plan`.
Fixture/oracle: records placed inside and outside each current absolute class
interval with a deterministic class draw.
Expected: class is drawn before source; only current-interval records qualify;
empty replay/review falls back to global in the same class.
Counterexample: stored class label or foreign-strength record bypasses interval.

### TEST-22D Formal Commit And Persistence

Production boundary: official formal transaction plus checkpoint save/load.
Fixture/oracle: one two-Scenario M4 transaction with distinct Actor/Critic
gradients and replay scores, then strict roundtrip and corrupted/v15 cases.
Expected: exactly one Actor, Critic, optimizer, replay and normalizer transition;
all identities restore; incompatible state rejects before mutation.
Counterexample: frozen Actor, duplicate update, partial replay mutation or
first-consumer divergence.

Human status: confirmed by the 2026-08-11 Design Inspector confirmation and
one-shot implementation authorization; cards introduce no new semantics.

## Stop Conditions

Stop before a training command on any Contract/Inspector mismatch, zero first-
transaction Actor delta, wrong phase score, DR interval leak, old-schema silent
load, open P0/P1, failed module case, incomplete official offline route or
unresolved persistence identity. Runtime policy quality is explicitly unclaimed.
