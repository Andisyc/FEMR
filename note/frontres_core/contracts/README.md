# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v016-physics-constrained-intent-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v010-intent-critic-k-curriculum.md` | active |
| Reward | `active/reward/FRS-GAIN-v005-vector-physics-constraints.md` | active |
| Optimization | `active/optimization/FRS-PPO-v004-grouped-constraint-gradient-projection.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v003-local-repair-composition-evaluation.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v016` / `Preserved Replay Authority` | `M-02` | Existing v015 evidence remains valid; P2 must prove the new loss path does not change scenario identity. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v016` / `Preserved Replay Authority` | `SR-01` | Existing sealed multi-Segment x M evidence remains valid; P2 must preserve one equal-mass committed update. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v016` / `Preserved Replay Authority`; `FRS-TRAIN-v010` / `Per-K Recalibration` | `M-06` | v009 runtime evidence is historical compatibility evidence only; P2 must implement the v010 target and checkpoint-v5 identity. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v016` / `Actor And Information Boundary` | `M-04` | The 158D/full-6D authority remains unchanged; P2 must ensure constraint evidence cannot enter actor observations. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v016` / `Preserved Replay Authority` | `M-10` | Existing one-action-K evidence remains valid; P2 changes only its loss-side interpretation. |
| `FRS-DP-06` | Paired Rollouts | `FRS-GAIN-v005` / `Paired Evidence Authority` | `Q-PAIR` | E-FI-81 preserves the same sealed paired roles while replacing formal ZMP proxy evidence with contact-wrench ZMP against immutable Clean-foot envelopes. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v005` / `Scalar Objective And Physics Constraints`; `FRS-PPO-v004` / `Grouped First-Order Projection` | `Q-01` | v005/PPO-v004 mathematics are source-connected; E-FI-81 closes estimator/carrier/persistence offline, with official raw-sensor S4 still open. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v010` / `Actor-Only Initialization` | `M-03` | HSL-v1 remains frozen and actor-only; P2 must not change its target or payload. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v010` / `Fresh Target Entry And Per-K Recalibration` | `M-05` | Fresh v010 target entry rejects v004 Critic state; each global K increase recalibrates the same v010 Critic with actor/std frozen before ramp and joint. Source support is P2-pending. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v016` / `Actor And Information Boundary` | `M-11` | Existing deployment/Noisy q29 H evidence remains valid and is not changed by P1. |

The physics-constrained Intent migration is governed by
`../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md` and its
paired checklist. `FRS-PPO-v004` preserves v003 one-row grouped equal-mass
reduction but replaces the actor update with a joint constraint-gradient
projection; `E-FI-27` closes the bounded S4 local identity and
exact-one-update route. `E-FI-71` additionally closes the bounded K-transition
and actual checkpoint-cadence gate. Long-training convergence, policy quality,
and Step 5B deployment composition remain unconfirmed. `E-FI-28` completes the immutable Step 5B-S1 deployment
request/report kernel, and `E-FI-29` completes only the S2A command-owned
current/H carrier plus read-only runtime connector. `E-FI-30` completes the S2
CPU formal executor with unclamped `T-max(H)` frames, frozen GMT, atomic report,
and zero training-state mutation. `E-FI-31` adds the v015-only server CLI and
proves registered-task/checkpoint/CUDA/config dispatch without Segment sampler
or training/update calls. `E-FI-32` records that no compatible trained v015
checkpoint or defined external Noisy file exists: the checkpoint must be
produced by the new-layout training loop, and composition must be paired
against frozen-GMT baseline. `E-FI-46` now closes the controlled-corruption
materializer: an external unexplained `Noisy.npz` is no longer a prerequisite.
The CLI is implemented-not-runnable and S4 is blocked. Simulator timing,
physical metric values, and live composition
remain open. No
long-training recommendation follows from contract activation or one bounded
sentinel alone.

`E-FI-42` closes G2 proposal-only HSL readiness with one bounded Main-2 run.
`E-FI-43` adds deterministic actor-only migration of that strict HSL-v1
artifact into the fresh Stage-3 928/158/770 q29/grouped/formal configuration;
critic, optimizer, sampler, transaction, and online HSL state remain excluded.
The resulting `model_warmup.pt` is still an initializer, not a trained Stage-3
repair policy. `E-FI-44` connects ordinary Stage-3 to complete multi-Segment x M
selection, sealed grouped exact-one update, and a matching committed-only save
trigger while isolating the legacy immediate-update route. `E-FI-45` connects
the same semantic 158D/6D policy through exact-one update, actual v015
`save_runner()`, and strict fresh-inference reload with exact q29, prefix
normalizer, and proposal identity. This closes G3 engineering readiness only;
bounded training and policy quality remain open under G5.
`E-FI-46` closes G4 with deterministic source/protocol/carrier/q29 identities,
one sealed no-resample lifecycle, and existing command current/H consumption;
it does not execute composition or expose corruption metadata to the actor.

Entrypoints, configuration, storage, checkpointing, diagnostics, probes, and
tests are implementation objects under these design points. They are not
additional top-level method designs.

## Active Contract / Implementation Stop

`E-FI-72` records the user-confirmed method change and `E-FI-73` activates its
coordinated METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010 contracts. Noisy
rollout remains the same-scenario
zero-action counterfactual. Contact, phase-conditioned ZMP, and survival must
remain separate K-step Physics constraints; paired Intent improvement minus
repair cost becomes the scalar objective and the only scalar-Critic target.

The human Concept Figure, active contracts and formal source route agree.
E-FI-74 closes deterministic METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010
connectivity; E-FI-75 live-confirms one constrained K8 transaction and
checkpoint-v5; E-FI-77 closes strict v5 full resume and raw quality evidence;
E-FI-78 advances the same K8 Critic to the iteration-200 actor-warmup boundary.
E-FI-79 records the later `model_251.pt` -> `model_2000.pt` run and exposes four
post-rescale recovery KKT violations. E-FI-80 repairs that implementation
mismatch without changing PPO-v004 semantics: recovery is reprojected after
norm scaling and the formal telemetry consumer enforces checkpoint-v5
tolerance. E-FI-81 closes the later Physics evidence-authority mismatch offline:
formal phase-ZMP now uses per-contact wrench data against sealed Clean-foot
support envelopes, and checkpoint-v5 binds that evidence identity. Official
IsaacLab raw-contact S4 remains unconfirmed. Silent v004/v003/v009 fallback
remains forbidden. Policy efficacy and the disposition of the pre-fix
checkpoint lineage remain unconfirmed.

## Reading Rule

1. Read this registry first.
2. Read only the active contracts required by the task.
3. Do not scan `history/` for additional context.
4. Read one named historical contract only when the user requests historical
   comparison, an active contract cites it, or a current contradiction cannot
   be resolved from active contracts and fresh code evidence.
5. History never overrides an active contract or current code/runtime evidence.

## Version Rule

Use category-local monotonic versions such as `FRS-METHOD-v010` or
`FRS-TRAIN-v002`.

- Create a new version when method semantics, variable ownership, training
  signal, authority boundary, or formal runtime path changes.
- Keep the version when only wording, paths, or evidence links change; update
  `updated_date` instead.
- Record `effective_date`, `status`, `scope`, and `supersedes` in every contract.
- Move replaced contracts to `history/<category>/` and mark them `superseded`.
- Use `rejected` for designs that should not be reused and `ablation` for
  intentionally retained experimental branches.
- Keep drafts in plans, not contracts.

## Historical Migration

`history/method/FRS-METHOD-v000-design-history-compendium.md` preserves the
original accumulated design document. It is not an active contract. Split it
into named historical contracts only after each section's semantic version and
status are reviewed.

The reviewed Method sequence and supersession chain are indexed in
`history/method/README.md`.

Raw Segment Replay plans, checklists, logs, intake snapshots, and reference
notes were migrated without semantic rewriting to
`history/sources/segment_replay/`. They are evidence sources, not active
contracts, and must not override the active set above.
