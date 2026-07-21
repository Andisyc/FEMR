# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v015-future-intent-single-action-k-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v007-proposal-only-hsl-future-intent-transaction.md` | active |
| Reward | `active/reward/FRS-GAIN-v003-intent-physics-local-repair.md` | active |
| Optimization | `active/optimization/FRS-PPO-v003-single-policy-row-k-evidence-grouped-reduction.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v003-local-repair-composition-evaluation.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v015` / `Local Root-Artifact Scenario` | `M-02` | `E-FI-27` live-confirms two root-artifact local scenarios with paired immutable hashes; the legacy full-tape route remains excluded. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v015` / `Frozen-Policy Multi-Attempt Transaction` | `SR-01` | `E-FI-27` live-confirms two Segment sources with M=2, one frozen snapshot, equal attempt mass, and exactly one update after all rows seal. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v015` / `Single-Action K-step Evidence` | `M-06` | `E-FI-27` live-confirms one action/PPO row per attempt, K=8, eight valid evidence steps, and no K-dependent actor mass. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v015` / `Future Intent Context` | `M-04` | `E-FI-27` live-confirms the local `870+58=928`, FEMR `158D`, critic `289D`, and frozen-GMT `770D` route; `E-FI-30` CPU-confirms one 6D FEMR action on every unclamped deployment frame; `E-FI-42` live-confirms the proposal-only HSL initializer on the same 158D actor interface. No Stage-3 v015 policy has yet been trained, so repair quality remains open. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v015` / `Method Boundary` | `M-10` | `E-FI-27` live-confirms one local t action, eight post-advance Clean-C GMT reads, and zero later FEMR actions; `E-FI-30` CPU-confirms the per-frame suffix route. A same-carrier frozen-GMT baseline versus FEMR+GMT paired composition owner is still missing at `E-FI-32`. |
| `FRS-DP-06` | Paired Rollouts | `FRS-GAIN-v003` / `Two-Role Pairing And Time` | `Q-PAIR` | `E-FI-27` live-confirms four Repair plus four Noisy role rows and M=2 identity pairing for two fixed scenarios; legacy quartet remains excluded. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v003` / `Core Decision` | `Q-01` | `E-FI-27` live-confirms the v003 carrier reaches four valid grouped rows and one update; component quality and sampler-state evolution remain unconfirmed. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v007` / `HSL Proposal-Only Initialization` | `M-03` | `E-FI-35` adds the minimal carrier; `E-FI-36` connects formal actor-only HSL; `E-FI-37` adds strict HSL persistence; `E-FI-38` proves offline fresh-runner equality; `E-FI-42` live-confirms real artifact/q29, 928/158/770, current anti-DR, actor-only gradient, zero critic delta, strict HSL-v1 save, and bounded CUDA/CPU reload. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v007` / `Formal Transaction Route` | `M-05` | HSL remains initialization-only and disabled in Stage 3; `E-FI-27` live-confirms separate 928D actor/289D critic routing and exact-one grouped update. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v015` / `Future Intent Context` | `M-11` | `E-FI-27` live-confirms H is deployment/Noisy q29 read once at local t and is not reopened as the eight-step Clean-C GMT reference; `E-FI-30` CPU-confirms command-owned q29 H across `T-max(H)` unclamped frames; `E-FI-46` materializes one deterministic q29-preserving carrier from ordinary `.npz` plus fixed protocol. |

The v015 implementation route is governed by
`../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md` and its
paired checklist. `FRS-PPO-v003` remains active because its one-row grouped
reduction is unchanged; `E-FI-27` closes the bounded S4 local identity and
exact-one-update route. Long-training convergence, policy quality, actual
checkpoint cadence/resume, and Step 5B deployment composition remain
unconfirmed. `E-FI-28` completes the immutable Step 5B-S1 deployment
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
