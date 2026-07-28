# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v016-physics-constrained-intent-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v011-coordinated-k-m-checkpointed-curriculum.md` | active |
| Reward | `active/reward/FRS-GAIN-v006-loaded-support-zmp-applicability.md` | active |
| Optimization | `active/optimization/FRS-PPO-v004-grouped-constraint-gradient-projection.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v003-local-repair-composition-evaluation.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v016` / `Preserved Replay Authority` | `M-02` | E-FI-74/E-FI-75 preserve scenario identity through the v016 loss and live route. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v016` / `Frozen-Policy Transaction`; `FRS-TRAIN-v011` / `Exact-M Frozen-Policy Transaction` | `SR-01` | E-FI-89 installs exactly two Segment sources x active M attempts and isolates state-driven sampler M. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v016` / `Preserved Replay Authority`; `FRS-TRAIN-v011` / `Global Coordinated K x M Schedule` | `M-06` | E-FI-89 closes the K8/M2 -> K16/M3 -> K32/M4 owner, role widths and checkpoint-v6 persistence offline; M3/M4 live evidence remains open. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v016` / `Actor And Information Boundary` | `M-04` | E-FI-74 confirms the 158D/full-6D actor authority and constraint-evidence isolation. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v016` / `Preserved Replay Authority` | `M-10` | E-FI-74/E-FI-75 preserve one-action-K and change only its loss-side interpretation. |
| `FRS-DP-06` | Paired Rollouts | `FRS-GAIN-v006` / `Evidence Authority` | `Q-PAIR` | The same sealed roles remain; valid physical loss of support is Contact failure rather than corrupt evidence. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v006` / `Loaded-Support Phase-ZMP`; `FRS-PPO-v004` / `Grouped First-Order Projection` | `Q-01` | E-FI-84 live-confirms sensor authority; E-FI-85 closes role-specific applicability offline; E-FI-86 live-confirms the larger fail-closed raw-contact capacity through iteration 2000. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v011` / `First Entry Into The New Identity` | `M-03` | HSL-v1 remains the only cold-start actor source; checkpoint-v5 cannot migrate into v011. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v011` / `Per-Stage Recalibration And Actor Ramp` | `M-05` | E-FI-89 closes the resolver/persistence identity offline; the first real K16/M3 and K32/M4 recalibrations remain unconfirmed. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v016` / `Actor And Information Boundary` | `M-11` | Deployment/Noisy q29 H remains unchanged and isolated from Physics evaluator evidence. |

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
E-FI-74 closes the original deterministic METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010
connectivity; E-FI-75 live-confirms one constrained K8 transaction and
checkpoint-v5; E-FI-77 closes strict v5 full resume and raw quality evidence;
E-FI-78 advances the same K8 Critic to the iteration-200 actor-warmup boundary.
E-FI-79 records the later `model_251.pt` -> `model_2000.pt` run and exposes four
post-rescale recovery KKT violations. E-FI-80 repairs that implementation
mismatch without changing PPO-v004 semantics: recovery is reprojected after
norm scaling and the formal telemetry consumer enforces checkpoint-v5
tolerance. E-FI-81 closes the later Physics evidence-authority mismatch offline:
formal phase-ZMP now uses per-contact wrench data against sealed Clean-foot
support envelopes, and checkpoint-v5 binds that evidence identity. E-FI-82
then activates GAIN-v006/schema-v2: valid actual no-load is a Contact violation
with role-specific ZMP N/A, while malformed payload and loaded-without-resultant
still fail closed. E-FI-84 live-confirms official IsaacLab raw-contact sensor
authority and E-FI-85 closes Repair/Noisy applicability carriers offline.
E-FI-86 increases the raw-contact view from 16 to 256 contacts per foot/env
while retaining saturation fail-closed, then live-confirms 1999 consecutive
K8 committed transactions through absolute iteration 2000 without capacity,
applicability or KKT failure. Silent v005/v004/v003/v009 fallback remains
forbidden. Independent policy efficacy and deployment composition remain
unconfirmed.

E-FI-88 activates FRS-TRAIN-v011 after the K8/M2 log audit established that
the prior formal route never increased M. The accepted campaign freezes
K8/M2 -> K16/M3 -> K32/M4, two Segment sources, absolute review boundaries
2000/3500/4825/6500/8000 and checkpoint-v6. METHOD-v016, GAIN-v006, PPO-v004,
HSL, 158D actor authority and one-action-K remain unchanged. E-FI-89 installs
the exact-M formal owner, active-M telemetry and strict checkpoint-v6
save/resume while rejecting checkpoint-v5 pre-mutation. The route is now
offline contract-confirmed; simulator/training and first M3/M4 evidence still
require separate material-runtime authorization.

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
