# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v025-current-visit-scenario-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v024-current-visit-target-cold-start.md` | active |
| Reward | `active/reward/FRS-GAIN-v008-recovery-aware-raw-evidence-utility-boundary.md` | active |
| Optimization | `active/optimization/FRS-PPO-v012-current-m4-mean-target.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v006-current-visit-policy-quality.md` | active |
| Engineering | `active/engineering/FRS-ENG-v001-interface-oriented-change-discipline.md` | active |

## Confirmed Design Rationale

`../plans/FRS-TRAIN-v024-current-visit-target-one-shot-engineering-plan.md`
records the confirmed DP02/DP09 correction: outer Replay schedules fresh
current-policy Scenario reruns, while the Critic target is only the current
transaction's exact-M4 utility mean. Active authority is
METHOD-v025/PPO-v012/TRAIN-v024.

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v025` / `Preserved boundaries`; `FRS-TRAIN-v024` / `Formal transaction` | `M-02` | Seeded Scenario identity remains unchanged. |
| `FRS-DP-01P` | Perturbation Probing | `FRS-TRAIN-v024` / `Preserved campaign` | `M-12` | The measured 2.381 ceiling remains fixed. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v025` / `Current-visit target and priority`, `Outer Replay state and lifecycle` | `SR-01` | Replay-v5 selects fresh Scenario reruns and never persists numerical target history. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v025`; `FRS-TRAIN-v024` / `Preserved campaign` | `M-06` | Latest priorities and visits remain K-specific; capacity and DR gates are unchanged. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v025` / `Preserved boundaries` | `M-04` | Direct finite `[B,6]` action semantics are unchanged. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v025` / `Preserved boundaries` | `M-10` | Frozen 770D GMT authority is unchanged. |
| `FRS-DP-06` | Paired Rollouts | `FRS-METHOD-v025`; `FRS-GAIN-v008`; `FRS-EVAL-v006` | `Q-PAIR` | Evaluation remains read-only; every selected Scenario is freshly rerun. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v008`; `FRS-PPO-v012`; `FRS-EVAL-v006` | `Q-01` | Raw Gain, symlog and current-attempt Actor credit are unchanged. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v024` / `Campaign identity` | `M-03` | HSL-v2 remains Actor-only initialization. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v024` / `Formal transaction`; `FRS-PPO-v012` | `M-05` | Critic uses current exact-M4 Scenario means; Actor LR/B8/M4 remain; checkpoint-v19. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v025` / `Preserved boundaries`; `FRS-TRAIN-v024` / `Formal transaction` | `M-11` | Actor 158D, Critic 449D and GMT 770D remain. |

## Active Recovery-Aware Contract Migration

Human review confirmed the DP07/DP09 utility change on 2026-08-10 after
TRAIN-v018 K8/M4 retained a large negative Critic bias under heavy-tailed finite
raw Gain. The Critic remains `V(s)` over the same 449D action-pre state. Each
attempt is mapped by fixed `sign(G)*log1p(abs(G))` before Actor advantage and
before the M4 Critic mean. Raw Gain and hard Physics diagnostics remain visible;
network, observations, K/DR ceiling, Gain arithmetic and simulator are unchanged;
TRAIN-v024 retains B8/M4 and bounded Replay breadth while removing
cross-visit numerical target evidence.
The coordinated active semantic authority is now:

```text
FRS-METHOD-v025
FRS-GAIN-v008
FRS-PPO-v012
FRS-TRAIN-v024
FRS-EVAL-v006
```

The accepted route executes one Clean and one fixed Noisy baseline per Scenario,
scores every valid Repair with
`G_total = G_I + lambda_RA G_P - beta C_repair`, and sends all attempts through
one grouped scalar PPO update. All attempts in one Segment share the same old
state value; the current transaction's exact-M4 Scenario mean supervises the Critic while individual
`U(G_total_m)-V_old(s)` values supervise Actor credit. Each transaction uses
eight distinct Scenario states and M4 attempts. Outer Replay schedules fresh
future visits through the bounded 64->128->256 active pool, ranks latest Critic
calibration error beyond the current-M4 confidence interval, and never changes
PPO mass. Checkpoint-v19 is the only compatible Stage-3 training/resume
identity and carries target moments/count plus replay-v5
active/archive/capacity/key/latest-score/visit/staleness/RNG, with no utility
window or policy anchor. The current
Engineering Plan and Module Test Cards are the executable-work and oracle
control surfaces; implementation, formal route and live evidence close only at
their own gates.

TRAIN-v024 local engineering evidence is recorded by the current plan,
checklist, `frontres_train_v024_module_alignment.json` and
`frontres_train_v024_formal_live_required.json`: TEST-25A--25D, strict
checkpoint-v19/replay-v5 and the 59/59 Contract suite pass. This is S0--S3
evidence only. Prior TRAIN-v023 manifests and reviews are historical and cannot
admit this route. The single remaining live fact is one official K8/B8/M4
transaction; long training and policy-quality claims remain unauthorized.

`FRS-EVAL-v006` defines the read-only interpretation required for a future
checkpoint-v19 held-out evaluation. This engineering unit closes strict v19
checkpoint inspection only; it intentionally leaves the existing
`FrontRESV018PolicyQualityManifest` and `policy_quality_eval` entrypoint bound
to historical EVAL-v004/checkpoint-v14. A new v19 held-out manifest and its
zero-Replay-mutation proof are post-checkpoint evaluation work, not a TRAIN-v024
startup dependency. No existing EVAL-v004 artifact is evidence for this route.
The confirmed proposal at
`../plans/FRS-METHOD-v018-state-value-future-context-proposal.md` is retained as
design rationale. The five coordinated method, Gain, optimization, training,
and evaluation contracts above are now the default semantic authority. No
prior v10 training or evaluation artifact is evidence for the new formal route
or policy quality.

## Superseded Runtime Evidence

The following evidence records the implementation route that preceded the
2026-08-01 contract activation. It is retained for traceability and does not
override the active contract set above.

The physics-constrained Intent migration was governed by
`../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md` and its
paired checklist. `FRS-PPO-v004` preserves v003 one-row grouped equal-mass
reduction but replaces the actor update with a joint constraint-gradient
projection; `E-FI-27` closes the bounded S4 local identity and
exact-one-update route. `E-FI-71` additionally closes the bounded K-transition
and actual checkpoint-cadence gate. Long-training convergence, policy quality,
and Step 5B physical deployment quality remain unconfirmed. `E-FI-28` completes the immutable Step 5B-S1 deployment
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
`E-FI-98` closes the deterministic same-state Baseline/Repair executor and its
runtime Gateway: one canonical route-start/RNG/carrier identity is restored
before each branch, Baseline never invokes FEMR, and the paired report cannot
feed training state. Simulator timing, physical metric values, and live
composition remain open. No
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
forbidden. Independent policy efficacy and physical deployment-composition
quality remain unconfirmed; deterministic paired composition connectivity is
closed by E-FI-98.

E-FI-88 activates FRS-TRAIN-v011 after the K8/M2 log audit established that
the prior formal route never increased M. The accepted campaign freezes
K8/M2 -> K16/M3 -> K32/M4, two Segment sources, absolute review boundaries
2000/3500/4825/6500/8000 and checkpoint-v6. METHOD-v016, GAIN-v006, PPO-v004,
HSL, 158D actor authority and one-action-K remain unchanged. E-FI-89 installs
the exact-M formal owner, active-M telemetry and strict checkpoint-v6
save/resume while rejecting checkpoint-v5 pre-mutation. The route is now
offline contract-confirmed; simulator/training and first M3/M4 evidence still
require separate material-runtime authorization.

The first P5-C K8/M2 block reached strict checkpoint-v6 `model_2000.pt` and
exposed an actual-update authority mismatch: 240 actor-enabled no-direction
transactions installed zero Actor gradients but historical Adam momentum still
changed Actor/std. E-FI-90 closes this mismatch without versioning a new
optimization contract. PPO-v004 now governs the committed post-Adam parameter
delta, restores Actor/std parameters and optimizer state for critic-only and
no-direction statuses, and checks actual-update KKT. The correction is
offline-confirmed; the first K16/M3 transaction remains the live boundary.

E-FI-91 is a behavior-preserving implementation ownership closure, not a new
contract version: `frontres_segment_ppo.py` owns the one shared optimizer call
and actual Actor/std commit; `frontres_segment_diagnostics.py` owns the final
read-only authority validator; runner files retain orchestration and
serialization only. METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011,
checkpoint-v6 and the 928/158/770 layout are unchanged.

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
