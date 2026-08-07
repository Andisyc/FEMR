# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v017-clean-anchored-recovery-aware-segment-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v015-fixed-split-lr-direct-full6-curriculum.md` | active |
| Reward | `active/reward/FRS-GAIN-v007-clean-anchored-recovery-aware-ranking.md` | active |
| Optimization | `active/optimization/FRS-PPO-v005-grouped-recovery-aware-scalar-update.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v004-clean-anchored-local-and-composition-evaluation.md` | active |
| Engineering | `active/engineering/FRS-ENG-v001-interface-oriented-change-discipline.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v017` / `Sealed Local Scenario`; `FRS-TRAIN-v015` / `Per-K Inner DR Curriculum` | `M-02` | TRAIN-v013 curriculum evidence remains valid; v015 changes only optimizer/checkpoint identity. |
| `FRS-DP-01P` | Perturbation Probing | `FRS-TRAIN-v015` / `Optional GMT Boundary Acquisition` | `M-12` | The current campaign directly configures the measured 2.381 boundary; optional re-probing for a changed setup remains outside this change. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v017` / `Frozen-Policy Transaction`; `FRS-PPO-v005` / `Grouped Equal-Mass Reduction`; `FRS-TRAIN-v015` / `Exact-M Frozen-Policy Transaction` | `SR-01` | E-FI-135 confirms the unchanged exact-one route with the new named optimizer identity offline. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v017` / `One-Action K Evidence`; `FRS-TRAIN-v015` / `Nested K x M x DR Schedule` | `M-06` | K/M/DR semantics are unchanged; checkpoint-v10 binds the same schedule. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v017` / `Actor And Information Boundary`; `FRS-TRAIN-v015` / `Design Delta` | `M-04` | Direct finite `[B,6]` action semantics are unchanged by the split-LR campaign. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v017` / `One-Action K Evidence And Frozen GMT` | `M-10` | E-FI-101 closes the one-action-K/frozen-770D route offline; simulator evidence pending. |
| `FRS-DP-06` | Paired Rollouts | `FRS-METHOD-v017` / `Clean/Noisy/Repair Evidence`; `FRS-GAIN-v007` / `Evidence Authority And Lifecycle`; `FRS-EVAL-v004` / `Local Clean/Noisy/Repair Evaluation` | `Q-PAIR` | E-FI-101 closes typed lifecycle; E-FI-102 proves authoritative capture cardinality offline; physical counts remain Step-2 evidence. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v007` / `Recovery-Aware Total Gain`; `FRS-PPO-v005` / `Scalar Actor And Critic Signal`; `FRS-EVAL-v004` / `Local Report` | `Q-01` | E-FI-101 closes v007 formula, scalar Critic/PPO, local report and projection retirement offline; beta quality pending. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v015` / `First Entry From HSL` | `M-03` | HSL-v2 remains Actor-only initialization; Critic and split optimizer start fresh. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v015` / `Critic Recalibration And Actor Ramp`; `Fixed Split-LR Optimizer Identity` | `M-05` | E-FI-135 proves `critic_only -> actor_ramp -> joint`, Actor freeze, one Adam with named Actor `3e-6` and Critic `1e-5` groups, exact-one count, telemetry and strict v10 persistence offline. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v017` / `Actor And Information Boundary` | `M-11` | E-FI-119 fixes exact deployment/Noisy offsets `(1,2)` at layout/config/formal/checkpoint boundaries and proves the unmocked offline 928/158/770 route; Phase B/live remains pending. |

## Active Recovery-Aware Contract Migration

Human review of all ten Design Inspector cards completed on 2026-08-01.
The reopened Perturbation Data, K-step Curriculum and Actor & Critic Warmup
details were re-confirmed on 2026-08-03. The subsequent DP04 Phase A review
confirmed that the active implementation must remove its older squashed action
coordinate. FRS-TRAIN-v015 retains that direct action and activates the fixed
split-LR/checkpoint-v10 campaign identity.
The coordinated active semantic authority is now:

```text
FRS-METHOD-v017
FRS-GAIN-v007
FRS-PPO-v005
FRS-TRAIN-v015
FRS-EVAL-v004
```

The accepted route executes one Clean and one fixed Noisy baseline per Segment,
scores every valid Repair with
`G_total = G_I + lambda_RA G_P - beta C_repair`, and sends all attempts through
one grouped scalar PPO update. The old independent Physics projection/KKT path
is superseded. The active source implements this contract set offline; remaining
formal-runtime and physical facts stay unconfirmed until Phase B. The coordinated
Engineering Plan is
`../plans/FRS-v017-clean-anchored-recovery-aware-engineering-plan.md`; planning
passed independent `engineering_plan_review` as E-FI-100. That READY verdict
does not itself authorize implementation, checkpoint migration, simulator, or
training.

The local/composition evaluation boundary is aligned by `FRS-EVAL-v004`.
The closed proposal at
`../plans/FRS-GAIN-v007-clean-anchored-recovery-aware-ranking-proposal.md`
is retained only as design rationale. The five coordinated method, Gain,
optimization, training, and evaluation contracts above are now the default
semantic authority. Evidence below that names v016/v006/v004/v011
describes the superseded implementation route and must not override them.

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
