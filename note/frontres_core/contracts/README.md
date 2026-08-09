# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v019-support-conditioned-state-value-segment-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v018-support-conditioned-m4-curriculum.md` | active |
| Reward | `active/reward/FRS-GAIN-v007-clean-anchored-recovery-aware-ranking.md` | active |
| Optimization | `active/optimization/FRS-PPO-v007-output-preserving-adaptive-value-scale.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v004-clean-anchored-local-and-composition-evaluation.md` | active |
| Engineering | `active/engineering/FRS-ENG-v001-interface-oriented-change-discipline.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v019` / `Sealed Local Scenario`; `FRS-TRAIN-v018` / `Per-K Inner DR Curriculum` | `M-02` | K/DR semantics are unchanged; prior runtime evidence is stale for the v018 formal route. |
| `FRS-DP-01P` | Perturbation Probing | `FRS-TRAIN-v018` / `Optional GMT Boundary Acquisition` | `M-12` | The campaign still configures the measured 2.381 boundary directly. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v019` / `Frozen-Policy Transaction`; `FRS-PPO-v007` / `Grouped Equal-Mass Reduction`; `FRS-TRAIN-v018` / `Exact-M Frozen-Policy Transaction` | `SR-01` | Exact-one behavior is retained; M=4 is now fixed at all K stages. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v019` / `One-Action K Evidence`; `FRS-TRAIN-v018` / `Nested K x M x DR Schedule` | `M-06` | K/DR timing is unchanged; checkpoint-v13 binds K8/M4 -> K16/M4 -> K32/M4. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v019` / `Actor And Information Boundary`; `FRS-TRAIN-v018` / `Design Delta` | `M-04` | Direct finite `[B,6]` action semantics are unchanged; action-conditioned Critic remains forbidden. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v019` / `One-Action K Evidence And Frozen GMT` | `M-10` | Frozen 770D GMT authority is unchanged; Critic-only support context stays isolated. |
| `FRS-DP-06` | Paired Rollouts | `FRS-METHOD-v019` / `Clean/Noisy/Repair Evidence`; `FRS-GAIN-v007` / `Evidence Authority And Lifecycle`; `FRS-EVAL-v004` / `Local Clean/Noisy/Repair Evaluation` | `Q-PAIR` | Held-out Evaluation preserves the same K16 Segment bank and now executes exact M4 from checkpoint-v13. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v007` / `Recovery-Aware Total Gain`; `FRS-PPO-v007` / `Scalar Actor And Critic Signal`; `FRS-EVAL-v004` / `Local Report` | `Q-01` | Gain remains per-attempt; Evaluation reports raw shared value against the exact-M Segment mean without changing Actor ordering. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v018` / `First Entry From HSL` | `M-03` | HSL-v2 remains Actor-only initialization; the 449D Critic and optimizer start fresh. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v018` / `Critic Recalibration And Actor Ramp`; `Adaptive Critic Value Scale`; `Fixed Split-LR Optimizer Identity`; `FRS-PPO-v007` / `Warmup Weight Boundary` | `M-05` | Phase counts/LRs remain; 449D support context, M4 and checkpoint-v13 require fresh evidence. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v019` / `Actor And Information Boundary`; `FRS-TRAIN-v018` / `Scalar Critic Authority` | `M-11` | Actor remains 158D; Critic adds action-pre support context for 449D total while GMT remains 770D. |

## Active Recovery-Aware Contract Migration

Human review confirmed the DP09 support-conditioning and M4 change on
2026-08-09 after TRAIN-v017 K8/M2 remained near a constant Critic despite finite
training. The Critic remains `V(s)`, now over 449D action-pre state, and still
predicts one arithmetic exact-M Segment mean. M=4 is fixed across K8/K16/K32.
The Actor, raw Gain/value/advantages, GMT, K/DR, loss scaling, LR and simulator
semantics are unchanged.
The coordinated active semantic authority is now:

```text
FRS-METHOD-v019
FRS-GAIN-v007
FRS-PPO-v007
FRS-TRAIN-v018
FRS-EVAL-v004
```

The accepted route executes one Clean and one fixed Noisy baseline per Segment,
scores every valid Repair with
`G_total = G_I + lambda_RA G_P - beta C_repair`, and sends all attempts through
one grouped scalar PPO update. All attempts in one Segment share the same old
state value; their mean `G_total` supervises the Critic while their individual
returns supervise Actor credit. Checkpoint-v13 is the only compatible Stage-3
persistence identity and carries the committed target moments/count. The current
Engineering Plan and Module Test Cards are the executable-work and oracle
control surfaces; implementation, formal route and live evidence close only at
their own gates.

The local/composition evaluation boundary is aligned by `FRS-EVAL-v004`.
Its 2026-08-10 compatibility revision binds the active Held-out Policy Quality
route to checkpoint-v13, K16/M4, the 449D support-conditioned state-value
Critic and the checkpoint's privileged-observation normalizer. The held-out
Segment bank, Gain and deployment-composition question are unchanged.
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
