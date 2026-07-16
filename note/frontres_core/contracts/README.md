# FrontRES Contract Registry

This registry is the only default entrypoint for FrontRES contracts.

## Active Contract Set

| Category | Active contract | Status |
| --- | --- | --- |
| Method | `active/method/FRS-METHOD-v011-segment-replay.md` | active |
| Training | `active/training/FRS-TRAIN-v003-segment-replay-warmup.md` | active |
| Reward | `active/reward/FRS-GAIN-v002-style-physics-repair.md` | active |
| Optimization | `active/optimization/FRS-PPO-v001-sign-preserving-advantage-scaling.md` | active |
| Evaluation | `active/evaluation/FRS-EVAL-v002-segment-evaluation.md` | active |

## Concept Figure Design Point Register

This table is the machine-readable counterpart of the human-facing FrontRES
Concept Figure. Canonical names and block IDs come from
`note/architecture/concept/03_frontres_concept_tabs.data.json`.

| Design ID | Canonical human name | Active contract section | Figure block ID | Current code/evidence gap |
| --- | --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v011` / `Perturbation Data` | `M-02` | formal family/strength runtime evidence remains required |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v011` / `Segment Replay Design` | `SR-01` | live source/priority evidence remains required |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v011` / `K-Step Curriculum` | `M-06` | formal multi-K live evidence remains required |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v011` / `Action Semantics` | `M-04` | full-6D runtime evidence remains required |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v011` / `Method Boundary` | `M-10` | frozen-parameter runtime evidence remains required |
| `FRS-DP-06` | Paired Rollouts | `FRS-GAIN-v002` / `Pairing And Time` | `Q-PAIR` | all paired components require S4 population evidence |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v002` / `Core Decision` | `Q-01` | all active consumers require shared-owner S4 evidence |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v003` / `HSL Warmup` | `M-03` | checkpoint identity and Stage 3 load evidence remain required |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v003` / `Actor And Critic Warmup` | `M-05` | formal-route connectivity and live phase evidence remain required |

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
