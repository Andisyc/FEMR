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
| `FRS-DP-01` | Perturbation Data | `FRS-METHOD-v015` / `Local Root-Artifact Scenario` | `M-02` | `E-FI-16` connects a root-only local scenario into the dedicated pre-live sentinel; the legacy full-tape route remains a contract mismatch. |
| `FRS-DP-02` | Segment Replay | `FRS-METHOD-v015` / `Frozen-Policy Multi-Attempt Transaction` | `SR-01` | `E-FI-16` connects the sealed local transaction to grouped exact-one update under a CPU fake; legacy route still mixes policy/search roles and updates immediately. |
| `FRS-DP-03` | K-step Curriculum | `FRS-METHOD-v015` / `Single-Action K-step Evidence` | `M-06` | One action/PPO row at t; K and actual evidence-step count are retained as non-mass metadata while GMT executes the shared continuation. |
| `FRS-DP-04` | FrontRES 6D Repair | `FRS-METHOD-v015` / `Future Intent Context` | `M-04` | Full-6D repair remains active; `E-FI-16` reaches q29-before-normalizer through the dedicated pre-live sentinel. Generic checkpoint dispatch and live resume remain unconfirmed. |
| `FRS-DP-05` | Frozen GMT | `FRS-METHOD-v015` / `Method Boundary` | `M-10` | `E-FI-9`/`E-FI-16` establish a one-action/frozen-FEMR K connector under deterministic evidence; real environment K execution remains unconfirmed. |
| `FRS-DP-06` | Paired Rollouts | `FRS-GAIN-v003` / `Two-Role Pairing And Time` | `Q-PAIR` | `E-FI-16` routes only Noisy and Repair scored roles with shared Clean continuation through the pre-live connector; legacy quartet remains a contract mismatch. |
| `FRS-DP-07` | Repair Gain | `FRS-GAIN-v003` / `Core Decision` | `Q-01` | `E-FI-10`--`E-FI-16` prove typed q29/Physics/6D-cost candidate evidence reaches an exact-one pre-live transaction provider. Real evaluation and sampler-state effects remain unconfirmed. |
| `FRS-DP-08` | HSL Warmup | `FRS-TRAIN-v007` / `HSL Proposal-Only Initialization` | `M-03` | H1 S1/S2 prove q29/current-target local boundaries and a CPU-only zero-HSL-loss connector; the Clean-quartet label remains forbidden, while formal/live/persistence stay unconfirmed. |
| `FRS-DP-09` | Actor & Critic Warmup | `FRS-TRAIN-v007` / `Formal Transaction Route` | `M-05` | HSL is initialization-only; `E-FI-16` leaves it disabled while reusing unchanged grouped reduction and exact-one transaction ownership. |
| `FRS-DP-10` | Future Motion Context | `FRS-METHOD-v015` / `Future Intent Context` | `M-11` | H is future 29DoF intent from deployment/Noisy provenance; it is not a 65D future tape or K execution reference. |

The v015 implementation route is governed by
`../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md` and its
paired checklist. `FRS-PPO-v003` remains active because its one-row grouped
reduction is unchanged; `E-FI-16` closes only the deterministic pre-live
connector, while actual environment and live Gain consumers remain unconfirmed.
No live training recommendation follows from contract activation alone.

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
