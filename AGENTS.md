# MOSAIC / FrontRES Working Notes

This file is the local working contract for AI coding assistants in this
repository. Keep it concise and update it when the experiment design changes.

## Project Context

FrontRES is a lightweight residual corrector placed before the frozen GMT
tracker. Under the active v017 route it receives the deployable 158D
FrontRES prefix: the current Noisy root artifact/state features plus the
deployment/Noisy q29 future-intent tail. Frozen GMT retains authority over the
original 770D suffix. FrontRES outputs task-space corrections:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

The old confidence, acceptance, rho, authority-critic, and active-action-mask
interfaces are retired. Perturbation family never narrows the six-dimensional
repair output.

The goal is not to replace GMT. The goal is to make corrupted reference frames
more executable by GMT, especially when visual/video extraction artifacts
consume robustness budget.

## Core Design Principles

- FrontRES should correct reference-frame artifacts, not learn a new tracker.
- Corrections must be executable by GMT. A correction that is geometrically
  closer but dynamically damaging is wrong.
- Use task-space `Delta SE(3)` rather than `Delta q` for the main FrontRES path.
- Root-level upward `dz` is dangerous because it can create dynamics
  discontinuities. The active route discourages it through the HSL proposal,
  Clean-anchored K-step consequence and full-6D repair cost; it does not add a
  per-axis mask, scale, clamp, clip or tanh authority.
- Root sink/penetration artifacts are only partially repairable by FrontRES.
  Prefer feasible corrections such as roll/pitch or contact-consistent changes.
- Composite perturbations are a later curriculum stage. Warmup should first
  learn clear single-family correction signals.

## Training Pipeline

The intended training flow is:

1. Stage 1 Segment Cache
   - Store replayable Clean dynamic states and discrete Noisy variants.
2. Stage 2 HSL
   - Initialize only the proposal actor from the deployable 158D FrontRES
     prefix. HSL-v1 is proposal-only and does not define the Stage-3 target.
3. Stage 3 Segment Replay PPO
   - Initialize the same 6D actor from HSL, then optimize direct Delta SE(3)
     repair with one-action-K paired evidence, sealed multi-Segment x M replay,
     and exactly one grouped optimizer update per committed transaction.
   - The scalar Critic predicts the single FRS-GAIN-v007 Recovery-Aware
     `G_total`. Intent and Physics are both Clean-anchored; Physics recovery
     pressure controls how strongly Physics improvement enters the same scalar.
     TRAIN-v017 preserves raw `G_total`, raw `V(s)` and Actor credit while
     dividing only the Critic value loss by a committed non-amplifying EMA
     target-variance scale before the existing independent gradient clip.
     There is no active constraint-gradient projection or KKT authority.

## Perturbation Curriculum

The current experiment samples one sealed `local_rp` perturbation per Segment
through `frontres_specialist_mode="rp"`. This restricts the corruption
distribution, not the policy output: Stage 2 and Stage 3 always retain full-6D
repair. TRAIN-v013 gives each active K its own lower-to-higher DR curriculum.
At every K transition DR returns to a lower informative distribution while the
same Critic recalibrates. For the current frozen GMT, robot and perturbation
definition, `dr_scale=2.381` is the experimentally measured maximum reliable
perturbation boundary. The current campaign configures that known value
directly and keeps Easy/Medium/Hard/Broken-tail sampling at 20/30/40/10. A
changed setup may remeasure the boundary with an offline frozen-GMT Noisy-only
survival probe. Probing is optional boundary acquisition, not a per-K `g_K`, an
online controller, or a Gain/PPO feedback path.

## Reward / Energy Notes

Do not use the full environment reward for Segment gain or PPO return.
Teleoperation, velocity-command, generic tracking, and unrelated task terms are
not repair evidence.

The accepted scalar Stage-3 objective is the unique FRS-GAIN-v007 value:

```text
G_total = G_I + lambda_RA * G_P - beta * C_repair
return_K = G_total
```

`G_I` and `G_P` are signed Noisy-to-Repair improvements measured relative to
the same executed Clean anchor. `lambda_RA` is determined by remaining Physics
pressure in the Noisy and Repair outcomes. Missing or malformed evidence fails
the transaction closed; valid actual no-load is a Contact violation with
role-specific ZMP N/A. Clean continuation is GMT/Physics-evaluator evidence only
and never actor input. Repair cost covers full-6D magnitude and temporal change.
There is no rho, second actor/Critic/optimizer, constraint projection, KKT dual,
scalar Physics fallback, or epsilon gate.

Important diagnostics:

- scalar target, return, value, raw/scaled advantage, and Critic calibration;
- raw/scaled Critic value loss, non-amplifying target scale, committed moments
  and exact update-count transition;
- expected/actual Contact and loaded-support phase-ZMP applicability/violation;
- survival, sustained lateral lean, and unplanned support changes;
- Recovery-Aware components `G_I`, `G_P`, `P_N`, `P_R`, `lambda_RA`, cost and
  final `G_total`;
- action magnitude/non-collapse and actor/Critic parameter deltas;
- scenario/noisy hash, group mass, exact-one update and committed receipt.

Negative scalar objective is not by itself a Physics failure. Inspect paired
Intent improvement and repair cost separately from Contact/phase-ZMP/survival
constraints, then locate the first invalid owner before changing the method.

## Validation Experiments

Validation is separate from FrontRES training. It demonstrates that reference
frame errors consume robustness budget.

Preferred story:

```text
reference-frame error epsilon increases
  -> post-push stability margin decreases
  -> push recovery rate drops
  -> FrontRES is motivated
```

Store each motion sequence independently so failures do not invalidate the
whole run. Plot scripts should read a results directory containing per-motion
subdirectories with both metadata and raw arrays.

For videos, RobotBridge/MuJoCo is preferred for presentation artifacts. For
training-side quantitative validation, IsaacLab remains acceptable if it matches
the training environment.

## Coding Rules For This Repo

- DO NOT send optional commentary.
- Do not revert user changes.
- Keep changes scoped to the current experiment.
- Follow the active Engineering Discipline at
  `note/frontres_core/contracts/active/engineering/FRS-ENG-v001-interface-oriented-change-discipline.md`
  for every non-trivial FrontRES/FEMR production, test, configuration,
  persistence, evaluation, or deployment change.
- Before editing a non-trivial boundary, state the requested and preserved
  behavior, one semantic owner, public input/output, dependency direction,
  state/transaction boundary, forbidden dependencies, tests, and stop condition.
- Interface-oriented means fewer facts for callers to know. Do not add a
  wrapper, Protocol, service, or class hierarchy unless it removes a named
  dependency or stabilizes a named boundary.
- Diagnose structure with the named gates in `FRS-ENG-v001`: Refactoring change
  smells; WELC Characterization Test, Seam/Enabling Point, Effect Sketch and
  Pinch Point; CCP/CRP/ADP/SDP; Humble Object; Service Layer, Gateway, Unit of
  Work, Aggregate, and Composition Root. Do not replace these with generic
  "SOLID violation" labels.
- Keep entrypoints and runners as thin orchestration. Put deterministic domain
  decisions in the existing semantic owner and isolate simulator/framework IO
  behind narrow adapters.
- Do not access another layer's private attributes, introduce stable cross-layer
  `dict` payloads, duplicate semantic ownership, or add silent fallback/zero-fill.
- Treat size, branch count, parameter count, fan-in/fan-out, and static hotspot
  scores only as prompts to run the named gates. They are not automatic split or
  rejection rules.
- Do not modify MOSAIC-owned host behavior for FrontRES convenience unless the
  user explicitly authorizes that host change.
- Use Semble first when the query is semantic or the exact wording/symbol is
  unknown. Use `--content all` when the answer may live in code, contracts,
  plans, notes, or configuration.
- Use `rg` when an exact literal, symbol, filename, or exhaustive occurrence
  list is required. Use `lean-ctx` to read known files and GitNexus for call
  graphs, dependency tracing, and impact analysis.
- Semble only locates candidate evidence. It does not decide which Contract,
  plan, Architecture, or historical note is authoritative; apply the existing
  workflow-governance rules after retrieval.
- Use Context Mode only for high-volume, read-only data that would otherwise
  flood the conversation, especially large logs, JSON, CSV, and test output.
  Prefer `ctx_execute_file` to compute focused evidence while retaining the raw
  artifact on disk.
- Use `ctx_batch_execute` when several independent read-only commands are each
  expected to produce substantial output. Keep small commands, known-file
  reads, source search, and normal code inspection on the existing tools.
- Do not use Context Mode for repository edits, mutation commands, external
  research, or Contract/Architecture authority decisions. Its indexed or
  remembered content is candidate evidence only and must not override active
  workflow-governance state.
- Do not enable Context Mode session hooks or automatic memory restoration for
  this repository; the MCP-only boundary is intentional.
- Use `apply_patch` for manual edits.
- Run at least `python -m py_compile` after Python code changes when practical.
- For large FrontRES/FEMR design changes, audit implementation against `./note`
  before recommending training. If the live code path diverges from the
  planned concept, stop and report the mismatch instead of silently finishing a
  partial implementation.
- Treat concept-code alignment as a required check: config flag -> rollout
  construction -> storage/recompute contract -> loss/update -> gradient
  boundary -> deployment behavior -> diagnostics.
- When touching FrontRES training logic, check:
  - resume/cold-start behavior;
  - debug mode overrides;
  - full-6D action identity;
  - perturbation schedule;
  - reward diagnostics.

## Common Pitfalls

- Warmup diagnostics can be misleading if the current perturbation family has no
  signal for a dimension. Always inspect `modes=(...)` together with
  `valid_pos/valid_rpy`.
- A high supervised cosine does not guarantee PPO reward alignment.
- Composite perturbations can create reward conflict if one scalar executable
  reward is asked to represent multiple repair cones.
- Broken samples should not dominate actor updates.
- If `broken_frac` is too high, reduce DR scale or simplify the perturbation
  curriculum before changing the network.
