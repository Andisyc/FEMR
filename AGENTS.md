# MOSAIC / FrontRES Working Notes

This file is the local working contract for AI coding assistants in this
repository. Keep it concise and update it when the experiment design changes.

## Project Context

FrontRES is a lightweight residual corrector placed before the frozen GMT
tracker. It receives the tracking observation plus anchor-error history and
outputs task-space corrections:

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
  discontinuities. Keep upward `dz` constrained unless a specific experiment
  intentionally relaxes it.
- Root sink/penetration artifacts are only partially repairable by FrontRES.
  Prefer feasible corrections such as roll/pitch or contact-consistent changes.
- Composite perturbations are a later curriculum stage. Warmup should first
  learn clear single-family correction signals.

## Training Pipeline

The intended training flow is:

1. Stage 1 Segment Cache
   - Store replayable Clean dynamic states and discrete Noisy variants.
2. Stage 2 HSL
   - Train a proposal-only full-6D actor from the 870D ZMP/balance observation.
3. Stage 3 Segment Replay PPO
   - Initialize the same 6D actor from HSL, then optimize direct Delta SE(3)
     repair with paired executable evidence and K-step replay.

## Perturbation Curriculum

The current experiment samples `local_rp` perturbations only through
`frontres_specialist_mode="rp"`. This restricts the corruption distribution,
not the policy output: Stage 2 and Stage 3 always retain full-6D repair.

## Reward / Energy Notes

Do not use the full environment reward for Segment gain or PPO return.
Teleoperation, velocity-command, generic tracking, and unrelated task terms are
not repair evidence.

The accepted Segment gain has two paired improvements and one regularizer:

```text
gain_total = w_style * style_gain
           + w_physics * physics_gain
           - w_repair * repair_cost
```

Style compares Noisy/Repaired robot execution against immutable Clean motion.
Physics compares paired frozen-GMT executability. Repair cost covers full-6D
magnitude and temporal change. There is no epsilon-style mechanism or extra
gate. The current RP-only Segment score is a known implementation mismatch,
not the accepted method.

Important diagnostics:

- `gap`: estimated executable damage before repair;
- `gain`: executable improvement from FrontRES;
- `ratio`: normalized repair gain;
- `positive_gain_frac`: fraction of samples with positive gain;
- `safe/fragile/broken`: distribution of sample difficulty;
- `damage/broken/actor_gate`: whether the actor is being updated on the right
  samples;
- `exec planar/vertical/task`: reward decomposition for mismatch debugging.

If gain becomes negative, first check whether the perturbation family and
repair-specific executability component are aligned.

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
- Use `rg` for search.
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
