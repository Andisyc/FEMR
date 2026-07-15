contract_id: FRS-TRAIN-v002
status: superseded
effective_date: 2026-07-13
updated_date: 2026-07-14
supersedes: FRS-TRAIN-v001
superseded_by: FRS-TRAIN-v003
scope: Formal Stage 3 Segment Replay training route

# Segment Replay Training Contract

## Ownership

This contract owns how the active Segment Replay method reaches formal Stage 3
training. Method semantics remain in `../method/FRS-METHOD-v011-segment-replay.md`.
The PPO objective remains owned by the algorithm module and its optimization
contract. The sampler owns segment selection and replay evidence, not policy
gradient semantics.

## Formal Route

```text
Stage 3 config
-> global/replay segment sampler
-> trial-plan and quartet construction
-> dynamic reset or faithful preroll
-> per-row K-step frozen-GMT rollout
-> paired Noisy/Repaired style and physics components
-> repair regularization
-> valid policy-row storage
-> full-6D Delta SE(3) PPO update
-> rollout-evidence-only sampler update
-> checkpoint and diagnostics
```

A helper, planner, sampler, or test implementation outside this route is not
enough to claim that a feature is part of formal training.

## Trial Roles And Credit

Trial rows must have explicit roles. Policy rows contain actions sampled from
the current policy and may receive PPO credit. Search, candidate, clean, and
counterfactual rows may contribute sampler evidence or baselines, but must not
be relabeled as policy actions.

Quartet construction may include train, candidate, baseline, and clean roles.
Batch diagnostics must report those role counts instead of assuming every
environment row is trainable.

## Horizon Contract

The sampler may assign different horizons such as `8/16/32/64` per row. The
assigned horizon must survive batch construction, reset, rollout termination,
return/evidence aggregation, sampler update, and diagnostics.

The K-step curriculum has two independent acceptance gates:

1. Implementation gate: modules can represent and process the intended K
   values under contract tests.
2. Integration gate: formal Stage 3 training executes multiple intended K
   values and reports them from the live route.

Passing the implementation gate alone must be reported as
`implemented-not-integrated`.

## Reset And State Fidelity

Segment reset must restore the local dynamic state required by frozen GMT, or
use a preroll that reconstructs it. Pose-only teleportation is not equivalent.
Reset-success masks and trial metadata must use the same row domain as the
captured rollout batch.

## PPO And Sampler Boundary

PPO consumes policy actions, old distribution statistics, returns, and
advantages from the same action representation. Segment priority consumes
rollout-time executable evidence. Post-update KL, parameter deltas, or stale
policy logits must not affect sampler priority.

For paired Segment rows, PPO returns, sampler evidence, periodic evaluation,
and sequence evaluation must use the same style/physics/repair decomposition
defined by `../reward/FRS-GAIN-v001-style-physics-repair.md`. Generic environment
reward has no fallback authority when a component is missing. Missing data is
`UNCONFIRMED` and blocks the corresponding acceptance claim.

The former RP-only live Segment score is a legacy implementation path and is
not an accepted reward, sampler score, difficulty heuristic, diagnostic, or
evaluation design. It may remain only as explicitly labeled compatibility data
outside active decisions; the current offline cross-consumer owner migration is
covered by E14/E15/E16, while real runtime population remains an S4 boundary.

The default advantage rule is specified by
`../optimization/FRS-PPO-v001-sign-preserving-advantage-scaling.md`.

## Required Diagnostics

- source and trial-role counts;
- sampled and effective K distributions;
- reset/preroll success;
- perturbation family and strength;
- valid policy-row count;
- Noisy/Repaired gain and harmful-repair fraction;
- full-6D action mean, sigma, and executed residual;
- PPO pre/post-update trust-region evidence;
- sampler evidence and replay-pool updates.

Missing evidence must be reported as `UNCONFIRMED`, never silently as zero.

## Acceptance

Every training feature requires both a module-level implementation test and a
formal-route connectivity test. Runtime-dependent behavior additionally
requires an S4 live sentinel before full training. Architecture must be updated
when this route, ownership, or interface changes.
