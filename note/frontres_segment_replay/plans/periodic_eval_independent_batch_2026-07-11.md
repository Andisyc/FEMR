# Periodic Eval Independent Batch Step Contract

## Problem

Stage 3 periodic eval runs after the live sampler step has cleared the current
segment batch. The evaluator therefore rolls out from residual training env
state instead of an independently sampled and reset segment. Its score path
also reads the row immediately after repair as Noisy, which is Candidate in a
quartet rollout.

## Design Delta

- Old design: periodic eval reuses current env state and reconstructs score
  fields locally.
- New design: periodic eval independently samples scorable rows, builds an eval
  batch, resets, rolls out, and uses the offline evaluator's quartet-aware score
  semantics.
- Changed semantic objects: eval sample, eval batch, runner current sample/batch,
  sampler RNG/seen/staleness, Noisy row offset, eval metadata.
- Forbidden old assumptions: a training batch still exists after a sampler
  step; `reward[n_train:]` begins with Noisy; eval may advance training sampler
  state.

## Scope

- Modify periodic eval orchestration and its human-readable diagnostics.
- Preserve training sampler state and pre-existing runner temporary fields.
- Restore runner train mode even when periodic rollout raises.
- Print motion IDs, start frames, perturbation families, and strength range.

## Non-scope

- No PPO, reward, perturbation curriculum, checkpoint, or offline/sequence eval
  method changes.
- No claim about IsaacLab physics until an S4 live sentinel is run.

## Core Parameter Path

`training sampler state -> independent eval sample -> eval batch -> reset -> rollout -> quartet-aware summary -> diagnostics -> restore training state`

## Steps

1. Add an S2 regression contract for independent batch wiring, role-correct
   score extraction, sampler isolation, and runner-state restoration.
2. Implement the evaluator route in the live-training owner module.
3. Add metadata diagnostics and update the test inventory/control board.

## Evidence Target

- S0: Python compilation.
- S1: diagnostics formatting values and metadata.
- S2: fake runner/sampler/batch route and lifecycle isolation.
- S3: evaluator lifecycle semantics are covered offline; real env reset remains
  S4 and unconfirmed.

## Test Evidence

- `frontres/bin/python -m py_compile ...`: PASS for evaluator, diagnostics, and
  pseudo-contract files.
- `frontres_segment_live_training_pseudo_contract.py`: PASS. The regression
  observes `sample -> build -> reset`, correct quartet Noisy offset, per-step
  score normalization, sampler/RNG restoration, runner-field restoration, and
  train-mode restoration on both success and rollout failure.
- `frontres_segment_diagnostics_contract.py`: PASS.
- `frontres_segment_all_contract_suite.py`: `contract_count=43 failed_count=0`.
- Sequence-eval compatibility regression: a `SimpleNamespace(segment_ids=...)`
  sample can pass through the shared batch builder without optional trial fields;
  full live samples still attach all provided trial metadata.

## Remaining Gap

S4 remains required to confirm real IsaacLab reset success and that printed
`motion_ids`, `start_frames`, perturbation families, and strengths vary as
expected across periodic evaluations.
