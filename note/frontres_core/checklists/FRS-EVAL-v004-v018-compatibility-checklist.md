# FRS-EVAL-v004 v018 Compatibility Checklist

Plan: `../plans/FRS-EVAL-v004-v018-compatibility-one-shot-engineering-plan.md`

## Authority

- [x] User authorized current v018 Evaluation compatibility on 2026-08-10.
- [x] Active method/training/optimization identities are v019/v018/v007.
- [x] Actor, Gain, GMT, simulator, training and deployment composition excluded.
- [x] TEST-19 through TEST-22 confirmed by the same explicit execution request.

## Implementation

- [x] Active immutable v018 K16/M4 manifest exists; v017 K16/M3 is historical.
- [x] Request seals HSL-v2 separately from v018 checkpoint-v13 policy identity.
- [x] Checkpoint Gateway installs/restores 449D Critic observation normalizer.
- [x] Segment calibration rows expose raw value, exact-M mean target and error.
- [x] Launcher admits exactly 16-env K16/M4 policy-quality Evaluation.
- [x] Result schema and identity are v018 and atomically written.

## Evidence

- [x] TEST-19 active manifest/request ordinary, invalid and identity cases pass.
- [x] TEST-20 Critic/normalizer success, exception and restoration cases pass.
- [x] TEST-21 M4 mean, permutation and fail-closed calibration cases pass.
- [x] TEST-22 official offline entrypoint and zero-write/persistence cases pass.
- [x] Focused policy-quality and checkpoint suites pass.
- [x] Aggregate FrontRES contract suite and `python -m py_compile` pass.
- [x] `code-review-expert` final gate has no open P0/P1.
- [x] Formal audit reaches R1 official offline and R2 checkpoint-normalizer state.

## Closeout

- [x] EVAL-v004 Contract, registry, Design Inspector/Register and architecture map align.
- [x] Evidence ledger records limits: no simulator fact and no policy-quality verdict.
- [x] One standalone server test command/script is ready; it is not executed locally.
