# EVAL-v004 v018 Compatibility Evidence Ledger

Date: 2026-08-10

## Verified Offline

- Active identity: METHOD-v019 / TRAIN-v018 / GAIN-v007 / PPO-v007 /
  EVAL-v004 / checkpoint-v13.
- Held-out shape: eight fixed K16 Segments, two per transaction, exact M4,
  eight policy rows and sixteen Repair/Noisy role rows per transaction.
- Critic boundary: 449D state-value model and saved 449D observation
  normalizer install inside one reversible checkpoint scope.
- Calibration: each Segment emits raw shared value, exact-M mean `G_total`
  target and raw error; permutation is invariant and malformed evidence fails.
- Zero-write: Actor, Critic, optimizer, value normalizer, observation
  normalizers, sampler, transaction, receipt, warmup and iteration owners are
  hashed before/after Evaluation.
- Persistence: output is one atomic `frontres-v018-policy-quality-report-v1`
  JSON artifact with checkpoint and Critic fingerprints.

## Commands And Results

- Python compilation of all changed Python owners: PASS.
- Focused v018 manifest/compatibility/evaluator/entrypoint/checkpoint tests:
  PASS.
- State-value PPO, v018 transaction, telemetry and formal-runtime tests: PASS.
- Aggregate FrontRES contract suite: `55/55`, PASS.
- `run_v018_policy_quality_model2000_gpu7.sh` local command-only preflight:
  PASS with `NUM_ENVS=16`, `MAX_ITERS=0`, K16/M4 and no resume argument.

## Not Yet Verified

- No server checkpoint bytes were inspected locally.
- No IsaacLab rollout or GPU Evaluation was executed in this change unit.
- No policy-quality, Critic calibration or repair-effect conclusion is
  admitted until the real atomic report is returned.
