# FRS-EVAL-v004 v019 Fixed-Segment Critic Repeat Probe Implementation Plan

> **For agentic workers:** Execute inline with TDD; do not train, update, commit, push, or run the simulator locally.

**Goal:** Add one bounded EVAL-v004 diagnostic that repeats two fixed K8/M4 held-out Segments eight times under a frozen checkpoint and reports target variability against one shared Critic prediction.

**Architecture:** Reuse the existing checkpoint installation, fixed-manifest materializer, Clean/Noisy/Repair collector, GAIN-v008 calculation, symlog utility, and atomic JSON writer. `repeat_count` is evaluation execution identity, not a manifest or method parameter. The default remains one pass.

**Tech Stack:** Python, PyTorch, existing EVAL-v004 runner, shell preflight, JSON manifest.

---

## Change Contract And Test Card

- Requested behavior: keep two manifest Segment identities fixed while sampling eight independent M4 Repair groups; report each group and one per-Segment repeat summary.
- Preserved behavior: single-pass EVAL-v004 output, Actor/Critic/GMT/Gain/PPO/checkpoint semantics, zero-write inference, M=4, full-6D actions, and atomic report production.
- Semantic owner: `rsl_rl.runners.frontres_policy_quality_eval.run_frontres_v018_policy_quality_heldout_eval`.
- Public input/output: positive bounded `repeat_count`; output adds `repeat_count`, `repeat_diagnostics`, and `repeat_index` while retaining complete transaction reports.
- Dependency direction: shell/CLI -> runner connector -> EVAL-v004 owner -> existing sampler/collector/Gain owners -> atomic JSON.
- State boundary: Actor/Critic/checkpoint and training owners are frozen; only global inference RNG may advance to sample independent actions. Fixed normalized Critic inputs use the reset-owner/normalizer-derived `1e-3` numeric tolerance rather than bitwise hashing.
- Forbidden behavior: optimizer or normalizer update, sampler/curriculum mutation, scenario resampling, Critic/action conditioning, raw Gain changes, silent K16 substitution, or identical-action repeats accepted as evidence.
- Semantic fixture: two asymmetric fixed K8 Segment identities, three repeats, M4 targets `(0,1,2)`, shared values `(0.25,-0.1)`, distinct action fingerprints.
- Independent oracle: hand-computed repeat mean, population standard deviation, min/max, and value-minus-repeat-mean error.
- Sensitivity: scenario/hash drift, Critic-input drift above `1e-3`, repeated action fingerprints, non-finite target, invalid repeat count, and K outside `{8,16}` must fail closed.
- Evidence: S1 C1/C2/C3/C4/C5 deterministic cases, official preflight connectivity, Python compilation, existing EVAL-v004 regression suite.
- Stop: any discrete identity drift, Critic-input drift above `1e-3`, action non-diversity, training-state mutation, non-finite result, or failing existing evaluation contract.

## Execution

### Task 1: Red Tests

- Add a deterministic repeat-summary test to `source/rsl_rl/rsl_rl/tests/frontres_v019_critic_repeat_probe_contract.py`.
- Extend the launch-command contract to require and validate `POLICY_QUALITY_REPEAT_COUNT`.
- Run both tests and require failure because the repeat API/CLI does not yet exist.

### Task 2: Minimal Implementation

- Generalize the fixed held-out materializer from K16-only to homogeneous K8/K16 while retaining M4 and 16 environment rows.
- Thread `repeat_count` through `train.py`, `on_policy_runner.py`, and `run_frontres_stage3_segment_hrl.sh`.
- Repeat existing read-only collection in one checkpoint context; assert fixed scenario/hash, bounded Critic-input drift, and diverse action fingerprints while reporting Critic-value variation.
- Add `note/testing/manifests/frontres_v019_critic_repeat_k8_m4_v1.json` with exactly two fixed Segment items.
- Add `run/run_v019_critic_repeat_probe_gpu7.sh` with repeat count eight and stable `log/` outputs.

### Task 3: Verification And Review

- Run the new semantic test, existing v018 evaluator test, launch preflight test, manifest/entrypoint contracts, and aggregate FrontRES suite if focused tests pass.
- Run `python -m py_compile` on changed Python files and shell syntax checks.
- Review the exact diff for owner duplication, training writes, private access, hidden defaults, K/M drift, and report identity.
- End at `BOUNDED_LIVE_READY`; provide one server command but do not run it or start training.
