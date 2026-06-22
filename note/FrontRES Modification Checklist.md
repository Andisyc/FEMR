# FrontRES Modification Checklist

Use this checklist after every nontrivial FrontRES change.  The goal is to keep
concept, code path, diagnostics, and short-run evidence aligned.  Do not mark a
change as ready for training until each relevant item has concrete evidence.

## Active Change Record: 2026-06-22 Update Memory Safety

- [x] Failure observed:
  CUDA OOM during `FrontRESUnified._update_ppo_supervised()` around iteration 200,
  while evaluating the critic inside `self.alg.update()`.
- [x] Cause narrowed:
  the active run uses a large 288k-sample FrontRES rollout.  The original live
  path compounded this with retained actor graphs; after that was fixed, the run
  still OOMed at iteration 207 during `value_term.backward()`.  This proves the
  previous retained-graph hypothesis was incomplete, but it does not yet prove
  whether the remaining cause is mini-batch peak memory, cross-iteration tensor
  retention, allocator fragmentation, or external GPU occupancy.
- [x] Algorithm memory fix:
  `FrontRESUnified` now calls `optimizer.zero_grad(set_to_none=True)` before
  backward and after skipped NaN-gradient updates.
- [x] Active experiment memory fix:
  `G1FlatFrontRESUnifiedRunnerCfg.algorithm.num_mini_batches` is increased from
  4 to 16, reducing per-minibatch activation memory without changing rollout
  sample count or the rho objective.
- [x] Region-direct graph fix:
  `FrontRESUnified` no longer builds PPO action-log-prob or per-dim rho-log-prob
  graphs when `frontres_structured_joint_rl_loss_mode="region_direct"`, because
  the active rho loss is a direct logit loss rather than a sample-log-ratio loss.
- [x] Retired legacy-branch graph cleanup:
  disabled legacy acceptance preference loss now returns a detached zero scalar
  instead of `mu_batch.sum() * 0`, so `frontres_acceptance_preference_weight=0`
  and `frontres_structured_joint_rl_keep_legacy_bce=False` do not leave a tiny
  obsolete actor graph on the live path.
- [x] Active update peak-memory fix:
  in active `region_direct` mode, `FrontRESUnified` now uses a memory-safe update
  route: critic value loss is backpropagated first, proposal supervised loss is
  backpropagated next, then the actor is recomputed for rho acceptance loss.
  This removes the previous live-path `retain_graph=True` peak while preserving
  the rho-only acceptance gradient boundary.
- [x] Update-path toy test:
  `source/rsl_rl/rsl_rl/tests/frontres_region_direct_update_path.py` runs one
  CPU `FrontRESUnified.update()` through the active region-direct branch and
  verifies `generic=0`, `repair_bce=1`, storage clear, and returned diagnostics.
- [x] Update memory pipeline test:
  `source/rsl_rl/rsl_rl/tests/frontres_update_memory_pipeline.py` repeatedly
  fills synthetic FrontRES storage and calls the formal `FrontRESUnified.update()`
  path.  It is used to separate algorithm-update memory retention from
  runner/environment/live-storage memory growth.  It supports both `--policy tiny`
  for isolated update logic and `--policy frontres` for the real
  `FrontRESActorCritic` network with synthetic storage.
- [x] CUDA cache fragmentation mitigation:
  `FrontRESUnified._update_ppo_supervised()` now releases unused CUDA cache at
  update entry and after storage clear, matching the OOM hint that reserved but
  unallocated memory was large.
- [x] Evidence-first OOM diagnostic:
  `frontres_cuda_memory_debug=True` prints compact CUDA allocated/reserved/peak/free
  sentinels at update entry, after the first mini-batch value/supervised/rho
  backward stages, and at the exact OOM stage if an OOM occurs.  This is
  diagnostic-only and must be inspected before changing batch count, allocator
  settings, or loss structure again.
- [ ] First short-run sentinel observed:
  the next run should print `mini_batches=16`, include `[FrontRES CUDA mem]`
  compact sentinel lines, and either pass iteration 207/220 or report the OOM
  stage with memory numbers.
- [ ] First short-run behavior checked:
  after iteration 200, inspect active-dim `rho by adv +/−` separation over a
  small window.

## Previous Change Record: 2026-06-21 Logit-Level Rho Repair Loss

- [x] Design note updated:
  `note/FrontRES Design Contract.md`, section
  `2026-06-21 Logit-Level Rho Repair Loss`.
- [x] Test reference updated:
  `source/rsl_rl/rsl_rl/tests/frontres_rho_low_recovery_mechanism.py`;
  `source/rsl_rl/rsl_rl/tests/frontres_storage_algorithm_loss.py`.
- [x] Core concept: keep the existing rho advantage evidence, but apply
  repairable-region rho learning to the rho logit with BCEWithLogits.
- [x] Forbidden change: do not add a new evidence source, do not merge boundary
  prior into `rho_adv`, and do not remove the old post-sigmoid loss ablation.
- [x] Config surface verified for
  `frontres_structured_joint_repair_loss_kind` and
  `frontres_structured_joint_repair_loss_scale`.
- [x] Algorithm path verified in `FrontRESUnified` region-direct loss.
- [x] Diagnostics verified to print `repair_bce` and `rscale` in the same live
  branch as `rho region loss`.
- [ ] First short-run sentinel observed: startup prints `repair=bce_logit` and
  log prints `repair_bce=1, rscale=1.000`.
- [ ] First short-run behavior checked: rho should recover from near-zero more
  quickly without making harmful Projected samples positive.

## 0. Design Delta

- [ ] Write the intended concept in one sentence.
- [ ] Name the variable being changed, such as rho authority, reward evidence,
  perturbation schedule, action cone, storage payload, or diagnostics.
- [ ] State what must not change.
- [ ] State whether this is a formal change, an ablation, or a diagnostic-only
  change.
- [ ] If the change affects a method idea, update
  `note/FrontRES Design Contract.md` before code is edited.

Evidence to record:

- Design section or note line:
- Old behavior:
- New behavior:

## 1. Config Surface

Check every config layer, not only the one being edited.

- [ ] Runner config class fields are updated.
- [ ] Algorithm config dataclass fields are updated.
- [ ] Algorithm constructor accepts the new field.
- [ ] Algorithm stores or prints the new field when useful.
- [ ] Active experiment config sets the intended value.
- [ ] Generic defaults are conservative and do not silently alter unrelated
  experiments.
- [ ] Legacy aliases or duplicated config names are either updated or explicitly
  left unchanged.

Required search:

```bash
rg -n "NEW_CONFIG_NAME|OLD_RELATED_CONFIG_NAME" source note
```

Evidence to record:

- Generic default:
- Active experiment value:
- Constructor field:
- Startup/log sentinel:

Common failure this prevents:

- Editing only `algorithm = RslRlFrontRESUnifiedAlgorithmCfg(...)` while the
  runner-level cfg still supplies the old value.
- Adding a config field to cfg dataclasses but not to `FrontRESUnified.__init__`,
  causing an unexpected keyword error.

## 2. Live Route

Prove the value reaches the live training path.

- [ ] Runner reads the intended config key.
- [ ] The helper/function receives the value as an argument.
- [ ] The helper/function uses the value in the intended formula or branch.
- [ ] Older fallback branches cannot overwrite the result.
- [ ] Cold-start and disabled-branch initialization set safe defaults.

Required search:

```bash
rg -n "NEW_SYMBOL|HELPER_NAME|OLD_VISIBLE_BEHAVIOR" source/rsl_rl/rsl_rl
```

Evidence to record:

- Config read location:
- Helper call location:
- Formula location:
- Disabled/default branch location:

## 3. Formula And Test Consistency

When a test module mirrors formal logic, explicitly compare both sides.

- [ ] Test formula and formal formula are intentionally identical, or the test
  clearly labels baseline versus proposed formula.
- [ ] Test inputs use the same default margins and relevant config constants as
  the formal code, unless the test is deliberately sweeping them.
- [ ] The test covers positive, negative, boundary, and low-signal cases.
- [ ] A known-bad case remains bad after the proposed change.
- [ ] A known-good low-signal case receives stronger evidence after the change.

Evidence to record:

- Test file:
- Formal file:
- Positive case:
- Negative case:
- Low-signal case:
- Sweep result:

## 4. Storage And Algorithm Contract

If the change affects training, verify the storage-to-loss path.

- [ ] Runner writes the changed value to the correct transition/storage field.
- [ ] Storage preserves shape, dtype, and batch order.
- [ ] Minibatch unpacking reads the correct field.
- [ ] Algorithm loss consumes the intended field.
- [ ] Gradient applies only to the intended policy dimensions.
- [ ] Old losses are disabled or kept only as deliberate ablations.

Required check:

```bash
rg -n "storage_field|transition_field|loss_name|gradient_boundary" source/rsl_rl/rsl_rl
```

Evidence to record:

- Transition write:
- Storage field:
- Minibatch read:
- Loss use:
- Gradient boundary:

## 5. Diagnostics And Log Sentinel

A mechanism is not live unless the log can prove it.

- [ ] Runner local values are initialized in cold-start/default paths.
- [ ] Runner local values are updated on the live path.
- [ ] Diagnostic sums include the new keys.
- [ ] Mean materialization exposes the keys with the expected names.
- [ ] Formatting function has access to the variables it prints.
- [ ] The log contains a one-line sentinel with the new value or mode.
- [ ] The sentinel appears in the same live branch as nearby old diagnostics.

Required search:

```bash
rg -n "NEW_DIAG_KEY|OLD_VISIBLE_LOG_LABEL|format_frontres" source/rsl_rl/rsl_rl/frontres
```

Evidence to record:

- Initialized key:
- Updated key:
- Accumulated key:
- Materialized mean:
- Printed line:
- First short-run log:

Common failure this prevents:

- Printing `locs` inside a formatter that only receives `loss_dict`.
- Adding a diagnostic to one log branch while the live branch prints a duplicate
  old label.

## 5A. OOM Debug Pipeline

Use this when CUDA OOM appears during FrontRES training.  The goal is to classify
the failure before changing any memory-related knob.

- [ ] Stage 0, route sentinel:
  confirm startup prints the intended `epochs`, `mini_batches`, objective, and
  structured rho mode.
- [ ] Stage 1, update-entry baseline:
  compare `[FrontRES CUDA mem] label=update_entry` across iterations.  If
  allocated/reserved memory rises monotonically at update entry, suspect
  cross-iteration tensor retention.
- [ ] Stage 2, first-mini-batch peak:
  inspect only the first mini-batch sentinel lines:
  `value_backward_after`, `actor_supervised_backward_after`, and
  `rho_backward_after`.  These show which update stage creates the largest peak
  without flooding the terminal.
- [ ] Stage 3, exact failure location:
  if OOM happens, use `oom_value_backward`, `oom_actor_supervised_backward`, or
  `oom_rho_backward` as the decisive stage label.
- [ ] Stage 4, branch audit:
  if update-entry memory grows, search for active branches that retain tensors,
  graphs, cached observations, diagnostics, or storage references across
  iterations.  Do not change `num_mini_batches` first.
- [ ] Stage 5, intervention:
  only after stages 1-4 identify the cause, choose one intervention:
  smaller per-mini-batch update, micro-batched critic/value backward, branch
  retirement, cache cleanup, allocator setting, or external GPU isolation.

Decision rules:

- Stable update-entry memory plus OOM at one backward stage:
  likely mini-batch peak memory in that stage.
- Rising update-entry memory:
  likely cross-iteration retention or an active branch keeping CUDA tensors.
- Large reserved/free mismatch with modest allocated memory:
  likely allocator fragmentation or external process pressure.
- Missing `[FrontRES CUDA mem]` lines:
  config route or algorithm constructor mismatch, not a memory conclusion.

Evidence to record:

- Startup sentinel:
- Update-entry trend:
- Peak stage:
- OOM stage:
- Decision:
- Intervention:

## 6. Static Verification

Run syntax and route checks before training.

- [ ] `python -m py_compile` covers every modified Python file.
- [ ] `rg` confirms the new key appears in all required layers.
- [ ] `git diff --stat` is reviewed for unrelated files.
- [ ] `git diff` is reviewed around each edited block.
- [ ] No unrelated user changes are reverted.

Suggested command pattern:

```bash
python -m py_compile FILES_CHANGED
rg -n "NEW_SYMBOL|NEW_CONFIG|NEW_LOG_LABEL" source note
git diff --stat
```

Evidence to record:

- Compile result:
- Search result:
- Diff reviewed:

## 7. First Short-Run Acceptance

The first short run proves the path, not final performance.

- [ ] Startup summary prints the intended config value or mode.
- [ ] The new log sentinel appears in the first relevant iteration.
- [ ] The old visible behavior changes only where expected.
- [ ] The key diagnostic moves in the expected direction.
- [ ] No exception occurs in rollout, logging, storage, or algorithm update.
- [ ] Performance is judged over a small window, not from one lucky iteration.

Evidence to record:

- Iteration:
- Startup sentinel:
- Live diagnostic line:
- Expected diagnostic movement:
- Unexpected behavior:

## 8. After-Run Decision

Classify the outcome before making another edit.

- [ ] Implementation worked and concept improved.
- [ ] Implementation worked but concept failed.
- [ ] Implementation did not reach the live path.
- [ ] Diagnostic was insufficient to judge.
- [ ] New failure belongs to current method core.
- [ ] New failure is a boundary diagnostic or future research problem.

Decision:

- Keep:
- Revert:
- Tune:
- Add diagnostic:
- Stop and rethink concept:
