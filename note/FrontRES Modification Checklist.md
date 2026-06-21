# FrontRES Modification Checklist

Use this checklist after every nontrivial FrontRES change.  The goal is to keep
concept, code path, diagnostics, and short-run evidence aligned.  Do not mark a
change as ready for training until each relevant item has concrete evidence.

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

