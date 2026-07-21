# Evidence Ledger: FRS-v015 Future Intent And Single-Action K

Date: 2026-07-19 through 2026-07-21
Scope: Accepted semantic migration and bounded implementation evidence,
including deterministic contracts, CPU formal-route/persistence checks, and
the successful bounded S4 local identity sentinel. It contains no long-training,
policy-quality, deployment-composition, or live checkpoint-resume evidence.

## E-FI-0: Confirmed Conceptual Decision

Evidence:

- Current conversation, 2026-07-19: the user confirmed that FEMR is a
  Noisy-to-Executable local repair policy; q29 internal motion is the trusted
  deployable motion intent; root/ground/global artifact is the repair target.
- Current conversation, 2026-07-19: the user confirmed a two-role main
  counterfactual, one FEMR action at the first K frame, frozen FEMR afterward,
  and GMT execution through Clean continuation.
- Current conversation, 2026-07-19: the user confirmed future context is
  retained to resolve conflicting repair gradients, but it must be future
  29DoF internal intent from Noisy/deployment provenance rather than future
  raw root/global reference or Clean provenance.

Facts:

- x_t remains a Clean dynamic reset state and is not actor-visible Clean
  reference.
- The active local perturbation changes only the current root-level artifact.
- The required invariant is:
  
  ```
  Pi_internal(R^N[t:t+H]) = Pi_internal(R^C[t:t+H])
  ```
  
  The actor reads the left-hand Noisy/deployment provenance even when the
  values equal the Clean calibration source.
- H is an actor-information horizon. K is a frozen-FEMR GMT evidence horizon.
- The local K pair is:
  
  ```
  Noisy: x_t -> uncorrected artifact at t -> GMT on common Clean continuation
  Repair: x_t -> artifact + Delta SE(3)_t -> GMT on the same continuation
  ```
  
  Clean is not a third scored rollout.
- The active reward meaning is intent realization relative to the shared q29
  intent plus paired physical executability minus repair cost.
- Direct Repair-vs-Noisy similarity and full-Clean global rollout Style are
  rejected as active actor reward semantics.

Decisions:

- Activate FRS-METHOD-v015, FRS-TRAIN-v006, FRS-GAIN-v003, and FRS-EVAL-v003.
- Retain FRS-PPO-v003 because one-row grouped reduction and its mass semantics
  do not change.
- Treat the existing full-65D Noisy-tape route, legacy quartet layout, and
  Clean-global Style code as contract-mismatch until a later bounded
  implementation step proves migration or isolation.
- Keep persistent full-sequence artifacts as a separate deployment composition
  evaluation, not as later noise in the main first-action K return.

Open risks:

- The q29-preservation assumption is accepted semantics but not code- or
  runtime-confirmed.
- The current actor/command route still uses a superseded full-65D tape
  interpretation.
- Current HSL target semantics may contain a Clean-oriented path and require
  a dedicated audit before HSL migration.
- No two-role one-action/frozen-FEMR lifecycle, v003 Gain consumer route,
  formal transaction route, checkpoint migration, or live sentinel has been
  run under v015.

Next:

- Execute only Step 1A of the refined v015 plan after explicit user
  authorization: materialize/seal the local-scenario kernel, prove its q29
  invariant and immutable identity, then stop before actor wiring.

## E-FI-1: Read-Only Owner Audit and Engineering Replan

Date: 2026-07-19
Scope: White-box source inspection used to split the v015 implementation route.
No source modification, test execution, simulator, training, optimizer update,
checkpoint operation, or live run occurred.

Evidence:

- Current-conversation CodeGraph read of
  `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py::materialize_frontres_fixed_noisy_tape`
  and `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`.
- Current-conversation CodeGraph read of
  `source/rsl_rl/rsl_rl/runners/frontres_runtime.py::append_frontres_fixed_noisy_future_context`
  and `apply_obs_normalizer`.
- Current-conversation CodeGraph read of
  `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py::set_frontres_quartet_baseline`,
  `set_frontres_paired_baseline`, and `_sync_frontres_pairs`.
- Current-conversation CodeGraph read of
  `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py::to_ppo_batch`,
  `to_grouped_ppo_candidate_batch`,
  `source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py`, and
  `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py::FrontRESFrozenPolicyTransactionAccumulator`.
- Current-conversation CodeGraph read of
  `source/rsl_rl/rsl_rl/runners/frontres_warmup.py::run_frontres_joint_warmup`
  and `frontres_hsl_rollout_target.py::build_frontres_hsl_rollout_target`.

Facts:

- The current scenario materializer is named as a fixed Noisy tape owner; the
  runtime actor helper requires a rank-2 `[B, |H|*65]` tail and labels the
  resulting actor/normalizer surface as the legacy v013 layout.
- `MultiMotionCommand` retains both paired and quartet synchronization. The
  quartet exposes projected, candidate, Noisy/base, and Clean groups, so it is
  not the accepted two-role local counterfactual.
- Current storage has two different adapters: `to_ppo_batch()` intentionally
  omits transaction metadata, whereas `to_grouped_ppo_candidate_batch()` keeps
  it for candidate/offline grouped loss. Therefore formal grouped routing is a
  separate integration task, not a consequence of the existing loss formula.
- The frozen-policy transaction accumulator already rejects optimizer steps
  during collection and requires exactly one injected update, but it declares
  itself candidate-only/offline. It is not formal-route evidence.
- Current HSL warmup obtains its target through
  `get_supervision_target_task_space()` and directly processes raw observations;
  its new future-intent interface and target semantics are not code-confirmed.
- The current Gain capture/formula family remains a Clean-global Style path;
  it cannot be treated as implementation evidence for v003 q29 intent Gain.

Decisions:

- Replace the coarse five-step implementation sequence with twelve bounded
  steps and one read-only HSL decision gate.
- Separate: scenario materialization (1A), actor H routing (1B), two-role reset
  (2A), one-action K collection (2B), Gain core/consumers/evaluation
  (3A/3B/3C), grouped metadata/formal route/persistence (4A/4B/4C), and
  user-gated live local/composition evidence (5A/5B).
- Use `G0`--`G5` for method migration gates and retain repository `S0`--`S4`
  solely for evidence tiers in the rewritten plan/checklist.

Open risks:

- The exact q29 extraction/projection and current-artifact representation need
  deterministic implementation evidence.
- New actor layout changes normalizer/checkpoint identity, but persistence is
  intentionally deferred to Step 4C.
- HSL must remain disabled/isolated under v015 until Gate H0 and an explicit
  user decision.

Next:

- Step 1A only, after explicit authorization; stop after local-scenario S1
  evidence and a Step End Report.

## E-FI-2: Step 1A Immutable Local-Scenario Kernel (S1)

Date: 2026-07-19
Scope: Deterministic local-module evidence only. No simulator, training,
formal runner, optimizer update, checkpoint/resume operation, or live run was
started.

Implementation evidence:

- `MultiMotionCommand.materialize_frontres_local_scenario()` now owns the
  selection-time split payload: current root artifact `[7]`, dense q29 intent
  `[H_max+1,29]`, and Clean continuation `[K,65]`.
- The command materializer rejects non-q29 input and insufficient future frames
  rather than clamping or carrying future root/global data into intent. It calls
  the physical perturber only for the current root artifact and never calls the
  joint-perturbation owner for q29 intent.
- `FrontRESLocalScenario` seals the five semantic inputs plus request identity
  and provenance into `noisy_segment_hash`; `FrontRESLocalScenarioLifecycle`
  materializes once per selected source, reuses the same object for M rows, and
  rejects rematerialization after close.
- The new sampler-to-command attachment carries separate named tensors for root
  artifact, q29 intent, and padded Clean continuation. It is intentionally not
  attached to the formal runner yet: the formal batch builder retains the old
  fixed-tape route until later authorized reset/actor/GMT work. The legacy
  fixed-tape reset owner fails closed when it receives a v015 local scenario.

Executed evidence:

- `source/rsl_rl/rsl_rl/tests/frontres_local_scenario_kernel_contract.py`
  passed all required deterministic checks:
  `T-schema`, `T-invariant`, `T-hash`, `T-provenance`, `T-metamorphic`, and
  `T-legacy-reject`.
- `python -m py_compile` passed for the four Step 1A owners and the focused
  contract test.

Facts established:

- The fixture proves `Pi_internal(Noisy) == Pi_internal(Clean)` numerically for
  q29 while the current root artifact differs; q29 still carries explicit
  `deployment_noisy_q29` provenance.
- Hash changes when `x_t`, current artifact, q29 values/source/window, Clean
  continuation, or K changes.
- A returned intent accessor is a copy, so caller mutation cannot change the
  sealed scenario; M rows reuse one scenario/hash and a closed identity cannot
  be rematerialized.

Open boundaries:

- No actor q29-H bridge, observation layout, normalizer, or checkpoint identity
  has been changed.
- No reset role, one-action K executor, GMT continuation consumer, Gain,
  PPO/transaction, formal runner, or live path has been connected.

Next:

- Stop at Step 1A. Step 1B remains pending explicit user authorization.

## E-FI-3: Step 1B Future-Intent Actor Bridge (S1)

Date: 2026-07-19
Scope: Deterministic actor-layout, provenance, and normalizer evidence only.
No simulator, training, formal runner, optimizer update, checkpoint/resume
operation, or live run was started.

Implementation evidence:

- `frontres_observation_layout.py` owns the explicit
  `frontres-v015-future-intent-q29-v1` layout: declared positive offsets map a
  sealed `[B,H_max+1,29]` deployment-q29 carrier to ordered
  `[B,|H|*29]` actor-tail values.
- `frontres_runtime.py` owns the runtime bridge. It consumes only the local
  scenario q29 carrier and its provenance; it does not read the current root
  artifact, Clean continuation, raw 65D tape, future root/global data, noise
  metadata, or perturbation timing.
- `OnPolicyRunner` and both FrontRES configuration owners now allocate the
  actor prefix from the versioned q29 layout rather than `|H|*65`. The rollout
  helper calls the new q29 bridge.
- `apply_obs_normalizer()` validates the selected layout and rejects
  unversioned or incompatible checkpoint-like prefix statistics when local
  future intent is active. Persistence of the new identity remains deferred to
  Step 4C.

Executed evidence:

- `python -m py_compile` passed for all Step 1B owners, both configuration
  owners, and the focused contract tests.
- `source/rsl_rl/rsl_rl/tests/frontres_future_intent_actor_context_contract.py`
  passed `T-shape`, `T-offset`, `T-provenance`, `T-clean-isolation`,
  normalizer-layout rejection, and `T-legacy-reject`.
- Existing deterministic regressions passed:
  `frontres_observation_layout_contract.py`,
  `frontres_fixed_noisy_actor_context_contract.py`, and the Step 1A
  `frontres_local_scenario_kernel_contract.py`.

Facts established:

- The actor H tail is exactly the ordered q29 values of the declared offsets;
  it contains neither future root/global values nor Clean continuation values.
- Numeric q29 equality with a Clean calibration source is insufficient without
  the required `deployment_noisy_q29` provenance; a Clean-labelled carrier is
  rejected fail-closed.
- A 65D carrier, absent local scenario, wrong layout version, or incompatible
  normalizer statistics is rejected rather than adapted.
- Clean `x_t` remains outside the actor reference carrier. The Step 1A sealed
  scenario is read-only across attempts; this step adds no reset, K, Gain, or
  PPO behavior.

Open boundaries:

- The formal batch/reset route is intentionally not connected to this offline
  bridge; it must not silently supply the legacy full-tape field as v015 input.
- Checkpoint serialization/resume does not yet persist the future-layout
  identity; the runtime rejection above is a temporary safety boundary, not
  Step 4C evidence.
- HSL remains isolated. Its interface and target semantics require the separate
  read-only H0 audit and a later user decision.

Next:

- Stop at Step 1B. Do not begin H0, Step 2A, formal-route integration,
  checkpoint work, or a live run without explicit user authorization.

## E-FI-4: Gate H0 HSL Interface and Target Audit (S0)

Date: 2026-07-20
Scope: Read-only source, layout, target, configuration, and checkpoint audit.
No source or contract change, test execution, simulator, training, formal
runner, optimizer update, checkpoint operation, or live run occurred.

Concept and contract boundary:

- `M-03` is HSL Warmup: it may initialize the 6D actor, but Training v006
  requires its actor interface to match the deployable q29 future-intent
  interface before re-enabling it.
- HSL is not allowed to put Clean future provenance or a full-Clean rollout
  target through the Stage 2 actor interface. Its target semantics were
  deliberately left for this audit and a subsequent user decision.

White-box chain:

```text
Stage-1 preset -> OnPolicyRunner.learn -> run_frontres_joint_warmup
  -> raw policy obs -> _apply_obs_normalizer -> residual_actor
  -> get_supervision_target_task_space -> supervised warmup loss

standard FrontRES rollout -> env.step -> build_frontres_hsl_rollout_target
  -> transition.supervised_target/weight/harm_weight -> RolloutStorage
  -> FrontRESUnified._compute_supervised_loss -> optimizer step
```

Code-confirmed findings:

- `run_frontres_joint_warmup()` never calls the v015
  `append_frontres_future_intent_context()` bridge. It normalizes raw policy
  observation and calls `residual_actor` directly. The v015 bridge, by contrast,
  requires a sealed q29 scenario and prepends its ordered q29 tail.
- With the current task-space configuration `num_frontres_obs=0`, a v015
  segment-replay runner expands `num_actor_obs` by the q29-tail width while the
  warmup tensor remains raw-width. The warmup fallback sets `_nfo` to the
  enlarged actor width, but slicing cannot manufacture the missing tail. This
  is a code-confirmed layout mismatch; no live exception was intentionally
  triggered in this audit.
- `get_supervision_target_task_space()` is a current-frame simulation-oracle
  anti-DR label: it reads the current perturbation delta and quaternion
  correction. It does not read a future q29/65D window or construct a
  full-Clean rollout target. It is nevertheless privileged training evidence,
  not deployment actor input or executable-return evidence.
- `build_frontres_hsl_rollout_target()` is a separate active legacy route. It
  reads FrontRES, Noisy, and Clean quartet root positions/orientations, forms a
  FrontRES-to-Clean residual, and writes it to the supervised loss path. The
  current agent configuration enables `frontres_hsl_rollout_label_enabled=True`;
  the Stage-3 preset zeros direct warmup iterations but does not clear this
  rollout-label flag.
- The legacy rollout-label route therefore reintroduces exactly the prohibited
  full-Clean/global supervision object. Its existing contract test proves that
  quartet/Clean behavior; it is historical coverage, not v015 acceptance.
- `load_runner()` restores the residual actor with `strict=True` and has no
  future-intent layout identity in its checkpoint payload. A legacy HSL actor
  has the old input first-layer shape, while v015 adds the q29 tail; v015
  normalizer code also rejects unversioned legacy prefix statistics once local
  future intent is active. No checkpoint was loaded here.

Classification:

- The current-frame anti-DR label is not itself a Clean-future leak. Whether it
  remains an allowed proposal-only initializer is a human semantic decision.
- The quartet Clean rollout label is a v015 contract mismatch and must remain
  disabled/legacy; it cannot be silently carried into formal Stage 3 or HSL.
- The existing HSL warmup and HSL checkpoint paths cannot initialize the v015
  actor as implemented because they do not supply the q29 tail or its layout
  identity.

Decision required:

- Decide whether to retain only the current-frame anti-DR oracle label as an
  explicitly proposal-only HSL initializer after a future q29-interface
  migration, or retire HSL from v015 entirely. No implementation action is
  authorized by this audit.

Open boundaries:

- No v015 HSL observation bridge, target migration, checkpoint migration, or
  formal-route isolation test exists.
- No runtime evidence establishes behavior after a migrated HSL path; this is
  S0 source evidence only.

Next:

- Stop at the H0 semantic decision. Do not re-enable HSL, modify its target,
  migrate checkpoints, or start a formal/live route without explicit user
  direction.

+## E-FI-5: H0-A Proposal-Only HSL Contract Closure

Date: 2026-07-20
Scope: User-confirmed semantic decision and governed document versioning only.
No training source, simulator, training, formal runner, optimizer update,
checkpoint/resume operation, or live run occurred.

Decision evidence:

- Current conversation, 2026-07-20: the user confirmed H0-A.
- Retain HSL only as Stage-1 proposal-direction initialization.
- Allow only the current-frame anti-DR oracle as a privileged training target.
- Require the actor input to be current Noisy root artifact plus future
  deployment/Noisy q29 intent.
- Forbid Clean actor input, Clean future H context, Clean Stage-1 target, and
  the quartet/Clean rollout label in Stage-3 storage, loss, PPO, or formal route.
- Forbid direct legacy HSL checkpoint migration or compatibility loading.

Governed contract result:

- FRS-TRAIN-v007 is active and supersedes FRS-TRAIN-v006.
- v006 is archived as superseded; the registry and Design Point Register map
  M-03 and M-05 to v007 without changing the Concept Figure.
- The v015 plan now records conditional HSL Migration Step H1 with explicit
  S1/S2 evidence and a stop condition. H1 is not active and does not block
  Step 2A when HSL remains disabled.

Facts retained from H0:

- The allowed current-frame target is privileged training evidence, not a
  deployable actor input or executable-return target.
- The existing rollout target is legacy Clean-quartet supervision and remains
  a contract mismatch until H1 isolates it.
- Old HSL checkpoints lack the v015 q29 layout identity and remain reject-only.

Open boundaries:

- No H1 source implementation, deterministic H1 test, offline connector,
  checkpoint migration, formal route, or live evidence exists.
- The current source remains legacy until a separately authorized H1
  implementation proves every v007 isolation rule.

Next:

- Stop after H0-A documentation closure. Await explicit authorization for H1
  before any training-source modification.

## E-FI-6: H1-S1a Proposal-Only HSL Deterministic Implementation

Date: 2026-07-20
Scope: H1-S1a only. It implements Stage-1 q29 input/current anti-DR target and
rejects Stage-3 legacy/direct HSL writes, nonzero v015 supervised loss, and
legacy HSL checkpoint load. No simulator, training, formal runner, optimizer
update, checkpoint/resume operation, or live run occurred.

Evidence:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py` exited 0 and printed all seven T-HSL checks.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_rollout_target_contract.py` exited 0.
- `frontres/bin/python -m py_compile` over all changed H1-S1a Python modules
  exited 0.
- Static consumer audit found the only remaining direct
  `transition.supervised_target` assignment in `frontres_rollout_step.py`; its
  v015 guard returns at zero lambda and raises at nonzero lambda. The generic
  storage tensors remain zero defaults, not HSL evidence.

Facts:

- `frontres_warmup.py` prepends the sealed q29 context before normalizer/actor
  use and validates a detached finite current anti-DR `[B,6]` target.
- The legacy rollout-label owner now raises before reading source data or
  mutating transition storage; the standard runner rejects its enabling flag.
- `frontres_unified.py` rejects nonzero `lambda_supervised` or floor whenever
  v015 future offsets are selected; the G1 config sets both to zero.
- `frontres_checkpointing.py` rejects a legacy `frontres_warmup_complete`
  payload before sampler, actor, normalizer, optimizer, or iteration restoration.

Open risks:

- S1 is deterministic/module-local only. It does not prove a real warmup has a
  sealed scenario carrier, physics behavior, or policy quality.
- The historical policy-quality/formal evaluator is explicitly out of H1-S1a;
  its retired HSL label now fails closed, but its custom checkpoint route has
  not been integrated or audited here.
- No new checkpoint identity exists; that remains a separate persistence step.

Next:

- Stop after H1-S1a. Await explicit authorization for H1-S2 fake connectivity.

## E-FI-7: H1-S2 Proposal-Only HSL Offline Connectivity

Date: 2026-07-20
Scope: CPU-only fake local-scenario connectivity for the already accepted H1
semantics. It does not construct an environment, call `run_frontres_joint_warmup`,
invoke PPO or an optimizer update, use a formal evaluator, load/save a
checkpoint, or start simulator, training, or live execution.

Evidence:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py` exited 0. It printed both owner traces.
- The S1 v007 contract and retired-label reject regression both exited 0 after
  the S2 test was added.
- `frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py` exited 0.
- Static consumer audit still finds one direct `transition.supervised_target`
  assignment, in the legacy branch of `frontres_rollout_step.py`; v015 returns
  at zero lambda and raises at nonzero lambda. The historical formal-quality
  caller reaches the retired label owner, which now raises before target output.

Facts:

- A semantic fake scenario carried detached `[2,4,29]` deployment-provenance
  q29 intent and a `[2,2,65]` Clean-continuation sentinel. The real q29 owner
  selected `[2,58]`, the real warmup owner passed `[2,63]` to the normalizer,
  and the fake residual actor received the real `[2,60]` FrontRES prefix. The
  Clean sentinel did not appear in either actor-facing tensor.
- The real target validator accepted only the detached current anti-DR `[2,6]`
  target. The fake actor prediction was compared to that target without an
  optimizer or backward call.
- The real v015 zero-lambda writer left all transition HSL fields `None`; the
  real storage then exposed zero target/default weight tensors to its batch;
  the real unified-loss helper returned scalar zero with no gradient and the
  fake optimizer recorded zero step calls.

Open risks:

- This is an offline connector proof, not a simulator, real warmup, physics,
  policy-quality, formal-route, checkpoint/resume, or live-runtime proof.
- The formal policy-quality evaluator remains outside H1. Its HSL target call
  fails closed, but no formal-route integration test has exercised that path.
- A sealed scenario still has no authorized reset/lifecycle installation in a
  real runner, and a new checkpoint identity is still undefined.

Next:

- H1 is complete. Stop and await explicit authorization for Step 2A, which
  must retain HSL disabled and remain separate from formal/live work.

## E-FI-8: Step 2A Two-Role Local Reset And Command Layout

Date: 2026-07-20
Scope: deterministic fake-reset implementation of the v015 Repair/Noisy layout
and immutable local command carrier. It does not route actor H or GMT K, sample
an action, execute a simulator, construct Gain/return/PPO, touch a checkpoint,
invoke a formal runner, or start training/live execution.

Evidence:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_two_role_reset_contract.py` exited 0. It printed `T-2A-role`, `T-2A-scenario-identity`, `T-2A-state`, and `T-2A-legacy-reject`.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_local_scenario_kernel_contract.py` exited 0 after its legacy assertion was rebased from the retired local-to-fixed-tape block to the active local/fixed-tape mixing rejection.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage1_env_hooks_contract.py`, `frontres_segment_motion_command_reference_contract.py`, `frontres_hsl_v007_s1_contract.py`, and `frontres_hsl_v007_s2_connectivity_contract.py` exited 0 as bounded legacy/HSL regressions.
- `frontres/bin/python -m py_compile` on the three Step 2A owners and the new deterministic contract exited 0; `git diff --check` exited 0.

Facts:

- `configure_frontres_pair_layout()` recognizes the frozen v015 q29 layout and installs only equal Repair and Noisy rows. Its legacy count carrier is `n_train=Repair`, `n_base=Noisy`, with candidate and Clean counts zero; explicit role IDs are retained separately.
- `apply_frontres_segment_index_reset()` accepts a v015 local request only with exactly `repair` and `noisy` role names. It expands the same source row into both roles, restores the physical robot from the Clean replay motion/frame (`x_t` dynamics only), and installs the same detached artifact, q29 intent, Clean continuation, K/lengths, identities, provenance, and hash into command-owned storage.
- `MultiMotionCommand.set_frontres_local_scenario()` clones the local carrier, requires every command row to be covered, requires one Repair plus one Noisy row per scenario, and rejects an active same-identity mutation. A retry with identical data reopens only the current-frame cache readiness; it does not call the perturber or rematerialize a scenario.
- The current root artifact is copied to the command cache with a zero Stage-3 supervised target. q29 remains stored as deployment provenance; Clean continuation remains stored-only. A generic future command read, a second current-cache refresh, fixed-tape installation, or reference-window installation fails closed before Step 2B.

Open risks:

- This is a fake reset proof only. It does not prove the formal runner selects the v015 layout, that actor observation consumes q29, or that GMT consumes C.
- Step 2A intentionally does not execute the K horizon, write an action, collect a policy row, calculate Gain/return, or reach transaction/PPO owners.
- A later transaction-close owner must explicitly close the complete command carrier before a different scenario may replace it; this prevents an unsafe mixed replacement in the current bounded implementation.

Next:

- Stop at Step 2A. Step 2B requires separate authorization for the one-action,
  frozen-FEMR, GMT-only Clean-continuation collector and its deterministic S1
  contract; formal/checkpoint/live paths remain outside scope.

## E-FI-9: Step 2B One-Action Frozen-FEMR Clean-C Collector (S1/S2)

Date: 2026-07-20
Scope: deterministic CPU fake reset-to-capture connectivity only. No simulator,
training, formal runner, optimizer update, Gain/return/priority/PPO operation,
checkpoint/resume operation, or live run was started.

Implementation evidence:

- `MultiMotionCommand` now owns an explicit candidate-only K execution phase.
  It advances only the sealed `[K,65]` Clean continuation after t, routes its
  q29/dq29/root fields to GMT command reads, zeros the one-time repair before
  every later GMT action, and returns an exact per-row K-valid mask. It does
  not route C before the actor action.
- `frontres_rollout_step.py` owns the authorization split: one t call to
  `alg.act()` can produce the full-6D Repair tuple; the frozen phase rejects a
  second call through the normal rollout helper and directly invokes the frozen
  GMT execution adapter without a later FrontRES correction write.
- `FrontRESV015OneActionKEvidence` is an immutable candidate-only carrier. It
  contains one Repair observation/action/log-prob/value/mean/sigma tuple per
  local scenario plus `[K,N,65]` C, `[K,N]` exact masks, GMT actions, and
  Repair/Noisy scenario/hash/x_t identities. It contains no reward, return,
  advantage, optimizer, or legacy PPO adapter.
- `collect_frontres_v015_one_action_k_evidence()` is deliberately separate from
  the legacy live collector. The latter now rejects an active v015 local
  scenario so it cannot silently repeat actor actions or enter formal storage.
- Completing the K capture calls the command close method: only the K cursor
  closes, while the sealed artifact/I/C/K/identity/hash remain intact for the
  next Clean-reset M attempt.

Executed evidence:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_one_action_k_contract.py` exited 0. It printed
  `T-action-count/T-frozen`, `T-continuation/T-row`, and
  `T-K-metamorphic/T-legacy-reject`.
- The deterministic fake chain used the real Step 2A reset hook and command
  carrier: Clean `x_t` reset -> one t actor/repair write -> command C cursor ->
  frozen GMT calls -> immutable evidence. It has no simulator or formal runner.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_two_role_reset_contract.py` and
  `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py` exited
  0 as focused reset and legacy-probe regressions.
- `frontres/bin/python -m py_compile` passed for the four Step 2B owners and
  the new contract test.

Facts established:

- For two scenarios with K=(3,2), exactly one actor sample and one command
  repair write occur. The three frozen-GMT frames read C q29 values
  `[1000,2000,1000,2000]`, `[1100,2100,1100,2100]`, and clamped final C with
  exact valid mask `[T,F,T,F]`; no later actor or repair write occurs.
- Changing K changes the number and validity of evidence frames, not the number
  of policy tuples: both K fixtures store two Repair tuples for two scenarios.
- The t GMT action sees deployment-q29 current intent, while every later GMT
  action reads Clean C q29/dq29/root from the command owner. C never enters the
  actor action at t.
- Re-arming the same carrier after capture preserves artifact/I/C/K/scenario
  hash without a perturber call or rematerialization; it requires a new Clean
  reset/current-cache installation before the next actor action.

Open boundaries:

- This proof has no Gain, return/advantage, priority, grouped-PPO, transaction,
  checkpoint, formal-route, simulator, training, or live evidence.
- The candidate collector is not connected to `OnPolicyRunner` or the legacy
  `run_frontres_segment_live_probe()` loop; that loop is intentionally
  fail-closed for an active v015 local scenario.
- Actual frozen GMT policy/physics execution is represented by the fake adapter
  only; a later user-gated formal route and live sentinel must prove it with the
  real policy/environment.

Next:

- Stop at Step 2B. Step 3A is separately authorizable only for the pure
root-invariant q29 intent/physics/cost Gain core; it must not connect return,
priority, PPO, checkpoint, formal route, simulator, training, or live work.

## E-FI-10: Root-Invariant Intent Gain Core S1

Date: 2026-07-20  
Tier: S1 deterministic module semantics only  
Authorization: user-authorized Step 3A.

Implementation:

- `frontres_gain.py::compute_intent_physics_local_repair_gain()` is the new
  typed, side-effect-free FRS-GAIN-v003 calculation owner.
- Its input has q29 intent/execution tensors shaped `[B,29]` or
  `[B,T,29]`, a required `intent_q29_provenance=deployment_noisy_q29`,
  a non-Clean/root/global q29 source string, paired scalar Physics facts, and
  executed full-6D action evidence shaped `[B,6]` or `[T,B,6]`.
- Its result is one `[B]` decomposition:
  `intent_gain + physics_gain - repair_cost -> gain_total`. The
  `style_gain` property is an explicit alias to `intent_gain`, not a
  Clean-global Style metric.
- qvel/qacc and one-action temporal terms stay `NaN` when unavailable;
  a partially supplied derivative triple or invalid provenance fails closed.
- The prior Physics and full-6D cost primitives are reused as shared pure
  components without routing their legacy Clean-global composition into v003.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_gain.py source/rsl_rl/rsl_rl/tests/frontres_intent_physics_gain_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_intent_physics_gain_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py

Observed output:

- `[T-value/T-sign]`: fixed-I q29 error gives the expected signed
  Noisy-to-Repair intent gain.
- `[T-noop/T-invariant]`: equal Noisy/Repair execution gives zero intent
  gain; changing I changes the result.
- `[T-root-exclusion/T-provenance]`: there is no Clean/root/global typed
  input channel, and prohibited provenance is rejected.
- `[T-unconfirmed]`: missing qvel/qacc and one-action temporal values
  remain `NaN`.
- `[T-pair/T-full6]`: survival is K-normalized and all six Delta SE(3)
  coordinates enter the repair cost.
- `frontres_intent_physics_gain_contract: ok`; historical
  `frontres_gain_components_contract: ok`.

Facts established:

- Direct Repair-vs-Noisy similarity cannot define the new intent component:
  both execution branches are measured only against the same fixed q29 I.
- Root translation/orientation, Clean positions, Clean root quaternions, and
  global-body metrics are structurally absent from the v003 typed input.
- Current one-action K evidence can later supply a finite magnitude cost even
  when temporal action difference is not yet observable; the missing temporal
  diagnostic is explicit rather than zero-filled.

Open boundaries:

- No actual q29 execution facts have been captured from the candidate collector.
- No Gain output reaches return/advantage, priority, diagnostics, evaluator,
  PPO, checkpoint, formal route, simulator, training, or live run.
- The v002 `compute_segment_gain()` route remains isolated legacy code; its
  preservation regression is not v003 consumer evidence.

Next:

- Stop at Step 3A. Step 3B requires separate authorization and must prove only
  candidate capture -> v003 Gain -> return/priority provenance with v002
  Clean-global rejection before any grouped PPO or formal-route work.

## E-FI-11: Candidate Gain-to-Return and Priority Connectivity S1

Date: 2026-07-20  
Tier: S1 deterministic candidate-only consumer connectivity  
Authorization: user-authorized Step 3B.

Implementation:

- `collect_frontres_v015_one_action_k_evidence()` now captures post-`t` robot
  q29 from `command.robot_joint_pos` after the first action, records t/K done
  and survival evidence, and retains the sealed deployment/Noisy q29 provenance.
- `pair_frontres_v015_gain_facts()` pairs each Repair policy row with its same-
  scenario Noisy row. Its intent target is only `I[:,0]`; H remains actor
  context and Clean C remains GMT-only executable evidence.
- `collect_frontres_v015_gain_return_priority_evidence()` invokes only
  `compute_intent_physics_local_repair_gain()`, then creates one immutable
  `return_k = gain_total`, `advantage_k = return_k - old_value` carrier and
  scenario-keyed priority evidence with the same v003 decomposition.
- Invalid post-t q29 rows remain invalid: the complete v003 decomposition,
  return, and advantage are `NaN`, while priority remains evidence only. There
  is no legacy storage batch,
  sampler-state update, actor-loss mass, PPO loss, optimizer, checkpoint,
  formal runner, simulator, training, or live entrypoint.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/tests/frontres_v015_gain_consumer_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_gain_consumer_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_one_action_k_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_intent_physics_gain_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py

Observed output:

- `[T-provenance/T-consumer/T-no-v002-fallback]` proves that sealed `I[t]` and
  post-t robot q29 reach v003 only, with `return=Gain`.
- `[T-priority-isolation/T-invalid]` proves invalid q29 rows fail closed and a
  priority artifact cannot mutate the return/loss carrier.
- The Step 2B one-action/K, Step 3A Gain-core, legacy storage, sampler, and
  legacy live-probe deterministic regressions all pass.

Facts established:

- Noisy and Repair executions are evaluated against the same fixed deployment/
  Noisy q29 intent, never against each other or a Clean-global target.
- Candidate return and priority preserve scenario id, noisy segment hash, x_t
  identity, K, q29 provenance, v003 decomposition, and the fail-closed row mask.
- The v002 `compute_segment_gain()` and `_capture_paired_gain()` paths are
  monkeypatched to fail in the candidate contract and are not invoked.
- Return and priority validators independently reject Clean/root/global q29
  sources; no valid row can retain only a partial v003 decomposition.
- Priority is a copied, scenario-keyed artifact rather than sampler state; it
  cannot change an actor loss or update count.

Open boundaries:

- The proof uses a deterministic fake two-role reset/capture chain. It does not
  prove real-policy robot timing, physical q29 accuracy, optional ZMP/contact
  capture, or a formal Stage-3 execution route.
- Candidate priority has no stable segment/trial identity and intentionally
  performs no sampler-state update; a later transaction/metadata step owns that
  connection.
- Diagnostics, periodic/local evaluation, composition evaluation, grouped PPO,
  checkpoint/resume, formal runner, simulator, training, and live evidence remain
  outside this step.

Next:

- Stop at Step 3B. Step 3C is separately authorizable only for diagnostic and
  evaluation isolation; it must not alter the sealed return/priority carrier or
  enter grouped PPO, formal runner, checkpointing, simulator, training, or live
  work.

## E-FI-12: v003 Diagnostics And Evaluation Isolation S1

Date: 2026-07-20  
Tier: S1 deterministic candidate-only diagnostic/protocol connectivity  
Authorization: user-authorized Step 3C.

Implementation:

- `frontres_segment_diagnostics.py::build_frontres_v015_local_evaluation_report()`
  projects the sealed Step 3B `FrontRESV015GainConsumerEvidence` into an
  immutable local-K report. It reads only v003 intent/physics/cost/total, q29
  provenance/source, scenario/hash/x_t/K, and valid policy rows; it has explicit
  false return/priority/PPO feedback flags.
- Missing valid rows remain `NaN`/`UNCONFIRMED`; the report rejects partial
  v003 decomposition, Clean/root/global q29 provenance, malformed identities,
  or any feedback flag.
- `FrontRESV015CompositionEvaluationProtocol` is a separately typed
  deployment-stream protocol with frame/action counts and false local return,
  replay-priority, and PPO feedback. It is not a sequence executor or metric.
- The legacy periodic/offline/sequence evaluators now reject the explicit
  `frontres-v015-future-intent-q29-*` layout before sampler access, reset,
  legacy rollout, or `FRS-GAIN-v002` capture.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py source/rsl_rl/rsl_rl/tests/frontres_v015_evaluation_isolation_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_evaluation_isolation_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_gain_consumer_contract.py

Observed output:

- `[T-diagnostic/T-evaluator/T-no-v002-fallback/T-no-zero-fill]` proves the
  sealed v003 candidate carrier formats intent/physics/cost/source only, while
  both legacy v002 functions are monkeypatched to fail and remain uncalled; an
  all-invalid report remains NaN/UNCONFIRMED and rejects any zero-filled value.
- `[T-composition-isolation]` proves the composition protocol cannot accept
  local return evidence, all feedback flags fail closed, candidate return and
  priority tensors remain unchanged, and all three legacy evaluators reject
  v015 before capture.
- Existing diagnostics, live-training pseudo, sequence-evaluation, and Step 3B
  candidate-consumer regression contracts pass. These are deterministic tests,
  not simulator or formal-route evidence.

Facts established:

- v015 local diagnostics have one active source: the sealed v003 candidate
  decomposition, not Clean-global Style, a v002 fallback, or a newly computed
  metric.
- The composition question is recorded as a distinct deployment-stream protocol,
  so it cannot silently alter local-K return, replay priority, PPO eligibility,
  actor loss, or optimizer state.
- Existing v002 evaluator APIs remain historical behavior for non-v015 inputs,
  but are now explicit non-consumers for v015.

Open boundaries:

- No formal periodic evaluator is wired to the v015 report, and no real
  full-sequence executor or composition metric has run.
- The protocol does not prove robot timing, q29 tracking accuracy, persistent
  artifact behavior, policy quality, checkpoint behavior, or live deployment.
- Formal transaction metadata, grouped PPO, checkpoint/resume, runner wiring,
  simulator, training, and live sentinel evidence remain outside this step.

Next:

- Stop at Step 3C. Step 4A is separately authorizable only for sealed
  transaction metadata and the grouped candidate adapter; it must not connect
  the formal runner, optimizer, checkpoint/resume, HSL, simulator, training,
  or live work.

## E-FI-13: Sealed Local Metadata And Grouped Candidate Adapter S1

Date: 2026-07-20  
Tier: S1 deterministic candidate-only storage/adapter connectivity  
Authorization: user-authorized Step 4A.

Implementation:

- `frontres_segment_storage.py::FrontRESV015GroupedCandidateMetadata` seals
  `transaction_id`, `policy_snapshot_id`, motion/start/Segment/source/trial,
  scenario/hash/`x_t`, q29 provenance/source, `horizon_k`, and
  `evidence_valid_step_count` for exactly one Repair policy row per attempt.
  One source may contain M attempts, but all must share the same local scenario
  identity and have unique trial indices.
- `build_frontres_v015_gain_return_evidence()` derives the evidence-step count
  from the Repair branch's actual frozen-GMT survival count and rejects a
  non-integer, negative, or over-K count.
- `build_frontres_v015_grouped_candidate_storage()` binds sealed Step 3B
  evidence to the metadata-bearing one-row storage object. The live-probe
  connector then creates a `FrontRESSegmentPPOBatch` only through
  `to_grouped_ppo_candidate_batch()`; it invokes no loss, backward, or update.
- `to_ppo_batch()` rejects v015 metadata rather than silently dropping it;
  `to_grouped_ppo_candidate_batch()` rejects legacy fixed-tape metadata,
  incomplete row sets, mixed local identities, and duplicate source/trial rows.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/tests/frontres_v015_grouped_candidate_adapter_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_grouped_ppo_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_grouped_candidate_adapter_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_grouped_ppo_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_gain_consumer_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_evaluation_isolation_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py

Observed output:

- `[T-schema/T-row/T-metadata/T-legacy-reject]` proves a sealed v003 carrier
  becomes one `[B,6]` Repair-row candidate batch with scenario/hash/`x_t`/q29/
  K/evidence identity, while the legacy adapter rejects it.
- `[T-permute/T-scale/T-k-isolation/T-fail-closed]` proves row permutation and
  changed K evidence metadata leave grouped actor mass unchanged; a partial
  transaction fails before reduction.
- The existing grouped-PPO contract re-proves equal motion -> Segment -> attempt
  mass with v015 metadata, sign-preserving scale, and no sampling/replay loss
  multiplier. Step 3B/3C, generic storage, and legacy fixed-tape isolation
  regressions pass.

Facts established:

- `noisy_segment_hash` is now carried by active candidate metadata as the sealed
  local-scenario identity; no Step 4A path treats it as a whole Noisy K tape.
- K and actual evidence-step count are retained separately from policy rows and
  are not read by the grouped reduction as actor-loss weights.
- The old fixed-tape/S1b transaction object may remain for historical reset
  tests, but it is an explicit non-consumer of the active v015 candidate adapter.

Open boundaries:

- No formal `on_policy_runner` caller selects this adapter, no real frozen
  policy transaction collects all M attempts, and no optimizer step has run.
- No checkpoint/resume identity, sampler-state mutation, simulator timing,
  training, real evaluator, or live deployment evidence is established.
- The live-sampler regression uses only fake callbacks and temporary fake
  checkpoint identifiers; it is not checkpoint/resume or optimizer evidence.

Next:

- Stop at Step 4A. Step 4B is separately authorizable only for formal route
  connection and exact-one update proof; it must not change grouped mathematics,
  HSL, checkpoint/resume, simulator, training, or live work.

## E-FI-14: Step 4B Fake Formal Transaction And Exact-One Update S2

Date: 2026-07-20  
Tier: CPU-only fake S2 formal-connectivity evidence  
Authorization: user-authorized Step 4B extension limited to sealed transaction
-> grouped PPO -> exact-one update. No simulator, real training/live run,
checkpoint/resume, HSL change, or grouped-formula change.

TDD evidence:

- The new `frontres_v015_transaction_route_contract.py` was first run before
  implementation and failed at the required missing owner:
  `AttributeError: ... has no attribute FrontRESV015FormalTransactionPlan`.
  This established that the test required a new v015 transaction owner rather
  than silently exercising the historical S1b accumulator.
- The first regression run exposed two unrelated legacy static-test stub import
  failures. Root cause: the new v015 symbols were imported eagerly by modules
  that historical tests load with minimal legacy stubs. The fix made the q29
  route predicate local to the probe and deferred the new probe import until the
  explicit fake-S2 dispatcher is called. The same legacy contracts then passed.

Implementation:

- `frontres_segment_live_sampler.py::FrontRESV015FormalTransactionPlan` seals
  one frozen policy snapshot and every expected `(source_index, trial_index)`
  row. It requires at least two selected sources and two contiguous policy
  attempts per source, preserving motion/start/Segment/scenario/hash/`x_t`/K
  identity and deployment-q29 provenance.
- `FrontRESV015FormalTransactionAccumulator` accepts only grouped-candidate
  adapter shards, rejects duplicate/foreign/mixed/partial rows, observes an
  explicit optimizer counter during collection, canonicalizes the completed
  candidate batch, and leaves optimizer ownership downstream.
- `on_policy_runner.py` exposes an opt-in public fake-S2 connector; its dedicated
  update-loop requires an injected provider and verifies no optimizer step before
  or during provider collection. `frontres_segment_live_probe.py` then applies
  unchanged v003 grouped PPO once and requires exactly one counter increment.
- The v015 live observation route now chooses deployment-q29 context before its
  normalizer and does not concatenate the historical 65D fixed tape. The fake
  formal config rejects HSL/rollout labels, nonzero supervised loss, legacy
  warmup, non-grouped normalization, and legacy live-update dispatch flags.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_transaction_route_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_grouped_candidate_adapter_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_one_action_k_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_future_intent_actor_context_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py

Observed output:

- `[T-connect/T-order/T-exact-one-update/T-no-legacy-route/T-diagnostic]`
  reports a sealed `2 Segment x 2 attempt` transaction, grouped v003 reduction,
  and optimizer counter delta exactly one.
- `[T-partial/T-warmup-isolation/T-fail-closed]` proves incomplete transactions,
  HSL initialization, and legacy `scale_only` reject before a step.
- `[T-q29-route]` proves the v015 actor route excludes the legacy fixed tail and
  appends q29 before normalizer use.
- Existing grouped-candidate, HSL q29, one-action K, future-intent layout,
  legacy live-probe, legacy live-sampler, and stage-entrypoint contracts pass.

Facts established:

- The only Step 4B update owner is the fake-S2 probe function after a complete
  sealed plan/accumulator; neither `to_ppo_batch()` nor
  `run_frontres_segment_single_update()` is on that route.
- A `noisy_segment_hash` together with scenario/`x_t`/q29/K identity remains
  fixed across all attempts of each source. K/evidence-step count is metadata,
  not additional PPO loss mass.
- The public runner method is not called by `train.py`, `learn_frontres_segment_live`,
  or the legacy update loop. Its provider is intentionally absent outside the
  CPU fake test.

Open boundaries:

- This is not a generic formal runner, simulator/reset, checkpoint/resume,
  sampler-state, real training, evaluator, or live-runtime proof.
- A real actor/critic privileged-observation carrier, formal selected-scenario
  materialization, and ordinary train-entry dispatch remain separate work; they
  must not be inferred from the injected fake provider.
- Step 4C persistence must version future layout and reject partial transaction
  resume before any later live gate.

Next:

- Stop after Step 4B. Await explicit authorization for Step 4C only.

## E-FI-15: Step 4C Future-Intent Checkpoint Identity And Transaction Atomicity S3

Date: 2026-07-20  
Tier: CPU-only deterministic fake checkpoint/resume evidence  
Authorization: user-authorized Step 4C-S1 persistence/atomicity only. No
grouped-PPO formula or HSL change, generic formal dispatch, simulator, real
training, or live run.

Implementation:

- `frontres_checkpointing.py` is the v015 persistence owner. Its envelope
  records `FRS-METHOD-v015`, `FRS-TRAIN-v007`, `FRS-GAIN-v003`,
  `FRS-PPO-v003`, exact future-intent H offsets/layout, one-row grouped-loss
  identity, and a value-sensitive q29-prefix-normalizer fingerprint.
- Before sampler, actor, normalizer, optimizer, or iteration mutation, v015
  resume rejects a missing/unversioned identity, old `[H,65]` payload,
  same-width but different H offsets, incompatible normalizer mode/shape, or a
  tampered prefix statistic. A valid prefix restore records the exact layout
  version for the runtime normalizer. The full v015 envelope is validated before
  the old-HSL marker guard, so a valid Stage-3 resume may retain completed-HSL
  history without turning a legacy HSL payload into a legal input.
- `frontres_segment_live_update_loop.py` opens `collecting` before its injected
  provider. `frontres_segment_live_probe.py` binds the immutable plan, changes
  it to `sealed` after all expected attempts, and publishes a receipt only
  after the unique optimizer-step delta equals one.
- Checkpoint save rejects `collecting`, `sealed`, and `failed` states. Resume
  accepts only `idle` or a metadata-only committed receipt; it records the
  receipt as history and resets the runtime barrier to `idle`, without
  reconstructing a provider request, candidate batch, or raw local scenario.

TDD correction:

- The first `T-v015-hsl-history` run failed because `load_runner()` applied the
  legacy HSL marker guard before validating the new v015 envelope. That would
  reject a legal post-warmup Stage-3 checkpoint. The fixed order validates an
  active v015 envelope before the legacy guard; malformed/missing envelopes
  still fail before state mutation, and the legacy HSL reject regression passes.

Command evidence:

    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_update_loop.py source/rsl_rl/rsl_rl/tests/frontres_v015_checkpoint_resume_contract.py source/rsl_rl/rsl_rl/tests/frontres_v015_transaction_route_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_checkpoint_resume_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_transaction_route_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_future_intent_actor_context_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py
    /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py

Observed output:

- `[T-checkpoint/T-layout/T-commit-receipt]` proves the saved payload contains
  the exact q29 layout, prefix statistics identity, and only an exact-one
  committed transaction receipt.
- `[T-resume/T-legacy-reject/T-normalizer]` proves bad H/legacy/tampered
  payloads reject before a fake sampler or model is mutated, while a valid
  committed resume restores only normalizer layout identity and receipt history.
- `[T-v015-hsl-history]` proves `frontres_warmup_complete=True` is accepted
  only when the complete v015 Stage-3 envelope validates; the separate legacy
  HSL rejection contract still passes.
- `[T-atomicity]` proves an in-flight save produces no checkpoint and a sealed
  resume rejects. `[T-connect/T-order/T-exact-one-update/T-checkpoint-barrier]`
  proves the provider observes `collecting`, then the completed fake route
  publishes a committed receipt whose optimizer delta is one.

Facts established:

- No v015 checkpoint contains raw Clean continuation, q29 intent, root artifact,
  `x_t`, or candidate batch as transaction-restart state. The only persisted
  transaction result is a narrow immutable receipt.
- `H=(1,3)` and `H=(1,2)` remain distinct even when their q29 tail widths are
  equal, so resume cannot silently reinterpret future offsets.
- The old 65D and legacy HSL checkpoint boundaries remain reject-only; this step
  does not define a new Stage-1 HSL checkpoint format. A completed-HSL marker
  is permitted only as history inside an already validated v015 Stage-3 resume.

Open boundaries:

- Evidence is a semantically complete CPU fake persistence path, not proof of
  generic `train.py` / `learn_frontres_segment_live` dispatch, real checkpoint
  cadence, environment reset, simulator timing, real training, or live resume.
- The legacy formal runner and persistent sampler state remain separate owners;
  neither becomes active merely because the fake transaction can persist its
  receipt.

Next:

- Stop after Step 4C. Step 5A requires separate user authorization for one
  bounded local live identity sentinel; it must not start long training or
  deployment-composition evaluation.

## E-FI-16: Step 5A-S0 Pre-Live Formal Sentinel Connectivity S2

Date: 2026-07-20  
Tier: CPU-only deterministic pre-live connectivity evidence  
Authorization: explicit Step 5A-S0 only. No simulator, real training, live
transaction, checkpoint/resume change, HSL change, or grouped-PPO/Gain formula
change.

Implementation:

- `scripts/rsl_rl/train.py` exposes the opt-in
  `--frontres_v015_local_sentinel_only` entrypoint. It requires explicit ordered
  q29 H offsets, disables HSL/warmup and generic live modes, selects
  `grouped_scale_only`, and exits after the dedicated sentinel owner.
- `frontres_segment_live_sampler.py::prepare_frontres_v015_local_sentinel_batch`
  selects two distinct Segment sources, seals the complete M-attempt plan with
  one frozen old-policy snapshot, and materializes only the split local carrier
  `{x_t, artifact[7], intent[H+1,29], Clean C[K,65], hash}`. A legacy 65D tape
  is rejected; a Repair-row count different from the planned complete
  transaction fails closed.
- `frontres_segment_live_probe.py` routes the split carrier through two-role
  reset, q29-before-normalizer actor input, one action plus frozen GMT K capture,
  v003 candidate adapter, the sealed grouped transaction, and the existing
  exact-one-update owner. It records x_t/scenario/hash, artifact norm, q29
  provenance, Clean-C length, roles, action counters, K, group mass, and update
  delta before closing the immutable scenario lifecycle.

TDD and command evidence:

- The new config contract first failed before the v015 config/entrypoint owner
  existed; the connectivity contract then first failed at missing local reset
  attachment, local batch materialization, and sentinel request builder owners.
- The following CPU-only contracts passed:

      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_local_sentinel_config_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_local_sentinel_connectivity_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_one_action_k_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_future_intent_actor_context_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_runner_boundary_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
      /Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py

Facts established:

- The opt-in v015 route is mutually exclusive with legacy sentinel/probe/update
  loops and cannot use `to_ppo_batch()` or an HSL writer.
- The local reset request carries separated artifact, deployment-q29 intent, and
  GMT-only Clean continuation; it never carries a full fixed Noisy tape.
- The fake route proves provider collection occurs under the transaction barrier,
  preserves `2 Segment x 2 attempt` group mass, and causes exactly one explicit
  optimizer counter increment only after complete candidate sealing.

Open boundaries and stop condition:

- This does not prove a real environment reset, actor/GMT execution, actual
  selected M cardinality, checkpoint cadence, simulator timing, or a live
  optimizer update. The live preflight must reject an environment whose Repair
  rows do not equal the selected complete transaction size.
- Stop and report on any absent identity, legacy tape, Clean actor input, later
  FEMR action, mixed scenario/hash, partial transaction, or update delta other
  than one.

Next:

- Stop at Step 5A-S0. A single bounded actual local transaction requires new
  user confirmation after an exact command/preflight is reported.

## E-FI-17: Step 5A-S1 Runtime Parameter Preflight

Date: 2026-07-20  
Tier: read-only host/artifact/config preflight; no S4 live evidence  
Authorization: user requested Step 5A-S1 execution. The preflight may inspect
runtime and artifact identity, but the sole simulator transaction may start
only on a host satisfying every required boundary.

Observed host facts:

- `uname -srm` returned `Darwin 25.5.0 arm64`.
- `nvidia-smi` was absent. The configured FrontRES Python reported
  `isaaclab_spec=None`; PyTorch was present.
- `/hdd1/cyx/AMASS_G1Segment`, `/hdd1/cyx/AMASS_G1NPZ_Final`, and
  `/hdd1/cyx/FEMR/model/model_warmup.pt` do not exist on this host.
- The local FEMR and MOSAIC `model_27000.pt` files are byte-identical
  (`sha256=3efcdb50df81465a1d3cbd0edb71cc9662e1e69f65e8f2e067f845607660c426`)
  and contain only legacy model/optimizer/normalizer keys, with no v015
  checkpoint identity. They are not legal v015 Stage-3 resume envelopes.

Code-confirmed command boundary:

- The active config auto-selects `/hdd1/cyx/MOSAIC/model/model_27000.pt` as the
  frozen GMT checkpoint on SUST_Main_2. The v015 sentinel itself must cold-start
  a fresh FrontRES policy and omit `--resume_student_checkpoint`; the existing
  Stage-3 launcher is legacy because it requires an HSL checkpoint.
- A fresh sampler begins with UNKNOWN Segment state. The planner assigns one
  attempt, then the frozen transaction clamps it to minimum M=2. Two selected
  Segments therefore require four Repair rows plus four Noisy rows, so the
  bounded cold-start sentinel uses `--num_envs=8`.
- Fresh CPU contracts passed for the sampler transaction plan, v015 checkpoint
  rejection/identity boundary, and the dedicated sentinel config entrypoint.

Prepared SUST_Main_2 command (not executed on this host):

    cd /hdd1/cyx/FEMR
    python scripts/rsl_rl/train.py --task=FrontRES-Unified-Tracking-Flat-G1-v0 --num_envs=8 --motion /hdd1/cyx/AMASS_G1NPZ_Final --headless --logger tensorboard --experiment_name g1_flat_frontres_stage3_v015_sentinel --run_name V015_LOCAL_SENTINEL_ONCE --max_iterations 0 --frontres_stage stage3_segment_hrl --frontres_specialist_mode rp --frontres_segment_cache_dir /hdd1/cyx/AMASS_G1Segment --frontres_segment_shard_cache_size 8 --frontres_v015_local_sentinel_only --frontres_v015_future_offsets 1,2

Stop result:

- The live transaction was not started. Running on Darwin without IsaacLab,
  GPU, GMT path, cache, or motion data would not exercise the formal route.
- The next execution requires the current dirty v015 worktree to be synchronized
  to SUST_Main_2 without reverting user changes. Stop immediately if the server
  lacks any required path, selects duplicate Segments, plans a row count other
  than four Repair attempts, emits a later FEMR action, exposes Clean actor
  input, or reports optimizer-step delta other than one.

## E-FI-18: R0 Formal Observation Contract Freeze

Date: 2026-07-20
Tier: S0 read-only source/log audit and governance correction
Authorization: user requested R0 / 7. Source and logs may be read, and the
v015 plan/checklist/evidence/canvas may be corrected. Concept Figure, active
method semantics, training source, tests, simulator, and live execution are
out of scope.

Raw evidence:

- `v015_step5a_s1.log` reports the real policy observation as `870D`:
  `30D` anchor-error history, `70D` balance history, and a `770D`
  GMT-compatible suffix. Its command term is `290D = 58D x 5 history`.
- The same log reports the selected v015 layout as `H=(1,2)`, q29 tail `58D`,
  raw observation `870D`, and combined actor observation `928D`. The loaded
  frozen GMT model and normalizer both retain `770D` input.
- The residual actor printed `Linear(in_features=928, ...)`. Source confirms
  `rsl_rl_mosaic_cfg.py` sets `num_frontres_obs=0`; the runner adds the q29 tail
  to `num_actor_obs` but changes `num_frontres_obs` only when it was already
  positive. `FrontRESActorCritic` therefore gives FEMR the full `928D` tensor
  when the configured prefix is zero.
- The live route reached balanced `repair=4, noisy=4`, completed the sealed
  local-scenario reset, and then failed in environment observation construction:
  `MultiMotionCommand.command -> _gather_future_by_motion()` rejected the
  active local scenario before the explicit K executor opened. The failure
  occurred before q29 append, normalizer, actor action, K execution, storage,
  loss, or optimizer update.
- `apply_frontres_segment_index_reset()` expands the selected motion indices
  and start frames onto every Repair/Noisy role row before installing the local
  scenario. Thus the command owner already has role-aligned selected deployment
  motion/frame identities at `t`.
- The ordinary command owner constructs one current command from
  `env_motion_indices` and `time_steps`: q29 and dq29 are gathered from the
  selected motion carrier and flattened to `[B,58]` when
  `motion_horizon=1`. The policy observation manager's five-frame history turns
  this term into `[B,290]` inside the existing `870D` observation.
- The local scenario separately stores role-aligned `[B,H+1,29]` q29 intent,
  but `frontres_runtime.py::_future_intent_context_batch()` currently reads the
  policy-attempt batch. In the bounded transaction that source has `B=4`, while
  environment observation and command rows have `B=8`.
- `frontres_v015_local_sentinel_connectivity_contract.py` replaces
  `_read_live_observations()` with a function returning `object()`. Therefore
  `E-FI-16` did not execute the command property, 870D observation construction,
  q29 append, normalizer, or FEMR/GMT consumers.

Frozen observation contract:

| Boundary | Shape | Provenance / consumer |
| --- | --- | --- |
| Environment policy observation | `[B,870]` | current robot/balance/artifact prefix `100D` plus unchanged GMT suffix `770D` |
| Role-aligned local intent | `[B,H+1,29]`, `H=2` | sealed deployment/Noisy q29 carrier for all Repair/Noisy rows |
| Positive-offset actor tail | `[B,58]` | q29 offsets `(1,2)` only; no root/global/Clean C |
| Combined observation | `[B,928]` | `[q29 tail 58, current prefix 100, GMT suffix 770]` |
| FEMR visible input | `[B,158]` | first `58+100`; current artifact plus deployable future q29 intent |
| Frozen GMT input | `[B,770]` | original final suffix and original checkpoint/normalizer identity |
| Current GMT command at `t` | `[B,58]` before history | current deployment-carrier `q29_t+dq29_t`, selected by role-aligned motion/frame identity |
| GMT continuation after action | `[B,K,65]` | Clean C, inaccessible until the explicit frozen-GMT K executor opens |

Decisions:

- This is an integration/evidence defect, not a v015 method change. The Concept
  Figure and active contracts remain unchanged.
- Current GMT q29/dq29 at `t` belongs to the selected deployment reference
  carrier. Numerical equality with Clean calibration does not change its
  provenance. Clean C remains `t+1:t+K` and cannot service the pre-action
  command.
- Step 1B is reclassified as partial: `E-FI-3` proves the isolated q29-tail
  builder, not role-aligned formal consumption or the FEMR `158D` boundary.
- Step 5A-S0 is reclassified as partial: `E-FI-16` proves config/transaction
  isolation and exact-one fake update, not formal observation connectivity.
- No further simulator/live run is allowed before R1--R5 complete, including
  an unmocked offline formal observation contract.

R1 Step Contract:

- Owner: `commands.py::MultiMotionCommand.command` and its local
  `_gather_future_by_motion()` branch.
- Scope: for an active local scenario before K execution, accept only
  `motion_horizon=1` and return current deployment q29/dq29 as `[B,58]` from
  the already installed role-aligned motion/frame identities.
- Non-scope: q29 H-tail, actor visibility, normalizer, K continuation, Gain,
  PPO, checkpoint, formal runner, simulator, training, or live run.
- Tests: S1 T-current-command/T-shape/T-provenance/T-role-identity/
  T-current-only/T-continuation-isolation/T-legacy-reject.
- Stop: any access to Clean C, future root/global, Clean actor data, horizon
  greater than one, mixed same-scenario role values, or weakened K gate.

Next:

- Stop after R0. R1 requires separate user authorization.

## E-FI-19: R1 Current-Frame GMT Command Route

Date: 2026-07-20
Tier: S1 deterministic command-owner evidence; no S2 formal observation or S4
live claim

Scope:

- Implement only the `MultiMotionCommand` pre-action current-command branch and
  deterministic S1 tests.
- Keep q29 H append, actor visibility, normalizer, checkpoint, K semantics,
  Gain, PPO, formal runner, simulator, training, and live execution unchanged.

RED evidence:

- The new `frontres_v015_current_gmt_command_contract.py` first failed at
  `MultiMotionCommand.command -> _gather_future_by_motion()` with the preserved
  runtime error: `v015 local scenario has no generic future command route...`.
  This reproduced the R0 command-owner defect before production code changed.
- The first GREEN attempt then stopped at the new q29 identity oracle because
  the inherited reset fixture used arbitrary `I[t]` values unrelated to its
  declared motion/frame. The fixture was corrected to use the same deployment
  motion/frame q29; the production assertion was retained.

Implementation facts:

- `commands.py::MultiMotionCommand._frontres_local_scenario_current_command_rows()`
  is the sole new behavior owner.
- It accepts only a transaction-wide active/current-frame-ready local scenario,
  K execution inactive, getter `joint_pos` or `joint_vel`, and
  `motion_horizon=1`.
- It gathers current q29/dq29 from the role-aligned `env_motion_indices` and
  `time_steps`, requires finite `[B,29]` rows, and asserts gathered q29 exactly
  matches sealed deployment `I[t]`.
- `MultiMotionCommand.command` requires `command_velocity=True` for this
  pre-action local route and returns `[B,58] = q29_t + dq29_t`.
- The existing K-active branch remains first and still reads Clean C through
  `_frontres_local_scenario_continuation_rows()` only after the explicit K
  executor opens. Mixed local/legacy rows remain rejected.

Verification:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_current_gmt_command_contract.py`
  exited 0 and printed:
  `T-current-command`, `T-shape`, `T-provenance`, `T-role-identity`,
  `T-current-only`, `T-continuation-isolation`, and `T-legacy-reject`.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_two_role_reset_contract.py`
  exited 0. Immutable Repair/Noisy reset and role balance remain valid.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_one_action_k_contract.py`
  exited 0. One action, later-FEMR freeze, Clean-C K execution, and one policy
  row remain valid.
- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py`
  exited 0. Legacy reference-window/fixed-tape command behavior remains valid.
- `python -m py_compile` over `commands.py` and the two changed/new R1 test
  files exited 0. `git diff --check` also exited 0.

Confirmed:

- For the semantic `B=8` fixture, current command shape is `[8,58]` and every
  Repair row equals its paired Noisy row from the same selected motion/frame.
- `motion_horizon>1` and `command_velocity=False` reject before exposing a
  future or q-only command.
- Poisoned/nonmatching Clean C and future q29 values do not service the
  pre-action command; once K opens, the unchanged command branch consumes C.
- R1 stop conditions were not triggered.

Unconfirmed:

- IsaacLab five-frame observation history, role-aligned future q29 H append,
  `928 -> 158/770` actor/GMT visibility, normalizer consumption, formal
  connectivity, simulator timing, and live behavior remain R2--R6 evidence.

Next:

- Stop after R1. R2 requires separate user authorization.

## E-FI-20: R2 Role-Aligned q29 H Bridge

Date: 2026-07-20
Tier: S1 deterministic command-snapshot/runtime-bridge evidence; no normalizer,
formal observation, simulator, or live claim

Scope:

- Add one read-only `MultiMotionCommand` accessor for the sealed role-aligned
  q29 intent and identity/provenance metadata.
- Make `frontres_runtime.py::append_frontres_future_intent_context()` consume
  that command snapshot rather than the policy-attempt batch.
- Keep actor visibility, normalizer behavior, checkpoint, K, Gain, PPO, formal
  runner, simulator, training, and live execution unchanged.

RED evidence:

- `frontres_v015_role_aligned_future_intent_contract.py` first failed with
  `AttributeError`: `MultiMotionCommand` had no
  `frontres_local_scenario_intent_snapshot()` accessor.
- The test fixture deliberately supplied a poisoned B=4 policy-attempt intent
  beside the real B=8 command carrier, so a batch fallback could not satisfy
  the contract.

Implementation facts:

- `commands.py::frontres_local_scenario_intent_snapshot()` requires one
  transaction-wide active/current-frame-ready scenario with K inactive.
- It returns cloned `intent_q29` plus scenario/hash/x_t/role/provenance metadata
  only. Current root artifact and Clean continuation are not in its schema;
  caller mutation cannot alter command-owned data.
- `frontres_runtime.py::_future_intent_context_snapshot()` resolves the motion
  command and requires that exact snapshot schema.
- `append_frontres_future_intent_context()` now selects offsets from the frozen
  `FrontRESFutureIntentLayout`, reads intent/provenance only from the command
  snapshot, and never reads the policy-attempt batch for actor H values.
- The existing layout builder still validates detached finite
  `[B,H_max+1,29]`, deployment/Noisy provenance, ordered positive offsets, and
  excludes root/global/Clean sources.

Verification:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_role_aligned_future_intent_contract.py`
  exited 0 and printed T-role-expand/T-offset/T-policy-batch-isolation,
  T-read-only/T-no-root/T-no-Clean/T-no-C, and
  T-permute/T-scenario-identity/T-provenance.
- `frontres_future_intent_actor_context_contract.py` exited 0 after its weak
  batch fixture was upgraded to expose the same data through a command
  snapshot; shape/offset/provenance/clean-isolation/legacy rejection remain
  valid.
- `frontres_hsl_v007_s1_contract.py` and
  `frontres_hsl_v007_s2_connectivity_contract.py` exited 0 after their fixtures
  were similarly rebased; no production HSL behavior changed.
- `frontres_v015_current_gmt_command_contract.py`,
  `frontres_v015_two_role_reset_contract.py`, and
  `frontres_v015_one_action_k_contract.py` exited 0. R1 current command,
  immutable reset, one action, and K-only Clean C remain valid.
- `python -m py_compile` over the two owner files and affected R2 tests exited
  0.

Confirmed:

- The semantic fixture carries `B=8`, `H+1=3`, q29=29; offsets `(1,2)` produce
  `[8,58]` in stable order.
- Four Repair rows equal their corresponding four Noisy rows; an arbitrary row
  permutation produces the same permutation of tail and identity.
- A poisoned B=4 policy-attempt batch is ignored for actor H values.
- Snapshot mutation, current-root mutation, Clean-C mutation, and an injected
  future-root/global field cannot change the H tail.
- Clean-labelled provenance rejects, and R2 stop conditions were not triggered.

Unconfirmed:

- `num_frontres_obs=0` still permits FEMR to consume the full `928D` tensor.
  Exact FEMR `158D` / frozen-GMT `770D` authority is R3.
- Prefix/suffix normalizer persistence, unmocked formal observation
  connectivity, simulator timing, and live execution remain R4--R6.

Next:

- Stop after R2. R3 requires separate user authorization.

## E-FI-21: R3 FEMR 158D / GMT 770D Authority Split

Date: 2026-07-20
Tier: S1 deterministic observation-layout/consumer-isolation evidence; no
checkpoint persistence, formal connection, simulator, training, or live claim

Scope:

- Change only the v015 observation config, runner layout resolution, and
  `FrontRESActorCritic` visibility boundary.
- Keep command provenance, q29 H values, normalizer persistence, checkpoint,
  K, Gain, PPO, formal runner, simulator, training, and live execution
  unchanged.

RED evidence:

- The new `frontres_v015_observation_authority_contract.py` first failed to
  import `resolve_frontres_v015_observation_authority`, proving that the
  `870+58 -> 928`, `58+100 -> 158`, and zero-prefix rejection contract had no
  owner before the implementation.

Implementation facts:

- `rsl_rl_mosaic_cfg.py` now declares the current FrontRES prefix as `100D`;
  it no longer permits the legacy zero-prefix/full-observation fallback.
- `frontres_observation_layout.py` owns an immutable v015 authority resolver.
  It requires `870 = 100 + 770`, resolves combined `928 = 870 + 58`, and
  resolves FEMR-visible `158 = 100 + 58`.
- `on_policy_runner.py` uses that resolver only on the requested v015 Segment
  Replay route, passes `num_actor_obs=928`, and passes
  `num_frontres_obs=158` to the policy.
- `FrontRESActorCritic` records the GMT checkpoint/normalizer policy dimension
  and rejects task-space construction or parsing unless
  `num_frontres_obs + gmt_policy_obs_dim == num_actor_obs`; zero is rejected.
- `_parse_observations()` caches the full tensor for GMT but returns only the
  prefix to FEMR. `_run_gmt_direct()` continues to slice the original final
  `770D` suffix using the frozen GMT normalizer dimension.

Verification:

- `frontres_v015_observation_authority_contract.py` exited 0 and printed
  T-928-layout/T-158-actor, T-num-frontres-zero-reject,
  T-770-GMT/T-frozen-GMT-isolation, and T-config/T-runner-resolution.
- `frontres_observation_layout_contract.py` exited 0; the existing
  `870 = 100 + 770` split and frozen GMT suffix normalization remain valid.
- `frontres_future_intent_actor_context_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`,
  `frontres_hsl_v007_s1_contract.py`, and
  `frontres_hsl_v007_s2_connectivity_contract.py` exited 0; q29 provenance,
  role alignment, Stage-1 routing, and Stage-3 HSL isolation remain valid.
- `frontres_segment_stage3_entrypoint_pseudo_contract.py` exited 0. This is a
  deterministic config/entrypoint regression only and did not run training.
- `python -m py_compile` over all R3 source/test owners exited 0.
  `git diff --check` exited 0.

Confirmed:

- The full policy observation remains `[B,928] = [58D q29 tail, 100D current
  FrontRES prefix, 770D GMT suffix]`.
- FEMR receives exactly the first `[B,158]`; a poisoned GMT-only suffix cannot
  appear in the actor slice.
- Frozen GMT receives exactly the original final `[B,770]`; no checkpoint,
  GMT input layer, or GMT normalizer reshaping was introduced.
- Both runner resolution and task-space actor parsing reject
  `num_frontres_obs=0`. R3 stop conditions were not triggered.

Unconfirmed:

- Prefix/suffix normalizer persistence and checkpoint/resume identity remain
  R4 S3 evidence.
- Unmocked formal observation connectivity, simulator timing, training, and
  live behavior remain R5--R6.

Next:

- Stop after R3. R4 requires separate user authorization.

## E-FI-22: R4 Exact Layout Persistence Revalidation

Date: 2026-07-20
Tier: S3 deterministic CPU checkpoint/resume identity and atomicity evidence;
no actual checkpoint cadence/resume, formal connection, simulator, training,
or live claim

Scope:

- Rebind the existing v015 Stage-3 persistence owner to the R3 observation
  authority: `H=(1,2)`, raw `870D`, q29 tail `58D`, combined `928D`, FEMR
  prefix `158D`, frozen-GMT suffix `770D`.
- Preserve save/load ownership, grouped-loss identity, metadata-only committed
  receipt, and pre-mutation transaction atomicity.
- Keep runtime normalizer math, optimizer/grouped PPO, formal runner,
  simulator, training, actual checkpoint cadence/resume, and live execution
  unchanged.

Observed pre-fix fact:

- The existing `frontres_v015_checkpoint_resume_contract.py` exited 0 while
  printing `Loaded FrontRES prefix normalizer stats (dims 0-61)`. Its fixture
  used offsets `(1,3)`, a `3D` current prefix, and a `7D` GMT suffix. This
  proved generic save/load mechanics but could not prove the R3
  `928 -> 158/770` identity.
- After rebasing the fixture, the first RED run stopped at
  `identity["format"] == "frontres-v015-checkpoint-v2"` because production
  still emitted `frontres-v015-checkpoint-v1`.

Implementation facts:

- `frontres_checkpointing.py::_v015_checkpoint_layout_fields()` remains the
  unique layout identity owner and now reuses the R3 authority resolver.
- The v2 identity requires offsets `(1,2)` and records raw `870D`, current
  FrontRES prefix `100D`, q29 tail `58D`, actor `928D`, FEMR prefix `158D`,
  and GMT suffix `770D`.
- The normalizer payload remains `[158D prefix stats | 770D frozen-GMT
  suffix] = 928D`. SHA-256 covers both mean and std of the complete `158D`
  prefix; tampering index 157 rejects.
- `frontres-v015-checkpoint-v1`, absent/unversioned identity, legacy `[H,65]`,
  `num_frontres_obs=0`, full-`928D` FEMR visibility, mismatched H, and partial
  transaction states fail before actor/sampler/normalizer/optimizer restore.
- Committed transactions still resume only as a metadata receipt and return
  runtime transaction state to idle; no scenario reference or candidate batch
  is serialized.

Multi-layer semantic audit:

| Layer | Files / symbols checked | Aliases checked | Evidence | Gap |
| --- | --- | --- | --- | --- |
| Architecture / contract | registry, Method v015, Training v007, R4 plan/checklist, Method-to-Code atlas | checkpoint identity, HSL checkpoint, persistence | note-confirmed | Concept Figure unchanged because method semantics did not change |
| Symbol | `save_runner`, `load_runner`, `_v015_checkpoint_layout_fields`, `_build_v015_checkpoint_identity`, `_validate_v015_checkpoint_resume` | actor/prefix/GMT dims, transaction receipt | code-confirmed | generic cadence/dispatch unconfirmed |
| Lexical | checkpoint, resume, `obs_norm_state_dict`, prefix mean/std/fingerprint, normalizer | `_frontres_extra_*`, layout version, `num_frontres_obs` | code-confirmed | policy-quality evaluator is outside the R4 resume owner |
| Dataflow | runner fields -> combined normalizer -> identity -> pre-mutation validation -> prefix restore | `870/58/928/158/770` | contract-confirmed | no real GMT checkpoint artifact loaded |
| Lifecycle | idle/committed save-resume; collecting/sealed reject | sampler, actor, normalizer, optimizer mutation | contract-confirmed | actual periodic/final checkpoint trigger untested |
| Semantic stress | H mismatch, v1, 65D, unversioned, zero/full prefix, tampered last prefix stat, partial transaction | legacy/mixed layouts | contract-confirmed | simulator/live timing untested |

Verification:

- `/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_checkpoint_resume_contract.py`
  exited 0. It printed T-checkpoint/T-layout/T-prefix-stats/
  T-commit-receipt, T-v015-hsl-history, T-resume/T-legacy-reject/
  T-prefix-stats, T-legacy-zero-reject, and T-atomicity.
- `frontres_v015_observation_authority_contract.py` exited 0 and reconfirmed
  `870+58=928`, `58+100=158`, GMT `770D`, and zero-prefix rejection.
- `frontres_v015_transaction_route_contract.py` exited 0 and reconfirmed the
  sealed `2 x 2` exact-one receipt and checkpoint barrier.
- `frontres_hsl_v007_s1_contract.py` exited 0; old HSL checkpoints and Stage-3
  HSL writes/loss remain rejected.
- `python -m py_compile` over the R4 owner and S3 contract exited 0.
  `git diff --check` exited 0.

Confirmed:

- The only accepted v015 Stage-3 persistence envelope is v2 with exact
  `(1,2)` / `928 -> 158/770` identity.
- The full `158D` prefix mean/std fingerprint round-trips and restores before
  runtime use; the frozen GMT normalizer object is not overwritten.
- All R4 legacy/mixed-layout and transaction-atomicity stop conditions reject
  before mutable restore. R4 stop conditions were not triggered.

Unconfirmed:

- Actual periodic/final checkpoint cadence, a real v2 checkpoint artifact,
  generic formal dispatch, and live resume remain outside this S3 claim.
- `frontres_policy_quality_eval.py` reads checkpoint normalizer state through
  a separate evaluation route; R4 did not certify that consumer.
- Unmocked formal observation connectivity, simulator timing, training, and
  live behavior remain R5--R6.

Next:

- Stop after R4. R5 requires separate user authorization.

## E-FI-23: R5 Unmocked Offline Formal Observation Connectivity

Date: 2026-07-20
Tier: S2 deterministic semantic-CPU connectivity evidence; no simulator,
training, live run, checkpoint mutation, grouped-PPO formula change, or HSL
change

Scope:

- Exercise the production `_read_live_observations()` owner without replacing
  or monkeypatching it.
- Connect current deployment command -> semantic `870D` observation ->
  command-owned `58D` q29 future-intent append -> normalization -> FEMR
  `158D` prefix / frozen-GMT `770D` suffix -> one-action K evidence -> sealed
  grouped transaction -> exactly one optimizer step.
- Use two Segment scenarios, two policy attempts per scenario, eight physical
  role rows (`4 Repair + 4 Noisy`), and four PPO policy rows.

Observed RED facts:

- The earlier local-sentinel fixture replaced `_read_live_observations` with a
  prepared object. It proved transaction plumbing but not the formal
  observation connector required by R5.
- The first unmocked route failed because K execution reopened the actor-only
  q29 snapshot after the command-owned Clean-C executor had opened.
- The next run carried the correct immutable identities
  (`scenario-a, scenario-a, scenario-b, scenario-b` for both roles), but
  storage rejected them by assuming one Repair and one Noisy row per scenario.
  That assumption contradicted M attempts sharing one sealed scenario.
- After those fixes, every compared GMT command-history element was stale
  (`464/464` mismatched): the runner read the K observation before advancing
  the command cursor to the requested Clean-C offset.

Implementation facts:

- `frontres_segment_live_probe.py` freezes the already normalized FEMR `158D`
  prefix at action time. During K it reads and normalizes only a fresh raw
  `770D` GMT suffix, so it does not reopen the actor H snapshot. The exact
  formal v015 route fails closed if the `158/770` authority is unavailable.
- `FrontRESV015OneActionKEvidence.validate()` now accepts M attempts per sealed
  scenario while requiring equal nonempty Repair/Noisy role counts and exact
  agreement of scenario, x_t, artifact/intent/continuation identities, hash,
  and K for every row.
- `pair_frontres_v015_gain_facts()` pairs Repair and Noisy rows by stable
  attempt order inside each scenario. The v003 Gain formula and grouped mass
  are unchanged.
- `prepare_frontres_v015_frozen_gmt_step()` advances the Clean-C command cursor
  before invoking a supplied GMT-observation provider. The provider reuses the
  production environment observation owner and split normalizer; it does not
  add an environment step, later FEMR action, or PPO row.

Connectivity trace:

| Boundary | Deterministic S2 fact |
| --- | --- |
| Current command | `[B,58] = q29_t + dq29_t`, deployment provenance |
| Command history in raw observation | `5 x 58 = 290D` inside semantic `870D` |
| Future intent append | command-owned offsets `(1,2)` -> `58D` |
| Combined observation | `870 + 58 = 928D` |
| FEMR authority | first `158D = 58D q29 tail + 100D current prefix` |
| GMT authority | final original `770D` suffix only |
| K lifecycle | one FEMR action at t; every K observation follows the current Clean-C cursor |
| PPO transaction | four attempts/four rows; sealed grouped update delta exactly one |

Verification:

- `frontres_v015_unmocked_observation_connectivity_contract.py` exited 0 and
  printed T-command-connect, T-history-layout, T-role-tail, T-normalizer,
  T-consumer, T-one-action, T-clean-C-order, and T-exact-one-update.
- `frontres_v015_one_action_k_contract.py` exited 0 and reconfirmed one actor
  action, no later actor action, and K evidence without extra policy rows.
- `frontres_v015_gain_consumer_contract.py` and
  `frontres_v015_grouped_candidate_adapter_contract.py` exited 0; v003 Gain,
  sign-preserving scale, and grouped mass remain unchanged.
- `frontres_v015_transaction_route_contract.py` exited 0 and reconfirmed the
  sealed `2 x 2` exact-one update boundary and legacy/HSL rejection.
- `frontres_v015_local_sentinel_connectivity_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`, and
  `frontres_v015_observation_authority_contract.py` exited 0; local scenario,
  role-aligned q29, and `928 -> 158/770` contracts remain valid.

Confirmed:

- R5 reaches the actual observation-reading function using a semantic CPU
  environment fixture; `_read_live_observations()` is not replaced.
- Clean x_t remains dynamics reset only. Actor context is current artifact plus
  deployment/Noisy q29 intent; Clean continuation is consumed only by frozen
  GMT after the single FEMR action.
- The sealed `2 Segment x 2 attempt` transaction preserves one PPO row per
  attempt and performs exactly one optimizer step after all attempts arrive.
- No checkpoint, grouped-PPO formula, HSL, simulator, training, or live path
  was modified or executed.

Unconfirmed:

- Isaac/real-environment timing, actual frozen GMT checkpoint behavior,
  simulator-side command refresh, live telemetry, and physical policy quality
  remain outside R5.

Next:

- Stop after R5. R6 bounded live identity sentinel requires separate user
  authorization.

## E-FI-24: R6-S0 Structured Live Snapshot And Remote Preflight

Date: 2026-07-21
Tier: deterministic telemetry contract plus read-only SUST_Main_2 preflight;
no source transfer, simulator launch, optimizer update, training, or S4 live
transaction

Scope:

- Make the already-authorized bounded sentinel emit one fail-closed structured
  snapshot containing the full observation authority and transaction identity.
- Verify the named SUST_Main_2 host, repository state, required assets, and
  runtime interpreter before paying for the single live transaction.
- Keep checkpoint, grouped-PPO math, HSL, environment semantics, and training
  behavior unchanged.

Observed gap and fix:

- Before R6-S0, the sentinel retained scenario/hash/x_t/K/update diagnostics
  but did not persist `870/58/928/158/770`, q29 provenance, Clean-C length, or
  optimizer identity as one structured log record. A successful run therefore
  could not satisfy the R6 stop condition.
- `_read_live_observations()` now records actual raw, appended, normalized,
  FEMR-visible, and GMT-suffix dimensions. The one-action collector records the
  live-only `58D` current command, while each post-advance K read records the
  actual `770D` GMT input and read count.
- `_build_frontres_v015_local_identity_sentinel_request()` rejects before the
  grouped update unless the exact `8/58/870/58/928/928/158/770/770` trace and at
  least one post-advance GMT read are present.
- `run_frontres_v015_local_identity_sentinel()` rejects a missing pre-update
  snapshot and prints `[FrontRES v015 Live Snapshot]` as sorted JSON after the
  exact-one update receipt.

Deterministic verification:

- `frontres_v015_unmocked_observation_connectivity_contract.py` exited 0 using
  the production `_read_live_observations()` and reconfirmed one action,
  post-advance Clean C, four attempts, and update delta one while checking the
  observation trace.
- `frontres_v015_local_sentinel_connectivity_contract.py` exited 0 and printed
  a structured snapshot with q29 provenance, Clean-C lengths, roles,
  scenario/hash/x_t, grouped masses, `later_femr_action_count=0`, and
  `optimizer_step_delta=1`.
- `frontres_v015_one_action_k_contract.py`, Python compilation, and
  `git diff --check` exited 0.

Read-only remote facts:

- SSH reached `SUST_Main_2` at `172.18.36.110`; repository HEAD is
  `2451409d8ce6682ceb77ca039e46d1b9d6f990ce` on `main`.
- The remote tracked worktree is clean. Its merge relative to local baseline
  `f8e14a4` changes only historical run artifacts plus `run_eval.sh` and
  `run_stage3.sh`; it does not change an R6 owner file.
- `/hdd1/cyx/MOSAIC/model/model_27000.pt`,
  `/hdd1/cyx/AMASS_G1Segment`, and `/hdd1/cyx/AMASS_G1NPZ_Final` exist.
- `/hdd1/cyx/miniconda3/envs/mosaic/bin/python` reports PyTorch
  `2.5.1+cu121`, CUDA available, and IsaacLab importable. The base interpreter
  lacks PyTorch and is not a valid R6 command owner.
- All ten local R1--R6 production-owner checksums differ from the remote clean
  checkout because the accepted changes remain in the local dirty worktree.

Blocked transfer fact:

- A scoped `rsync -avR` of the ten owner files and two deterministic contracts
  was rejected by the execution security policy before transfer. The policy
  requires explicit informed user authorization because this copies private
  workspace code to a remote host not yet declared trusted.
- No remote file changed and no simulator/live transaction started. Running
  the stale remote owners would violate the R6 contract, so no fallback run is
  permitted.

Prepared command after synchronization:

    cd /hdd1/cyx/FEMR
    /hdd1/cyx/miniconda3/envs/mosaic/bin/python scripts/rsl_rl/train.py --task=FrontRES-Unified-Tracking-Flat-G1-v0 --num_envs=8 --motion /hdd1/cyx/AMASS_G1NPZ_Final --headless --logger tensorboard --experiment_name g1_flat_frontres_stage3_v015_sentinel --run_name V015_R6_LIVE_SENTINEL_ONCE --max_iterations 0 --frontres_stage stage3_segment_hrl --frontres_specialist_mode rp --frontres_segment_cache_dir /hdd1/cyx/AMASS_G1Segment --frontres_segment_shard_cache_size 8 --frontres_v015_local_sentinel_only --frontres_v015_future_offsets 1,2

Confirmed:

- R6 now has a reviewable one-record telemetry surface and an exact runtime
  interpreter/asset boundary.
- No method, Concept Figure, checkpoint, grouped-PPO, HSL, or live behavior was
  changed or claimed.

Unconfirmed / blocker:

- S4 simulator timing and the single live transaction remain unconfirmed.
- The next action requires explicit informed authorization to transfer the
  named local owner files to trusted host SUST_Main_2, or an equivalent manual
  synchronization by the user.

## E-FI-25: R6-F1 Local-vs-Legacy Command-Clock Isolation

Date: 2026-07-21
Tier: one failed S4 live boundary plus deterministic S1 lifecycle regression;
no second simulator run, training, checkpoint, grouped-PPO, HSL, or method
change

Raw live evidence:

- `v015_r6_live_sentinel.log` reached the exact v015 layout
  `870 + 58 -> 928 -> 158/770`, selected two Segments with M=2, installed four
  Repair plus four Noisy rows, preserved paired scenario hashes, and completed
  reset with `success_frac=1.0`.
- The unique actor action was prepared, then the first `runner.env.step()`
  entered IsaacLab `command_manager.compute -> MultiMotionCommand._update_command`.
- `_update_command()` unconditionally executed the legacy clock
  `time_steps += 1 -> reference/tape advance -> current-cache refresh`.
  `refresh_frontres_reference_cache_current_frame()` correctly rejected the
  duplicate local-cache installation before the t transition returned.
- K capture, Gain, storage, grouped PPO, and optimizer update were not reached.
  CUDA/IOMMU/GLFW/shader-cache warnings were not the Python failure owner.

Root cause:

- The formal environment had two active reference clocks: IsaacLab's automatic
  legacy command clock and Step 2B's explicit sealed current/C cursor. R5's
  semantic CPU fixture did not execute the real command-manager callback, so
  this simulator lifecycle boundary remained unmodeled.
- The failing guard was the detector, not the writer. Removing it would allow
  silent reference drift; the invalid operation was legacy clock advancement
  while a transaction-wide local scenario owned the reference.

Implementation:

- `MultiMotionCommand._advance_frontres_command_clock()` is now the single
  per-command-step dispatcher.
- If a local scenario is active, it requires all rows active and current-ready,
  rejects mixed current/K execution, increments only the global simulator step,
  and holds either the sealed current reference or the explicitly installed
  Clean-C offset.
- If no local scenario is active, it preserves the original ordered legacy
  path: `time_steps += 1`, reference-window advance, fixed-tape advance, then
  one cache refresh.
- `_update_command()` calls this owner exactly once. The direct duplicate local
  refresh guard remains unchanged and fail-closed.

RED/GREEN and regression evidence:

- The new lifecycle test first failed with
  `AttributeError: MultiMotionCommand has no attribute _advance_frontres_command_clock`,
  proving the live command-clock boundary had no owner.
- `frontres_v015_current_gmt_command_contract.py` then exited 0 and printed
  T-t-clock-hold, T-K-clock-hold, T-legacy-clock, and
  T-duplicate-refresh-reject.
- `frontres_v015_two_role_reset_contract.py`,
  `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_unmocked_observation_connectivity_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`,
  `frontres_segment_motion_command_reference_contract.py`, and
  `frontres_v015_local_sentinel_connectivity_contract.py` exited 0.
- The Stage-1 AST contract initially failed because it still required legacy
  advancement directly inside `_update_command()`. After rebasing ownership to
  `_advance_frontres_command_clock()`,
  `frontres_segment_stage1_env_hooks_contract.py` exited 0 and reconfirmed one
  legacy refresh, perturbation draw, and pair sync.
- Python compilation, both Architecture JSON parses, and `git diff --check`
  are required in the final R6-F1 verification gate.

Confirmed:

- Local t and K command computes no longer advance `time_steps`, mutate the
  current artifact, or move the Clean-C cursor.
- Legacy command rows retain their original clock behavior.
- This repair does not suppress the duplicate-install detector and does not
  change Clean x_t, q29 H, K, Gain, PPO, checkpoint, or HSL semantics.

Unconfirmed / next:

- The repaired path has not been rerun under IsaacLab. Synchronize the updated
  `commands.py` and current-command contract to SUST_Main_2 before the one
  remaining R6 live sentinel attempt.

## E-FI-26: R6-F2 Sealed Critic-Observation Route

Date: 2026-07-21
Tier: one failed S4 loss boundary plus deterministic S1/S2 carrier,
row-order, fail-closed, and exact-one-update evidence; no additional simulator,
training, checkpoint mutation, grouped-PPO formula, HSL, or method change

Raw live evidence:

- `v015_r6_live_sentinel_gpu3.log` confirms R6-F1 succeeded: reset, the unique
  t action, Clean-C K execution, Gain/candidate sealing, and grouped PPO entry
  were reached for four policy attempts.
- The loss-side trace then called the frozen critic with the actor observation:
  `mat1 and mat2 shapes cannot be multiplied (4x928 and 289x1024)`.
- No optimizer step occurred. CUDA-visible-device, IOMMU, GLFW, and shutdown
  warnings were not the Python failure owner.

Root cause:

- `FrontRESUnified.act()` correctly stored both t tensors:
  `transition.observations [8,928]` and
  `transition.privileged_observations [8,289]`.
- The v015 one-action carrier preserved only actor observations. Candidate
  storage and the formal transaction therefore lost the Repair critic rows.
- `FrontRESSegmentLivePolicyAdapter` received `None` for privileged
  observations and fell back to actor observations. The critic layer detected
  the mismatch; it was not the writer.

Implementation:

- `FrontRESV015OneActionKEvidence` now seals the Repair-role t critic rows as
  detached `[B,C]` data alongside the actor tuple.
- `build_frontres_v015_grouped_candidate_storage()` and the candidate PPO
  schema carry the critic rows. `FrontRESV015FormalTransactionAccumulator`
  concatenates and reorders them by the same `(source_index, trial_index)`
  order as every actor/loss row.
- The v015 formal evaluator consumes the sealed critic tensor and rejects a
  missing, empty, row-misaligned, or conflicting request tensor. It no longer
  falls back to actor observations. Generic legacy adapter behavior and the
  grouped PPO formula are unchanged.
- R6 telemetry now records `critic_observation_dim=289` and prints both actor
  and critic evaluator shapes.

RED/GREEN and regression evidence:

- The strengthened unmocked S2 contract first reproduced the live failure as
  `mat1 4x928 and 289x1` using a real `Linear(289,1)` critic.
- After routing the sealed carrier, the same contract exited 0 and printed
  `actor_obs_shape=(4, 928) critic_obs_shape=(4, 289)` plus
  `step_delta=1`.
- The contract also checks exact Repair-row values and rejects missing or
  three-row critic tensors before loss.
- `frontres_v015_grouped_candidate_adapter_contract.py` confirms candidate
  critic equality; `frontres_v015_transaction_route_contract.py` confirms
  multi-shard critic order `[source0/attempt0, source0/attempt1,
  source1/attempt0, source1/attempt1]` and one update.
- One-action K, Gain consumer, local sentinel connector, checkpoint/resume,
  grouped PPO, legacy sampler/storage, local sentinel config, Python compile,
  JSON parse, and `git diff --check` regression gates pass.

Confirmed:

- The deterministic formal path now keeps actor and critic authorities
  separate: actor `[4,928]`, critic `[4,289]`, one policy row per attempt, and
  exactly one optimizer update after the sealed transaction.
- R6-F2 does not change Clean x_t, q29 H, Clean-C K, Gain, grouped reduction,
  checkpoint identity, HSL, or Concept Figure semantics.

Unconfirmed / next:

- The repaired critic carrier has not yet been rerun in IsaacLab. Synchronize
  the R6-F2 owner files to SUST_Main_2, then run the single authorized bounded
  transaction again.
- `frontres_segment_live_probe_ppo_contract.py` has a pre-existing import-stub
  drift for `_append_future_intent_actor_context`; it fails before reaching
  this critic route and is not evidence against R6-F2.

## E-FI-27: R6 / Step 5A Successful Bounded Live Identity Sentinel

Date: 2026-07-21
Tier: S4 bounded live runtime identity, authority, K-lifecycle, grouped-mass,
and exact-one-update evidence; no long training, policy-quality, deployment-
composition, checkpoint-cadence/resume, or sampler-evolution claim

Raw evidence identity:

- Repository-root log: `v015_r6_live_sentinel_gpu3.log`.
- SHA-256:
  `d67ed9327d8166ef7617b61f1cd746ee1f4b94710277b28cff6a825b6483f15b`.
- The hash distinguishes this successful rerun from the earlier failed log
  that used the same filename before R6-F2 was synchronized.
- Exact scans found no `Traceback`, `RuntimeError`, Hydra job error, CUDA
  runtime error, fatal signal, assertion failure, or standalone NaN/Inf token.

Observed S4 facts:

| Boundary | Runtime fact |
| --- | --- |
| Runtime owner | SUST_Main_2, Isaac Sim headless, `cuda:0` after CUDA-visible-device remapping |
| Role layout | eight physical rows: four Repair and four Noisy; no candidate/Clean scored rows |
| Scenario transaction | two Segment sources, two policy attempts per source, four policy rows |
| Immutable identity | each source's two attempts share one scenario ID, Noisy hash, motion/frame `x_t`, artifact norm, and K |
| Observation authority | raw `870D` + q29 tail `58D` = actor observation `928D`; FEMR prefix `158D`; frozen GMT suffix/input `770D`; critic `289D` |
| q29 provenance | `deployment_noisy_q29` from `motion_internal_q29` |
| One-action K | `actor_forward_count=1`, `later_femr_action_count=0`, K and Clean-continuation length are 8 for all attempts |
| K evidence | `post_advance_gmt_read_count=8`; every attempt has `evidence_valid_step_count=8` |
| Grouped reduction | attempt mass `(0.25,0.25,0.25,0.25)`; motion and Segment mass `(0.5,0.5)` |
| Optimizer boundary | four valid rows, `optimizer_step_delta=1`, `exact_one_update=true` |

Warning classification:

- CUDA/Omniverse enumeration, skipped non-selected GPU, IOMMU, headless GLFW,
  no-viewport, and one shader-cache lifecycle message were present.
- They did not stop environment setup or the transaction; the structured live
  snapshot and exact-one-update receipt were emitted afterward. They are not
  evidence of a v015 method-route failure in this bounded run.
- The log alone does not prove the host's physical GPU ordinal because
  Omniverse and CUDA-visible-device enumeration differ; it does prove the
  successful policy path ran on the remapped `cuda:0` device.

Acceptance decision:

- R6 is complete at S4 for the bounded local identity sentinel.
- Step 5A is complete. The current v015 local route runtime-confirms sealed
  scenarios, Clean dynamics reset, deployment-q29 actor context, one FEMR
  action, Clean-C K execution, separate actor/critic authorities, grouped
  equal-mass reduction, and exactly one optimizer update.
- R6-F1 command-clock isolation and R6-F2 critic-observation routing are both
  runtime-confirmed by this log. No R6 stop condition triggered.
- The result does not establish long-training convergence, physical policy
  quality, actual checkpoint cadence/resume, persistent-sequence composition,
  or replay-priority evolution.

Next:

- Close Step 5A across plan/checklist/canvas/registry/Architecture.
- Step 5B deployment-composition evaluation is the only remaining Stage-3
  engineering step and requires separate explicit user authorization.

## E-FI-28: Step 5B-S1 Immutable Deployment Composition Kernel

Date: 2026-07-21
Tier: deterministic S1 schema, identity, report, no-feedback, and legacy
isolation evidence; no command/actor/GMT connectivity, simulator, training,
live evaluation, checkpoint, sampler, storage, PPO, or optimizer execution

Implementation:

- `frontres_segment_sequence_eval.py` now owns an explicit v015 deployment
  composition config, persistent-corruption protocol, validated request, and
  immutable per-frame report. Existing legacy plan/reset functions are
  unchanged.
- The request reads one explicit `.npz` with `fps`, q29/dq29, body pose,
  quaternion, linear velocity, and angular velocity arrays. It validates frame,
  body, q29, finite-value, fps, and H coverage before sealing the absolute path,
  file SHA-256, and `deployment_reference_stream` provenance.
- Corruption metadata is scalar-only, canonicalized by family and parameter
  order, fixed to `persistent_full_sequence`, and sealed by SHA-256. This hash
  identifies the protocol; it does not claim that an S2 materialized corrupted
  frame stream has already executed.
- The report requires one row per deployment frame for FEMR action use, q29
  intent error, physics success/fall, ZMP margin, and contact consistency. Its
  accumulated failure count is derived from per-frame physics success.
- Return, priority, PPO, sampler, optimizer, Clean continuation, and local
  scenario are absent from the report dataclass. Feedback properties are
  immutable false, and passing a local return carrier is rejected by the type.
- Config validation rejects disabled mode, non-`.npz`, invalid H offsets, and
  any legacy mode mixing before a file or runner path is consumed.

Fresh verification:

- RED: the new contract initially failed at the missing
  `build_frontres_v015_persistent_corruption_protocol` owner API.
- `frontres_v015_deployment_composition_s1_contract.py`: exit 0;
  `T-npz-schema/T-identity/T-corruption-protocol`,
  `T-report/T-no-feedback`, and
  `T-config-fail-closed/T-legacy-reject` passed.
- `frontres_segment_sequence_eval_contract.py`: exit 0; the legacy v002
  plan/reset/evaluator contract remains unchanged.
- `frontres_v015_evaluation_isolation_contract.py`: exit 0; every legacy
  evaluator still rejects v015 before capture.
- `python -m py_compile` for the owner and new test: exit 0.

Confirmed:

- Step 5B-S1 is complete at S1. One structured deployment file and one
  persistent corruption protocol now have immutable, reviewable identities;
  per-frame report semantics cannot carry local training state.
- No Concept Figure or method semantic change was made.

Unconfirmed / next:

- Step 5B-S2A must first connect this request only to a command-owned deployment
  current/H snapshot. Step 5B-S2B then connects repeated per-frame FEMR, frozen
  GMT, metrics, report, and the dedicated entry while proving zero sampler/
  storage/PPO/optimizer mutation.
- S2 formal config/runner dispatch, materialized corrupted-stream identity,
  report persistence, simulator timing, and S4 deployment behavior remain
  unconfirmed and require separate authorization.

## E-FI-29: Step 5B-S2A Deployment Carrier And H Snapshot

Date: 2026-07-21
Tier: deterministic S1 owner plus S2 read-only connector evidence; no local
scenario, actor/GMT execution, metrics/report production, formal runner,
simulator, training, storage, return, priority, PPO, or optimizer execution

Root-cause boundary and RED evidence:

- E-FI-28 sealed the `.npz` and corruption-protocol identities, but
  `MultiMotionCommand` had no deployment sequence owner and
  `frontres_runtime.py` could read only the local-Segment intent snapshot.
- The new semantic contract first failed with
  `AttributeError: MultiMotionCommand has no attribute
  set_frontres_v015_deployment_sequence`, localizing the missing S2A owner.

Implementation:

- `MultiMotionCommand.set_frontres_v015_deployment_sequence()` validates the
  E-FI-28 request and protocol, verifies the file SHA-256 before and after the
  safe `.npz` read, then copies detached finite q29/dq29 `[T,29]` arrays into
  one command-owned immutable sequence.
- One explicit `[B]` cursor starts at frame zero. Snapshot reads return current
  q29+dq29 `[B,58]`, dense q29 intent `[B,H+1,29]`, row ids, frame indices,
  future offsets, reference/file/protocol identity, and deployment provenance.
- Snapshot values and provenance are cloned. Reads do not move the cursor;
  explicit advance changes all rows by exactly one frame and rejects before H
  would require a clamp. Reinstall, changed hash, active local scenario, fixed
  tape, and legacy reference window all fail closed.
- `frontres_runtime.py::read_frontres_v015_deployment_context()` validates and
  clones that exact schema. The existing actor append, GMT command property,
  and command-clock dispatcher do not reference the deployment carrier.

Fresh verification:

- `frontres_v015_deployment_carrier_s2a_contract.py`: exit 0;
  `T-install/T-current/T-H/T-identity/T-provenance`,
  `T-frame-order/T-cursor/T-boundary/T-read-only`,
  `T-row-alignment/T-mixed-reference/T-hash`, and
  `T-no-execution/T-no-training-state/T-close` passed.
- `frontres_v015_deployment_composition_s1_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`,
  `frontres_v015_current_gmt_command_contract.py`,
  `frontres_v015_two_role_reset_contract.py`, and
  `frontres_segment_motion_command_reference_contract.py`: exit 0.
- `python -m py_compile` for `commands.py`, `frontres_runtime.py`, and the new
  contract: exit 0. `git diff --check`: exit 0.

Confirmed:

- Step 5B-S2A is complete at deterministic S1/S2. One request owns one
  command sequence; `[B,58]` current command and `[B,H+1,29]` q29 intent share
  the same cursor, file/stream identity, corruption-protocol identity, row
  order, and deployment provenance.
- No Concept Figure or method semantic change was made.

Unconfirmed / next:

- The corruption protocol hash remains declared request identity; S2A does not
  claim a corruption materializer or physical artifact execution.
- Step 5B-S2B must separately connect per-frame FEMR, frozen GMT, metrics,
  immutable report, config/runner dispatch, and zero-write isolation. S4
  simulator timing and deployment composition behavior remain unconfirmed.

## E-FI-30: Step 5B-S2B Formal Composition Executor

Date: 2026-07-21
Tier: semantic CPU S2 formal connectivity, report persistence, and zero-write
evidence; no simulator, training, live evaluation, checkpoint, sampler update,
storage write, return, priority, PPO, or optimizer step

Boundary and RED evidence:

- E-FI-29 exposed command-owned current/H snapshots but intentionally had no
  actor, GMT, metric, report, or runner consumer.
- `frontres_v015_deployment_composition_s2b_contract.py` first failed at the
  missing `FrontRESV015DeploymentCompositionRunConfig`, proving it did not
  silently reuse the legacy v002 sequence evaluator.
- The no-clamp S2A contract implies a reference with T frames and Hmax lookahead
  has exactly `T-Hmax` evaluated frames. S2B records that boundary explicitly
  instead of fabricating final future frames.

Implementation:

- `FrontRESV015DeploymentCompositionRunConfig` requires one absolute report
  path and `source=pre_materialized_deployment_npz`. The formal owner consumes
  an already artifact-bearing stream; it does not draw corruption from scalar
  protocol parameters.
- `MultiMotionCommand` now retains q29/dq29 and body pose/velocity arrays from
  the same sealed file. Its current q29/dq29/body/root properties follow one
  explicit cursor, while Isaac command compute returns
  `deployment_current_hold`; only the sequence executor advances after metrics.
- `frontres_runtime.py::build_frontres_v015_deployment_observation()` verifies
  row/cursor/stream/protocol identity and builds `870D + 58D = 928D`. It selects
  only q29 at declared H offsets; future root/global arrays are not appended.
- `run_frontres_v015_deployment_composition_eval()` performs one deterministic
  `[B,6]` correction and one frozen-GMT action per evaluated frame, then records
  mean absolute q29 intent error, success/fall, ZMP margin, and contact
  consistency. It writes one immutable JSON through a temporary-path replace.
- The formal OnPolicyRunner method is a thin connector. It never calls the
  legacy sequence evaluator. Before/after hashes cover optimizer, sampler,
  storage, and transition state; any difference rejects report production.

Fresh verification:

- `frontres_v015_deployment_composition_s2b_contract.py`: exit 0;
  `T-connect/T-per-frame/T-frozen-GMT/T-report/T-zero-write` and
  `T-formal-entry/T-legacy-isolation` passed. The semantic fixture observed
  `T=6,Hmax=2 -> 4` FEMR actions, four `770D` GMT reads, four report rows,
  eight normalization calls, four command-clock holds, and zero optimizer or
  sampler writes.
- S1/S2A/evaluation isolation, local q29/current/reset, legacy reference,
  Stage-3 entrypoint/boundary, v015 checkpoint, HSL-v007, transaction,
  observation-authority, policy-quality isolation, live-sentinel static, and
  full-6D contracts all exited 0 in the focused regression set.
- Python compilation, Architecture JSON parse, and `git diff --check` are part
  of the final S2B closeout gate.

Confirmed:

- Step 5B-S2B is complete at semantic CPU S2. One pre-materialized deployment
  stream now reaches per-frame FEMR, frozen GMT, metrics, atomic report, and the
  dedicated runner entry with no training-state mutation.
- Clean continuation, local Segment Gain, grouped PPO, checkpoint identity,
  HSL, and Concept Figure semantics are unchanged.

Unconfirmed / next:

- The CPU metric provider proves schema and aggregation, not physical ZMP,
  contact, fall, auto-reset, or command-manager timing. The default physical
  metric reader and actual report values remain S4-only.
- Step 5B-S4 requires one separately authorized bounded simulator evaluation
  using an explicitly hashed pre-materialized deployment `.npz` and report
  path. No generic training or long evaluation is authorized by E-FI-30.

## E-FI-31: Step 5B-S4-S0 Dedicated Live Composition Entrypoint

Date: 2026-07-21
Tier: deterministic S2 config/checkpoint/dispatch evidence only; no IsaacLab
import or launch, simulator step, training, live evaluation, sampler creation,
PPO, optimizer step, checkpoint write, or Concept Figure change

Boundary and RED evidence:

- The first S4-S0 contract run failed because
  `scripts/rsl_rl/frontres_v015_deployment_composition.py` did not exist,
  proving S2B had no server-callable owner.
- White-box task registration then showed the old
  `FrontRES-RLFinetune-Tracking-Flat-G1-v0` name is commented out. The
  strengthened contract failed until the CLI selected the registered
  `FrontRES-Unified-Tracking-Flat-G1-v0` task.
- The formal transaction owner rejects nonzero Stage-3 warmups. A second RED
  assertion required both algorithm critic/actor warmup counters to be zero
  before the config passed.

Implementation:

- The dedicated CLI validates absolute existing FEMR/GMT `.pt` files, one
  absolute pre-materialized Noisy `.npz`, a new absolute `.json` report path,
  ordered H offsets, scalar persistent-corruption metadata, positive even env
  rows, and a valid CUDA device exposed by `CUDA_VISIBLE_DEVICES` before
  AppLauncher construction.
- Agent config binds the explicit frozen-GMT checkpoint before
  `OnPolicyRunner` construction. It enables only the existing formal v015
  layout/checkpoint identity and disables Segment Replay, live train/probe,
  sampler-requesting modes, HSL labels/loss, and all legacy evaluators.
- `OnPolicyRunner` now resolves the accepted q29 H layout when the formal v015
  checkpoint identity is requested even if Segment Replay is not requested.
  `FrontRESSegmentRunnerBoundary.requested` remains false, so the Segment
  sampler initializer is not reached.
- FEMR loads through `frontres_checkpointing.load_runner` with
  `load_optimizer=False, load_critic=False`; the only subsequent dispatch is
  `run_frontres_v015_deployment_composition_eval`. The tracked Adam step count
  must remain unchanged or the CLI fails.
- Completion sentinel records reference/protocol hashes, T, evaluated
  `T-max(H)` frames, H offsets, FEMR action count, failures, optimizer delta,
  no-feedback state, and report identity.

Fresh verification:

- `frontres_v015_deployment_live_cli_s4s0_contract.py`: exit 0;
  `T-path/T-gpu/T-protocol`, `T-config/T-dispatch/T-zero-update`, and
  `T-owner/T-formal-entry/T-no-training` passed.
- `frontres_v015_deployment_composition_s2b_contract.py`,
  `frontres_v015_observation_authority_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_v015_local_sentinel_config_contract.py`, and
  `frontres_segment_runner_boundary_contract.py`: exit 0.
- Python compilation of the CLI, runner connector, and S4-S0 contract: exit 0.

Confirmed:

- S4-S0 is complete. The repository now has one fail-closed v015-only command
  that can construct the formal IsaacLab runner and connect explicit
  checkpoint/reference/report identities to S2B without a training/update
  dispatch.
- No evaluator formula, 928/158/770 observation authority, PPO, sampler,
  optimizer, checkpoint format, or Concept Figure semantic was changed.

Unconfirmed / next:

- No real v015 FEMR checkpoint or pre-materialized Noisy deployment stream was
  opened in this step. Their server identities and hashes remain S4 inputs.
- Step 5B-S4 remains separately user-gated. One bounded simulator run must
  produce the expected sentinel and atomic report before deployment
  composition is runtime-confirmed.

## E-FI-32: Post-Observation-Change Test-Path Rebase

Date: 2026-07-21
Tier: user-confirmed dependency correction plus S0 documentation/governance
evidence; no source code, Concept Figure, simulator, training, checkpoint IO,
artifact generation, or live run

Raw decision evidence:

- The user stated: the observation was changed substantially and the policy
  has not been retrained, therefore no compatible v015 FEMR checkpoint exists.
- The user stated: no `Noisy.npz` exists and this object had never been defined
  for them before the S4 command requested it.
- The user requested a replanned test path and then explicitly authorized the
  documentation update before returning to the main conversation.

Corrected facts:

- `FRONTRES_CKPT` is a future product of new-layout HSL/Stage-3 training,
  checkpoint save, and fresh-runner reload. It is not an available S4 input.
- Deployment consumes an ordinary reference `.npz`; it has no special
  user-facing `Noisy.npz` type. For controlled synthetic evaluation, a planned
  selection-time owner must materialize one fixed artifact carrier from an
  existing Clean/reference `.npz` and seal source/protocol/carrier hashes.
- Corruption metadata is report-only and cannot enter FEMR observation. The
  carrier is created once and shared unchanged by every comparison branch.
- Scientific composition evidence requires frozen-GMT baseline versus
  per-frame-FEMR plus frozen-GMT under the same carrier, initial conditions,
  and GMT identity. A repair-only sequence is insufficient to claim benefit.
- E-FI-28--E-FI-31 remain valid deterministic interface/config evidence. S2B
  and the CLI are now classified `implemented-not-runnable`, not ready S4.

Replanned order:

1. G0 document/test-path rebase;
2. G1 read-only Training Readiness Audit;
3. G2 new-layout proposal-only HSL persistence and smoke;
4. G3 Stage-3 one-transaction save/fresh-reload smoke;
5. G4 controlled fixed-carrier materializer;
6. G5 formal training and policy-quality gate;
7. G6 paired composition connectivity;
8. G7 one bounded live composition run.

Acceptance:

- G0 is complete. Step 5B-S4 is blocked behind G1--G6.
- The next authorized action is read-only G1 only. No training recommendation
  follows from prior route/connectivity tests.

## E-FI-33: G1 v015 Training Readiness Audit And Gap Rebase

Date: 2026-07-21
Tier: S0 static owner/layout/checkpoint/train-dispatch audit plus documentation
closeout; no source code, active contract, Concept Figure, test, checkpoint IO,
simulator, training, artifact generation, or live run

Read-only evidence:

- `scripts/rsl_rl/train.py::_apply_frontres_stage_preset()` configures
  proposal-only HSL without enabling the v015 formal transaction/layout or
  supplying nonempty q29 future offsets. `OnPolicyRunner` resolves the v015
  layout only for Segment Replay or an explicit v015 formal identity.
- `frontres_warmup.py::run_frontres_joint_warmup()` writes
  `model_warmup.pt` through generic `save_runner()`. The active training
  contract and `reject_legacy_frontres_hsl_checkpoint()` both state that no
  accepted proposal-only v015 HSL checkpoint identity exists.
- ordinary Stage-3 training dispatches
  `run_frontres_segment_live_update_loop()`, which repeatedly invokes the
  legacy `run_frontres_segment_sampler_step()` path. The sealed grouped v015
  transaction owner remains a separate explicit dispatch.
- `frontres_checkpointing.save_runner()` emits the exact
  `frontres-v015-checkpoint-v2` envelope only when the v015 formal identity and
  layout are active. Ordinary Stage-3 training does not enable that route; the
  bounded local sentinel is not a formal training checkpoint producer.

Confirmed gaps:

1. Stage-1 formal HSL has not enabled q29 v015 layout; legacy `870D` input
   remains reachable.
2. A new proposal-only HSL checkpoint identity is undefined.
3. Ordinary Stage-3 training bypasses the sealed grouped transaction through
   the legacy sampler/update loop.
4. Ordinary training cannot produce and fresh-reload an exact v015 checkpoint.

Plan rebase:

- G2 owns the new HSL identity, mandatory Stage-1 q29 formal route, HSL
  actor/prefix-normalizer save/reload, deterministic evidence, and one bounded
  proposal-only smoke.
- G3 owns actor-only migration, sealed formal grouped dispatch, exact-one
  transaction/update smoke, the exact v015 checkpoint producer, and fresh
  inference reload.
- G4--G7 retain controlled carrier materialization, formal policy quality,
  paired composition connectivity, and bounded live composition respectively.

Acceptance:

- G1 is completed as a stopped audit, not as training readiness approval.
- G2-S0 is the only next user-gated step. No code change or training may begin
  until its owner/schema/evidence/stop contract is confirmed.

## E-FI-34: G2-S0 Proposal-Only HSL Persistence White-Box Audit

Date: 2026-07-21
Tier: S0 multi-layer semantic audit and planning evidence; no source code,
active contract, Concept Figure, test, checkpoint IO, simulator, training,
artifact generation, or live run

Design and owner evidence:

- Concept Figure `M-03` defines HSL as supervised initialization of the 6D
  actor; `M-05` separately owns Actor & Critic Warmup.
- Active `FRS-TRAIN-v007` allows only current-frame anti-DR Delta SE(3) as the
  Stage-1 HSL target and reserves a separately authorized new HSL identity.
- `train.py::_apply_frontres_stage_preset()` does not enable a v015 HSL layout
  mode or install `(1,2)` future offsets for `stage1_hsl`.
- `OnPolicyRunner` resolves `928/158/770` only when Segment Replay or the
  Stage-3/deployment formal-transaction identity requests it.

Carrier/dataflow finding:

- `prepare_frontres_hsl_actor_observation()` calls the shared q29 bridge.
- The bridge reads only
  `MultiMotionCommand.frontres_local_scenario_intent_snapshot()`.
- That accessor requires a transaction-wide active/current-frame-ready local
  scenario. Formal HSL neither samples nor installs such a scenario.
- Therefore merely enabling the layout would change the failure from legacy
  `870D` reachability to a missing-carrier exception; it would not complete the
  Stage-1 q29 route.

Target/gradient finding:

- `run_frontres_joint_warmup()` validates the actor target as current anti-DR
  `[B,6]`, but also constructs `_energy_target` from executable and feasible
  oracle scores.
- Its warmup optimizer contains both `residual_actor` and `critic` parameters,
  and the total loss includes Huber energy loss on `policy.evaluate()`.
- Generic `model_warmup.pt` consequently saves actor, critic, policy noise,
  optimizer, critic normalizer, Gain/sampler state when present, and no HSL
  identity. This is not a proposal-only migration artifact.

Persistence/lifecycle finding:

- `frontres_checkpointing.py` is the unique runner persistence owner, but only
  the Stage-3 `frontres-v015-checkpoint-v2` envelope is implemented.
- `load_runner()` has no proposal-only HSL validation branch. It either
  validates the Stage-3 envelope or reaches the legacy-HSL reject boundary.
- Legacy normalizer extraction can pad missing prefix statistics with identity
  values; that compatibility behavior must not be reachable after a new HSL
  identity is selected.
- `FrontRESActorCritic` loads the frozen GMT checkpoint and its 770D normalizer,
  but the proposed HSL artifact must explicitly bind that GMT artifact identity
  to make a fresh reload reproducible.

Decision and proposed schema:

- Recommended: define a separate immutable Stage-1 proposal carrier containing
  only current artifact identity, deployment/Noisy q29 H, provenance, and a
  proposal-context identity. Do not add x_t, Clean continuation, K, Segment
  roles, attempts, return, priority, or PPO state.
- Not recommended: reuse the full Stage-3 local-scenario carrier and keep C/K
  unused. This introduces Segment Replay objects into proposal initialization.
- Proposed checkpoint key/format:
  `frontres_v015_hsl_checkpoint_identity` /
  `frontres-v015-hsl-proposal-v1`.
- Allowed migration payload: residual actor, distribution std/log_std, and the
  exact 158D FEMR-prefix normalizer state.
- Forbidden payload: critic/critic normalizer, HSL optimizer, sampler,
  transaction, Gain/return/priority/PPO, Clean continuation, or rollout label.

Acceptance and stop:

- G2-S0 is blocked, not complete, because v007 currently names the shared
  local-scenario carrier while the recommended minimal proposal carrier is a
  distinct responsibility. This requires human clarification before contract
  activation or code.
- After confirmation, G2 is split into S1a carrier, S1b actor-only formal
  route, S2 identity/save/reload, S3 fresh-runner connectivity, and separately
  authorized S4 bounded smoke.

## E-FI-35: G2-S1a Existing-Module HSL Proposal Carrier

Date: 2026-07-21
Tier: S1 deterministic command/runtime/layout contract evidence; no new source
or test module, formal Stage-1 dispatch, checkpoint IO, simulator, training,
artifact generation, or live run

Human decision and scope:

- The user selected the minimal Stage-1-only proposal carrier and required the
  implementation to modify existing owners instead of creating another module.
- The active v007 contract was clarified in place: Stage-1 and Stage-3 share
  the versioned q29 layout/provenance validator, but Stage-1 owns only current
  root-artifact identity, deployment/Noisy q29 `I[t:t+H]`, and an immutable
  proposal-context identity.
- The carrier excludes Clean `x_t`, Clean continuation `C`, `K`, Segment roles,
  attempts, return, priority, PPO state, and every future root/global field.

Fail-first evidence:

- The extended existing `frontres_hsl_v007_s1_contract.py` initially failed at
  `MultiMotionCommand.frontres_hsl_proposal_intent_snapshot`: the accessor did
  not exist before this step.
- No separate carrier module or separate test module was introduced.

Implementation evidence:

- `commands.py` now owns one vectorized deployment-q29 row extractor reused by
  both local-scenario materialization and the Stage-1 proposal snapshot. The
  snapshot returns detached `[B,H+1,29]` q29, motion/frame identities, current
  artifact hashes, proposal-context hashes, offsets, and deployment provenance.
- `frontres_runtime.py` selects that command-owned snapshot only under the
  explicit HSL proposal-context flag and routes it through the existing future
  q29 append path. The local-scenario route remains separate.
- `frontres_observation_layout.py` reuses the existing provenance validator for
  `carrier_kind=hsl_proposal` and rejects Clean-continuation provenance.
- The command snapshot rejects out-of-range windows instead of clamping and
  rejects mixing with local-scenario, fixed-tape, or deployment-eval state.

Deterministic verification:

- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py`
  passed, including carrier schema, immutability, no-C/K, runtime isolation,
  layout/provenance, current anti-DR target, Stage-3 label/write rejection, and
  legacy checkpoint rejection.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_future_intent_actor_context_contract.py`
  passed.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_role_aligned_future_intent_contract.py`
  passed.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_local_scenario_kernel_contract.py`
  passed.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_two_role_reset_contract.py`
  passed.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py`
  passed.
- `frontres/bin/python -m py_compile` passed for the three modified source
  owners and the extended HSL S1 contract test.

Acceptance and remaining boundary:

- G2-S1a is complete at S1. Existing owner modules and the existing HSL S1
  test carry the implementation; no duplicate module was created.
- This evidence does not prove formal Stage-1 config/runner connectivity. The
  runner flag is deterministic-fixture-only, the warmup optimizer still owns
  the legacy critic/energy path, and the proposal-only checkpoint identity is
  not implemented.
- G2-S1b is the next independently authorized step: formal Stage-1
  `928/158/770` route plus actor-only proposal warmup and critic-state
  invariance. G2-S2 checkpoint identity/save/reload remains out of scope.

## E-FI-36: G2-S1b Formal Proposal-Only HSL Route

Date: 2026-07-21
Tier: S1 deterministic config/layout/core-parameter and regression evidence;
no checkpoint IO, simulator, training, artifact generation, or live run

Fail-first evidence:

- The extended existing HSL S1 contract initially failed after the Stage-1
  preset printed `segment_replay=True`, `future_offsets=()`, and retained
  `frontres_warmup_energy_loss_weight=1.0`.
- This isolated the two authorized defects: the formal Stage-1 preset did not
  request the v015 q29 layout, and the warmup owner still admitted the legacy
  executable-energy critic route.

Implementation evidence:

- `train.py::_apply_frontres_stage_preset()` now fixes Stage-1 HSL offsets to
  `(1,2)`, rejects another offset layout, disables Segment Replay and Stage-3
  formal transaction, selects the v015 q29 layout, keeps online supervised
  algorithm loss and rollout labels at zero/off, fixes the current FEMR prefix
  at `100D`, and sets the HSL energy weight to zero.
- `OnPolicyRunner` derives an explicit proposal-context route only from the
  Stage-1 exit plus `supervised_restore` identity. It rejects Segment/formal
  mixing, resolves `870+58=928`, writes `num_frontres_obs=158`, preserves the
  frozen GMT `770D` suffix, and appends proposal q29 before the first
  normalization call. It also bypasses the inactive privileged/teacher
  normalizers so proposal HSL cannot update critic-side running statistics.
- `frontres_warmup.py` now builds its optimizer only from
  `residual_actor.parameters()`. The executable-energy target, critic
  observation collection, critic forward/loss, and energy diagnostics were
  removed. A before/after tensor guard fails closed on any critic mutation.
- The FrontRES config default and nearby owner comments now record
  proposal-only actor initialization instead of the retired joint HSL
  actor/energy-critic interpretation.

Observed S1 facts:

- The actor-only pseudo step changed residual-actor parameters.
- Every critic parameter retained `grad is None` and exact tensor equality.
- The formal runner contains no Stage-1 privileged/teacher normalizer write.
- A deliberate critic mutation triggered the invariance guard.
- The formal preset produced `(1,2)`, `segment_replay=False`, zero online HSL
  loss, and the deterministic `928D -> FEMR 158D / GMT 770D` contract.

Fresh verification:

- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py`
  passed all carrier, config, layout, target, actor-only, critic-invariance,
  Stage-3 isolation, and legacy-reject assertions.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py`
  passed the q29-normalizer-actor-current-target and zero Stage-3 HSL paths.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py`
  and `frontres_segment_stage3_entrypoint_pseudo_contract.py` passed.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_observation_authority_contract.py`
  passed `928/158/770` authority and zero-prefix rejection.
- `python -m py_compile`, Architecture JSON parsing, and `git diff --check`
  passed for the modified step surface.

Acceptance and remaining boundary:

- G2-S1b is complete at deterministic S1. This proves formal config/owner
  selection and the local actor-only gradient boundary; it is not simulator or
  live HSL evidence.
- Generic `model_warmup.pt` persistence is still not an accepted v015 HSL
  artifact. G2-S2 must define and validate
  `frontres-v015-hsl-proposal-v1` before any Stage-1 checkpoint IO or bounded
  smoke is authorized.

## E-FI-37: G2-S2 Strict Proposal-Only HSL Persistence

Date: 2026-07-21
Tier: deterministic S3 semantic checkpoint fixture with temporary local files;
no fresh formal runner, simulator, training, Stage-3 PPO, artifact production,
or live run

Fail-first evidence:

- The extended existing HSL contract first failed because generic
  `save_runner()` produced the ordinary runner payload rather than the required
  three-field HSL artifact.
- The old route admitted critic, optimizer, Gain, warmup marker, normalizer,
  sampler, and other generic state before any HSL-specific identity existed.

Implemented owner boundary:

- `frontres_checkpointing.py` is the sole format owner for
  `frontres-v015-hsl-proposal-v1`.
- `frontres_warmup.py` remains a thin connector through existing `self.save()`;
  it marks runtime warmup completion only after the strict save succeeds.
- The HSL save branch returns before generic critic/optimizer/sampler/Gain/
  transaction payload construction. The deterministic optimizer fixture raises
  if `state_dict()` is read, while HSL save succeeds.

Exact payload and identity:

- Top-level keys are exactly `frontres_v015_hsl_checkpoint_identity`,
  `model_state_dict`, and `frontres_prefix_norm_state_dict`.
- `model_state_dict` is exactly `residual_actor` plus one `std` or `log_std`
  tensor. The prefix payload is the complete 158D empirical-normalizer state:
  `_mean`, `_var`, `_std`, and `count`.
- Identity binds FRS-METHOD-v015/FRS-TRAIN-v007, proposal-only current anti-DR
  objective, `(1,2)` q29 layout, `870/928/158/770`, full 6D Delta SE(3), GMT
  checkpoint SHA-256, frozen GMT 770D normalizer fingerprint, and value-sensitive
  actor/distribution/prefix fingerprints.

Pre-mutation evidence:

- Load validates the exact top-level and nested field sets, current layout,
  action, GMT artifact/normalizer, tensor keys/shapes/dtypes/finiteness, and all
  fingerprints before the first actor/distribution/prefix write.
- Successful reload changed only actor, distribution, and prefix-normalizer
  state to the source values. Critic, privileged/critic normalizer, and
  optimizer load count remained unchanged.
- Five rejected cases were injected: forbidden optimizer key, actor-value
  tamper, prefix-stat tamper, GMT identity tamper, and legacy/unversioned
  warmup payload. Every case preserved actor, critic, distribution, prefix,
  privileged normalizer, optimizer count, warmup marker, and loaded-path state.

Fresh verification:

- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py`
  passed the combined S1/S3 HSL contract.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_checkpoint_resume_contract.py`
  passed, proving the existing Stage-3 `frontres-v015-checkpoint-v2` format and
  transaction atomicity path were not migrated or regressed.
- HSL S2 connectivity and future-intent actor-context contracts passed.
- `python -m py_compile`, Architecture JSON parsing, and `git diff --check`
  passed for the modified surface.

Acceptance and remaining boundary:

- G2-S2 is complete at deterministic S3. The accepted HSL artifact is strictly
  migration-only and cannot enter a non-HSL or Stage-3 formal checkpoint route.
- This fixture does not prove a newly constructed formal runner reproduces the
  same normalized 158D input and actor proposal. That is the separately
  authorized G2-S3 offline fresh-runner connectivity boundary.

## E-FI-38: G2-S3 Offline Fresh-Runner HSL Connectivity

Date: 2026-07-21
Tier: deterministic S2/S3 semantic connectivity with two independent fake
runner objects and temporary checkpoint files; no production source change,
simulator, training, Stage-3 migration/PPO, or live run

Core parameter path:

```text
fixed current artifact in raw [B,870]
+ fixed deployment q29 [B,3,29] at offsets (1,2)
-> command-owned 58D tail
-> combined [B,928]
-> existing prefix/GMT normalizer split
-> normalized FEMR input [B,158]
-> residual actor
-> bounded full-6D Delta SE(3) proposal
```

Fixture evidence:

- The source and fresh runner are independently initialized with different
  residual-actor weights and different 158D prefix-normalizer statistics.
- Both use the same frozen GMT artifact, GMT 770D normalizer, current artifact,
  raw 870D observation, q29 window, offsets, proposal-context identities, and
  deployment/Noisy provenance.
- Before reload, their normalized 158D actor inputs and 6D proposals differ,
  proving the target is not accidentally identical by construction.
- The source saves through `frontres-v015-hsl-proposal-v1`; the fresh runner
  reloads through the strict HSL pre-mutation branch.

Observed equality after reload:

- Combined observation is exactly `[2,928]`; its first 58 values equal q29
  offsets `(1,2)`, and the following current-artifact values equal the fixed
  raw input.
- Normalized full observation, normalized FEMR `[2,158]` input, and bounded
  `[2,6]` proposal are elementwise identical with `rtol=0, atol=0` between the
  source-before-save and fresh-after-reload routes.
- Fresh reload uses the live 158D prefix-normalizer state directly:
  `_frontres_extra_mean`, `_frontres_extra_std`, and legacy layout-padding state
  remain absent.
- Critic parameters remain at the fresh runner's pre-load values and optimizer
  load count remains zero.

Fresh verification:

- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py`
  passed the new T-fresh-runner/T-output/T-zero-leak trace plus prior S1/S3
  HSL contracts.
- HSL S2 connectivity, Stage-3 v2 checkpoint/resume, future-intent actor
  context, and v015 observation-authority contracts passed.
- `python -m py_compile` and `git diff --check` passed.

Acceptance and remaining boundary:

- G2-S3 is complete at deterministic S2/S3. No production code change was
  required; the existing strict checkpoint, q29 bridge, normalizer, and actor
  owners already compose correctly offline.
- This does not prove IsaacLab lifecycle timing, real current-artifact/q29
  values, optimizer execution during bounded HSL, or real checkpoint output.
  Those are the separately authorized G2-S4 bounded formal Stage-1 smoke.

## E-FI-39: G2-S4-S0 Bounded HSL Telemetry And Fresh-Reload Connector

Date: 2026-07-21
Tier: deterministic S1/S3 config, telemetry-schema, persistence, and shadow
reload contracts; no simulator, training, live run, Stage-3 PPO, deployment
composition, or checkpoint-format change

Implemented owner path:

```text
--frontres_hsl_live_smoke
-> Stage-1-only bounded config checks
-> command-owned real artifact/q29 snapshot
-> 870D + 58D = 928D -> FEMR 158D / GMT 770D telemetry
-> current anti-DR target
-> residual-actor nonzero gradient / critic zero gradient and zero delta
-> frontres-v015-hsl-proposal-v1 exact save identity
-> independent pre-warmup CPU shadow strict reload
-> normalized 158D and bounded 6D exact equality
```

Observed deterministic evidence:

- `frontres_hsl_v007_s1_contract.py` exited 0. Its new
  T-HSL-live-smoke/T-telemetry/T-shadow-reload case observed `[2,928]`,
  `[2,158]`, `[2,770]`, deployment q29 provenance, exact HSL-v1 keys,
  `forbidden_payload=0`, pre-reload proposal inequality, and post-reload
  `normalized_158_equal=1 proposal_6_equal=1`.
- The bounded config rejects non-Stage-1 use, odd or more than eight envs,
  nonzero PPO iterations, warmup iterations/steps other than one, and resume.
- The shadow owns only residual actor, distribution, frozen GMT normalizer,
  and 158D prefix normalizer. It has no critic or optimizer attribute.
- `frontres_hsl_v007_s2_connectivity_contract.py`,
  `frontres_future_intent_actor_context_contract.py`,
  `frontres_v015_observation_authority_contract.py`, and
  `frontres_v015_checkpoint_resume_contract.py` all exited 0.
- `python -m py_compile` for all changed Python files and `git diff --check`
  exited 0.

Acceptance and remaining boundary:

- G2-S4-S0 is complete at deterministic S1/S3. It changes no method semantic,
  grouped PPO formula, checkpoint format, or Concept Figure, and creates no new
  source module.
- G2-S4 remains open. Only the user-gated G2-S4-S1 Main-2 IsaacLab smoke can
  runtime-confirm real artifact/q29 values, actor gradient, zero critic delta,
  real HSL-v1 artifact output, and fresh reload equality in one formal run.

## E-FI-40: G2-S4-S0a Full-6D Diagnostic Mask Regression Repair

Date: 2026-07-21
Tier: deterministic S1 regression repair after the first bounded live attempt;
no simulator, training, live retry, Stage-3 PPO, deployment composition, or
checkpoint-format change

Observed failure:

- Repository `log.txt` stopped in `run_frontres_joint_warmup()` final
  diagnostics with `NameError: name '_sup_mask' is not defined`.
- Historical source confirmed `_sup_mask` belonged to the retired
  `frontres_active_task_dims` partial-dimension warmup. Proposal-only v007 had
  removed its definition and training use but left one diagnostic read.

Repair and regression:

- Deleted the three diagnostics-only `_sup_mask` lines. No replacement mask,
  fallback, active-dimension route, clamp, or skip was introduced.
- The existing `frontres_hsl_v007_s1_contract.py` now asserts that
  `run_frontres_joint_warmup()` contains neither `_sup_mask` nor
  `frontres_active_task_dims`, preserving full-6D HSL semantics.
- HSL S1, HSL S2 connectivity, v015 observation authority, future-intent actor
  context, and v015 checkpoint/resume contracts all exited 0.
- `python -m py_compile` and `git diff --check` exited 0.

Acceptance and remaining boundary:

- G2-S4-S0a is complete. The specific `NameError` now has a deterministic
  regression test and the old partial-dimension mechanism remains absent.
- G2-S4-S1 is still open. The same bounded Main-2 command must be explicitly
  re-authorized; no live retry was executed during this repair.

## E-FI-41: G2-S4-S0b Cross-Device Proposal Reload Verification

Date: 2026-07-21
Tier: live-failure diagnosis plus deterministic S1/S3 numerical and persistence
regression; no simulator, training, live retry, Stage-3 PPO, deployment
composition, or checkpoint-format change

Observed failure and boundary:

- Repository `log.txt` reached strict HSL-v1 save/reload and failed only at
  `normalized_158_equal=1 proposal_6_equal=0`.
- The checkpoint loader had already validated the exact residual-actor
  fingerprint and loaded it with `strict=True`. The residual actor is a
  deterministic Linear/ELU MLP with no dropout or running-state layer.
- The live source proposal is computed on CUDA while the independent shadow is
  intentionally CPU-only. A bitwise `torch.equal()` comparison therefore
  treated normal float32 backend reduction-order differences as checkpoint
  corruption.

Repair and discrimination:

- Actor/checkpoint fingerprints and normalized 158D input remain strict and
  exact. The checkpoint format and allowed payload are unchanged.
- Only the final CUDA/CPU bounded 6D proposal comparison now uses
  `torch.allclose(rtol=1e-5, atol=1e-6)` and records source/shadow devices,
  bitwise equality, and `max_abs_error` in both PASS and failure sentinels.
- The HSL S1 regression observed exact same-device error `0`, accepted a
  hand-checkable `5.066e-7` differential, and rejected a `1e-3` differential.
  This distinguishes backend roundoff from real reload drift.
- HSL S1/S2, v015 observation authority, future-intent actor context, and v015
  checkpoint/resume contracts all exited 0. `python -m py_compile` and
  `git diff --check` exited 0.

Acceptance and remaining boundary:

- G2-S4-S0b is complete. The reported failure is fixed without relaxing state,
  layout, normalizer, identity, or payload validation.
- G2-S4-S1 remains open. Its next log must report `proposal_6_close=1` plus the
  actual CUDA/CPU `max_abs_error`; an error beyond the fixed tolerance remains
  fail-closed.
