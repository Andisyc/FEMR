# Evidence Ledger: FRS-v015 Future Intent And Single-Action K

Date: 2026-07-19 through 2026-07-22
Scope: Accepted semantic migration and bounded implementation evidence,
including deterministic contracts, CPU formal-route/persistence checks, and
successful bounded S4 local identity and one-transaction training evidence. It
contains no long-training, held-out live policy-quality, deployment-composition,
or live checkpoint-resume evidence.

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

## E-FI-42: G2-S4 Bounded Proposal-Only HSL Live Closure

Date: 2026-07-21
Tier: bounded live S4 on SUST_Main_2/Main-2 with eight IsaacLab envs, one HSL
warmup iteration, one environment step, three actor epochs, zero PPO
iterations, real GMT checkpoint, and strict HSL-v1 save/reload

Raw evidence:

- `v015_g2_s4_hsl_smoke_gpu3.log`
- GMT artifact: `/hdd1/cyx/MOSAIC/model/model_27000.pt`
- HSL artifact:
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-07-21_17-06-12_G2_S4_BOUND_HSL/model_warmup.pt`

Observed formal chain:

```text
stage1_hsl config: max_iterations=0, warmup=1, Segment/live train disabled
-> real current root artifact + deployment_noisy_q29 [8,3,29], offsets (1,2)
-> raw 870D + q29 tail 58D = combined 928D
-> FEMR 158D / frozen GMT 770D
-> current anti-DR Delta SE(3) [8,6]
-> residual-actor-only backward
-> critic gradient count 0 and parameter delta 0
-> frontres-v015-hsl-proposal-v1 exact three-field payload
-> CPU shadow strict reload
-> exact normalized 158D + bounded-close 6D proposal
-> explicit exit before PPO
```

Runtime facts:

- Current artifact and q29 provenance were present at log lines 458--459;
  `q29_provenance=deployment_noisy_q29`, raw/combined/FEMR/GMT shapes were
  `870/928/158/770`.
- The current anti-DR target was finite `[8,6]`. Actor gradient norm was
  `9.7264719`; `critic_grad_count=0` and `critic_max_abs_delta=0`.
- The HSL identity was `frontres-v015-hsl-proposal-v1`, offsets `(1,2)`, GMT
  SHA-256 `3efcdb50df81465a1d3cbd0edb71cc9662e1e69f65e8f2e067f845607660c426`,
  exact top-level keys were actor/distribution plus 158D prefix normalizer, and
  `forbidden_payload=0`.
- Fresh reload observed `normalized_158_equal=1`, `proposal_6_close=1`,
  `proposal_6_bitwise_equal=0`, CUDA/CPU `max_abs_error=2.79396772e-09`, below
  fixed `rtol=1e-5, atol=1e-6`, and `pre_reload_proposal_equal=0`.
- Completion sentinels were `bounded_hsl=1 ppo_entered=0` and
  `Stage 1 HSL warmup-only run complete; exiting before PPO loop.`
- Each of the eight required `G2-S4-*` sentinels plus the final exit appeared
  exactly once. No Traceback, exception, legacy fallback, reload failure,
  forbidden payload, nonzero critic delta, or `Entering PPO loop` appeared.

Interpretation and remaining boundary:

- G2 / 7 is complete. This runtime-confirms proposal-only HSL input, target,
  actor-only optimization, strict persistence, and fresh-reload behavior.
- The generic `FrontRESUnified` optimizer is constructed during runner setup,
  but the live HSL owner used its separate actor-only optimizer; the observed
  critic gradient and parameter delta were both zero, and no PPO loop ran.
- `model_warmup.pt` is an initialization artifact. It does not prove a trained
  Stage-3 policy, grouped formal training dispatch, policy quality, controlled
  deployment carrier, or paired composition. G3-S0 is the next user-gated
  read-only owner audit.

## E-FI-43: G3-S1A Explicit Actor-Only HSL Migration

Date: 2026-07-21
Tier: deterministic S1/S3 config and temporary-checkpoint semantic fixtures;
no simulator, optimizer step, Stage-3 checkpoint save, training, or live run

Fail-first evidence:

- The ordinary Stage-3 preset still produced empty q29 offsets, `scale_only`,
  implicit HSL state, and nonzero legacy warmup counts.
- Generic `load_runner()` correctly rejected HSL-v1 on Stage 3 because no
  explicit actor-initializer boundary existed.

Implemented owner path:

```text
--frontres_v015_hsl_initializer_checkpoint + explicit offsets (1,2)
-> q29/grouped/formal Stage-3 config, HSL flags and supervised loss closed
-> OnPolicyRunner thin initializer connector
-> frontres_checkpointing strict HSL-v1 pre-mutation validation
-> residual actor + std/log_std + complete 158D prefix normalizer restore
-> explicit stop before G3-S1B transaction/training dispatch
```

Observed deterministic evidence:

- Successful migration reproduced the source actor, 6D distribution, and 158D
  prefix-normalizer tensors exactly.
- Critic, privileged/critic normalizer, optimizer load count, sampler identity,
  and transaction state remained unchanged; no HSL flag or supervised loss
  remained active.
- Generic Stage-3 `load_runner`, post-iteration migration, full-resume mode,
  open HSL flags, and an active transaction all rejected before mutation.
- Ordinary Stage-3 config now resolves `(1,2)`, `grouped_scale_only`, formal
  transaction identity, zero legacy actor/critic warmups, and no online HSL.
  It raises before `learn_frontres_segment_live()` until G3-S1B is connected.

Fresh verification:

- `frontres_segment_stage3_entrypoint_pseudo_contract.py` exited 0, including
  T-HSL-explicit/T-dispatch-stop.
- `frontres_hsl_v007_s1_contract.py` exited 0, including strict Stage-3
  actor-only migration and all earlier HSL S1/S3 regressions.
- `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_v015_observation_authority_contract.py`,
  `frontres_future_intent_actor_context_contract.py`, and
  `frontres_hsl_v007_s2_connectivity_contract.py` all exited 0.
- `python -m py_compile` passed for all five modified Python files.

Acceptance and remaining boundary:

- G3-S1A is complete at deterministic S1/S3. The strict Stage-1 artifact now
  has one explicit Stage-3 consumer and cannot be mistaken for resume state.
- G3-S1B remains user-gated: connect the ordinary formal training owner to one
  complete sealed transaction, exact-one update, committed receipt, and save
  trigger. No Stage-3 training or policy checkpoint is yet authorized.

## E-FI-44: G3-S1B Formal Transaction Dispatch And Commit-Only Save

Date: 2026-07-21
Tier: deterministic S2/S3 semantic CPU connectivity; no simulator, training,
fresh inference, checkpoint-format change, or live run

Implemented owner path:

```text
ordinary Stage-3 train dispatch
-> exact Repair-row budget -> distinct whole Segment M budgets, no truncation
-> immutable local artifact/q29/C/hash request under one frozen policy
-> sealed accumulator -> unchanged grouped v003 loss -> optimizer delta=1
-> matching committed receipt
-> iteration advance and checkpoint trigger only after commit
```

Observed deterministic evidence:

- The source selector preserved sampler-owned M/K budgets and selected a whole
  multi-Segment subset whose M counts exactly filled the Repair role rows;
  partial final Segment attempts were not admitted.
- The ordinary provider was called only after the collecting barrier opened,
  collection performed zero optimizer steps, and carrier cleanup ran after the
  exact-one owner returned a committed receipt.
- Two Segments x two attempts retained equal grouped attempt mass, one PPO row
  per attempt, deployment q29 provenance, and optimizer `step_delta=1`.
- The ordinary training loop did not call
  `run_frontres_segment_live_update_loop()` or legacy `to_ppo_batch()`. One
  formal iteration advanced the iteration once and invoked one save trigger.
- The save trigger accepted only the matching committed transaction identity
  with `optimizer_step_delta=1`; collecting state rejected before `runner.save`.
- Formal config now fixes `frontres_segment_live_update_steps=1`, because one
  outer iteration is one complete transaction and one optimizer update.

Fresh verification:

- `frontres_v015_transaction_route_contract.py` exited 0 with T-provider,
  T-complete-transaction, T-grouped, T-exact-one-update, T-legacy-isolation,
  T-commit, and T-save evidence.
- `frontres_v015_local_sentinel_connectivity_contract.py` exited 0 with
  T-formal-owner, T-complete-transaction, and T-no-partial evidence.
- `frontres_segment_stage3_entrypoint_pseudo_contract.py` exited 0 and proved
  the explicit HSL initializer precedes ordinary formal dispatch without the
  former G3-S1B stop.
- `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_v015_unmocked_observation_connectivity_contract.py`,
  `frontres_v015_grouped_candidate_adapter_contract.py`,
  `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_two_role_reset_contract.py`,
  `frontres_v015_real_optimizer_counter_contract.py`,
  `frontres_v015_observation_authority_contract.py`, and
  `frontres_future_intent_actor_context_contract.py` all exited 0.
- `python -m py_compile` passed for the nine touched Python files.

Acceptance and remaining boundary:

- G3-S1B is complete at deterministic S2/S3. The formal training branch now
  owns provider -> seal -> grouped exact-one -> commit -> save order, while the
  legacy immediate-update branch remains separate.
- The save-trigger test uses a fake `runner.save`; existing S3 persistence tests
  separately prove the v015 checkpoint schema. G3-S2 must connect the actual
  save producer to a fresh inference runner and prove exact proposal equality
  before any simulator/training/live smoke is authorized.

## E-FI-45: G3-S2 Exact Save Producer And Fresh Inference Reload

Date: 2026-07-21
Tier: deterministic offline S3 semantic CPU fixture; no HSL change, grouped-PPO
formula change, checkpoint-format change, simulator, training loop, or live run

Fail-first evidence:

- The prior checkpoint fixture instantiated its `residual_actor` with a 928D
  input even though `num_frontres_obs=158`. The new inference trace failed at
  `mat1 and mat2 shapes cannot be multiplied (2x158 and 928x6)`, locating the
  missing acceptance boundary in the fixture rather than the persistence owner.
- The fixture actor was corrected to consume only the 158D FEMR prefix; its
  critic retains the full 928D semantic state for checkpoint coverage.

Observed owner path:

```text
semantic 158D/6D policy + frozen policy snapshot
-> existing two-Segment x two-attempt grouped candidate request
-> existing formal grouped exact-one owner
-> real Adam step counter: 0 -> 1
-> committed metadata-only receipt
-> actual frontres_checkpointing.save_runner()
-> independently initialized fresh semantic inference runner
-> strict frontres_checkpointing.load_runner()
-> command-owned deployment q29 append -> normalization -> 158D actor -> 6D proposal
```

Facts established:

- The policy saved by `save_runner()` is the same policy instance that the
  formal transaction verified against its frozen snapshot and updated once;
  the test does not splice a receipt from a different policy.
- The committed receipt records two Segment sources, four policy attempts,
  four valid rows, and `optimizer_step_delta=1`. Resume returns transaction
  state to `idle` while preserving the exact committed receipt as history.
- The v015 envelope remains `frontres-v015-checkpoint-v2` with exact
  `928/158/770`, H offsets `(1,2)`, deployment q29 provenance, grouped identity,
  and a complete 158D prefix-stat fingerprint.
- Before reload, independently seeded actor weights and deliberately different
  prefix statistics produce different normalized 158D input and 6D proposal.
  After strict reload, combined 928D observation, normalized 928D observation,
  158D actor input, and bounded 6D proposal are elementwise identical with
  `rtol=0, atol=0`.
- q29 tail values are exactly `[B,58] = intent_q29[:,(1,2),:]`; Clean
  continuation values do not enter the checkpoint identity or actor trace.
- Existing partial/legacy tests still reject collecting/sealed transactions,
  v1/unversioned/65D layouts, zero/full actor visibility, and tampered prefix
  statistics before mutable restore. No fallback or padding was added.

Fresh verification:

- `frontres_v015_checkpoint_resume_contract.py` exited 0, including
  T-save-producer/T-v015-identity/T-commit-receipt/T-fresh-runner/
  T-prefix-normalizer/T-proposal-equality/T-legacy-reject.
- `frontres_v015_transaction_route_contract.py`,
  `frontres_v015_observation_authority_contract.py`, and
  `frontres_future_intent_actor_context_contract.py` all exited 0.
- `python -m py_compile` passed for the modified checkpoint contract fixture.

Acceptance and remaining boundary:

- G3-S2 and G3 engineering readiness are complete at offline S2/S3. The exact
  actor migration, sealed update, commit, actual save, and strict fresh reload
  chain is now contract-confirmed.
- This does not produce a trained Stage-3 policy or prove simulator timing,
  policy quality, long-run checkpoint cadence, or live deployment behavior.
  G4 owns controlled carrier materialization; G5 separately owns bounded
  training and policy-quality evidence.

## E-FI-46: G4 Controlled Artifact Carrier Materializer

Date: 2026-07-21
Tier: deterministic S1/S2 semantic CPU materialization and current/H
connectivity; no actor execution, composition executor, metrics/report,
training, simulator, or live run

Fail-first evidence:

- The focused S1 contract failed because
  `FrontRESV015DeploymentCarrierLifecycle` did not exist. Existing code could
  validate a user-supplied pre-materialized `.npz`, but no owner transformed an
  ordinary reference and fixed protocol into that carrier.
- The first root-index patch accidentally changed the existing deployment
  request schema. Both old S1 and S2A contracts failed immediately; the field
  was moved to the new carrier receipt, restoring the unchanged request schema.

Implemented owner path:

```text
ordinary reference NPZ + canonical family/seed/scale/root_body_index protocol
-> safe required-array load + source sha256
-> one RNG draw of persistent Delta SE(3)
-> rigid root/global transform of body pose/velocity arrays
-> bitwise unchanged q29/dq29
-> deterministic atomic NPZ archive with no metadata arrays
-> carrier sha256 + q29 hash + materialization sha256
-> sealed lifecycle; second materialize call rejects
-> existing strict request -> command current [B,58] / H [B,H+1,29]
```

Facts established:

- `frontres_segment_sequence_eval.py` is the sole G4 materializer owner. No new
  source module, command sampling path, runner path, or training owner was
  introduced.
- Input/output arrays retain `q,dq=[T,29]`, body position/linear/angular
  velocity `[T,J,3]`, body quaternion `[T,J,4]`, and scalar fps. q29/dq29 are
  byte-identical; the fixed artifact changes only body-frame reference data.
- `root_body_index` is explicit protocol identity because the `.npz` schema has
  no body-name metadata. It is sealed into the materialization hash but is not
  stored in the deployment archive or exposed as actor input.
- The archive contains exactly the eight required numeric arrays. Corruption
  family, seed, protocol, label, truth, and Clean metadata remain only in the
  immutable receipt/report boundary.
- The deterministic writer gives identical carrier file hashes, sampled
  Delta SE(3), q29 hashes, and materialization hashes for the same source and
  protocol across different output paths. Changing source or seed changes the
  correct identities.
- Planar, yaw, global-z, and local-RP parameter branches are covered. Missing
  root identity, missing family scale, unknown parameters, existing output,
  or a second lifecycle materialization fail closed.
- The generated carrier is accepted by the existing strict deployment request
  and command carrier, producing current q29+dq29 `[B,58]` and dense q29 intent
  `[B,H+1,29]` with the carrier file hash and deployment provenance.

Fresh verification:

- `frontres_v015_deployment_composition_s1_contract.py` exited 0 with
  T-materialize/T-hash/T-determinism/T-q29-invariant/T-no-label/T-no-resample.
- `frontres_v015_deployment_carrier_s2a_contract.py` exited 0 with the new
  T-G4-S2/T-current-H/T-carrier-identity path plus all existing carrier
  lifecycle regressions.
- `python -m py_compile` for the owner and both focused contracts, and
  `git diff --check`, exited 0.

Acceptance and remaining boundary:

- G4 is complete at deterministic S1/S2. An unexplained external `Noisy.npz`
  is no longer a prerequisite: it is a deterministic output/cache of ordinary
  reference plus fixed protocol.
- This does not execute FEMR/GMT, compare baseline and repair, produce metrics,
  or prove physical artifact quality. G5 owns trained-policy quality; G6 owns
  same-carrier paired composition connectivity.

## E-FI-47: G5-S0 Formal Training And Policy-Quality Preflight

Date: 2026-07-21
Tier: read-only S0 owner/shape/persistence/quality-route audit; no code or active
contract change, test, checkpoint IO, simulator, training, or live run

Code-confirmed formal chain:

```text
Stage-3 preset + explicit HSL-v1 initializer
-> 870D raw + 58D q29 tail = 928D
-> FEMR 158D / frozen GMT 770D / critic 289D
-> ordinary whole-M v015 request
-> sealed grouped transaction
-> exactly one optimizer update
-> matching committed receipt
-> actual v015 save trigger
```

Owners read:

- `scripts/rsl_rl/train.py::_apply_frontres_stage_preset()` and `main()` own
  formal configuration, explicit HSL input, and Stage-3 dispatch.
- `frontres_segment_live_training.py::run_frontres_segment_live_training_loop()`
  is the unique ordinary iteration/save-order owner.
- `frontres_segment_live_update_loop.py::run_frontres_v015_formal_training_update_loop()`
  owns provider/collection order.
- `frontres_segment_live_probe.py::run_frontres_v015_formal_transaction_update()`
  owns grouped exact-one update and committed transaction diagnostics.
- `frontres_checkpointing.py` owns strict HSL-v1 migration and exact Stage-3
  v015 save/load identity.

Confirmed boundaries:

- The prior S4 HSL log records the server initializer
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-07-21_17-06-12_G2_S4_BOUND_HSL/model_warmup.pt`
  as `frontres-v015-hsl-proposal-v1`. This audit did not reopen the artifact;
  current server existence remains unconfirmed.
- One formal training iteration is one complete transaction and one optimizer
  step. Iteration advance and save require the matching committed receipt.
- The actual v015 save producer exists, but ordinary training does not create an
  independent post-save fresh runner or compare normalized 158D input and 6D
  proposal after reload. G3-S2 proves that boundary only in an offline fixture.
- `_v015_formal_update_summary()` exposes transaction, grouped mass, loss, and
  update counts. It does not expose full-6D action distribution, v003
  intent/physics/cost/total Gain, positive/negative Gain fractions, or an atomic
  policy-quality report.

Quality-route stop facts:

- `frontres_policy_quality_formal_owners.py` configures policy/candidate/noisy/
  clean quartet roles and invokes `build_frontres_hsl_rollout_target()` during
  repeated K-step actor execution. This is incompatible with the active v015
  Repair/Noisy, proposal-only HSL, one-action-K contract.
- The checked manifests identify `FRS-METHOD-v011+FRS-GAIN-v002`, not
  v015/v003.
- `FrozenFrontRESTaskActor.from_checkpoint_payload()` requires generic
  `obs_norm_state_dict`; the strict HSL-v1 artifact instead owns
  `frontres_prefix_norm_state_dict`. The evaluator cannot consume the accepted
  initializer without an unauthorized fallback.
- The old quality report reconstructs v002 Style/Physics/repair-cost and
  requires quartet role identity. It cannot be treated as v015 policy quality.

Decision and plan effect:

- G5-S0 is completed with stop conditions, not a training authorization.
- G5 is locally rebased into S1 transaction telemetry, S2A strict checkpoint/
  manifest identity, S2B two-role held-out evaluation, S3 save/fresh-reload/
  report connectivity, and S4 bounded live training/quality.
- G0-G4 and G6-G7 remain unchanged. Active contracts and the Concept Figure do
  not change because the audit found implementation/acceptance gaps, not a new
  method semantic.
- Numeric policy-quality acceptance thresholds must be explicitly confirmed
  before G5-S4; route connectivity alone cannot satisfy the gate.

Next:

- G5-S1 is the earliest user-gated code step. Reuse
  `frontres_segment_diagnostics.py::build_frontres_v015_local_evaluation_report`
  as the sole read-only quality projection owner and
  `frontres_segment_live_probe.py` as its formal transaction connector. No new
  rollout, Gain recomputation, feedback path, checkpoint work, or live run is
  allowed.

## E-FI-48: G5-S1 Transaction-side v003 Action/Gain/Harm Telemetry

Date: 2026-07-21
Tier: deterministic S1/S2 CPU diagnostics and sealed-transaction connectivity;
no simulator, training, live run, checkpoint, held-out evaluator, or method
change

Fail-first evidence:

- Both focused contracts failed only because the existing local evaluation
  projection had no transaction identity or row-level telemetry interface.
- No failure entered Gain computation, grouped PPO, optimizer ownership, or a
  legacy v002/Clean-global route.

Implemented owner path:

```text
sealed FrontRESV015GainConsumerEvidence
-> build_frontres_v015_local_evaluation_report(transaction_id=...)
-> immutable action/mask/v003 component rows + sign fractions
-> one report per candidate shard in FrontRESV015FormalTransactionRequest
-> transaction/scenario/noisy-hash validation
-> grouped exact-one update and committed receipt
-> post-update diagnostics publication only
```

Facts established:

- `frontres_segment_diagnostics.py` remains the sole projection owner. It
  copies sealed `policy_actions [B,6]`, valid-row mask, `intent_gain`,
  `physics_gain`, `repair_cost`, and `gain_total`; it does not recompute Gain.
- Invalid policy-row component values must remain NaN/UNCONFIRMED. Missing
  actions, mask, components, or transaction identity fail closed instead of
  being silently filled with zero.
- Positive and negative Gain fractions are computed only from the already
  sealed `gain_total` rows selected by the valid mask; zero Gain remains
  neutral.
- `frontres_segment_live_probe.py` requires one frozen report per candidate
  shard and verifies transaction, scenario, noisy-hash, and row-count identity
  before collection. The reports are not included in PPO batches or loss
  inputs and are published only after the exactly-one optimizer step commits.
- No v002 Style/Clean-global field, return/priority mutation, sampler write,
  HSL path, checkpoint path, or held-out evaluation path was added.

Fresh verification:

- `frontres_segment_diagnostics_contract.py` exited 0 with row-level action,
  v003 component, positive/negative fraction, identity, no-feedback, and
  missing-field fail-closed assertions.
- `frontres_v015_transaction_route_contract.py` exited 0 for a two-Segment by
  two-attempt transaction, preserving all report identities and
  `optimizer_step_delta=1`.

Acceptance and remaining boundary:

- G5-S1 is complete at deterministic S1/S2. It establishes observable training
  evidence but does not establish checkpoint/manifest identity, held-out
  quality, fresh-reload report atomicity, or live policy quality.
- G5-S2A remains separately user-gated. This step did not enter it.

## E-FI-49: G5-S2A Strict Quality Checkpoint And Manifest Identity

Date: 2026-07-21
Tier: deterministic S1/S3 CPU checkpoint/manifest/request identity; no actor
restore, evaluator execution, optimizer, training, simulator, or live run

Fail-first evidence:

- The new contract failed because no v015-only quality request owner existed.
  The legacy builder checked only file existence and accepted the old v1
  manifest plus uninspected checkpoint placeholders.
- Existing HSL and Stage-3 persistence formats were already strict for their
  own runner routes, but no read-only quality receipt distinguished them before
  evaluator state construction.

Implemented identity path:

```text
held-out manifest JSON
-> strict frontres-v015-policy-quality-manifest-v1 parse
HSL file -> exact proposal-only payload/fingerprint inspect
Stage3 file -> exact v015-v2/v003/grouped/layout/transaction inspect
-> manifest/file SHA-256 receipts
-> immutable FrontRESV015PolicyQualityEvalRequest
```

Facts established:

- `frontres_checkpointing.py::inspect_frontres_v015_quality_checkpoint()` is a
  CPU read-only inspector. It never calls `load_runner()` and exposes no
  optimizer, critic, sampler, or mutable normalizer state.
- The HSL route requires `frontres-v015-hsl-proposal-v1`, exact actor/std plus
  `frontres_prefix_norm_state_dict`, embedded actor/distribution/prefix
  fingerprints, 928/158/770, offsets `(1,2)`, and 6D Delta SE(3). Generic
  `obs_norm_state_dict` cannot substitute for the HSL prefix key.
- The policy route requires `frontres-v015-checkpoint-v2`, v015/v007/v003/v003
  contracts, grouped one-row identity, exact 928D observation normalizer with
  158D prefix fingerprint, a finite 6D distribution/output identity, and only
  idle or valid committed transaction state.
- `FrontRESV015PolicyQualityManifest` is separate from the retained legacy v1
  class. Its strict schema binds v015/v003, 870/928/158/770, offsets `(1,2)`,
  and full-6D action identity. The v015 request accepts only this schema.
- HSL and policy files remain distinct immutable receipts with exact file
  hashes. Route swap, HSL actor tamper, policy prefix tamper, partial
  transaction, v1 schema, and `FRS-GAIN-v002` all reject before any runner
  mutation.
- The formal `run_frontres_policy_quality_eval()` now detects an active v015
  runner before the legacy builder. It validates only the strict request and
  then stops on the intentionally absent G5-S2B executor; a legacy executor
  cannot consume v1/v011/v002 inputs on the active v015 route.
- Existing checkpoint formats and old evaluator execution were not modified.
  The old evaluator remains explicitly legacy/incompatible until G5-S2B.

Fresh verification:

- `frontres_v015_policy_quality_identity_contract.py` exited 0 for strict
  identity, layout, fingerprints, tamper, route-swap, partial-transaction, v1,
  v002, and active-route legacy-bypass rejection.
- `frontres_hsl_v007_s1_contract.py` exited 0, preserving proposal-only HSL
  save/reload/pre-mutation behavior.
- `frontres_v015_checkpoint_resume_contract.py` exited 0, preserving Stage-3
  v2 save/resume, committed receipt, fresh inference, and tamper rejection.
- `frontres_policy_quality_manifest_contract.py` and
  `frontres_policy_quality_entrypoint_contract.py` exited 0, proving the legacy
  classes remain isolated rather than silently reinterpreted as v015.

Acceptance and remaining boundary:

- G5-S2A is complete at deterministic S1/S3. Identity validation is ready for
  the future held-out owner, but no checkpoint was loaded into an evaluator and
  no quality rollout or report was produced.
- G5-S2B is the next separately user-gated step. It must replace, not adapt,
  the legacy quartet/Clean/repeated-action execution route.

## E-FI-50: G5-S2B Repair/Noisy One-Action-K Held-Out Quality

Date: 2026-07-21
Tier: deterministic S1/S2 semantic CPU evaluator and atomic-report
connectivity; no fresh runner, simulator, training, optimizer, PPO, sampler,
checkpoint mutation, or live run

Fail-first evidence:

- The strict G5-S2A request had no v015 held-out owner bundle or execution
  function. Active v015 therefore stopped before the legacy evaluator.
- The first positive implementation review found that a naked one-action
  carrier could be labeled HSL/policy without proving its checkpoint or
  manifest-item origin. The route carrier was strengthened before closeout.

Implemented owner path:

```text
strict v015 manifest + HSL-v1/Stage3-v015 checkpoint receipts
-> fixed item order
-> zero / HSL / policy route+checkpoint+comparison carrier
-> validated Repair/Noisy FrontRESV015OneActionKEvidence
-> pair_frontres_v015_gain_facts
-> compute_intent_physics_local_repair_gain (FRS-GAIN-v003)
-> state/identity checks
-> atomic frontres-v015-heldout-quality-report-v1 JSON
```

Facts established:

- `frontres_policy_quality_eval.py::run_frontres_v015_policy_quality_heldout_eval()`
  is the unique deterministic execution owner. The formal v015 entry uses it
  only when an exact active-owner bundle is installed; legacy executor
  attributes are ignored.
- Input evidence must be `[B,928]` policy observations, `[B,289]` critic
  observations, `[B,6]` Repair actions, and `2B` ordered Repair/Noisy role rows.
  Its existing schema enforces exactly one actor forward, zero later FEMR
  actions, deployment q29 intent, and `[K,2B,65]` frozen-GMT continuation.
- Every route carrier binds the manifest item comparison signature and the
  exact expected checkpoint SHA. Zero has no checkpoint; HSL and policy cannot
  exchange identities.
- Zero/HSL/policy must share scenario ID, noisy hash, `x_t`, roles, intent,
  continuation, valid mask, and K. Mixed route identity rejects before report
  production.
- Gain is computed only through `FRS-GAIN-v003` from the active paired facts.
  No return, advantage, priority, sampler, PPO, legacy v002, or HSL target path
  is called. Unavailable component values serialize as `null`, never zero.
- A training-state signature is checked before and after every route. The
  semantic mutation fixture rejects and leaves no partial report. Success uses
  a temporary file followed by atomic replace.
- The semantic fixture uses hand-checkable q29/survival/action values and real
  active evidence validation and v003 math. It does not prove real checkpoint
  actor inference, Isaac reset timing, GMT physics, or policy quality.

Fresh verification:

- `frontres_v015_policy_quality_heldout_contract.py` exited 0 for route order,
  exact shapes, two roles, same scenario/K, one action, v003 values,
  checkpoint/item binding, state isolation, mixed-identity rejection, and
  atomic success/failure behavior.
- `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_gain_consumer_contract.py`, and
  `frontres_intent_physics_gain_contract.py` exited 0.
- Legacy `frontres_policy_quality_eval_contract.py`, entrypoint contract, and
  executor contract exited 0, confirming isolation rather than reinterpretation.

Acceptance and remaining boundary:

- G5-S2B is complete at deterministic S1/S2. The evaluator semantics and
  report are contract-confirmed, not runtime-confirmed.
- G5-S3 is the next user-gated step: actual committed save, independent fresh
  runner, strict actor/normalizer reload, proposal equality, and the same
  atomic held-out report.

## E-FI-51: G5-S3 Actual Save To Fresh Reload To Atomic Quality Report

Date: 2026-07-21
Tier: deterministic S2/S3 semantic CPU persistence and evaluator connectivity;
no simulator, training loop, live run, checkpoint-format change, or physical
policy-quality claim

Fail-first and owner correction:

- The first mixed snapshot assertion failed because Python dictionary equality
  attempted an ambiguous tensor boolean; the contract was corrected to compare
  q29 tensors exactly and immutable metadata field by field.
- The next run reached a real persistence mismatch: the existing Stage3 test
  normalizer stored `count` as a plain tensor attribute, so real
  `save_runner()` omitted the strict schema field expected by the policy-quality
  inspector. The semantic fixture now registers `count` as a buffer, matching
  the real empirical-normalizer and existing HSL fixture contract. No production
  checkpoint format or inspector rule changed.

Verified path:

```text
semantic 2-Segment x M ordinary transaction
-> grouped exact-one optimizer update and committed receipt
-> actual Stage3-v015 save_runner()
-> independently initialized fresh inference runner
-> strict load_runner(load_optimizer=False)
-> exact deployment q29 + 928/158/770 + prefix normalizer + 6D proposal

strict proposal-only HSL source
-> actual HSL-v1 save_runner()
-> independently initialized strict HSL fresh runner
-> exact normalized 158D input + 6D proposal

strict manifest + exact HSL/Stage3 file identities
-> G5-S2B zero/HSL/policy Repair/Noisy one-action-K evaluator
-> atomic frontres-v015-heldout-quality-report-v1
```

Facts established:

- The ordinary transaction collected four policy attempts from two sources,
  committed with `optimizer_step_delta=1`, and the actual save payload carried
  the matching committed receipt.
- The independent Stage3 fresh runner started from different actor/normalizer
  state, then strict reload reproduced the exact combined `[2,928]`, FEMR
  `[2,158]`, GMT suffix `[2,770]`, deployment q29 offsets `(1,2)`, normalized
  input, and bounded 6D proposal bit for bit.
- The independently saved/reloaded HSL-v1 route reproduced its exact normalized
  158D input and 6D proposal. HSL and Stage3 consumed the same raw artifact and
  deployment q29 combined observation; their learned proposals remained route
  specific.
- Strict quality request inspection accepted the actual HSL-v1 and committed
  Stage3-v015-v2 files, bound their exact SHA-256 identities and the immutable
  manifest identity, and rejected no schema through fallback or padding.
- The held-out report contains the actual fresh HSL and Stage3 proposal values,
  preserves route/checkpoint/item identities, uses the existing v003 evaluator,
  and is atomically identical to the JSON artifact.
- Full optimizer state including Adam moments and step count, sampler state,
  committed transaction/receipt state, warmup flag, actor state, and 158D
  prefix statistics produced the same signature before and after evaluation.
  No evaluator feedback reached training state.
- The one-action-K Repair/Noisy evidence remains a semantic CPU fixture. This
  proves connectivity and isolation, not IsaacLab execution, GMT physics, or
  learned policy quality.

Fresh verification:

- `frontres_v015_policy_quality_save_reload_contract.py` exited 0 with the
  `G5-S3/T-commit/.../T-atomic-report/T-isolation PASS` sentinel.
- `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_v015_policy_quality_identity_contract.py`,
  `frontres_v015_policy_quality_heldout_contract.py`, and
  `frontres_hsl_v007_s1_contract.py` all exited 0 after the fixture correction.

Acceptance and remaining boundary:

- G5-S3 is complete at deterministic S2/S3. No production formula, checkpoint
  schema, simulator, training, or live route changed.
- G5-S4 remains user-gated. Before one bounded live transaction, its exact
  command, artifacts, telemetry, and numeric action/Gain/harm acceptance
  thresholds must be frozen in a separate S0 preflight.

## E-FI-52: G5-S4-S0 Readiness Audit And Repair Plan Rebase

Date: 2026-07-21
Tier: read-only S0 code/plan audit and documentation rebase; no code, active
contract, Concept Figure, Architecture, test, checkpoint IO, simulator,
training, or live run

Code-confirmed chain:

```text
Stage3 preset
-> explicit HSL-v1 initializer + q29 offsets
-> 8 env Repair/Noisy local transaction
-> 2 Segments x 2 policy attempts
-> grouped exact-one update + committed receipt
-> actual model_1.pt save
-> [missing formal independent fresh runner]
-> [missing formal v015 held-out owner bundle/manifest]
-> [missing live atomic quality report]
```

Confirmed blockers:

1. `run/run_frontres_stage3_segment_hrl.sh` passes the HSL artifact through
   `--resume_student_checkpoint`; `cli_args.py` converts this to `resume=True`.
   Ordinary v015 instead requires explicit
   `--frontres_v015_hsl_initializer_checkpoint`, offsets `(1,2)`, and rejects
   combining the initializer with resume. The launcher does not currently
   express that contract.
2. `run_frontres_v015_formal_transaction_update()` already returns immutable
   `v003_action_gain_harm_reports`, but
   `frontres_segment_live_training.py::_v015_formal_update_summary()` projects
   only PPO/count/mass values. The action rows, valid mask, v003 components,
   sign fractions, scenario/noisy hash, and provenance are therefore not
   available in the bounded live log.
3. `frontres_policy_quality_eval.py` requires an installed
   `FrontRESV015PolicyQualityOwnerBundle` on the formal runner. Repository search
   found installation only in deterministic contracts, not in `train.py` or
   `OnPolicyRunner`; formal evaluation stops before route collection.
4. The repository contains no checked-in
   `frontres-v015-policy-quality-manifest-v1`. Existing manifests are legacy
   v011/v002. No formal owner connects an actually saved committed checkpoint
   to an independent fresh runner and the existing atomic v015 report.

Artifact facts:

- The prior bounded HSL log names
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-07-21_17-06-12_G2_S4_BOUND_HSL/model_warmup.pt`
  and records strict HSL-v1/fresh-reload success. Current server existence was
  not checked in this S0.
- The intended motion and cache roots remain
  `/hdd1/cyx/AMASS_G1NPZ_Final` and `/hdd1/cyx/AMASS_G1Segment`; their current
  server existence is unconfirmed.
- With eight envs the formal local owner requires four Repair rows and four
  Noisy rows, with at least two distinct Segments and two attempts per Segment.
- A committed one-iteration run saves
  `g1_flat_frontres_stage3_segment_hrl/<timestamp>_<run_name>/model_1.pt` only
  after `optimizer_step_delta=1` and a matching receipt.

Candidate numeric gate, pending explicit user confirmation:

- transaction: two Segments, four attempts, `valid_rows=4/4`, one update, and
  `optimizer_step_delta=1`;
- action: all 24 values finite, at least two row L2 norms above `1e-4`, and at
  least one cross-row dimension std above `1e-5`;
- saturation: row fraction above `0.285` position or `0.38` rotation no more
  than `0.25`;
- trained quality: `gain_total_mean > 0`, positive fraction at least `0.50`
  and no lower than HSL, negative fraction at most `0.25` and no higher than
  HSL;
- harmful-repair fraction is not a new variable: it is the existing
  `gain_total < 0` fraction unless the user explicitly changes this boundary;
- reload: normalized 158D input equal and 6D proposal within `rtol=1e-5`,
  `atol=1e-6`;
- committed receipt, manifest SHA, HSL/policy route SHA, and atomic JSON output
  must match with no partial artifact.

Plan effect:

- G5-S4-S0 is complete with stop conditions. It does not authorize the live
  command shown during the audit.
- G5-S4 is split into S1A explicit training launch/live telemetry, S1B formal
  held-out/fresh-report dispatch, S2 final read-only preflight, and S4 one
  user-confirmed bounded live run.
- G5-S4-S1A is the only ready code step. G6/G7 remain blocked.

## E-FI-53: G5-S4-S1A Explicit Training Launch And Transaction Telemetry

Date: 2026-07-21
Tier: deterministic S1/S2 launcher, diagnostics, and sealed-transaction
contracts; no checkpoint IO, simulator, training, held-out evaluation, fresh
runner, or live run

Fail-first evidence:

- The launcher contract failed because the Stage3 command did not contain the
  explicit HSL-v1 initializer or future offsets and still used student-resume
  semantics.
- The transaction-route contract failed because the formal update summary
  discarded `v003_action_gain_harm_reports` and exposed no sealed transaction
  telemetry.

Implemented route:

```text
bounded launcher
-> explicit HSL-v1 initializer + offsets (1,2)
-> 8 envs + 1 iteration + 1 update + checkpoint interval 1
-> sealed 2-Segment x 2-attempt grouped transaction
-> exact-one optimizer update
-> read-only v003 action/Gain/identity telemetry JSON
```

Facts established:

- `run_frontres_stage3_segment_hrl.sh` passes the strict initializer and
  offsets directly. Bounded mode requires the exact 8/1/1 dimensions and adds
  checkpoint interval one plus the existing formal runtime audit.
- Both Stage3 launcher layers default legacy periodic evaluation off and reject
  resume, student-checkpoint, full-resume, and periodic-evaluation overrides.
- `_v015_formal_update_summary()` consumes only the already sealed immutable
  v003 reports after the transaction update. It does not call a Gain owner or
  alter return, priority, PPO, sampler, optimizer, or storage state.
- The emitted telemetry preserves four `[6]` policy-action rows, valid mask,
  intent/physics Gain, repair cost, total Gain, positive/negative fractions,
  scenario/noisy/x_t identity, K, q29/Gain provenance, grouped mass,
  update-count, and optimizer-step delta.
- Missing reports, mixed transaction/provenance, feedback-bearing reports,
  invalid identity, non-finite valid values, and zero-filled invalid component
  rows reject fail-closed. Unavailable invalid-row components serialize as
  `null`, never as a fabricated zero.

Fresh verification:

- `frontres_segment_stage3_launch_command_contract.py` exited 0, including the
  strict bounded command and forbidden-override cases.
- `frontres_stage_entrypoint_contract.py` exited 0 for the updated launcher and
  v015 preset authority.
- `frontres_segment_diagnostics_contract.py` exited 0 for the immutable v003
  diagnostic report contract.
- `frontres_v015_transaction_route_contract.py` exited 0 for four attempts,
  exact-one update, JSON telemetry, missing-field rejection, feedback
  rejection, and no legacy route.
- Python compilation, both launcher `bash -n` checks, and `git diff --check`
  exited 0.

Acceptance and remaining boundary:

- G5-S4-S1A is complete at deterministic S1/S2. This proves command and
  telemetry contracts only; it does not prove simulator execution, learned
  policy quality, or live logging.
- G5-S4-S1B is now the next separately user-gated step. No manifest, formal
  quality-owner bundle, held-out evaluator dispatch, fresh runner, simulator,
  training, or live execution was entered here.

## E-FI-54: G5-S4-S1B Formal Held-Out Owner And Fresh-Report Dispatch

Date: 2026-07-21
Tier: deterministic S1/S2/S3 manifest, formal-owner, persistence, and atomic
report connectivity; no simulator, training loop, live run, checkpoint-format
change, or physical policy-quality claim

Fail-first evidence:

- `frontres_v015_policy_quality_heldout_contract.py` failed because no formal
  v015 owner factory existed.
- `frontres_v015_policy_quality_save_reload_contract.py` failed because no
  strict temporary checkpoint-route actor owner existed.
- The first route-context fixture exposed two contract prerequisites: the
  Stage3 runner must carry the same frozen-GMT identity as HSL-v1, and HSL and
  Stage3 must expose the same residual-actor schema. The fixture was corrected
  to represent those accepted invariants; production validation was not
  weakened.

Implemented formal path:

```text
fixed frontres-v015-policy-quality-manifest-v1 item
-> exact Stage1-index motion/frame/K resolution
-> one seeded immutable local scenario shared by 4 Repair + 4 Noisy rows
-> Clean x_t reset before zero/HSL/policy route
-> strict temporary HSL-v1 or committed Stage3-v015 actor/prefix install
-> one deterministic 6D proposal, then frozen FEMR and Clean-C GMT K execution
-> existing FRS-GAIN-v003 held-out owner
-> atomic frontres-v015-heldout-quality-report-v1
-> exact actor/normalizer/training-state restoration
```

Facts established:

- `frontres_v015_policy_quality_heldout_v1.json` fixes 16 item identities over
  eight held-out motions and seeds 42/43, with local_rp, K=8, q29 offsets
  `(1,2)`, 928/158/770, full-6D Delta SE(3), and v015/v007/v003/v003 identity.
- The manifest materializer does not call the training sampler or curriculum.
  One item resolves to one loaded Stage1 index row; four Repair attempts share
  one source/scenario/hash/x_t. Seeded materialization restores CPU/CUDA RNG
  state and repeated construction reproduces the same identity.
- The formal evaluator installs `FrontRESV015PolicyQualityOwnerBundle` only
  after strict request inspection. Legacy executor attributes are not read on
  the active v015 route.
- Zero/HSL/policy each execute exactly one deterministic 6D proposal at t.
  No later FEMR action is admitted; K remains frozen-GMT executable evidence.
- HSL-v1 and Stage3-v015 checkpoints are SHA-bound before mutation. Only the
  residual actor, 6D distribution, and 158D prefix statistics are installed;
  the source actor, distribution, prefix normalizer/statistics, and mode are
  restored in `finally`.
- The evaluator hashes actor, critic, optimizer, prefix normalization,
  sampler, transaction/receipt, warmup, and iteration state around every route.
  Missing/mixed identity, partial checkpoint, fallback, padding, scenario
  drift, repeated action, or mutation rejects before an atomic report remains.

Fresh verification:

- `frontres_local_scenario_kernel_contract.py` exited 0, including fixed-item
  4-attempt identity, hash equality, and RNG restoration.
- `frontres_v015_one_action_k_contract.py` exited 0, including deterministic
  zero/policy proposal, one actor forward, and zero later FEMR actions.
- `frontres_v015_policy_quality_identity_contract.py` exited 0 for the fixed
  16-item manifest and strict checkpoint/manifest rejection cases.
- `frontres_v015_policy_quality_heldout_contract.py` exited 0 for formal
  auto-install, one prepared batch reused by all routes, route/checkpoint
  identity, training-state isolation, and atomic report.
- `frontres_v015_policy_quality_save_reload_contract.py` exited 0 for actual
  committed save, independent fresh reload, exact 928/158/770 q29-normalized
  proposal equality, strict temporary route installation/restoration, and the
  atomic v003 report.
- `frontres_v015_checkpoint_resume_contract.py` and
  `frontres_policy_quality_entrypoint_contract.py` exited 0, preserving strict
  persistence and legacy-entry isolation.
- Python compilation and `git diff --check` exited 0.

Acceptance and remaining boundary:

- G5-S4-S1B is complete at deterministic S1/S2/S3. The formal route is
  contract-confirmed, not runtime-confirmed.
- G5-S4-S2 is the next separately user-gated read-only preflight. Real server
  artifact existence, exact command/sentinels, numeric thresholds, simulator
  execution, training, and live policy quality remain unconfirmed.

## E-FI-55: G5-S4 Bounded Train And Held-Out Index/K Resolver Repair

Date: 2026-07-22
Tier: one bounded S4 training transaction plus deterministic S1/S2 resolver,
held-out, persistence, and atomic-report contracts; held-out live quality did
not execute

Runtime evidence:

- `v015_g5_s4_train_gpu3.log` records strict HSL-v1 actor-only initialization,
  928/158/770 observation authority, two Segment sources, four valid policy
  attempts, equal grouped mass, `update_count=1`, `optimizer_step_delta=1`, a
  matching committed receipt, and successful `model_1.pt` persistence.
- The same log records a Stage-1 index with cache `horizon_k=4`, while the
  selected transaction independently carries executable budget K8 and
  materializes K8 local-scenario evidence.
- `log.txt` records that the first held-out launch stopped before evaluation:
  same-command shell expansion supplied an empty positional HSL path, so
  `run_stage3.sh` selected its legacy default checkpoint and the strict HSL-v1
  inspector rejected the missing identity. The new HSL-v1 artifact was not
  shown to be corrupt.
- Training telemetry is connectivity evidence, not held-out acceptance:
  actions were finite/non-collapsed and unsaturated, but the four training rows
  had positive Gain fraction `0.25`, harm fraction `0.75`, and zero physics
  Gain. No policy-quality conclusion is claimed.

Fail-first evidence:

- The focused local-scenario contract reproduced the server incompatibility:
  a K4 cache spec plus a K8 manifest item produced `matches=0` because the
  held-out resolver incorrectly included cache `spec.horizon_k` in x_t
  identity.

Implemented correction:

```text
Stage-1 cache identity: motion_id + start_frame
-> exactly one Segment/x_t row
-> manifest effective_horizon_k remains K8
-> local-scenario materializer returns Clean continuation [K8,65]
-> one-action-K held-out evaluator
```

- `prepare_frontres_v015_policy_quality_item_batch()` no longer equates the
  cache index construction window with executable-evidence K.
- Zero or duplicate `(motion_id,start_frame)` identities still reject. There is
  no fallback, clamp, manifest rewrite, or cache rebuild.

Fresh deterministic verification:

- `frontres_local_scenario_kernel_contract.py` exited 0 with
  `T-heldout-manifest`: K4 index identity -> K8 budget/continuation, repeated
  construction retains identity, and duplicate motion/start rejects.
- `frontres_v015_policy_quality_heldout_contract.py` and
  `frontres_v015_policy_quality_identity_contract.py` exited 0.
- `frontres_v015_policy_quality_save_reload_contract.py` exited 0 through an
  actual committed save, independent fresh reload, strict checkpoint identity,
  exact proposal equality, and atomic held-out report.
- Python compilation exited 0.

Acceptance and remaining boundary:

- The resolver defect is contract-confirmed fixed. No Concept Figure or active
  method semantics changed: cache K identifies neither actor context H nor
  executable-evidence K.
- The bounded training half is runtime-confirmed. The corrected fresh held-out
  quality command remains live-unconfirmed and must not repeat training.
- Numeric Gain/harm acceptance thresholds remain a human decision; G5-S4 and
  G5 are not complete until the atomic live quality report is inspected.

## E-FI-56: G5-S4-S1D Quality Inference-Mode Isolation

Date: 2026-07-22
Tier: deterministic S1/S2 held-out evaluator, persistence, and observation
contracts; no simulator, training, checkpoint-format change, or live rerun

Runtime symptom and root cause:

- The corrected server quality command passed strict HSL/policy identity,
  manifest resolution, and the K4-index/K8-execution resolver, then stopped at
  `v015 quality evaluation mutated training state` after the first route.
- Code order showed that every route called `_read_live_observations()` before
  entering the temporary checkpoint actor context. On a fresh live runner, the
  158D empirical prefix and privileged normalizers remained in training mode;
  the zero route therefore updated running state before the post-route
  signature check. The guard correctly detected this write.

Fail-first regression:

- The focused held-out contract installed training-mode semantic normalizers
  whose forward pass increments a registered running-state buffer. Without an
  outer inference guard, the zero route reproduced the same mutation error.

Implemented isolation:

```text
held-out evaluator entry
-> snapshot every policy/prefix/GMT/privileged/teacher submodule mode
-> set every captured module to inference mode
-> zero -> HSL -> policy observation/action/K routes
-> unchanged actor/critic/optimizer/sampler/transaction/warmup/normalizer signatures
-> atomic report or exception
-> restore each original submodule mode exactly
```

- The evaluator does not call recursive `runner.train_mode()` during restore;
  it restores individual module flags so an already-frozen GMT/dropout child
  is not accidentally switched to training.
- The training-state signature now includes 158D prefix, GMT, privileged, and
  teacher normalizer state dictionaries. The mutation guard was strengthened,
  not removed or bypassed.

Fresh verification:

- `frontres_v015_policy_quality_heldout_contract.py` exited 0 with zero running-
  state writes and exact mixed-mode restoration after success and intentional
  mutation failure.
- `frontres_v015_policy_quality_save_reload_contract.py` exited 0 through real
  save, independent fresh reload, exact proposal identity, and atomic report.
- `frontres_v015_unmocked_observation_connectivity_contract.py`,
  `frontres_v015_observation_authority_contract.py`, and
  `frontres_future_intent_actor_context_contract.py` exited 0.
- Python compilation exited 0.

Acceptance and remaining boundary:

- S1D is contract-confirmed. No active contract or Concept Figure semantics
  changed; this is an inference lifecycle defect under the existing zero-write
  evaluation boundary.
- The same corrected fresh held-out quality command remains live-unconfirmed.
  Bounded training must not be repeated. G5-S4 stays partial until the atomic
  16-item quality JSON is produced and inspected against human-confirmed gates.

## E-FI-57: G5-S4-S1E Manifest Item Lifecycle Isolation

Date: 2026-07-22
Tier: deterministic S1/S2 evaluator lifecycle and persistence contracts; no
simulator, training, checkpoint-format change, or live rerun

Runtime symptom and root cause:

- The next server quality attempt completed repeated resets for the first
  manifest identity, then failed at
  `v015 local scenario reset attempted to mutate an active sealed scenario`.
- The live traceback reached `collect_one_action_k -> reset ->
  set_frontres_local_scenario`. Code inspection confirmed that K-execution end
  intentionally retained the sealed scenario for matched attempts, while the
  held-out evaluator had no item-close callback after its zero/HSL/policy
  counterfactual set. Item two therefore attempted to install a new seed/hash
  over item one's still-active command carrier.

Fail-first regression:

- A two-item semantic manifest required the event order `zero -> HSL -> policy
  -> close` for each item. The old owner bundle rejected the required
  `close_item` callback, reproducing the absent lifecycle boundary before the
  implementation changed.

Implemented isolation:

```text
manifest item
-> materialize and seal one immutable scenario
-> zero reset/action/K
-> same sealed scenario HSL reset/action/K
-> same sealed scenario policy reset/action/K
-> command.clear_frontres_local_scenario()
-> close immutable batch lifecycle
-> clear evaluator transient batch/sample pointers
-> next manifest item may install a new scenario/hash
```

- The evaluator invokes item close in `finally`, so a route exception cannot
  leak an active command or batch lifecycle.
- `commands.py` remains unchanged and continues to reject replacement of an
  active sealed scenario. No clear occurs between the three matched routes.
- A post-close training-state signature rejects any lifecycle callback that
  feeds back into policy, optimizer, sampler, transaction, warmup, or
  normalizer state.

Fresh verification:

- `frontres_v015_policy_quality_heldout_contract.py` exited 0. It proves exact
  two-item route/close order, formal command and batch close, exception close,
  and rejection of close-side training-state mutation.
- `frontres_v015_policy_quality_save_reload_contract.py` exited 0 through real
  save, independent fresh reload, exact proposal identity, and atomic report.
- Python compilation for the evaluator and both focused contracts exited 0.

Acceptance and remaining boundary:

- S1E is contract-confirmed. This is an evaluator lifecycle defect, not a
  method/Concept Figure or command sealed-carrier semantic change.
- The original live symptom is not yet runtime-confirmed absent. Do not repeat
  bounded training; rerun only the corrected held-out quality command after
  server sync, and stop on any active-carrier replacement or missing atomic
  report.

## E-FI-58: G5-Q0 Quality-Gap Plan Rebase

Date: 2026-07-22
Tier: S0 governance and prior bounded-runtime evidence classification; no code,
active contract, Concept Figure, test, checkpoint IO, simulator, training, or
live run in this step

Raw evidence recalled from the completed bounded server execution:

- `/hdd1/cyx/FEMR/v015_g5_s4_quality_gpu3.log` completed all 48 route resets
  for 16 manifest items x zero/HSL/policy, without traceback, scenario-lifecycle
  replacement, training-state mutation, later FEMR action, or optimizer update.
- `/hdd1/cyx/FEMR/v015_g5_s4_policy_quality_gpu3.json` was emitted atomically
  with SHA-256
  `98eaf4c2932df12e291177186f4c3308bf6849730e70a990d5a3b8326e29cf1b`.
- The report used `frontres-v015-heldout-quality-report-v1` and
  `FRS-GAIN-v003-intent-physics-local-repair`; all 64 policy actions were finite
  and non-collapsed.
- Aggregate policy `gain_total` had mean `0.0117027`, positive fraction
  `0.453125`, and negative/harm fraction `0.546875`. These values are recorded
  as observations, not accepted quality thresholds.

Earliest authority contradictions:

- Zero action produced nonzero `gain_total` with range approximately
  `[-0.162, 0.428]`; zero-route `physics_gain` and `repair_cost` were zero, so
  the variation came from `intent_gain`. The zero route is therefore not yet a
  proven no-op oracle.
- `physics_gain` was identically zero for zero, HSL, and policy routes. The
  bounded report did not supply discriminative survival/ZMP/contact evidence
  required to judge paired executability.
- The evaluator proved scenario/hash/x_t and lifecycle identity, but did not
  prove equality of complete role-aligned root/joint pose and velocity,
  command/cache state, and RNG state before each route observation/action.
- Therefore the observed policy fractions cannot distinguish policy failure
  from evaluation-state or Gain-evidence mismatch. Runtime connectivity passed;
  evaluation authority is blocked; policy efficacy remains unconfirmed.

Governance decisions:

- Preserve completion of G0--G4 and G5-S1 through G5-S4-S1E. Do not erase the
  exact-one training, save/fresh-reload, lifecycle, or atomic-report evidence.
- Rebase only the policy-quality branch as G5-Q1 through G5-Q6: full dynamic-
  state identity; zero-action no-op oracle; Physics survival/ZMP/contact;
  matched rerun; Gain-to-advantage-to-update causality; checkpoint trajectory
  and long-training admission.
- Mark all v011/v002 and Q-E evidence in the Quality Audit Atlas as historical
  and incompatible with current v015 acceptance.
- Do not change active contracts or the Concept Figure. The contradiction is
  currently an evaluation-authority gap under accepted v015 semantics, not a
  confirmed method change.

Next:

- G5-Q1 only, after explicit user authorization. Add a read-only probe after
  every active v015 canonical reset and before `_read_live_observations()`;
  prove full dynamic-state/cache/RNG equality across zero/HSL/policy in S1/S2,
  then stop before any live rerun.

## E-FI-59: G5-Q1 Full Dynamic-State Identity S1/S2

Date: 2026-07-22
Tier: deterministic S1 core-parameter and S2 held-out connectivity evidence;
no simulator, training, full quality rerun, live sentinel, reset modification,
or active-contract/Concept-Figure change

Fail-first evidence:

- The focused held-out contract initially failed because the active v015 route
  exposed no `capture_frontres_v015_policy_quality_dynamic_state_identity`
  owner and route evidence had no complete state identity.
- The first connector implementation then failed because report serialization
  incorrectly read the identity from inner one-action-K evidence rather than
  the route wrapper. Passing the wrapper identity explicitly fixed that owner
  boundary without weakening validation.

Implemented core path:

```text
canonical Clean x_t reset
-> read-only B=8 state capture
-> 12 field-level SHA-256 identities
-> route evidence before observation/action
-> exact zero/HSL/policy equality check
-> atomic report or fail-closed field-name diagnostic
```

Captured fields:

- `root_state_w` including root pose and linear/angular velocity;
- 29DoF `joint_pos` and `joint_vel`;
- env origins and episode length;
- command cursor/perturbation/correction cache and all per-env perturber tensors;
- Python, NumPy, Torch CPU, and all CUDA RNG states;
- command-owned current artifact, q29 intent, Clean continuation, K/length,
  scenario/hash/x_t, role layout, and deployment provenance.

Fresh deterministic evidence:

- `frontres_v015_policy_quality_heldout_contract.py` exited 0. Its semantic
  B=8 fixture proves repeat capture equality, exact 4 Repair + 4 Noisy order,
  unchanged source tensors, row-permutation sensitivity, role-mix rejection,
  route-state mismatch rejection, no partial report, and lifecycle cleanup.
- `frontres_v015_policy_quality_save_reload_contract.py` exited 0, preserving
  actual save/fresh-reload and atomic quality connectivity with the new required
  route identity. This is a regression check, not new checkpoint evidence.
- Python compilation of the owner and both focused contracts exited 0.

Confirmed:

- The unique active owner is
  `frontres_policy_quality_eval.py::build_frontres_v015_policy_quality_owner_bundle`.
  It captures identity immediately after `_apply_current_segment_reset()` and
  before `_read_live_observations()`.
- The probe does not restore or mutate env, RNG, optimizer, sampler,
  transaction, warmup, normalizer, or checkpoint state.
- Missing schema, non-4+4 roles, row misalignment, changed field hash, mixed
  comparison signature, and missing command scenario snapshot fail closed.

Open boundary:

- S1/S2 prove the owner and comparison semantics on a meaningful CPU fixture;
  they do not prove that the real IsaacLab reset reproduces all 12 hashes.
- G5-Q1 remains partial until one separately authorized bounded S4 identity
  sentinel passes. Q2 and all further quality/training work remain blocked.

Next:

- Execute G5-Q1-S4 only after user confirmation. Use one matched manifest item,
zero/HSL/policy routes, no optimizer update, and stop at the first differing
field or missing identity.

## E-FI-60: G5-Q1-S4 Live Identity Sentinel Fail-Closed

Date: 2026-07-22

Tier: S4 one-item live sentinel negative evidence; no training, PPO entry,
optimizer update, full 16-item evaluation, Q2 work, or atomic quality report

Raw evidence:

- Server log: `/hdd1/cyx/FEMR/v015_g5_q1_s4_identity_gpu3.log`.
- SHA-256: `c74d3b264f879a8df80558d1d20c131a148188f9a4625c5d33bb1f4672704534`.
- Size: `47480` bytes; mtime: `2026-07-22 02:35:16.417000961 +0800`.
- Runtime failure: `route=hsl differing_fields=('cuda_rng_state',)`.
- Two canonical Segment resets ran. `Entering PPO loop` count was `0`,
optimizer-update sentinel count was `0`, and
`v015_g5_q1_s4_identity_gpu3.json` was absent.

Facts:

- Zero and HSL route-start identities matched for root state, joint position,
joint velocity, env origins, episode length, command state, perturber state,
Python RNG, NumPy RNG, Torch CPU RNG, and sealed local scenario.
- CUDA RNG was the only differing field. The second canonical reset restored
Clean x_t and reused the sealed scenario but did not reproduce the CUDA RNG
state consumed after the zero route.
- The fail-closed comparison stopped before the policy route and prevented an
atomic report. No PPO or optimizer update was reached.

Decision:

- Keep CUDA RNG in the required identity. Do not silence, omit, or tolerate
the mismatch.
- Freeze G5-Q1-S4A as a separate implementation step: capture one complete
route-start snapshot after the first canonical reset and restore the same
physical, command, perturber, Python/NumPy/Torch/CUDA RNG, and sealed
local-scenario identity before zero/HSL/policy.
- Q2--Q6, G6/G7, additional training, full quality rerun, and deployment
composition remain blocked.

Open boundary:

- Deterministic S1/S2 contracts have not yet proven route-order invariance,
CUDA RNG restoration after deliberate consumption, exception cleanup, or zero
training-state feedback for the proposed restore path.
- A replacement one-item S4 sentinel is not authorized until those contracts
pass and requires separate user confirmation.

Next:

- Implement G5-Q1-S4A only after explicit user authorization. Do not enter Q2
or launch simulator/training/live execution in that implementation step.

## E-FI-61: G5-Q1-S4A Route-Start Snapshot Restore S1/S2

Date: 2026-07-22

Tier: deterministic S1 core-parameter and S2 held-out lifecycle evidence; no
simulator, training, live sentinel, Q2, Gain/PPO/HSL/checkpoint, active-contract,
or Concept-Figure change

Fail-first evidence:

- The focused held-out contract deliberately consumed a modeled CUDA RNG value
after each route. Before the implementation change it reproduced the live
failure at HSL with `differing_fields=('cuda_rng_state',)`.

Implementation:

- `build_frontres_v015_policy_quality_owner_bundle()` now owns one
`FrontRESPolicyQualityScoringState` plus expected dynamic identity per manifest
item.
- The owner materializes and resets a sealed item once, captures one complete
post-reset route-start, and calls the existing strict state restore before each
zero/HSL/policy observation/action path.
- A fresh 12-field dynamic identity is compared with the expected identity
before `_read_live_observations()`. Field or role drift still fails closed.
- Item close removes the route-start snapshot before closing the command-owned
carrier and immutable batch. It does not reinstall, mutate, or resample the
sealed local scenario.

Deterministic evidence:

- `frontres_v015_policy_quality_heldout_contract.py`: PASS. One reset and one
snapshot feed three restores; deliberate RNG consumption produces identical
route hashes; `policy -> zero -> hsl` permutation remains identical; an HSL
exception closes the item, emits no report, and a retry captures a fresh state.
- `frontres_v015_policy_quality_save_reload_contract.py`: PASS, including the
existing strict actual-save/fresh-runner/atomic-report connectivity fixture.
- `frontres_policy_quality_state_contract.py`: PASS; the complete scoring state
capture/restore hash remains closed offline.
- `python -m py_compile` for the modified owner and held-out contract: PASS.
- `git diff --check`: PASS.

Confirmed:

- The implementation fixes the owner-local cause observed at `E-FI-60`
without deleting CUDA RNG identity or weakening mismatch rejection.
- Route order, deliberate RNG consumption, exception cleanup, retry lifecycle,
sealed-scenario identity, and training-state zero-write are contract-confirmed.

Open boundary:

- CPU fixtures cannot prove IsaacLab simulator write-back or real CUDA RNG
restoration. G5-Q1 remains partial until one separately authorized replacement
one-item S4 sentinel matches all zero/HSL/policy field hashes.
- Q2--Q6, full quality rerun, additional training, G6/G7, and deployment
composition remain blocked.

Next:

- Perform a read-only replacement G5-Q1-S4 artifact/command preflight only
after explicit user authorization; report the exact command and stop before
launching the live sentinel.

## E-FI-62: One-Shot HRL Engineering Plan Rebase

Date: 2026-07-22

Tier: user-confirmed execution-governance and current-plan rebase; no training
source, active contract, Concept Figure, checkpoint, simulator, training, or
live execution change

User decision:

- HSL is an already validated auxiliary initializer. Do not repeatedly audit it
when training diagnostics provide no fresh contradiction.
- The engineering terminal outcome is to connect the HRL path, remove visible
bugs, run a bounded training smoke, and judge readiness from decisive metrics.
- Repeated fragmented preflights, per-assertion gates, and user handoffs waste
time and tokens. Internal assertions must remain inside one engineering run.
- `formal-runtime-audit` is reserved for visible official-route/runtime bugs.
`policy-quality-audit` is reserved for poor, no-op, regressing, or contradictory
learned-policy metrics.

Skill evidence:

- Updated `/Users/chengyuxuan/.codex/skills/one-shot-execution/SKILL.md`.
- Final `one-shot-execution` SHA-256:
  `450059d376e91162cff5fcdb85d32ec921d08edad0ee4900de366ccaa77f8ebf`.
- Final `workflow-governance` SHA-256:
  `ea69eb7923b2c3c1fc1a1823acbff71dfe26127902bba1321d53cc1b5280de03`.
- Added the planning compression gate, deletion test, ML/RL engineering closure
unit, internal-assertion rule, validated auxiliary-path freeze, and conditional
routing to formal-runtime/policy-quality audits.
- Updated workflow governance so lifecycle stages and checklist/evidence layers
may close inside one authorized engineering unit; crossing owners, test types,
or offline/live evidence no longer creates an automatic user-visible step.
- Ruby replication of every `quick_validate.py` frontmatter/name/description
check plus required-body checks: PASS. The official
`quick_validate.py` could not run because available Python environments lacked
PyYAML and restricted network access prevented temporary installation; this is
a validator dependency limitation, not a claimed official-validator PASS.

Plan decision:

- Replace the active G5-Q1--Q6 prerequisite chain with G5-E0 One-Shot HRL
Engineering Closure.
- Batch official-route inspection, obvious repair, focused verification,
bounded formal training, log inspection, and routine evidence refresh into one
authorized unit.
- Keep `E-FI-58`--`E-FI-61` as valid historical/current facts and reusable
conditional diagnostics; do not discard their regression protections.
- A clean bounded Stage3 smoke with finite non-degenerate action/update,
plausible Gain/advantage/gradient diagnostics, and a committed checkpoint closes
engineering. G6/G7 then become experiment/composition work.

Next:

- Execute G5-E0 as one unit. Pause only for a true semantic decision, a costly
or destructive action, an unresolved official-route contradiction after one
repair cycle, or abnormal learned-policy metrics that trigger a conditional
audit.

## E-FI-63: Planning Compression Closeout

Date: 2026-07-22

Tier: governance/documentation only; no training source, active contract,
Concept Figure, checkpoint, test, simulator, training, or live-run change

Applied rules:

- The upgraded `one-shot-execution` skill now governs engineering planning as
  well as debugging. Owners, lifecycle stages, evidence tiers, tests, and
  routine document refresh are embedded checks rather than automatic steps.
- The upgraded `workflow-governance` skill permits one authorized engineering
  unit to cross implementation, integration, offline/live verification, and
  documentation stages when no new human decision or high-cost boundary is
  crossed.
- HSL remains frozen as a validated auxiliary initializer unless fresh
  official-run evidence identifies it as the first broken owner.
- `formal-runtime-audit` and `policy-quality-audit` are conditional tools, not
  mandatory prerequisite chains.

Deletion-test result:

- Replaced the chronological 2,086-line engineering plan with one current
  engineering closure unit, `G5-E0`.
- Replaced the micro-step checklist with embedded terminal assertions.
- Replaced the historical task canvas with the current cursor, stop rule, and
  next true boundary.
- Merged the former G6/G7 setup sequence into `X1 Formal Experiments And
  Composition`; only its long/costly experiment boundary requires separate
  authorization.
- Retained `E-FI-58--E-FI-61` and the policy-quality Architecture as reusable
  conditional evidence. Historical evidence was not deleted or rewritten.

Files updated:

- `note/frontres_core/plans/FRS-v015-future-intent-single-action-k-engineering-plan.md`
- `note/frontres_core/checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- `note/frontres_core/plans/FRS-v015-future-intent-single-action-k-task-canvas.md`
- `note/architecture/runtime/05_policy_quality_audit.data.json`
- this append-only evidence ledger

Current cursor:

- `G5-E0` is ready for explicit execution authorization as one complete unit.
- This closeout did not execute G5-E0 and did not authorize X1.

## E-FI-64: G5-E0 Bounded Official HRL Smoke

Date: 2026-07-22

Tier: S4 official Stage3-v015 runtime evidence; 8 envs, one iteration, one
complete transaction, one optimizer update; no long training, multi-seed,
deployment composition, or paper experiment

Raw evidence:

- Log: `/hdd1/cyx/FEMR/v015_g5_e0_train_gpu3.log`.
- Log SHA-256: `895c0938b5cdf20bc5f202a8e572f1651d7aba813370ce20b30bd894ebee11bd`.
- Checkpoint:
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-22_10-35-53_G5_S4_BOUND_V015/model_1.pt`.
- Checkpoint SHA-256:
  `1aafafc541f9abba1562f7cf4c62731d80a41a9edba015f88b7d12db4e24546d`;
  size `935964068` bytes.

Engineering repair inside G5-E0:

- The formal route already computed returns, advantages, and gradients but did
not project them into bounded telemetry. `frontres_segment_live_probe.py` now
captures finite valid-row return statistics and pre/post-clip gradient norms;
`frontres_segment_live_training.py` publishes them with action-scale and PPO
advantage diagnostics without recomputing Gain or feeding training state.
- The focused transaction contract first failed on missing `return_mean`, then
passed after the connector repair. A zero-gradient weak observation fixture
exposed an over-strict diagnostic rejection; the one permitted repair cycle
retained zero-gradient telemetry as an observable quality result while keeping
non-finite gradients fail-closed.
- Launcher, entrypoint, HSL-v1, observation authority, unmocked connectivity,
v003 Gain consumer, transaction, checkpoint/resume, save/reload, pycompile, and
diff checks passed after the repair.

Runtime-confirmed route:

- Strict HSL-v1 restored actor/std/158D prefix only; critic, optimizer, sampler,
and transaction were not restored from HSL.
- Observation authority was `928 = 158 FEMR + 770 frozen GMT`, with finite raw
and normalized tensors.
- Two sealed Segments x two attempts produced four full-6D policy rows and
K=8 evidence. Group mass was equal: motion/Segment `(0.5,0.5)`, attempt
`(0.25,0.25,0.25,0.25)`.
- One grouped transaction produced `update_count=1`,
`optimizer_step_delta=1`, a committed receipt, and one saved Stage3-v015
checkpoint. No Traceback or non-finite sentinel appeared.

Decisive telemetry:

- Action was finite and non-collapsed: absolute mean `0.00408867`, maximum
`0.01544815`, mean L2 `0.01178368`.
- `gain_total=(-0.0947941, 0.0194770, -0.0918342, -0.0853486)`;
positive fraction `0.25`, negative/harm fraction `0.75`.
- `physics_gain=(0,0,0,0)`. Gain was therefore driven by intent improvement
minus repair cost, without discriminative live Physics evidence.
- Return mean was `-0.063125`. Every advantage was negative: mean `-0.912162`,
minimum `-1.365667`, maximum `-0.367361`.
- Gradient was connected, not no-op: `18/18` parameter tensors were nonzero;
pre-clip norm `611.187561`, post-clip norm `0.49999998`.

Verdict:

- Official engineering connectivity, identity, grouped reduction, exact-one
update, diagnostics, and persistence passed.
- G5-E0 does not close because its declared quality stop condition fired. The
sampled attempts were predominantly harmful, all advantages were negative, and
Physics Gain was constant zero. This is not evidence of a disconnected
optimizer; PPO correctly discouraged the sampled actions.
- X1, additional training, multi-seed, deployment composition, and paper
experiments remain blocked. The smallest next action is a read-only
policy-quality audit of survival/ZMP/contact evidence through v003 Physics Gain
and Gain -> return -> advantage causality.

## E-FI-65: G5-E0 Physics Evidence Closure S1/S2

Date: 2026-07-22

Tier: deterministic S1 core-parameter and S2 formal-connectivity evidence; no
simulator, training, live run, checkpoint operation, X1, Gain/PPO/HSL formula,
active-contract, or Concept-Figure change

Root cause confirmed:

- The formal v015 one-action-K collector dynamically captured `done_any` and
  `survival_steps`, but its immutable evidence schema had no ZMP/support or
  contact fields. The v003 input therefore received only success/survival.
- Missing ZMP/contact stayed internal `NaN` and the Gain owner's available-mean
  reduced only the delivered components. Aggregate `physics_gain=0` could not
  prove a complete paired physical tie.
- The bounded telemetry also omitted raw Physics components and per-row policy
  value/raw advantage. PPO scaled advantages were available only as aggregate
  statistics, so critic-to-return causality was not row-auditable.

Implementation:

- `collect_frontres_v015_one_action_k_evidence()` now captures the existing
  paired Physics frame after every frozen-GMT step. It seals Repair/Noisy ZMP,
  height-contact consistency, and a common alive/horizon mask as detached
  `[K,B]` tensors. Missing or malformed evidence fails closed.
- The two-role contact adapter now compares each Repair/Noisy robot row with
  its role-aligned command-owned K continuation; it does not require a third
  scored Clean role or expose Clean to the actor.
- Immutable storage computes masked per-attempt Physics means and passes them
  to the unchanged `compute_intent_physics_local_repair_gain()` input.
- The sealed return/report carrier exposes Repair/Noisy success, raw survival,
  K-normalized survival quality, ZMP, contact, all paired component gains,
  policy value, return, and raw advantage without recomputing Gain.
- PPO math is unchanged. Its result now retains the already-computed prepared
  advantages for diagnostics. Formal telemetry uses an explicit
  `(source_index, trial_index)` permutation to align report rows with the
  transaction accumulator's source-major PPO order before publishing scaled
  advantages.

Fresh deterministic evidence:

- `frontres_v015_one_action_k_contract.py`: PASS; unequal, exact tie, missing
  fail-closed, mixed-K mask, scenario-row permutation, and two-role contact
  reference cases.
- `frontres_v015_gain_consumer_contract.py`: PASS; complete paired Physics
  reaches v003 with no v002 fallback.
- `frontres_segment_diagnostics_contract.py`: PASS; missing fields reject and
  raw Physics/critic diagnostics remain immutable.
- `frontres_v015_transaction_route_contract.py`: PASS; raw advantage equals
  return minus value, scaled signs are preserved after explicit row alignment,
  and exact-one update remains intact.
- `frontres_intent_physics_gain_contract.py`: PASS; unchanged v003 formula.
- `frontres_v015_unmocked_observation_connectivity_contract.py`: PASS; the
  existing 928/158/770 -> one-action-K -> grouped exact-one CPU route remains
  connected with semantic Physics evidence.
- `frontres_v015_policy_quality_heldout_contract.py`: PASS.
- `python -m py_compile` for all modified production owners: PASS.
- `git diff --check`: PASS.

Confirmed:

- The first invalid owner is repaired without changing Gain, PPO reduction,
  HSL, actor visibility, checkpoint identity, K semantics, or optimizer count.
- Missing Physics cannot silently appear as a complete zero tie.
- Raw and grouped-scaled credit diagnostics are now aligned to the same
  transaction policy-row identity.

Open boundary:

- CPU fixtures prove schema, masking, pairing, failure, and consumer semantics;
  they do not prove real IsaacLab ZMP/contact availability or discriminative
  values.
- E-FI-64's policy-quality verdict is not overturned by offline repair.
  Additional training and X1 remain blocked until one bounded live Physics
  sentinel records the new raw/component telemetry.

Next:

- After explicit user authorization, run one bounded 8-env, one-transaction
  live Physics sentinel. Stop on missing/nonfinite Physics, row-identity drift,
  later FEMR action, or any training beyond the single authorized boundary.

## E-FI-66: Physics-Intent Design Rebase And FRS-GAIN-v004 Activation

Date: 2026-07-22

Tier: user-confirmed semantic decision plus read-only source-owner audit;
document-only governance update; no source modification, test, simulator,
training, checkpoint I/O, live run, or deployment composition

Decision evidence:

- The user confirmed that Physics must prevent the observed sustained-lean / extra-step
  shortcut, not merely reward survival.
- Expected left/right support mode is deterministically derived from the same
  sealed scenario's GMT-only Clean continuation and remains Physics-evaluator
  evidence only.
- Actual Contact must use the existing `contact_forces` ContactSensor.
- Contact timing alignment distinguishes planned steps from extra, missed,
  dragging, and out-of-tolerance switches.
- ZMP remains a core Physics variable and is interpreted by Contact phase:
  support-domain evaluation, planned transition recovery, and flight `N/A`.
- Physics admissibility is non-compensatory: Contact/ZMP/survival failure cannot
  be offset by Intent; after admissibility, Intent is optimized and full-6D
  repair cost remains bounded inside the same tier.
- One 6D actor, one scalar Gain, one scalar Critic, H/K, one-action-K, sealed
  transaction, grouped PPO, and HSL remain unchanged. No rho, dual network,
  predictor, new actor input, Noisy prefix, label, or perturbation time is added.

Fresh source audit:

- `frontres_segment_live_probe.py::_capture_physics_frame()` captures paired ZMP
  and delegates Contact to `_height_contact_consistency_pair()`.
- `_height_contact_consistency_pair()` thresholds both reference and robot foot
  heights. Its owner docstring explicitly identifies this as a foot-height
  support proxy rather than a contact-force sensor.
- `tracking_env_cfg.py` configures and enables `contact_forces =
  ContactSensorCfg(... force_threshold=10.0 ...)`.
- `MultiMotionCommand.materialize_frontres_local_scenario()` gathers reference
  `body_pos_w/body_quat_w` for the same ordered Clean frames while constructing
  the `[K,65]` GMT continuation. This makes selection-time expected-support
  derivation feasible without pretending the 65D command contains Contact and
  without adding actor-visible future root/global data.
- `FrontRESRobotRolloutState.contact_state` exists as an optional immutable
  carrier, but production population from `contact_forces` is not code-confirmed.
- `frontres_gain.py::compute_paired_physics_gain()` forms paired success,
  K-normalized survival, ZMP, and contact differences and combines available
  components by mean.
- `compute_intent_physics_local_repair_gain()` still uses the superseded
  additive weighted Intent + Physics - repair-cost formula.
- `frontres_segment_storage.py` seals `[K,B]` ZMP/contact evidence and averages
  it to paired facts, but its return builder and downstream identities still
  require FRS-GAIN-v003.

Governance changes:

- Activated `FRS-GAIN-v004-support-mode-physics-admissibility` and moved v003 to
  contract history as superseded.
- Preserved Concept Figure blocks `Q-PAIR` and `Q-01`; updated only the existing
  Repair Gain summary. No Contact Phase module was added.
- Rebased the active plan/checklist/canvas and current Architecture around one
  G5-P1 engineering closure.
- Marked current source as `contract-mismatch`; E-FI-64/E-FI-65 remain valid
  evidence of the old additive route and carrier connectivity, not v004
  implementation evidence.

Open evidence:

- S1 deterministic v004 semantic contracts are absent.
- S2 formal consumer migration and v002/v003 isolation are absent.
- S4 real ContactSensor / phase-conditioned ZMP / v004 Gain evidence is absent.
- X1, further training, multi-seed, deployment composition, and paper
  experiments remain blocked.

Next:

- After explicit user authorization, execute G5-P1 as one implementation,
  connectivity, focused-test, bounded-live, evidence, and Architecture closure.
  Stop on actor information leakage, unavailable sensor authority, phase
  identity drift, tier inversion, mixed Gain versions, frozen-boundary change,
  or harmful/no-op bounded evidence after one repair cycle.

## E-FI-67: FRS-TRAIN-v008 Critic-Ready Actor Curriculum Rebase

Date: 2026-07-22

Tier: user-confirmed training semantic plus read-only formal-owner audit;
document-only governance update; no source modification, test, checkpoint I/O,
simulator, training, live run, or deployment composition

Decision:

- Preserve one full-6D actor, one scalar Gain, one scalar Critic, and one
  optimizer.
- Preserve proposal-only HSL as actor/distribution/158D-prefix initialization;
  HSL does not restore or supervise the Critic.
- A cold v004 Stage-3 run must first calibrate the fresh scalar Critic while
  actor/std remain frozen, then linearly release actor PPO weight while the
  Critic keeps learning, then enter joint PPO.
- Merge this M-05 closure into the existing one-shot G5-P1 rather than creating
  a separate training subsystem or approval chain.

Fresh code evidence:

- `frontres_segment_warmup.py::frontres_segment_warmup_phase()` already defines
  `critic_only`, `actor_warmup`, and `joint` from persisted iteration and
  `N_c/N_a`; actor weights are `0`, `(j+1)/N_a`, and `1` respectively.
- Legacy `run_frontres_segment_single_update()` consumes that phase, passes the
  actor weight to PPO, and clears non-Critic gradients during critic-only.
- Focused contracts cover critic-only actor invariance and linear actor weight.
- Historical E68--E70 live evidence shows 200 critic-only iterations, a
  500-iteration actor ramp, joint weight 1, and resume continuity, but those
  runs used the older update/v002 Gain route.
- Formal v015 `_require_v015_formal_transaction_config()` explicitly rejects
  every nonzero critic/actor Warmup duration.
- Formal v015 `_v015_formal_ppo_config()` hardcodes
  `actor_loss_weight=1.0`.
- `run_frontres_v015_formal_transaction_update()` does not resolve a Warmup
  phase or clear actor/std gradients before its exact-one step.
- `v015_g5_e0_physics_s4_gpu3.log` records
  `segment_critic_warmup=0` and `segment_actor_warmup=0`; it proves scalar
  Critic/value connectivity, not current v015 Warmup.

Governance changes:

- Activated `FRS-TRAIN-v008-critic-ready-v004-actor-curriculum` and moved v007
  to training-contract history as superseded.
- Preserved Concept Figure `M-03` and `M-05`; their human method meaning was
  already correct.
- Rebased G5-P1 to close v004 Physics/Gain and v008 Critic-ready Actor
  Curriculum in one engineering unit.
- Updated checklist/canvas, Method-to-Code, formal Runtime Audit, and
  policy-quality Architecture to mark the formal v015 Warmup route as
  `contract-mismatch`.
- Removed one stale Method-to-Code gap that still claimed the actor lacked v015
  q29 future intent; E-FI-27/E-FI-42/E-FI-64 already runtime-confirm that route.

Required implementation evidence:

- S1: phase boundaries, actor-weight formula, critic-only actor/std invariance,
  nonzero Critic update, actor-ramp monotonicity, joint weight, and mixed
  identity rejection.
- S2: complete v004 transaction reaches the formal v015 phase-aware grouped
  update without legacy fallback and still takes exactly one optimizer step.
- S3: HSL cold start enters iteration 0 critic-only; v008/v004 save/reload
  preserves schedule/phase; v003/v007/unversioned resume rejects before
  mutation.
- S4: one bounded 8-env critic-only transaction records real v004 Physics,
  actor/std zero delta, nonzero Critic delta, exact-one update, and committed
  v008/v004 checkpoint.
- Live progression through actor-ramp and joint belongs to X1; deterministic
  fixtures may cover those boundaries before the costly run.

Next:

- After explicit user authorization, execute the rebased G5-P1 as one
 implementation/test/persistence/bounded-live closure. Stop if formal v015
 requires legacy update fallback, actor/std changes in critic-only, Critic has
 no v004 gradient, phase crosses an open transaction, resume restarts Warmup,
 or incompatible checkpoints can mutate state.

## E-FI-68: G5-P1 v004/v008 Engineering Closure And Critic-Only S4

Date: 2026-07-22

Tier: S1/S2/S3 deterministic contracts plus one S4 official Stage3-v015
critic-only transaction; 8 envs, one iteration, one exact optimizer update; no
actor-ramp progression, long training, multi-seed, deployment composition, or
paper experiment

Raw evidence:

- Local log: `v015_g5_p1_critic_s4_gpu3.log`.
- Log SHA-256:
  `e8e8a693b837c06585322f7b44abab9cbe670891ff8ddbcabd9e20cff9131936`.
- Runtime checkpoint:
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-22_20-05-22_G5_S4_BOUND_V015/model_1.pt`.
- Source implementation commit: `02adc2f` (`implement v004 physics gain and
  v008 critic warmup`); the server ran pulled merge identity `ad068c5`.

Deterministic implementation evidence:

- Expected-support schema/provenance/hash, real Contact owner, Contact timing,
  phase-conditioned ZMP, v004 ordering, missing/mask/permutation, v004 consumer
  isolation, formal grouped transaction, v008 phase/gradient,
  checkpoint/resume, held-out evaluation, and observation-connectivity focused
  contracts passed.
- Python production modules compiled and `git diff --check` passed before the
  implementation commit.
- Stage3 checkpoint identity is `frontres-v015-checkpoint-v3`, binding
  `FRS-TRAIN-v008`, `FRS-GAIN-v004`, warmup durations, iteration, and phase;
  incompatible old identities reject before mutation.

S4 runtime facts:

- Formal config resolved `critic_warmup_iterations=200`,
  `actor_warmup_iterations=500`, `phase=critic_only`, and
  `actor_loss_weight=0.0` at iteration 0.
- The transaction contained two Segments x two attempts, four valid policy
  rows, K=8 evidence, equal motion/Segment mass `(0.5,0.5)`, equal attempt mass
  `(0.25,0.25,0.25,0.25)`, and one sealed hash per Segment across attempts.
- The log recorded expected support `[4,8,2]`, actual Repair/Noisy Contact from
  the live sensor, survival, ZMP, per-role admissibility/deficit, utility,
  Intent quality, repair cost, Gain, return, raw/scaled advantage, gradients,
  group mass, and transaction/checkpoint identity.
- Critic-only isolation passed: actor/std changed `0/8` parameter tensors with
  max delta `0`; Critic changed `10/10`, first at `critic.0.weight`, with L2
  delta `0.00139620` and max absolute delta `1.00024e-6`.
- Gradient was finite and connected: `10/18` parameter tensors nonzero;
  pre/post-clip norm `0.10401675`.
- Exact-one and persistence passed: `optimizer_step_delta=1`,
  `update_count=1`, committed receipt `1`, and `save.status: OK` at iteration 1.
- No Python Traceback, RuntimeError, assertion, non-finite sentinel, legacy
  Gain fallback, actor drift, or partial transaction occurred. The lone
  `DriverShaderCacheManager` startup error did not prevent Isaac Sim, the
  transaction, update, or save from completing.

Policy-quality observation, not an engineering failure:

- Action remained finite/nonzero: absolute mean `0.00408867`, maximum
  `0.01544815`, mean L2 `0.01178368`.
- All four Noisy and Repair roles were Physics-inadmissible with deficit `1`;
  both utilities were `-2`. Consequently v004 correctly prevented Intent or
  ZMP improvements from compensating the Physics tier, and each total Gain was
  only the negative bounded repair penalty:
  `(-0.00113385,-0.00147449,-0.00275787,-0.00161156)`.
- Positive Gain fraction was `0`, harm fraction `1`. Raw and scaled advantages
  were all negative because the fresh Critic values exceeded these returns.
  This is valid Critic calibration evidence while the actor is frozen; it does
  not establish actor-ramp efficacy.

Verdict:

- G5-P1 engineering closure passes FRS-GAIN-v004 and FRS-TRAIN-v008 S1/S2/S3/S4.
- The run proves authoritative Physics evidence, non-compensatory scalar Gain,
  critic-only gradient isolation, exact-one grouped update, and committed
  persistence on the formal route.
- X1 is now the next separate high-cost decision. Actor-ramp policy quality,
  checkpoint trajectory, long training, multiple seeds, deployment
  composition, and paper evidence remain unconfirmed and unauthorized.

## E-FI-69: K-Stage Critic Curriculum Contract And White-Box Rebase

Date: 2026-07-22

Tier: user-confirmed training semantics, read-only source-owner audit, and
document-only C0 governance rebase; no training-source modification, test,
checkpoint I/O, simulator, training, live run, or deployment composition

Decision:

- Preserve one full-6D actor, one scalar Gain, one scalar Critic, and one
  optimizer. Multi-Critic and K-conditioned actor input are rejected.
- K values are ordered curriculum approximations toward one final executable
  evidence horizon, not simultaneous policy objectives.
- Each training stage has one global active K. Every Segment and every M
  attempt in a sealed transaction uses that same K.
- The same Critic continues its weights across stages. After every K increase,
  actor/std freeze again for a stage-local critic-only phase, followed by an
  actor ramp and joint PPO at the same K.
- A K-stage transition may occur only after the current exact-one transaction
  is committed. H, one-action-K, FRS-GAIN-v004, grouped PPO, HSL, and actor
  visibility remain unchanged.

Fresh code evidence:

- `frontres_segment_sampler.py::plan_rollout_budget()` currently assigns K by
  `FrontRESSegmentState`: UNKNOWN/HOPELESS use 8, PROMISING uses 16, FRONTIER
  uses 16 or 32, DELAYED_REGRET uses 32 or 64, and SOLVED uses 64.
  Consequently different Segments may carry different K values in one formal
  training period.
- `frontres_segment_warmup.py::frontres_segment_warmup_phase()` is an
  absolute-global-iteration v008 owner. It has no schedule fingerprint,
  K-stage index, active K, or stage-local iteration and therefore cannot
  re-enter critic-only when K changes.
- Stage3 configuration currently hardcodes `frontres_segment_k=8` with maximum
  horizon 64 rather than parsing an explicit frozen K-stage schedule.
- Stage3 checkpoint identity `frontres-v015-checkpoint-v3` binds v008,
  absolute iteration, warmup durations, and phase, but not schedule
  fingerprint, K stage, active K, or stage-local iteration.
- E-FI-68 remains valid evidence for the single-K K=8 critic-only foundation;
  it is not evidence of a K transition curriculum.

Governance changes:

- Activated `FRS-TRAIN-v009-k-stage-critic-curriculum` and superseded v008
  without silently changing its historical evidence.
- Updated Method v015, the contract registry, Concept Figure M-06/M-05 edge,
  the active engineering plan/checklist/task canvas, and current Architecture.
- The implementation plan contains one preparatory C0 gate and four bounded
  execution steps: C1 pure schedule/config identity; C2 homogeneous-K formal
  transaction and phase transition; C3 strict v009 persistence/resume; C4 one
  bounded official K-transition sentinel.
- Current per-Segment adaptive K may remain only as historical/ablation logic;
  it is marked `contract-mismatch` for the v009 formal route.

Stop conditions:

- Stop on mixed K within a transaction, stage change before commit,
  actor/std mutation in new-K critic-only, Critic reinitialization, Multi-Critic
  expansion, K entering actor observation, implicit schedule defaults, or
  resume identity drift.
- Long training and policy-quality claims remain blocked until C1-C4 agree on
  one schedule fingerprint and a real bounded K transition.

Next:

- C0 cross-file JSON/reference/diff validation passed. C1 / 4 requires separate
  implementation authorization and must not touch formal
  transaction, checkpoint, simulator, training, or live-run paths.

## E-FI-70: C1-C3 K-Stage Critic Curriculum Offline Engineering Closure

Date: 2026-07-22

Tier: deterministic S1 schedule/config, S2 formal transaction connectivity,
and S3 checkpoint persistence evidence; no simulator, training, optimizer
outside semantic CPU fixtures, live run, deployment composition, or C4

C1 implementation:

- `frontres_segment_warmup.py` now owns strict schedule parsing, normalization,
  SHA-256 fingerprinting, and committed-update mapping to
  `(stage_index, active_k, stage_iteration, phase, actor_loss_weight)`.
- The schedule is explicit CLI/config identity with rows
  `(K,N_c,N_a,N_joint)`. Empty, malformed, non-increasing, over-max,
  zero-recalibration, and non-final zero-joint schedules reject.
- Stage3 launch requires `FRONTRES_V015_K_CURRICULUM`; no production default
  silently selects the final method schedule. Legacy global warmup fields
  remain non-authoritative compatibility data.

C2 implementation:

- The formal sampler retains Segment-dependent trial budgeting but overrides
  its legacy per-Segment K with one `active_horizon_k` from the sealed v009
  stage identity.
- `FrontRESV015FormalTransactionRequest` binds the schedule fingerprint,
  stage, active K, stage-local iteration, absolute iteration, phase, and actor
  weight. Every plan row must equal the active K before rollout/loss/update.
- The formal update recomputes and matches the sealed identity, performs the
  unchanged grouped PPO exact-one update, and clears actor/std gradients in
  every new-K critic-only phase.
- The committed receipt records the exact curriculum identity consumed by the
  update. Formal telemetry must match that receipt field by field.

C3 implementation:

- Stage3 persistence is now `frontres-v015-checkpoint-v4` with
  `FRS-TRAIN-v009`, the ordered schedule/fingerprint, stage index, active K,
  stage-local/absolute iteration, phase, actor weight, v004/v003 identities,
  normalizer/layout, optimizer/sampler state, and committed receipt.
- A committed receipt must be adjacent to the checkpoint iteration and match
  the preceding stage identity. At a boundary, the receipt records the old
  stage commit while checkpoint curriculum records the new-K critic-only
  starting state.
- v008/checkpoint-v3, unversioned, different-schedule, tampered phase/K,
  collecting/sealed/failed, and non-adjacent receipt paths reject before actor,
  normalizer, sampler, optimizer, or iteration mutation.

Executed deterministic evidence:

- `frontres_segment_warmup_contract.py`: PASS.
- `frontres_segment_sampler_contract.py`: PASS.
- `frontres_v015_transaction_route_contract.py`: PASS, including mixed-K
  rejection and new-K actor/std zero delta with nonzero Critic delta.
- `frontres_v015_checkpoint_resume_contract.py`: PASS, including checkpoint-v4
  transition resume, optimizer restore, different schedule and v008 rejection.
- Stage3 entrypoint and launch-command contracts: PASS.
- Unmocked observation and v015 local-sentinel connectivity contracts: PASS.
- Observation authority, HSL isolation, policy-quality strict identity, and
  actual save/fresh-reload contracts: PASS.

Boundary:

- C1-C3 are offline engineering-complete. The retained legacy adaptive-K
  planner is not deleted, but it cannot decide K on the v009 formal route.
- No official simulator transaction has crossed a K boundary. C4 remains a
  separate user-authorized live gate; long training and policy-quality claims
  remain blocked.

## E-FI-71: C4 Bounded Official K-Transition And Quality Blocker

Date: 2026-07-23

Tier: S4 bounded official runtime evidence; 8 environments, four formal
iterations, one optimizer update per sealed transaction, checkpoint interval
one; no long training, multi-seed, deployment composition, or paper experiment

Raw evidence:

- repository-root `v015_c4_k_transition_gpu3.log` (779 lines);
- explicit schedule `8:1:1:1,16:1:1:0`;
- no `Traceback`, `RuntimeError`, `AssertionError`, NaN, or CUDA failure; final
  GLFW/IOMMU messages are headless-host warnings after successful completion.

Engineering facts established:

- the four committed transactions have the exact order K8 critic-only, K8
  actor-warmup, K8 joint, and K16 critic-only;
- each transaction contains four valid policy rows with group mass
  `[0.25,0.25,0.25,0.25]` and exactly one optimizer step;
- both critic-only phases preserve exact actor/std zero delta while changing ten
  Critic parameters; actor-warmup and joint change actor and Critic;
- `model_1.pt` through `model_4.pt` are strict checkpoint-v4/v009 saves;
  `model_3.pt` resolves to K16 critic-only and `model_4.pt` to K16
  actor-warmup;
- two pre-success integration defects were closed before this run: stale v008
  checkpoint audit ownership and command-carrier cleanup after committed
  training. Neither failure is present in the final log.

Policy-quality facts exposed:

- all four transactions report `positive_gain_fraction=0`,
  `negative_gain_fraction=1`, and `harm_fraction=1`: 0/16 policy rows have
  positive v004 Gain;
- Repair and Noisy Physics admissibility are both 0/16;
- only 3/16 raw advantages are positive: 0 in K8 critic-only, 2 in K8
  actor-warmup, 1 in K8 joint, and 0 in K16 critic-only;
- in both critic-only phases, Repair/Noisy Physics deficits equal `1` and both
  unsafe utilities equal `-2`, reducing Gain to the negative repair penalty;
- actor-warmup and joint pre-clip gradient norms are approximately `700.11`
  and `648.66`, both clipped to `0.5`; critic-only norms are approximately
  `0.104` and `0.0856`.

Interpretation and boundary:

- C1-C4 K-stage engineering is complete. The one-update phase durations prove
  ordering, isolation, exact-one update, and persistence; they do not prove
  Critic calibration or a usable long-training schedule.
- Negative Gain with positive raw advantage is legal PPO semantics when the
  return is less harmful than the Critic prediction. It is not itself a wiring
  failure, but an uncalibrated Critic can make that relative credit unreliable.
- Longer Critic warmup cannot recover action ranking if the v004 unsafe tier
  maps both roles to the same saturated worst-violation utility. The next
  decision must distinguish target saturation from insufficient calibration.
- Do not admit actor-ramp long training, weaken Physics non-compensation, or
  allow Intent to offset a critical violation without a separate human design
  decision and focused evidence.

Next:

- perform a bounded read-only causal projection over the 16 C4 rows: identify
  the worst Contact/ZMP/survival component, the component-level Repair-Noisy
  differences hidden by the `max` deficit, Critic values, returns, and raw
  advantages. Then choose between a richer Physics-first unsafe-tier distance,
  a sampling/curriculum adjustment, or longer Critic calibration only after the
  target is shown distinguishable.

## E-FI-72: Noisy Counterfactual Retention And Vector-Physics Design Rebase

Date: 2026-07-23

Tier: user-confirmed method concept, existing-log causal evidence, read-only
CodeGraph owner/consumer audit, and document-only governance rebase. No source
code, test, checkpoint I/O, simulator, training, or live run.

Raw evidence:

- repository-root `v015_zmp_plateau_s4_gpu3.log`;
- E-FI-71 bounded official C4 evidence;
- active v004 source at
  `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`;
- explicit user confirmation that FEMR should seek the smallest Intent-faithful
  repair inside the executable space defined by Contact, ZMP, and survival.

Observed facts:

- v004 normalizes violation severity, reduces Contact/ZMP/survival with
  `amax`, maps both inadmissible roles into a saturated unsafe utility tier,
  and then constructs one scalar paired Gain;
- the plateau log contains raw ZMP improvement that collides after this
  reduction, so the scalar return/advantage cannot preserve the corresponding
  policy-direction distinction;
- CodeGraph traces the active consumer chain from
  `compute_intent_physics_local_repair_gain()` through storage return,
  `compute_frontres_segment_ppo_loss()`, the exact-one formal update,
  diagnostics, and checkpoint identity;
- Critic calibration can fit the scalar target but cannot recover Physics
  evidence deleted before return construction.

Confirmed decision:

- retain one Noisy rollout per sealed scenario as the zero-action
  counterfactual shared by all M Repair attempts;
- Noisy supplies paired causal/Intent/no-op evidence but is not the Physics
  admissibility threshold;
- retain ordered ContactSensor, phase-conditioned ZMP, and survival evidence
  until the optimization boundary and consume them as separate actor
  constraints;
- paired Intent improvement minus full-6D repair cost becomes the scalar
  objective and the only scalar-Critic target;
- preserve one full-6D actor, one scalar Critic, one optimizer, H, one-action-K,
  global K curriculum, sealed grouped transaction, exact-one update, frozen
  GMT, and proposal-only HSL;
- do not introduce rho, a second network/Critic, Contact predictor, Clean actor
  input, Noisy physical prefix, or deployment Noisy rollout.

Governance result:

- created the v005 proposal and rebased the current plan/checklist/canvas;
- updated Concept Figure Q-01 without adding a Contact/constraint module;
- marked current v004/v003/v009 source and active contracts as an explicit
  figure/contract mismatch before further training;
- compressed the remaining work into P1-P4: one semantic contract decision,
  one offline engineering closure, one bounded official sentinel, and one
  policy-quality/long-training decision.

Open semantic boundary:

- the recommended implementation is a grouped first-order constraint-gradient
  projection: scalar Intent PPO/Critic plus separate Contact/ZMP/survival
  surrogates, with the actor direction projected into the local Physics-feasible
  set before the existing exact-one optimizer step;
- the exact residual units, time statistic, projection/recovery rule, and
  infeasible-case behavior require human confirmation in P1 before coordinated
  METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010 activation.

Stop:

- no source modification, checkpoint migration, simulator, or training while
  the Concept Figure and active v004/v003/v009 contracts intentionally differ.

## E-FI-73: P1 Physics-Constrained Intent Contract Activation

Date: 2026-07-23

Tier: user-authorized mathematical/identity contract activation and
document-only governance closeout. No source code, tests, checkpoint I/O,
simulator, training, or live run.

Activated decision:

- `FRS-METHOD-v016` preserves the sealed Noisy zero-action counterfactual,
  158D full-6D actor, one-action-K frozen-GMT evidence, grouped transaction,
  and exact-one update while separating scalar Intent from Physics authority;
- `FRS-GAIN-v005` defines the scalar Critic target as paired Intent improvement
  minus full-6D repair cost and retains Contact, phase-ZMP, and survival as
  separate physical-unit, non-saturating Repair constraint residuals;
- `FRS-PPO-v004` defines a joint grouped first-order projection of the Intent
  actor direction into the Contact/ZMP/survival non-worsening cone, with
  KKT-checked corrective recovery and actor/std zero update when no common
  descent direction exists; the Critic may still update in the same exact-one
  optimizer step;
- `FRS-TRAIN-v010` rejects old v004/v009 Critic/checkpoint identity before
  mutation, fresh-initializes Critic state on first entry to the new target
  with actor/std frozen, and reuses the same v010 Critic for critic-only
  recalibration after every global K increase before actor ramp and joint;
- replay priority remains selection-only and cannot recreate scalar Physics
  reward or change grouped actor-loss mass;
- no second actor, Critic, optimizer, learned dual, rho, new actor input, HSL
  change, or Clean/Physics leakage is introduced.

Identity and governance result:

- active identities are METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010 and
  checkpoint-v5 with explicit scalar-target, constraint-schema, projection,
  K-stage, transaction, and committed-receipt identity;
- METHOD-v015 / GAIN-v004 / PPO-v003 / TRAIN-v009 were moved to history and
  marked superseded, but remain the current incompatible source route;
- registry, plan, checklist, task canvas, and current Architecture now mark P1
  complete and P2 ready but not authorized;
- Concept Figure was not rewritten in P1; its P0 Q-PAIR/Q-01/M-05 mapping was
  verified against the activated contracts.

Stop:

- source/runtime/checkpoint execution remains blocked until P2 atomically
  implements the new evidence, loss, update, diagnostics, and persistence
  route and rejects all v004/v003/v009 fallback before state mutation.

## E-FI-74: P2 One-Shot Offline Physics-Constrained Intent Engineering Closure

Date: 2026-07-23

Tier: user-authorized source implementation plus deterministic S1/S2/S3
contracts. No simulator, real training, live run, deployment composition, long
training, multi-seed experiment, or policy-quality claim occurred.

Implemented authority split:

- `frontres_gain.py` now makes `gain_total` exactly paired root-invariant
  Intent improvement minus the existing full-6D repair cost. Contact,
  phase-conditioned ZMP and survival remain named physical-unit residuals;
  they are not added to the scalar Critic target.
- The one-action-K/storage path carries ordered raw Physics evidence, explicit
  ZMP applicability, residual levels and detached per-attempt constraint
  evidence. Constraint advantages remain unsealed/NaN at isolated-attempt
  scope and are centered only after all M rows of the same sealed scenario are
  present in the transaction accumulator.
- `frontres_segment_ppo.py` computes equal-mass grouped Intent and per-family
  constraint surrogates, then installs one KKT-checked joint projected/recovery
  actor gradient. Scalar-Critic gradients are derived only from value loss;
  actor/std and Critic parameter authority remain disjoint inside the same
  runner-owned exact-one optimizer step.
- Explicit ZMP N/A is excluded. Mixed support/flight rows retain equal mass
  among applicable motion/Segment/attempt groups; an all-flight transaction
  excludes the ZMP family without treating missing evidence as observed zero.
- Formal diagnostics expose contract IDs, raw constraint levels/advantages,
  applicability, projection status, gradients, dual coefficients, Gram matrix,
  directional derivatives and KKT violation without feeding diagnostics back
  into training state.
- The formal runtime checkpoint audit was migrated from the historical
  checkpoint-v4/scalar-Gain payload to the same checkpoint-v5 coordinated
  contract, solver and no-persistent-dual assertions, preventing the enabled
  audit path from rejecting the new official save boundary.
- Stage3 persistence is `frontres-v015-checkpoint-v5` and binds METHOD-v016,
  GAIN-v005, PPO-v004, TRAIN-v010, future layout, K curriculum, grouped loss,
  solver schema, transaction and committed receipt. The active payload has no
  legacy scalar `frontres_gain_config` and no persistent/learned dual state.
  v004/v009, solver-tampered and partial transaction resumes reject before
  state mutation.
- TRAIN-v010 deterministic coverage proves first-entry actor-only HSL migration
  leaves the newly constructed Critic/optimizer outside the HSL payload,
  critic-only formal updates preserve actor/std, and every homogeneous global
  K increase re-enters critic-only recalibration on the same v010 Critic before
  actor ramp/joint.

Executed deterministic evidence (all PASS):

- S1 Gain/Physics: `frontres_intent_physics_gain_contract.py`, including scalar
  purity, physical severity, flight N/A, provenance, missing and permutation
  rejection.
- S1/S2 optimization: `frontres_segment_grouped_ppo_contract.py`, including
  equal group mass, sign-preserving scale, joint projection/recovery,
  no-direction/no-common-descent, KKT, actor/Critic gradient authority and
  mixed/applicable ZMP N/A semantics.
- S2 formal route: `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_grouped_candidate_adapter_contract.py`,
  `frontres_v015_gain_consumer_contract.py`, and
  `frontres_v015_transaction_route_contract.py`; the last proves a complete
  2-Segment x 2-attempt transaction, one projected formal update, phase
  isolation, commit-before-save and no legacy adapter.
- S3 persistence/evaluation: `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_formal_runtime_audit_contract.py`,
  `frontres_segment_diagnostics_contract.py`, and the three strict
  `frontres_v015_policy_quality_{identity,heldout,save_reload}_contract.py`
  contracts.
- Regression: `frontres_segment_warmup_contract.py`,
  `frontres_segment_sampler_contract.py`, `frontres_hsl_v007_s1_contract.py`,
  `frontres_future_intent_actor_context_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`,
  `frontres_v015_observation_authority_contract.py`, and Stage3 entrypoint and
  launch-command contracts. Production Python compilation and Architecture
  JSON parsing also passed.

Verdict and remaining boundary:

- P2 is offline engineering-complete. The formal source route now matches the
  coordinated active contracts and Concept Figure authority split.
- Deterministic fixtures do not prove that real ContactSensor/ZMP/survival
  evidence is finite and informative, that the projected direction is useful
  on a real transaction, or that checkpoint-v5 is produced by IsaacLab.
- P3 remains a separate simulator/material-cost boundary. It requires explicit
  user authorization for one 8-env, one-transaction, one-update official
  sentinel and must stop on missing raw evidence, scalar-Critic Physics
  contamination, nonfinite projection, adverse constraint direction, legacy
  fallback, or optimizer step count other than one.

## E-FI-75: P3 Bounded Official Physics-Constraint Sentinel And Checkpoint-v5 Identity

Date: 2026-07-24

Tier: user-authorized S4 live sentinel review plus read-only checkpoint payload
inspection. No source modification, simulator rerun, actor-ramp, long training,
multi-seed experiment, deployment composition, or paper claim occurred in this
closeout.

Raw evidence:

- repository-root `v015_p3_physics_constraint_s4_gpu3.log`, produced by one
  official 8-env, one-iteration, one-update Stage3-v015 run;
- read-only `torch.load(..., map_location="cpu")` inspection on the authenticated
  server of
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-23_20-43-56_G5_S4_BOUND_V015/model_1.pt`;
- no `Traceback` or `RuntimeError`; headless GLFW/display and shader-cache
  shutdown messages did not interrupt the committed update or save.

Live transaction facts:

- one sealed transaction contained 2 Segments x 2 policy attempts, 4/4 valid
  rows, K8 evidence, grouped attempt mass `(0.25, 0.25, 0.25, 0.25)`,
  `optimizer_step_delta=1`, and `update_count=1`;
- final serialized telemetry carries METHOD-v016 / GAIN-v005 / PPO-v004 /
  TRAIN-v010, `paired-intent-minus-repair-v1`, the physical constraint schema,
  projection schema, row-aligned Contact/ZMP/survival advantages, Gram matrix,
  directional derivatives and `constraint_kkt_max_violation=0`;
- scalar purity is live-confirmed: `gain_total = intent_gain - repair_cost`
  with maximum numerical error `3.26e-9`; Physics does not enter the scalar
  Critic target. `raw_advantage = return - policy_value` holds with maximum
  error `3.73e-9`;
- Contact and ZMP were active while survival residual was zero. The unconstrained
  Intent direction had positive Contact/ZMP derivatives (`635.68`, `3190.09`),
  whereas `CONSTRAINT_RECOVERY` produced negative derivatives (`-4218.51`,
  `-11096.43`), so the selected first-order direction reduced both violated
  Physics channels;
- TRAIN-v010 was in `critic_only`: actor/std maximum parameter delta was exactly
  zero, ten Critic parameters changed with maximum delta `1.00024e-6`, and the
  committed update advanced the K8 phase from iteration 0 to 1;
- this single frozen-actor transaction reported positive Gain fraction `0.25`
  and harm fraction `0.75`. Those values describe the HSL-initialized proposals
  before actor-ramp learning; they do not prove actor efficacy or failure.

Checkpoint identity facts:

- the saved payload contains `format=frontres-v015-checkpoint-v5` and binds
  METHOD-v016, GAIN-v005, PPO-v004, TRAIN-v010,
  `paired-intent-minus-repair-v1`,
  `contact-phase_zmp-survival-physical-v1`, and
  `grouped-first-order-constraint-projection-v1`;
- curriculum identity is K8 with schedule `((8, 200, 500, 0),)`,
  `phase=critic_only`, `phase_iteration=1`, and `actor_loss_weight=0`;
- transaction state is `committed`; its receipt binds 4 collected/valid policy
  rows, optimizer step `0 -> 1`, scenario/plan hashes and the same contract
  identities. `persistent_dual_state=false`.

Verdict and next boundary:

- P3 is complete. The formal runtime, final telemetry consumer, exact-one
  optimizer boundary and checkpoint-v5 producer are live-confirmed.
- P3 does not admit long training. P4 must first decide the actor-ramp admission
  evidence and smallest justified horizon. It must stop on no feasible/corrective
  direction, no-op collapse, systematic Physics regression, sustained lean,
  unplanned stepping, or an Intent-only shortcut; such a result returns to P1
  rather than coefficient tuning.

## E-FI-76: P4-S0 Policy-Quality And Actor-Ramp Admission Audit

Date: 2026-07-24

Tier: read-only S0 contract/code/evidence audit. No source or Concept Figure
change, test, checkpoint mutation, simulator, training, live run, actor-ramp,
multi-seed evaluation or deployment composition occurred.

Evidence inspected:

- E-FI-75 and repository-root
  `v015_p3_physics_constraint_s4_gpu3.log`;
- active TRAIN-v010, PPO-v004 and EVAL-v003 contracts plus the current
  checkpoint-v5 identity recorded by P3;
- formal Stage3 shell/`train.py` dispatch, `load_runner()` persistence owner,
  `run_frontres_segment_live_training_loop()` and the active v015 held-out
  report projection;
- `frontres_v015_policy_quality_heldout_v1.json` and the one-action-K evidence
  carrier.

Confirmed facts:

- the current P3 artifact is
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-23_20-43-56_G5_S4_BOUND_V015/model_1.pt`,
  K8 `critic_only`, phase/absolute iteration `1/200`, actor weight zero and
  schedule `((8,200,500,0),)`; therefore the fixed contract boundary requires
  199 additional committed critic-only updates before actor-ramp can begin;
- `load_runner()` has a strict checkpoint-v5 full-resume owner, but the formal
  launcher rejects `--resume`, `--resume_student_checkpoint` and
  `--is_full_resume`, always passes an HSL-v1 initializer, while `train.py`
  requires that initializer for ordinary Stage3 and forbids combining it with
  resume. No honest official continuation command currently exists;
- one-action-K evidence already retains ordered expected support, actual
  ContactSensor rows, phase-ZMP values/masks and survival, but the atomic held-
  out report serializes only actions, identity, valid counts and aggregate
  Intent/repair/Contact/ZMP/survival components;
- the report has no sustained lateral-lean trajectory and no raw
  expected/actual Contact or phase-ZMP applicability/violation/recovery
  trajectory, so it cannot prove the requested non-hacking Demo-quality gate;
- P3 remains valid mechanism/connectivity evidence. These gaps block only
  checkpoint continuation and policy-quality admission; they do not reopen
  METHOD-v016, GAIN-v005, PPO-v004, TRAIN-v010 or the Concept Figure.

Decision and next boundary:

- P4-S0 is complete and actor-ramp remains blocked;
- P4-S1 is frozen as one readiness-closure implementation unit owned by
  `scripts/rsl_rl/train.py::main()` at the orchestration boundary, reusing the
  existing persistence, training-loop and held-out-report semantic owners;
- P4-S1 may implement strict HSL-XOR-v5-resume connectivity and read-only raw
  quality evidence plus evaluator-only sustained-lean reporting, with focused
  deterministic S1/S2/S3 contracts only;
- P4-S1 must stop before simulator/training/live execution. The remaining 199
  critic-only updates, actor-ramp length and numerical admission thresholds
  require a later explicit human decision after readiness closes.

## E-FI-77: P4-S1 Readiness Closure

Date: 2026-07-24

Tier: deterministic S1/S2/S3 implementation evidence. No simulator, training,
live run, remaining critic-only update, actor-ramp, HSL/Gain/PPO/TRAIN-v010
change, actor-observation change, checkpoint-format change or Concept Figure
change occurred.

Implemented boundaries:

- `scripts/rsl_rl/train.py::main()` remains the sole orchestration owner and
  accepts an explicit `--frontres_v015_resume_checkpoint`; ordinary Stage3
  requires exactly one of that strict v5 route or HSL-v1 initialization;
- the shell launcher is only a connector and emits exactly one identity flag;
  legacy generic resume arguments remain rejected;
- `load_runner()` validates v5 actor/Critic tensor identity, optimizer
  compatibility, sampler presence, 928/158/770 normalizer identity,
  TRAIN-v010 schedule/absolute iteration and committed-or-idle receipt before
  mutable restore. Optimizer mismatch is fatal for v5 rather than downgraded
  to a warning;
- a restored committed receipt remains available after runtime transaction
  state returns to idle and is retained by a subsequent coordinated v5 save;
- the existing atomic v015 held-out report now carries ordered expected and
  actual Contact, phase-ZMP applicability/N-A/violation/recovery and margins,
  raw Repair/Noisy survival, and evaluation-only actual root-roll plus
  cumulative lateral-lean trajectories;
- lateral lean is captured only when the held-out quality route is active;
  training one-action-K keeps that optional carrier absent. Missing quality
  evidence fails closed and the existing training-state signature proves no
  optimizer, sampler, transaction or normalizer feedback.

Deterministic evidence:

- `frontres_segment_stage3_launch_command_contract.py`: PASS, including HSL XOR
  v5 resume, 199-update command budget and missing-checkpoint rejection;
- `frontres_v015_one_action_k_contract.py`: PASS, including actual paired
  root-roll evaluation evidence and unchanged one-action-K semantics;
- `frontres_v015_policy_quality_heldout_contract.py`: PASS, including raw
  trajectory projection, missing-evidence rejection, atomic report and state
  isolation;
- `frontres_v015_checkpoint_resume_contract.py`: PASS, including actor, Critic,
  Adam moments, sampler, 158D prefix/770D GMT authority, schedule, absolute
  iteration, committed receipt and resumed-save persistence;
- `frontres_v015_policy_quality_save_reload_contract.py`: PASS, preserving
  actual save/fresh reload/atomic report isolation;
- Python compilation of all changed Python owners and focused contracts: PASS.

Verdict: P4-S1 is complete. The formal continuation route and policy-quality
evidence surface are engineering-ready. The checkpoint remains K8
`critic_only 1/200`; no claim is made about Critic calibration or policy
quality. Running the remaining 199 critic-only updates and deciding actor-ramp
admission remain separate human-controlled steps.

## E-FI-78: P4 K8 Critic-Only Continuation Live Closeout

Date: 2026-07-24

Tier: user-authorized S4 live continuation plus read-only log review. No source
or Concept Figure change, actor-ramp update, long training, multi-seed,
deployment composition or paper claim occurred in this closeout.

Raw evidence:

- repository-root `v015_p4_critic_k8_to_200_gpu3.log`;
- final persistence sentinel for
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-24_14-58-03_P4_CRITIC_K8_TO_200/model_200.pt`;
- no `Traceback`, `RuntimeError`, `ValueError` or `FAILED` marker was found.

Live-confirmed facts:

- 199 serialized transaction telemetry records and 199 formal-train receipts
  cover unique training iterations 1--199 and advance absolute iteration 1 to
  200;
- all 199 accepted transactions remain K8 `critic_only`, use actor loss weight
  zero, preserve actor/std maximum parameter delta exactly zero, and report a
  nonzero finite Critic delta;
- every accepted transaction has four valid policy rows, equal attempt mass
  `(0.25, 0.25, 0.25, 0.25)`, `optimizer_step_delta=1` and `update_count=1`;
  rejected scenario attempts report zero optimizer-step delta and do not
  consume the 199 committed-update budget;
- every telemetry record carries METHOD-v016 / GAIN-v005 / PPO-v004 /
  TRAIN-v010, finite scalar objective/value/advantage and finite Contact,
  phase-ZMP and survival constraint evidence. Observed projection states are
  `CONSTRAINT_RECOVERY`, `PROJECTED_INTENT`,
  `NO_COMMON_FIRST_ORDER_DESCENT` and `NO_EMPIRICAL_DIRECTION`;
- the final formal receipt is transaction
  `frontres-v015-local-training:i199:n207`, with two Segments, four attempts,
  four valid rows, exact-one update and committed identity at absolute
  iteration 200;
- `[AUDIT-PERSIST-01]` and the final save report bind checkpoint-v5 persistence
  to METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010, K8 schedule
  `((8,200,500,0),)`, model/optimizer/normalizer/sampler state and successful
  `model_200.pt` save. The persisted `phase=actor_warmup` denotes the phase of
  the next update; none of the 199 executed updates used a nonzero actor weight.

Verdict and limitation:

- P4-S2 is complete. The fixed K8 Critic-only schedule, actor/std freeze,
  scalar-Critic update authority, sealed exact-one transaction lifecycle and
  coordinated persistence are live-confirmed through the iteration-200
  boundary;
- this changing-scenario training log is not a matched calibration experiment.
  Aggregate return/value drift cannot establish Critic calibration or policy
  efficacy, and no actor-ramp update has yet occurred;
- P4-S3 is therefore frozen as exactly one actor-ramp transaction from
  `model_200.pt`, starting at weight `1/500 = 0.002`. It requires explicit
  simulator/live authorization and must not expand into a second update or
  longer training without a new policy-quality decision.

## E-FI-79: P4 Long Training Runtime Review And KKT Contradiction

Date: 2026-07-24

Tier: read-only L0 log audit of repository-root
`v015_train_to_model2000_gpu1.log`. No source, checkpoint, simulator or
training state was modified during this review.

Live-confirmed facts:

- strict full resume loads `model_251.pt`; 1749 unique telemetry records cover
  training iterations 251--1999 and save `model_2000.pt` at absolute iteration
  2000;
- every accepted transaction has two Segments, four valid attempts, equal
  attempt mass `(0.25, 0.25, 0.25, 0.25)`, equal motion/segment mass and exactly
  one optimizer step; METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010 remain
  present throughout;
- actor-ramp ends at iteration 699 and joint training covers 700--1999. Actor
  and Critic parameter deltas remain finite and nonzero; action L2 remains near
  `0.0103`, so the training actor does not numerically collapse to zero;
- late-joint scalar Critic value mean falls to `0.00254` and raw-advantage mean
  improves to `-0.01492`, but late Gain mean remains `-0.01238`, positive-Gain
  fraction `0.369`, Intent mean `-0.00210`, Physics mean `+0.00271`, and repair
  cost mean `0.01028`. This supports Critic calibration but not a policy-
  efficacy claim;
- Repair and Noisy full K8 survival are each `0.999142` over all rows; no
  sustained-lean telemetry is serialized by this training log.

Contradiction:

- iterations 445, 653, 1309 and 1394 report `CONSTRAINT_RECOVERY` with KKT
  residuals `0.929612`, `57.627438`, `37.050545` and `24.576648` respectively;
- source reconstruction identifies the first invalid owner as recovery norm
  scaling after a tolerance-feasible projection. Positive float32 residuals
  that were within tolerance before scaling were amplified without a second
  feasibility solve;
- the formal telemetry consumer accepted any finite nonnegative KKT value, so
  these four contract violations did not fail closed.

Verdict: the long run completed mechanically, but the pre-fix `model_2000.pt`
is not a contract-clean final artifact. It may be retained as diagnostic or
warm-start evidence pending a human checkpoint-lineage decision.

## E-FI-80: Recovery Postscale KKT Source And Consumer Closure

Date: 2026-07-24

Tier: deterministic S1 core-parameter regression plus S2 formal-consumer
contract. No simulator, training, checkpoint I/O or live run occurred.

Source repair:

- `frontres_segment_ppo.py` reprojects the recovery direction after target-norm
  scaling and accepts it only when all active directional derivatives are at
  most `projection_tolerance` and at least one is strictly negative;
- failure to recover a finite common descent falls through to the existing
  `NO_COMMON_FIRST_ORDER_DESCENT` zero-actor direction;
- `frontres_segment_live_training.py` rejects KKT above the checkpoint-v5
  `1e-8` tolerance and rejects disagreement between serialized directional
  derivatives and the reported maximum violation.

Fresh evidence:

- the recorded near-opposing float32 fixture produced KKT `0.015625` before
  repair and now returns `CONSTRAINT_RECOVERY`, KKT zero, Contact derivative
  `-0.015625` and ZMP derivative `-173.5625`;
- `frontres_segment_grouped_ppo_contract.py`: PASS, including inactive,
  feasible, recovery, no-direction, no-common, permutation, postscale KKT,
  actor/Critic authority and ZMP N/A cases;
- `frontres_v015_transaction_route_contract.py`: PASS, including explicit
  rejection of over-tolerance and inconsistent KKT telemetry;
- `python -m py_compile` for the two source owners and two contracts: PASS.

Verdict: producer and final formal consumer are contract-confirmed offline.
No post-fix live transaction or checkpoint has yet been produced.

## E-FI-81: Contact-Wrench ZMP And Expected-Support Authority Closure

Date: 2026-07-27

Tier: deterministic S1 estimator/carrier, S2 formal connectivity and S3 strict
persistence contracts, including temporary-directory save/reload fixtures. No
simulator, training, external checkpoint mutation or live run occurred.

First-invalid closure:

- the formal Physics capture no longer calls the root/capture-point balance
  proxy or evaluates ZMP against a Repair-created foot polygon;
- G1 formal config installs separate filtered left/right foot-to-ground sensors
  with raw contact-point tracking;
- `contact-wrench-zmp-v1` computes world-frame ZMP/CoP from per-contact point,
  normal and normal-force magnitude;
- the sealed Clean continuation deterministically derives and hashes
  `clean-foot-pose-oriented-box-v1` `[K,6]` support/recovery envelopes, marked
  `clean_gmt_physics_only`; they do not enter actor or GMT observations;
- supported phases without a finite ground resultant fail closed and flight is
  explicit ZMP N/A.

Persistence identity:

- checkpoint-v5 now binds `zmp_estimator_id`, `support_envelope_id`,
  `actual_contact_id` and `expected_phase_id`;
- a previous checkpoint-v5 lacking this evidence identity rejects before actor,
  Critic, optimizer, sampler or normalizer mutation.

Focused evidence:

- `frontres_contact_wrench_zmp_contract.py`: PASS for golden resultant, world
  translation, contact/row permutation, changed application point, signed
  envelope margin, flight N/A, missing resultant and raw PhysX buffer unpack;
- `frontres_local_scenario_kernel_contract.py`: PASS for schema, provenance,
  hash mutation, horizon and immutability;
- `frontres_v015_two_role_reset_contract.py`,
  `frontres_v015_current_gmt_command_contract.py`,
  `frontres_v015_role_aligned_future_intent_contract.py`,
  `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_unmocked_observation_connectivity_contract.py`: PASS;
- `frontres_intent_physics_gain_contract.py` and
  `frontres_v015_transaction_route_contract.py`: PASS with v005/PPO-v004
  mathematics unchanged;
- `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_v015_policy_quality_identity_contract.py` and
  `frontres_v015_local_sentinel_connectivity_contract.py`: PASS;
- Python compilation for all modified production owners: PASS.

Verdict: source producer, immutable carrier, formal consumer isolation and
persistence are contract-confirmed offline. The official server IsaacLab raw
contact API, terrain filter resolution and finite supported-phase values remain
S4-unconfirmed. A bounded sensor-authority sentinel is required before any new
training lineage; proxy fallback, missing supported resultant, role/hash drift,
flight misclassification or Clean geometry reaching the actor is a hard stop.

## E-FI-82: Loaded-Support ZMP Applicability And GAIN-v006 Closure

Date: 2026-07-27

Tier: deterministic S1 evidence semantics, S2 formal connectivity and S3 strict
persistence contracts. No simulator, training, checkpoint mutation or live run
occurred.

First-invalid runtime fact:

- the bounded log exposed a row with valid raw contact-point payload but no
  vertical foot-ground resultant; the former v005 path incorrectly treated
  expected support alone as requiring a finite actual ZMP;
- contact-point count is not support load. A valid, actually unloaded foot is a
  physical outcome, not a malformed sensor payload.

Semantic closure:

- expected-supported plus actual-unloaded remains a scored Contact violation;
- ZMP applicability is role-specific and requires both expected support and
  actual ContactSensor load; an unloaded role carries ZMP N/A, never a zero
  margin or a fabricated support polygon;
- malformed/nonfinite raw sensor payload still fails closed, and
  expected-supported plus actual-loaded without a finite contact-wrench
  resultant also fails closed;
- scalar Intent-minus-repair Critic target, vector Physics constraints,
  one-action-K, grouped PPO projection and HSL remain unchanged.

Formal connectivity and identity:

- separate Repair/Noisy applicability is preserved from contact-wrench capture
  through one-action-K evidence, paired facts, return evidence, sealed
  transaction telemetry and the atomic held-out report;
- the active identities are `FRS-GAIN-v006` and
  `contact-loaded-phase_zmp-survival-physical-v2`;
- checkpoint format remains `frontres-v015-checkpoint-v5`, but strict resume
  rejects v005/schema-v1 before state mutation;
- atomic held-out reports use `frontres-v015-heldout-quality-report-v2` and
  serialize valid ZMP N/A as `null`, never as zero or a dropped row.

Focused evidence:

- `frontres_contact_wrench_zmp_contract.py`: PASS for valid no-load N/A,
  malformed/nonfinite payload rejection and loaded resultant behavior;
- `frontres_intent_physics_gain_contract.py`: PASS for expected-supported plus
  actual-unloaded Contact violation, role-specific ZMP N/A and loaded-missing
  fail-closed;
- `frontres_v015_one_action_k_contract.py`,
  `frontres_segment_diagnostics_contract.py` and
  `frontres_v015_transaction_route_contract.py`: PASS for end-to-end K evidence,
  separate role applicability and formal telemetry;
- `frontres_v015_checkpoint_resume_contract.py`: PASS, including explicit
  v005/schema-v1 rejection before mutation;
- `frontres_v015_policy_quality_heldout_contract.py`,
  `frontres_v015_policy_quality_save_reload_contract.py`,
  `frontres_v015_policy_quality_identity_contract.py` and
  `frontres_policy_quality_manifest_contract.py`: PASS for strict identity,
  fresh reload, role-specific N/A and atomic report v2;
- `frontres_v015_unmocked_observation_connectivity_contract.py`,
  `frontres_v015_local_sentinel_connectivity_contract.py`,
  `frontres_v015_grouped_candidate_adapter_contract.py`,
  `frontres_formal_runtime_audit_contract.py` and
  `frontres_policy_quality_atlas_contract.py`: PASS for formal route isolation,
  exact-one update, grouped metadata and source-linked governance;
- `frontres_design_contract_sentinel.py`: PASS for the current coordinated
  METHOD-v016 / TRAIN-v010 / GAIN-v006 / PPO-v004 / EVAL-v003 registry and
  archived v005 isolation;
- production-owner `py_compile`, modified JSON validation and
  `git diff --check`: PASS.

Verdict: GAIN-v006 loaded-support applicability is contract-confirmed end to end
offline. One bounded official IsaacLab sensor-authority S4 remains required to
confirm raw contact APIs, filter resolution, vertical-load thresholding and
finite loaded ZMP values. No new training lineage is admitted before that S4;
the pre-fix model lineage remains evidence-only.

## E-FI-83: Final Snapshot And Checkpoint-v5 Consumer Closure

Date: 2026-07-27

Tier: deterministic final-consumer and persistence contracts. No simulator,
training, checkpoint mutation outside temporary fixtures or live run occurred.

First-invalid owner and repair:

- the formal sentinel result already carried immutable GAIN-v006 per-step
  Repair/Noisy Contact and phase-ZMP evidence, but
  `run_frontres_v015_local_identity_sentinel()` removed the raw reports before
  the terminal JSON adapter and emitted no structured replacement;
- `scripts/rsl_rl/train.py::main()` returned immediately after the sentinel,
  bypassing the adjacent committed checkpoint-v5 save;
- the final snapshot now reuses the ordinary sealed-transaction serializer,
  and the sentinel exit calls the existing checkpoint owner, validates the
  complete v5/v016/v010/v006/v004 identity plus exact-one receipt, and removes
  a newly written artifact if post-serialization validation fails.

Focused evidence:

- `frontres_v015_local_sentinel_connectivity_contract.py`: PASS for role/hash
  alignment, Repair/Noisy K-step arrays, role-specific ZMP applicability/N/A,
  final JSON conversion, exact-one receipt, strict checkpoint identity,
  iteration rollback and invalid-artifact cleanup;
- `frontres_v015_local_sentinel_config_contract.py`,
  `frontres_v015_transaction_route_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py` and
  `frontres_v015_policy_quality_heldout_contract.py`: PASS;
- modified Python owners compile, and `git diff --check` passes;
- `frontres_stage_entrypoint_contract.py` still stops on its older hard-coded
  `frontres_segment_k=8` source assertion before reaching the new sentinel
  checks; the formal source now derives K from TRAIN-v010 curriculum. This is a
  separate stale static-contract boundary, not evidence against the final
  consumer route.

Verdict: final snapshot and checkpoint persistence are `contract-confirmed`.
They are not yet `live-confirmed`; the same bounded official sensor-authority
sentinel must emit both `[FrontRES v015 Live Snapshot]` with
`sealed_transaction_evidence` and `[FrontRES v015 Checkpoint Sentinel]`, and
must leave the named checkpoint-v5 artifact on disk.

## E-FI-84: Sensor-Authority Live Closure And Return N/A Repair

Date: 2026-07-27

Tier: bounded official live sentinel plus deterministic storage/transaction
regression. The subsequent long run stopped before its first optimizer update.

Live-confirmed facts from `v015_p4_zmp_sensor_s4b_gpu3.log`:

- one 2-Segment x 2-attempt transaction produced four valid policy rows, one
  FEMR action, K8 evidence and exactly one optimizer update;
- the final snapshot serialized Repair/Noisy Contact, role-specific phase-ZMP
  applicability and one legitimate Noisy ZMP N/A without zero filling;
- Contact and ZMP constraints were active, projection KKT violation was zero,
  Actor/std remained frozen in critic-only and Critic delta was nonzero;
- the adjacent `model_1.pt` passed strict checkpoint-v5 identity for
  METHOD-v016 / TRAIN-v010 / GAIN-v006 / PPO-v004 and contact-wrench evidence.

First long-run failure and repair:

- the first post-fix long transaction reached
  `FrontRESV015GainReturnEvidence.validate()` and stopped because its generic
  finite-on-valid loop still required `repaired_zmp_margin` to be finite on an
  otherwise valid row whose Repair ZMP was correctly N/A;
- the return validator now preserves one-dimensional shape checks, excludes
  role-specific ZMP margins from the generic finite loop, requires Repair ZMP
  to be finite exactly when `zmp_constraint_applicable` is true, and keeps it
  N/A otherwise;
- loaded Repair ZMP missing remains fail-closed; no Gain, constraint, PPO,
  Critic, curriculum or checkpoint formula changed.

Focused contracts PASS:

- `frontres_v015_grouped_candidate_adapter_contract.py`, including N/A when
  not applicable plus rejection of both finite-when-inapplicable and
  N/A-when-applicable;
- `frontres_intent_physics_gain_contract.py`,
  `frontres_v015_one_action_k_contract.py`,
  `frontres_segment_diagnostics_contract.py`,
  `frontres_v015_transaction_route_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py` and
  `frontres_v015_local_sentinel_connectivity_contract.py`;
- modified owners compile and `git diff --check` passes.

Verdict: official sensor authority is `live-confirmed`; the return-evidence N/A
repair is `contract-confirmed`. The post-fix long lineage remains unrun beyond
the failed first transaction and must resume again from the live-confirmed
`model_1.pt` after synchronization.

## E-FI-85: Repair/Noisy ZMP Applicability Carrier Closure

Date: 2026-07-27

Tier: deterministic S1/S2/S3 cross-module closure; no simulator, training or
live run.

First-invalid structural owner:

- the phase evaluator already produced separate Repair and Noisy per-step ZMP
  applicability, but `FrontRESIntentPhysicsGainResult` and
  `FrontRESV015GainReturnEvidence` retained only the Repair/PPO aggregate mask;
- downstream validators and reports therefore inferred Noisy applicability
  from finite/NaN margins. This duplicated authority and caused successive
  stale `finite-on-valid` failures when a valid role was physically unloaded.

Implemented closure:

- GainResult and ReturnEvidence now carry explicit
  `zmp_applicable_repaired` and `zmp_applicable_noisy`; the existing
  `zmp_constraint_applicable` is asserted to be the Repair-only PPO alias;
- Repair and Noisy margins are finite exactly under their own valid-row masks;
  paired `physics_zmp_gain` is finite exactly when both role masks are true;
- local diagnostics recompute K-step applicability only as an oracle, require
  equality with ReturnEvidence, and preserve both aggregates in the final live
  snapshot; the held-out atomic report performs the same identity check;
- no Gain/PPO/Critic/HSL/checkpoint formula or identity changed.

Focused evidence PASS:

- `frontres_intent_physics_gain_contract.py` and
  `frontres_v015_grouped_candidate_adapter_contract.py`: role-asymmetric N/A,
  both-applicable/both-inapplicable, malformed finite/N-A and Repair alias;
- `frontres_segment_diagnostics_contract.py` and
  `frontres_v015_policy_quality_heldout_contract.py`: K-step-to-aggregate
  identity and role-specific final report serialization;
- `frontres_v015_transaction_route_contract.py`,
  `frontres_v015_local_sentinel_connectivity_contract.py`,
  `frontres_segment_grouped_ppo_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py` and
  `frontres_v015_policy_quality_save_reload_contract.py`: formal grouped update,
  Repair-only PPO authority, exact-one update, final snapshot and persistence
  remain connected;
- modified Python owners compile and `git diff --check` passes.

Repository-chain audit:

- the first aggregate run found one real misplaced assertion
  (`expected_support` referenced from the legacy fixed-tape materializer) and
  eight stale test/Architecture adapters left behind by the v015/v006 route;
- the assertion now belongs to the local-scenario materializer, current PPO and
  rollout symbols are represented by the offline stubs, launcher assertions
  follow dynamic K/checkpoint ownership, and both runtime Architecture source
  anchors resolve to their current `B*` comments;
- `frontres_segment_all_contract_suite.py` completes with 63/63 markers and
  zero failures after these repairs.

Verdict: Repair/Noisy applicability carrier closure is
`contract-confirmed`. The next runtime action is a single post-E-FI-85 server
validation/continuation; no further local probe or semantic redesign is
required unless that run exposes a different first-invalid owner.

## E-FI-86: Raw-Contact Capacity And Formal K8 Long-Training Closure

Date: 2026-07-28

Tier: source/focused deterministic plus official live-confirmed formal training.

Evidence:

- the relevant source/test diff in commit
  `648e1db9555804cf5841ebc20181b5dcfc204aeb` changes the existing raw-contact
  view owner and its focused contract for this boundary; unrelated repository
  bookkeeping in that commit is not method evidence;
- `frontres_segment_live_probe.py::_ensure_frontres_raw_contact_view()` now
  provisions 256 contacts per foot/environment, giving capacity `2048` for the
  current 8-env route instead of `128`;
- `frontres_contact_wrench_zmp_contract.py` asserts the 8-env capacity and adds
  `test_raw_contact_capacity_saturation_remains_fail_closed`, so an exactly
  saturated PhysX payload is still rejected rather than treated as complete;
- `v015_gain_v006_postfix_to_model2000_gpu3.log` is 123549 lines / 32 MB and
  contains no Traceback, RuntimeError, ValueError, assertion, CUDA fault,
  capacity-saturation error or applicability failure;
- the log contains 1999 transaction telemetry records, 1999 formal-train
  receipts and 1999 `step_delta=1 committed=1` receipts. Iterations 1--1999
  reach `absolute_iter=2000`; 40 periodic saves end with
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-28_14-06-10_V015_GAIN_V006_POSTFIX_TO_MODEL2000/model_2000.pt`;
- all 1999 rows use METHOD-v016 / GAIN-v006 / PPO-v004 / TRAIN-v010,
  `paired-intent-minus-repair-v1`, the v2 Contact/loaded-phase-ZMP/survival
  schema, deployment-Noisy q29 provenance and K8;
- every transaction has four valid policy rows, sealed pairwise scenario/hash
  identity and equal attempt/Segment mass. The two `[1.0]` motion-mass cases
  are correct because both Segments came from one underlying motion;
- curriculum phase counts are 199 critic-only, 500 actor-warmup and 1300 joint.
  Critic delta is nonzero in all 1999 updates; actor/std delta is nonzero in all
  1800 actor-warmup/joint updates and zero during critic-only;
- projection counts are 625 constraint-recovery, 1120 projected-Intent, 7
  Intent-feasible, 69 no-common-first-order-descent and 178 no-empirical-
  direction. `constraint_kkt_max_violation` is exactly zero for all updates.

Facts:

- this change increases evidence-transport headroom; it does not change the
  contact-wrench ZMP estimator, GAIN-v006 applicability, PPO-v004 projection,
  TRAIN-v010 curriculum, actor observation, or checkpoint identity;
- E-FI-86 is `live-confirmed`: the larger buffer eliminates the prior capacity
  stop without weakening truncation detection;
- formal K8 training connectivity, exact-one persistence, applicability and
  postscale KKT are closed through iteration 2000;
- aggregate training telemetry is informative but not independent quality
  evaluation. First-to-final 200-row summaries change as follows:
  `gain_mean -0.0170 -> -0.0104`, positive Gain `0.341 -> 0.386`,
  `intent_mean -0.00656 -> 0.000077`, positive Intent `0.465 -> 0.483`;
  action L2 remains non-collapsed near `0.0104`; Repair-vs-Noisy Contact gain
  stays positive, survival is approximately one, and Physics gain is positive
  on average. These are weak improvements, not a deployment-quality claim;
- no additional matched-route or all-variable comparison is required for
  engineering closure.

Open risks / stop:

- the emitted remote `model_2000.pt` was not copied locally in this evidence
  pass, so its payload was not independently reopened; the live log proves the
  formal save receipt and prior contracts prove the strict v5 producer/loader;
- policy efficacy, sustained-lean/unplanned-step behavior, fresh deployment
  reload and deployment composition remain separate unresolved claims;
- the old pre-KKT-fix `model_2000.pt` remains historical evidence only. The
  authoritative post-fix artifact is distinguished by the dated output path
  above.

Next: use the existing policy/deployment route with the dated post-fix
`model_2000.pt` to judge actual motion quality. Do not reopen capacity,
applicability or matched-route debugging without a new first-invalid runtime
fact.

## E-FI-87: Policy-Only Deployment Final-Consumer Readiness

Date: 2026-07-28

Scope: close the existing v015 deployment-composition final consumer for the
post-fix `model_2000.pt`; no matched zero/HSL route, training, optimizer update,
Gain/PPO/HSL change, or live execution.

Evidence:

- the five focused deployment/Contact contracts exit 0;
- `frontres_segment_all_contract_suite.py` exits 0 with
  `contract_count=63 failed_count=0 total_marker_count=63`;
- `python -m py_compile` and `bash -n run/run_v015_model2000_deployment_demo.sh`
  exit 0.

Facts:

- the formal CLI separates the ordinary Clean source used only for frame-0
  physical reset and expected Physics evidence from the immutable Noisy carrier
  consumed by FEMR/GMT;
- the policy-only report uses filtered ContactSensor forces and raw
  contact-wrench ZMP against Clean expected Contact/support envelopes; the old
  foot-height actual Contact and root/capture-point proxy are no longer the
  formal deployment report authority;
- atomic JSON/S4 telemetry now exposes action, Intent q29 error,
  expected/actual Contact, timing-aware mismatch, phase-ZMP
  applicability/violation/recovery, survival, signed lateral-roll cumulative
  lean, and unplanned Contact events with training feedback fixed at zero;
- the evaluator freezes the policy and every 158D/GMT/privileged/teacher
  observation normalizer before the first observation read, fingerprints their
  state, and restores their original modes on success or exception;
- `run/run_v015_model2000_deployment_demo.sh` materializes one fixed `local_rp`
  carrier and dispatches the dated post-fix `model_2000.pt` through the existing
  formal IsaacLab evaluator.

Open risk: this is deterministic S1/S2/final-serializer evidence. Real
simulator values and policy/Demo quality remain unconfirmed until the single
deployment script is run and its JSON/log is reviewed.

## E-FI-88: Coordinated K x M Checkpointed Curriculum Activation

Date: 2026-07-28

Tier: user-confirmed semantic decision, source audit and document-only contract
activation. No training source, test, simulator, checkpoint IO, optimizer
update, deployment or live run occurred.

Observed source/runtime facts:

- E-FI-86 contains 1999 committed K8 transactions and every transaction has
  exactly two Segment sources, four Repair policy attempts and equal attempt
  mass, so M remained two for the entire pilot;
- the current launcher supplies only `FRONTRES_V015_K_CURRICULUM=8:200:500:0`;
- `frontres_segment_live_sampler.py` calls
  `plan_frozen_policy_transaction(..., minimum_policy_attempts=2)` and derives
  wider transaction layouts from sampler `rollout_trial_count` rather than an
  explicit global M identity;
- the generic legacy sampler step calls `sampler.update_with_probe()`, while
  the v015 formal transaction/update path does not use that owner to advance M;
- changing only `num_envs` therefore cannot prove more attempts per Segment:
  it may select additional M2 Segment sources instead;
- checkpoint-v5 binds the complete v010 K-only schedule fingerprint and rejects
  a different schedule on strict resume.

Confirmed decision:

- activate FRS-TRAIN-v011 while preserving METHOD-v016, GAIN-v006, PPO-v004,
  EVAL-v003, HSL-v1, 928/158/770 authority, one-action-K and one scalar Critic;
- freeze the coordinated schedule
  `(K8,M2,200,500,1300) -> (K16,M3,300,300,900) ->
  (K32,M4,400,300,625)` with exactly two Segment sources;
- freeze absolute review boundaries `2000,3500,4825,6500,8000` and maximum
  absolute iteration 8000;
- define role env widths `8,12,16`, one policy row per Repair attempt, equal
  motion -> Segment -> attempt mass and exactly one update per sealed
  transaction;
- bind schedule/max/review/stage/K/M/row/receipt identity in strict
  `frontres-v015-checkpoint-v6`;
- preserve E-FI-86 `model_2000.pt` as the K8/M2 checkpoint-v5 pilot only. The
  v011 campaign starts fresh from strict HSL-v1 and checkpoint-v5 rejects
  before mutation.

Governed artifacts:

- active contract and registry now point to
  `FRS-TRAIN-v011-coordinated-k-m-checkpointed-curriculum.md`;
- FRS-TRAIN-v010 is archived as superseded;
- Concept Figure keeps the existing Segment Replay, K-step Curriculum and
  Actor & Critic Warmup blocks while their concise text reflects coordinated
  K x M stages;
- Method-to-Code and Policy Quality Architecture mark the current v010/v5
  source as contract-mismatch rather than implemented v011;
- the current plan, checklist and task canvas freeze P5-B as one offline
  engineering closure and P5-C as the later material-runtime boundary.

Open implementation boundary:

- no v011 parser, active-M carrier, exact-M formal owner, stage env-width
  guard, checkpoint-v6 producer/loader, launcher or deterministic test exists;
- no v011 training command is authorized until one P5-B engineering closure
  implements and verifies this complete route;
- P4 policy-only deployment quality remains independently open and does not
  prove or block the exact-M engineering route.

Next: execute P5-B in one unit across the existing schedule, sampler,
transaction, telemetry, orchestration and checkpoint owners; run focused
S1/S2/S3 contracts and refresh governance evidence; stop before simulator,
training or live execution.

## E-FI-89: Exact-M And Checkpoint-v6 Offline Engineering Closure

Date: 2026-07-28

Tier: deterministic S1/S2/S3 source and persistence evidence. No simulator,
training, optimizer-bearing live run or deployment composition occurred.

Implemented facts:

- the only accepted formal schedule is
  `(8,2,200,500,1300) -> (16,3,300,300,900) ->
  (32,4,400,300,625)`, with maximum absolute iteration 8000 and review
  boundaries 2000/3500/4825/6500/8000;
- the formal planner selects exactly two Segment sources and expands each to
  exact active M attempts; state-driven sampler trial counts cannot change the
  formal width;
- formal role width is fail-closed at `num_envs=4*M`, yielding policy rows
  `2*M` and Repair/Noisy role rows `4*M`;
- sealed transaction request, receipt, final telemetry and live snapshot carry
  active K/M, selected-Segment count, policy-row count and role-row count;
- grouped PPO-v004 still performs exactly one optimizer update after the whole
  transaction is sealed; METHOD-v016, GAIN-v006, PPO-v004, HSL and Physics
  mathematics are unchanged;
- strict `frontres-v015-checkpoint-v6` binds the full schedule, stage/K/M,
  maximum/review boundaries, row counts, optimizer/sampler/normalizers and
  committed receipt. Checkpoint-v5 and malformed schedules reject before
  mutable restore.

Fresh evidence:

- `frontres_segment_warmup_contract.py`: PASS;
- `frontres_segment_sampler_contract.py`: PASS for M2/M3/M4 exact expansion
  and sampler-state immutability;
- `frontres_v015_transaction_route_contract.py`: PASS for sealed grouped
  exact-one update and active TRAIN-v011 identity;
- `frontres_v015_unmocked_observation_connectivity_contract.py`: PASS for the
  unchanged 928/158/770 and one-action-K route through exact-one update;
- `frontres_v015_local_sentinel_connectivity_contract.py`: PASS through the
  final active-M serializer and checkpoint-v6 sentinel;
- `frontres_v015_checkpoint_resume_contract.py`: PASS for v6 round-trip,
  tamper rejection, v5 pre-mutation rejection and fresh inference equality;
- launcher, entrypoint, formal-runtime Atlas, policy-quality identity/save-
  reload and design-contract focused contracts: PASS;
- `frontres_segment_all_contract_suite.py`:
  `contract_count=63 failed_count=0 total_marker_count=63`;
- `git diff --check`: PASS.

Verdict: P5-B is offline-complete. P5-C is now blocked only by explicit
material runtime authority. First real K16/M3 and K32/M4 transactions,
checkpoint-boundary quality and long-run policy efficacy remain unconfirmed.

## E-FI-90: Actual Actor-Update Authority Closure

Date: 2026-07-29

Tier: runtime-rooted defect audit plus deterministic S1/S2/S3 closure. No new
simulator, training, live run or checkpoint IO occurred.

First-invalid runtime fact:

- `v015_p5c_v011_k8m2_to_2000_gpu3.log` completed the strict K8/M2 block at
  absolute iteration 2000, but 240 actor-enabled transactions reported
  `NO_EMPIRICAL_DIRECTION` or `NO_COMMON_FIRST_ORDER_DESCENT` while Actor/std
  parameters still changed;
- those rows had no installed Actor gradient. Historical Adam momentum and
  coordinate-wise state therefore remained an ungoverned update authority:
  zeroing the current gradient was insufficient to enforce the PPO-v004
  no-direction contract;
- this is an implementation mismatch, not a change to METHOD-v016, GAIN-v006,
  PPO-v004 mathematics, TRAIN-v011, exact-M, exact-one or checkpoint-v6.

Implemented closure:

- the existing PPO-v004 owner now carries the exact active constraint-gradient
  vectors to a post-optimizer authority check;
- the existing formal transaction owner still performs exactly one optimizer
  call, then measures Adam's actual Actor/std candidate increment;
- critic-only, `NO_EMPIRICAL_DIRECTION` and
  `NO_COMMON_FIRST_ORDER_DESCENT` restore Actor/std parameters and their Adam
  state bit-exactly while retaining the legal Critic update;
- permitted Actor increments are projected into the same active Physics
  halfspaces before commit, and KKT/postconditions are computed from the
  committed parameter delta rather than only the installed gradient;
- formal telemetry now distinguishes gradient-projection KKT from
  actual-update KKT and records optimizer-candidate norm, committed norm and
  optimizer-state restoration.

Deterministic evidence:

- the new historical-momentum regression failed before the repair because
  Actor/std changed under `NO_EMPIRICAL_DIRECTION`, then passed after the
  repair with exact parameter and Adam-state equality plus nonzero Critic
  delta;
- a critic-only K8 recalibration fixture with pre-existing Actor Adam momentum
  also preserves Actor/std parameters and optimizer state exactly;
- a valid actor-enabled fixture proves nonzero Adam candidate, nonzero
  committed Actor delta, actual-update directional derivatives within
  tolerance, finite Critic delta and exactly one optimizer call;
- grouped PPO, strict checkpoint-v6 resume, formal-runtime Architecture and
  policy-quality Architecture focused contracts pass;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=63 failed_count=0 total_marker_count=63`; modified Python
  owners compile and `git diff --check` passes.

Runtime boundary and lineage limitation:

- the new actual-update telemetry and restoration behavior remain offline-
  confirmed until the next official transaction;
- the existing strict checkpoint-v6 `model_2000.pt` remains the authorized
  TRAIN-v011 parent for K16/M3, but its K8 lineage contains the 240 pre-fix
  momentum-bypass Actor deltas. The repair is not retroactive and this fact
  must remain attached to later policy-quality claims;
- the next material action is the declared K16/M3 critic-only recalibration.
  Its first transaction must show M3/12-env identity, Actor/std and Actor Adam
  state freeze, same-Critic update, actual-update KKT, exact-one update and
  unchanged checkpoint-v6 curriculum fingerprint.

## E-FI-91: Formal Update Maintainability Closure

Date: 2026-07-29

Tier: behavior-preserving ownership refactor plus deterministic S1/S2/S3
regression. No simulator, training, live run or checkpoint IO occurred.

Problem and boundary:

- E-FI-90 closed the actual Actor-update defect, but its exact-one optimizer
  call, Adam-state restoration, actual-delta commit and postcondition checks
  were split across the already-large formal runner files;
- this did not invalidate the method, but it increased integration debt and
  made future optimizer-authority repairs unnecessarily dependent on the live
  orchestration layer;
- METHOD-v016, GAIN-v006, PPO-v004 mathematics, TRAIN-v011, HSL,
  checkpoint-v6, 928/158/770 observation authority and all public telemetry
  identities were frozen for this closure.

Implemented ownership closure:

- the existing `frontres_segment_ppo.py` algorithm owner now performs the one
  shared optimizer call, measures Adam's candidate Actor/std delta, applies the
  actual parameter-space projection/restoration authority and returns an
  immutable commit result;
- `frontres_segment_live_probe.py` retains sealed-transaction orchestration,
  snapshots, Critic/non-Critic delta accounting and detached telemetry only;
- the final actual-update telemetry validator is centralized in the existing
  `frontres_segment_diagnostics.py` owner, and
  `frontres_segment_live_training.py` delegates to it before serialization;
- no new module, runner, optimizer step, checkpoint field, method formula or
  training-state feedback path was introduced.

Evidence:

- Python compilation passes for the PPO, diagnostics, live-probe and
  live-training owners;
- `frontres_segment_diagnostics_contract.py`,
  `frontres_v015_transaction_route_contract.py`,
  `frontres_segment_grouped_ppo_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_formal_runtime_audit_contract.py` and
  `frontres_policy_quality_atlas_contract.py` pass;
- both runtime Architecture files point to current source boundaries after the
  ownership move;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=63 failed_count=0 total_marker_count=63`.

Verdict and remaining boundary:

- the repository does not require a broad rewrite before continuing. The
  highest-risk optimizer-authority responsibility is now isolated in its
  algorithm owner and diagnostics no longer duplicates its serializer checks;
- the two live runner files remain large orchestration modules and may receive
  further behavior-preserving extraction when a concrete owner boundary is
  touched, but that is not a prerequisite for K16/M3;
- runtime behavior of the refactored owner remains unconfirmed until the next
  already-planned official K16/M3 transaction. No extra standalone live probe
  is authorized or required.

## E-FI-92: Interface-Oriented Formal Route Closure

Date: 2026-07-29

Tier: behavior-preserving ownership refactor plus deterministic S1/S2/S3
regression. No simulator, training, deployment, live run or checkpoint IO
occurred.

Implemented boundary:

- the MOSAIC host files remain byte-identical; existing public wrapper calls
  are the compatibility shell rather than implementation owners;
- `frontres_interfaces.py` is the lightweight authority for v016/v006/v004/
  v011 identity, 928/158/770 observation authority, exact two-Segment x M row
  shape, committed-update receipt, final telemetry and the lifecycle/update/
  mode ports actually consumed by the engine;
- `FrontRESStage3Engine` is the unique public formal transaction state machine;
  it opens the barrier, forbids optimizer mutation during collection,
  recollects only declared invalid evidence, commits one update, validates its
  receipt and closes the ordinary training lifecycle;
- `MosaicFrontRESStage3Backend` is the only adapter that knows the existing
  live-probe primitives. The public update-loop dispatcher no longer imports
  those private functions;
- `frontres_segment_runner_boundary.py` now exposes one explicit run mode and
  rejects mixed modes; final transaction telemetry is validated through the
  typed identity/shape view before serialization;
- the FrontRES package facade now lazily resolves its existing public exports,
  allowing interface imports without eagerly constructing IsaacLab-dependent
  modules.

Evidence:

- `frontres_interface_refactor_contract.py` passes under both system Python and
  the project runtime, covering K8/M2, K16/M3, K32/M4, exact-one update,
  provider-time mutation rejection, bounded recollection, exception cleanup,
  lazy import and mixed-mode rejection;
- `frontres_v015_transaction_route_contract.py`,
  `frontres_segment_runner_boundary_contract.py`,
  `frontres_segment_live_sentinel_contract.py`,
  `frontres_segment_live_update_loop_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py`,
  `frontres_segment_diagnostics_contract.py` and
  `frontres_formal_runtime_audit_contract.py` pass;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=64 failed_count=0 total_marker_count=64`;
- modified Python modules compile and `git diff --check` passes;
- SHA-256 values for `train.py`, `on_policy_runner.py`, `commands.py`, base
  `ppo.py` and `mosaic.py` match the frozen pre-refactor baseline.

Residual boundary:

- `frontres_segment_live_probe.py` remains a large concrete compatibility
  backend. It is no longer the public orchestration owner, so internal physical
  splitting can be performed by responsibility when a concrete backend change
  requires it; doing so is not necessary to add a new engine/backend or to
  change formal transaction sequencing.

## E-FI-93: Interface Discipline One-Shot Completion

Date: 2026-07-29

Tier: behavior-preserving S1/S2/S3 interface and package-composition closure.
No simulator, training, deployment or live run occurred. Deterministic tests
used only fake/CPU owners and temporary checkpoint artifacts.

Implementation evidence:

- `frontres_interfaces.py` now exposes consumer-shaped request, committed
  receipt, telemetry, observation-authority and exact K x M records. Invalid
  identities, K/M rows, checkpoint format and exact-one counts reject before
  the final consumer proceeds.
- `frontres_stage3_engine.py` owns the only public transaction Unit of Work:
  `idle -> collecting -> committing -> idle`. Its backend and bindings are
  injected by `frontres_segment_live_update_loop.py`; the engine does not
  import or locate live-probe implementations.
- formal training, sampler, checkpoint, warmup, Gain, storage, diagnostics and
  runtime owners use normal package imports. This removes duplicate module
  identities for the v011 schedule, formal audit switch and Physics evaluator.
- live training consumes public reset/observation/rollout/Gain and legacy-eval
  batch seams. Physics consumers use public `frontres_balance.py` gateways.
  Compatibility aliases remain only for historical callers and contracts.
- isolated contracts use `frontres_contract_imports.py` as a lightweight test
  composition root. They no longer depend on production file-path fallback or
  empty pseudo-packages that contradict normal owner imports.

Executed evidence:

- modified Python owners passed `python -m py_compile`;
- focused interface, transaction, one-action-K, sampler, checkpoint,
  diagnostics, formal-runtime, policy-quality and sequence contracts passed;
- `frontres_segment_all_contract_suite.py` reported
  `contract_count=64 failed_count=0 total_marker_count=64`;
- `git diff --check` passed;
- the diff for frozen host files `train.py`, `on_policy_runner.py`,
  `commands.py`, base `ppo.py` and `mosaic.py` was empty;
- runtime and policy-quality Architecture source links were mechanically
  realigned to their unchanged `# Bn:` evidence markers and both Atlas
  contracts passed.

Verdict:

- the formal FEMR route is interface-oriented at its change-prone boundaries:
  MOSAIC host adapter, transaction lifecycle, request/receipt identity,
  telemetry, Physics gateway and package composition;
- METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, checkpoint-v6,
  928/158/770, one-action-K and exact-M/exact-one semantics are unchanged;
- the remaining size of `frontres_segment_live_probe.py` is localized
  compatibility-facade debt. Further splitting is justified only when a
  concrete internal responsibility changes; it is not a blocker for formal
  training or adding a new backend implementation.

## E-FI-94: Live-Probe Responsibility-Owner Extraction

Date: 2026-07-29

Tier: behavior-preserving S1/S2 architecture and deterministic connectivity
closure. No simulator, training, deployment or live run occurred.

Change evidence:

- the 5840-line `frontres_segment_live_probe.py` implementation was separated
  into runtime-types, probe-logging, policy-update, reset, storage/Gain,
  one-action-K, formal-transaction, Physics, rollout, reporting and legacy-probe
  owners. The old module is now a 225-line import-only compatibility facade;
- active live-training, transaction-loop and policy-quality consumers import
  public owner seams directly. The frozen MOSAIC `on_policy_runner.py` remains
  the sole production facade consumer for its legacy probe entry;
- cross-owner imports use public names, the extracted dependency graph is
  acyclic and every owner is below 1000 lines. Physics and one-action-K expose
  patchable public gateways instead of requiring facade-global replacement;
- legacy characterization contracts now replace dependencies at the actual
  owner boundary. No tensor formula, transaction ordering, checkpoint payload,
  role layout, K/M schedule or actor/GMT observation authority changed;
- repository, formal-runtime and policy-quality Architecture links now name the
  extracted owners rather than stale lines in the former large module.

Executed evidence:

- `frontres_interface_refactor_contract.py` passes its facade-body, public
  import, acyclic dependency, owner-size and direct-consumer assertions;
- focused live-probe, one-action-K, Contact/ZMP, transaction, grouped-PPO,
  formal-runtime and policy-quality contracts pass;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=64 failed_count=0 total_marker_count=64`;
- modified Python files compile, Architecture JSON parses, both source-link
  Atlas contracts pass and `git diff --check` passes;
- the frozen MOSAIC host file diff remains empty.

Verdict:

- Phase B is complete. A change to reset, one-action-K, Physics, optimizer
  update, transaction assembly or reporting now starts in one named owner and
  does not require reading the compatibility facade;
- E-FI-93 remains the typed outer transaction/Unit-of-Work evidence; E-FI-94
  closes the inner responsibility split. The active training cursor remains
  P5-C K16/M3 and is unaffected by this offline refactor.

## E-FI-95: Adjacent Runner-Owner Interface Closure

Date: 2026-07-29

Tier: behavior-preserving S1/S2/S3 ownership and deterministic connectivity
closure. No simulator, training, deployment or live run occurred.

Change evidence:

- `frontres_segment_transaction.py` is the single frozen-policy/exact-M
  plan/accumulator owner; runtime records and formal commit no longer depend on
  transaction definitions inside the sampler facade;
- `frontres_segment_sampler_reporting.py` owns evidence projection and human
  reporting, while `frontres_segment_live_sampler.py` retains selection,
  materialization and compatibility orchestration;
- `frontres_segment_training_telemetry.py` owns fail-closed committed telemetry
  and removes the reverse formal-transaction dependency on the training loop;
  legacy periodic/offline evaluation is isolated in
  `frontres_segment_training_evaluation.py`;
- `frontres_checkpoint_quality.py` owns read-only strict HSL-v1/Stage3-v6
  identity inspection. `frontres_checkpointing.py` remains the only mutable
  save/load gateway, consumes only the public identity API, and preserves the
  payload format unchanged;
- active v015 deployment and held-out evaluators contain no legacy sequence
  plan or counterfactual actor implementations. Those compatibility paths live
  in `frontres_segment_legacy_sequence_eval.py` and
  `frontres_policy_quality_legacy.py`;
- isolated tests now patch/load the real owner and lightweight package seam.
  Production code gained no file-path fallback, service locator or second
  mutable transaction owner.

Executed evidence:

- focused exact-M sampler/transaction, pseudo-training, telemetry,
  sequence-evaluation, policy-quality identity/held-out/state/executor,
  checkpoint-v6 resume and actual save/fresh-reload contracts pass;
- `frontres_interface_refactor_contract.py` passes the Phase C owner and reverse
  dependency assertions, including active/legacy policy-quality import in both
  clean-process load orders and rejection of cross-owner private checkpoint
  imports;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=64 failed_count=0 total_marker_count=64`;
- modified Python compiles; repository/runtime/policy-quality Architecture JSON
  parses; formal-runtime and policy-quality source-link contracts pass;
- `git diff --check` passes and frozen MOSAIC hosts `train.py`,
  `on_policy_runner.py`, `commands.py`, base `ppo.py` and `mosaic.py` have no
  Phase C diff.

Verdict:

- Phase C is complete. Changes to exact-M lifecycle, sampler reporting,
  transaction telemetry, quality artifact inspection, deployment composition
  or legacy evaluation start in one named owner rather than a multi-purpose
  hotspot;
- METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, checkpoint-v6,
  928/158/770, one-action-K and exact-M/exact-one semantics are unchanged. The
  active runtime cursor remains P5-C K16/M3.

Reviewed hotspot exceptions:

- `frontres_segment_live_sampler.py` (1312 lines) remains the single
  selection/materialization imperative shell; sampler, fixed-scenario and
  exact-M contracts contain it. Re-audit before adding a non-materialization
  responsibility or at 2000 lines;
- `frontres_segment_training_evaluation.py` (1549 lines) is an isolated legacy
  evaluation owner. Sequence-evaluation characterization contains it; retire
  it when the v002 compatibility route is removed rather than splitting it
  into new public helpers;
- `frontres_checkpointing.py` (1926 lines) remains one mutable HSL-v1/Stage3-v6
  Unit-of-Work gateway. Resume, pre-mutation reject and actual save/fresh-load
  contracts contain it; re-audit before any new format or persistence backend;
- `frontres_segment_sequence_eval.py` (1587 lines) and
  `frontres_policy_quality_eval.py` (1454 lines) remain deep active evaluator
  owners. Their S1/S2/S3 evaluation contracts contain them; re-audit when a
  second production entrypoint or independent report schema is introduced.

## E-FI-96: Domain, Evidence, Projection And Dependency Owner Closure

Date: 2026-07-29

Tier: behavior-preserving S1/S2/S3 ownership and deterministic connectivity
closure. No simulator, training, deployment or live run occurred.

Change evidence:

- `frontres_segment_planning.py`, `frontres_local_scenario.py` and
  `frontres_segment_legacy_scenario.py` own transaction planning, active v015
  immutable local scenarios and retired fixed-Noisy lifecycle respectively;
  `frontres_segment_sampler.py` retains stateful selection, priority, budget
  and persistence;
- `frontres_segment_storage_records.py`, `frontres_segment_evidence.py`,
  `frontres_segment_grouped_adapter.py` and
  `frontres_segment_rollout_storage.py` own records, paired one-action-K facts,
  grouped adaptation and mutable storage. `frontres_segment_storage.py` is a
  41-line compatibility facade with no semantic body;
- `frontres_constraint_projection.py` owns grouped constraint projection and
  actual Adam Actor/std commit/restore. `frontres_segment_ppo.py` retains
  grouped loss/reduction. Update validation, local evaluation and report
  formatting moved to named diagnostic owners behind a 51-line facade;
- production FrontRES modules use normal package imports. Formal transaction
  dispatch uses the runner public port, and policy-quality request/state
  interfaces break the former evaluator/formal-owner dependency cycle;
- active consumers import their concrete public owner. Compatibility re-exports
  remain only for frozen-host and historical contract surfaces.

Executed evidence:

- scenario lifecycle/provenance, sampler, storage, grouped candidate,
  one-action-K, grouped PPO, v015 transaction, diagnostics, policy-quality
  state/evaluator/held-out and Stage-1 cache contracts pass;
- `frontres_interface_refactor_contract.py` rejects semantic bodies in storage
  and diagnostics facades, production file-path dynamic loading, reverse formal
  transaction dispatch and the policy-quality owner cycle;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=64 failed_count=0 total_marker_count=64`;
- Runtime and Policy-Quality Atlas source links resolve to exact current
  `# Bn:` boundaries; their contracts and JSON parse pass;
- modified Python compiles, `git diff --check` passes, and frozen MOSAIC hosts
  `train.py`, `on_policy_runner.py`, `commands.py`, base `ppo.py` and
  `mosaic.py` have no Phase D diff.

Verdict:

- Phase D is complete. Scenario, evidence/storage, projection/actual commit,
  diagnostics and policy-quality shared state changes now start in one named
  owner instead of a multipurpose facade;
- METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, EVAL-v003, checkpoint-v6,
  928/158/770, one-action-K and exact-M/exact-one semantics are unchanged;
- the active runtime cursor remains P5-C. No additional refactor gate is needed
  before the existing K16/M3 campaign resumes.

## E-FI-97: Evaluation Legacy-Isolation And Public-Dependency Closure

Date: 2026-07-30

Tier: behavior-preserving S1/S2/S3 interface and deterministic connectivity
evidence. No simulator, training, deployment or live run occurred.

Change evidence:

- runner evaluation connectors now depend directly on
  `frontres_segment_training_evaluation.py`; live-training no longer re-exports
  Evaluation functions or owns duplicate legacy Gain-name tables;
- legacy sequence planning is consumed directly from
  `frontres_segment_legacy_sequence_eval.py`; the active v015 deployment
  evaluator no longer exposes legacy sequence-plan compatibility exports;
- the active policy-quality entry is v015-only and fails closed instead of
  selecting legacy semantics from a runtime boolean. Historical matched-route
  evaluation has an explicit legacy entry;
- policy-quality tensor snapshots and JSON conversion now cross module
  boundaries through public owner names. Static contracts reject private and
  reverse dependencies.

Executed evidence:

- `python -m py_compile` passes for every changed production module;
- focused interface-refactor, v015 evaluation-isolation, policy-quality
  state/entrypoint/executor/counterfactual/real-owner/Q2D, and sequence
  contracts pass;
- the aggregate deterministic suite reports 63 passing contracts and one
  unrelated static Atlas failure because
  `note/architecture/runtime/05_policy_quality_audit.data.json` is absent in
  the current user worktree. The file was not restored or recreated.

Verdict:

- Evaluation dependency direction and legacy isolation are closed without
  changing METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, checkpoint-v6,
  observation authority, transaction identity, update behavior or serialized
  report values;
- FRS-EVAL-v003 still requires paired full-sequence Baseline/Repair evaluation,
  but the current v015 composition executor remains Repair-only. Baseline
  initial-state/reset and stochastic-pairing semantics are unconfirmed; this is
  a method contract mismatch, not an implementation claim.
- `frontres_segment_sequence_eval.py` still crosses the active runtime boundary
  through private runner/provider access. E-FI-97 does not claim that Gateway
  boundary closed; it remains a P1 `Inappropriate Intimacy` finding under
  FRS-ENG-v001.

## E-FI-98: Paired Deployment Composition And Runtime Gateway Closure

Date: 2026-07-30

Tier: deterministic S1/S2 interface and offline connectivity evidence. No
simulator, training, deployment or live run occurred.

Change evidence:

- `frontres_runtime.py::FrontRESV015DeploymentRuntimeGateway` is the sole active
  owner of runner-private connectors, observation/normalizer access, FEMR/GMT
  inference, environment step, ContactSensor/contact-wrench ZMP capture,
  route-start snapshot/restore and training-state fingerprint;
- the sequence evaluator captures one canonical route-start image and restores
  it before Baseline and Repair. Both branches reuse the same carrier and state
  hash; Baseline uses exact-zero 6D correction without invoking FEMR, while
  Repair invokes FEMR once per evaluated frame;
- the atomic report now contains both branch trajectories and paired intent
  error/failure-count improvements. It fails closed on route-hash drift,
  Baseline FEMR use, nonzero Baseline action, shape/row misalignment or mutable
  training state;
- dead evaluator-local runner/simulator helpers were removed. The active branch
  collector has no direct runner-private access and depends on the public
  runtime Gateway.

Executed evidence:

- `python -m py_compile` passes for the changed runtime, evaluator and focused
  contracts;
- `frontres_v015_deployment_composition_s1_contract.py` passes immutable report,
  same-state identity and malformed-Baseline rejection;
- `frontres_v015_deployment_carrier_s2a_contract.py` passes command carrier,
  cursor, provenance, row alignment and no-resample boundaries;
- `frontres_v015_deployment_composition_s2b_contract.py` passes Baseline/Repair
  connectivity, zero/one FEMR call counts, frozen GMT, Gateway dependency,
  inference-mode restoration, atomic JSON and zero training-state writes;
- `frontres_v015_evaluation_isolation_contract.py` and
  `frontres_interface_refactor_contract.py` pass legacy/training isolation and
  typed formal-boundary checks.

Verdict:

- the E-FI-97 P1 `Inappropriate Intimacy` and Repair-only contract mismatch are
  closed at deterministic S1/S2;
- METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, checkpoint-v6, HSL,
  928/158/770 authority and training behavior are unchanged;
- real simulator pairing, physical composition efficacy and Demo quality remain
  S4 evidence and are not claimed here.

## E-FI-99: Clean-Anchored Recovery-Aware Contract Activation

Date: 2026-08-01

Tier: human-confirmed document and semantic-authority evidence. No training
source, test, checkpoint, simulator, deployment, or live run was changed or
executed.

Decision evidence:

- human review of all ten Design Detail Inspector cards completed;
- METHOD-v017 makes one executed Clean Rollout and one fixed zero-action Noisy
  Rollout read-only Segment baselines for all M Repair attempts;
- GAIN-v007 defines fixed-unit Clean-conditioned Intent and Physics remaining
  problems, direction-sensitive K-step evidence, separate smooth worst-item
  family aggregation, signed Noisy-to-Repair improvement, continuous remaining
  Physics pressure, full-6D repair cost, and one scalar Recovery-Aware
  `G_total`;
- PPO-v005 sends every valid attempt's complete scalar return through grouped
  equal-mass exact-one PPO and retires the independent Physics projection/KKT
  route;
- TRAIN-v012 preserves K8/M2 -> K16/M3 -> K32/M4, K-conditioned perturbation
  frontiers, Critic-only/recalibration/Actor-ramp/Joint Optimize, HSL-v1 Actor
  initialization, provisional `beta_init=0.02`, and introduces strict
  checkpoint-v7 identity;
- EVAL-v004 makes the executed Clean/Noisy/Repair local evidence lifecycle and
  v007 decomposition authoritative while preserving isolated no-feedback
  full-sequence deployment composition;
- METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011 and their checkpoint-v6
  campaign are historical and cannot resume the changed scalar objective.

Documentation evidence:

- the contract registry, history indexes, Concept Figure mapping, Design Detail
  Inspector/register, Architecture README, proposal rationale, and historical
  v015 plan/checklist/task canvas identify the same active quartet and lifecycle;
- the historical engineering controls are explicitly superseded rather than
  silently reused;
- current training source is recorded as contract-mismatched. Contract
  activation does not authorize implementation.

Verdict:

- Concept Figure -> human Inspector review -> Active Contract is complete;
- the workflow stops before Engineering Plan. No implementation, deterministic
  test, checkpoint migration, simulator, training, or live evidence is claimed.

## E-FI-100: Recovery-Aware Engineering Plan And Discipline Review

Date: 2026-08-02

Tier: document-only Engineering Plan and independent engineering-discipline
review. No training source, test, checkpoint, simulator, deployment, or live run
was changed or executed.

Planning evidence:

- the active METHOD-v017 / GAIN-v007 / PPO-v005 / TRAIN-v012 / EVAL-v004 set is
  mapped to existing Gain, evidence, transaction, PPO, schedule, checkpoint,
  diagnostics, evaluation and composition owners;
- one Engineering Boundary Record freezes requested and preserved behavior,
  public typed inputs/outputs, dependency direction, transaction lifecycle,
  failure/retry/resource rules, non-scope and stop conditions;
- the plan uses one offline implementation/S1-S3/review/formal-audit closure and
  one separately authorized bounded S4 transaction. Long training and beta
  selection remain outside the plan;
- Characterization Test, Effect Sketch, Pinch Points, Seam/Enabling Point,
  CCP/CRP/ADP/SDP, pattern admission, hotspot rules and legacy removal are
  explicit under FRS-ENG-v001.

Independent review evidence:

- review mode: `engineering_plan_review`;
- initial verdict: NEEDS_REVISION for ambiguous one-time Noisy versus `4*M`
  logical rows, a proposed monolithic evidence Data Clump, contradictory
  checkpoint-IO scope, and unspecified canonical source root;
- revisions distinguish one authoritative baseline capture from read-only row
  aliases, admit lifecycle-specific frozen Value Objects in the existing module,
  allow cleanup-bound temporary S3 persistence tests, and freeze `source/` as
  the production root while preserving the dirty `src/` mirror;
- re-audit verdict: READY with no remaining P0/P1 discipline blocker.

Verdict:

- Step 1 / 2 is disciplined and executable but remains unimplemented and
  unauthorized;
- READY does not prove formal connectivity, runtime behavior, persistence
  compatibility, beta calibration, policy quality, or deployment quality.

## E-FI-101: Recovery-Aware Step 1 Offline Engineering Closure

Date: 2026-08-02

Tier: deterministic S1/S2/S3 implementation, engineering-review and formal-route
evidence. No simulator, training, live run, deployment composition or production
checkpoint artifact was executed.

Implemented evidence:

- the tracked `source/rsl_rl` formal root now binds METHOD-v017, GAIN-v007,
  PPO-v005, TRAIN-v012, EVAL-v004 and checkpoint-v7;
- one authoritative executed Clean and fixed zero-action Noisy baseline per
  Segment is sealed and reused by exact-M typed Repair attempts;
- `frontres_gain.py` uniquely computes fixed-scale, late-weighted,
  smooth-worst Clean-anchored Intent/Physics remaining problems, signed
  Noisy-to-Repair improvements, continuous recovery pressure, full-6D cost and
  `G_total` with beta 0.02;
- every valid Repair row reaches grouped equal-mass scalar PPO-v005 and the one
  scalar Critic; old Physics projection/KKT cannot reach the active optimizer;
- v004 local reports consume owner-produced evidence without recomputation;
  legacy v002/v006/quartet local-evaluation modes fail closed while independent
  full-sequence composition remains available;
- checkpoint-v7 strictly persists contract/layout/beta/curriculum/sampler/
  normalizer/receipt identity and rejects v6, tampered or partial payloads
  before mutation.

Executed evidence:

- `frontres_gain_v007_contract.py`, `frontres_v017_step1_contract.py`,
  `frontres_segment_grouped_ppo_contract.py`,
  `frontres_v015_transaction_route_contract.py`,
  `frontres_v015_checkpoint_resume_contract.py` and
  `frontres_formal_runtime_audit_contract.py` pass;
- preserved observation, q29, two-role reset, one-action-K, frozen GMT and local
  scenario contracts pass;
- the active aggregate reports `contract_count=53 failed_count=0`;
- changed Python modules pass `py_compile`, the launcher passes `bash -n`, and
  `git diff --check` passes;
- construction and final `code-review-expert` gates report no open P0/P1.

Verdict:

- Step 1 / 2 is complete at deterministic S1/S2/S3 and formal offline scope;
- Step 2 remains not authorized. Real baseline counts, Contact/ZMP evidence,
  exact-one update, production checkpoint-v7, fresh reload and policy effect
  require one separately authorized bounded 8-env K8/M2 transaction.

## E-FI-102: Recovery-Aware Repository-Health P1 Closure

Date: 2026-08-02

Tier: behavior-preserving S1/S2/S3 interface, lifecycle and deterministic
connectivity evidence. No simulator, training, live run, deployment composition
or production checkpoint artifact was executed.

Implemented evidence:

- baseline collection accepts two authoritative Segment rows and projects raw
  Contact/ZMP/dynamic capture before constructing persistent trajectory frames;
- one `FrontRESStage3TransactionAggregate` now owns execution/checkpoint phase,
  sample/batch context, observation trace and immutable pre-update projection;
- explicit provider dispatch and public command cleanup replace runner-private
  provider/diagnostic/batch flags and command-private active-state reads;
- the active runner imports formal, legacy and single-update functions from
  their actual owners and cannot import the compatibility facade or retired
  projection transitively.

Executed evidence:

- focused baseline, interface, scalar exact-one transaction, checkpoint-v7 and
  formal-runtime contracts pass;
- changed Python modules pass `py_compile`;
- aggregate suite: `contract_count=53 failed_count=0`;
- final `code-review-expert` verdict: `APPROVE`, P0=0 and P1=0.

Verdict:

- E-FI-101 Step 1 remains complete with its later repository-health P1 findings
  closed; Step 2 remains separately authorized S4 work;
- P2 mixed-generation owners, pickle-capable checkpoint loading and v015 active
  naming remain explicit maintenance debt and were not modified.

## E-FI-103: Active-Generation, Safe-Load And Public-Naming Closure

Date: 2026-08-02

Tier: behavior-preserving deterministic S1/S2/S3 maintenance evidence. No
simulator, training, live run, deployment composition or production checkpoint
was executed.

Implemented evidence:

- active v007 Gain and v017 evidence retain their existing semantic owners but
  historical implementations now live in explicit legacy modules; active
  runtime imports do not execute those historical generations;
- supported HSL-v1 and checkpoint-v7 entrypoints share the restricted loader in
  `frontres_checkpoint_quality.py`; `weights_only=True`, recursive payload
  validation and foreign-object rejection occur before actor, Critic,
  optimizer, sampler, normalizer or receipt mutation;
- active Composition Root, transaction, telemetry, checkpoint and observation
  interfaces use version-neutral public names. Serialized checkpoint-v7 keys
  and format values remain unchanged for strict wire compatibility;
- no wrapper layer, new runner, semantic service, mutable carrier or method
  parameter was introduced.

Executed evidence:

- interface isolation, v007 Gain, v017 transaction, diagnostics, Stage-3
  entrypoint, local-sentinel config and checkpoint-v7 malicious-payload/
  roundtrip contracts pass;
- changed production and contract modules pass `py_compile`;
- the aggregate deterministic suite reports
  `contract_count=53 failed_count=0`;
- `git diff --check` passes after the final source cleanup;
- `code-review-expert` final-gate verdict is APPROVE with P0=0 and P1=0.

Verdict:

- the three repository-health P2 findings recorded by E-FI-102 are closed;
- this closure does not produce S4 runtime evidence. Step 2 remains the next
  separately authorized boundary: one bounded 8-env K8/M2 official
  transaction, one update, committed checkpoint-v7 and fresh reload.

## E-FI-104: Human-Confirmed Module Test Atlas Execution

Date: 2026-08-03

Tier: module-level S0/S1 and module-owned deterministic S2/S3 evidence. No
simulator, training, live run, policy-quality run, or deployment composition
was executed.

Execution evidence:

- all 18 Module Test Atlas cards were human-confirmed before execution;
- result is `8 passed / 6 partial / 4 blocked`;
- passed: Training Config, Motion Command, Observation Layout, Stage 3 Live
  Loop, Repair Gain, Segment Storage, Segment PPO, and Checkpointing;
- partial: Launch Entry, OnPolicyRunner, Perturbation Data, Trial Plan & Reset,
  FrontRES Policy, and Runtime Diagnostics;
- blocked: Segment Cache, Segment Sampler, Task-Space Correction, and
  Evaluation;
- focused active contracts and `py_compile` pass for the corrected translation
  and residue paths;
- the broad aggregate regression is explicitly not accepted as proof for all
  cards because it contains historical v002/v006/v013 characterization.

Repaired evidence:

- one-action-K tests now use the public transaction collection context and test
  raw evidence at the collector boundary rather than a retired Gain API;
- deployment composition no longer exposes constructible `legacy_modes`;
- HSL and fail-closed diagnostics use TRAIN-v012/METHOD-v017 identities.

Open first-invalid facts:

- cache does not own immutable Clean continuation and expected support/contact;
- sampler has no owner for earlier-Segment soft preference and candidate-order
  invariance;
- task-space application still performs XY scaling, dz clamp, and local-frame
  quaternion right multiplication against the active card;
- active held-out evaluation still reaches v006 phase identity/owners;
- six partial cards retain named independent-oracle or seam gaps.

Verdict:

- `code-review-expert`: REQUEST_CHANGES before formal-runtime audit;
- module correctness is not complete and no live admission is claimed;
- the next boundary is one reviewed Module Readiness Closure, then a complete
  rerun of the unchanged 18 cards.

Detailed execution ledger:
`note/testing/frontres_module_test_execution_2026-08-03.md`.

## E-FI-105: Module Readiness Closure

Date: 2026-08-03

Tier: module-level S0/S1 and module-owned deterministic S2/S3 evidence. No
simulator, training, live run, policy-quality run, or deployment composition
was executed.

Implemented evidence:

- the existing composition seam exposes observable config -> layout -> one
  HSL/resume load -> one dispatch lifecycle and wraps the real authority
  resolver, load and dispatch operations;
- Cache owns immutable x_t identity, Clean continuation, expected support and
  support envelope; it rejects missing, short, source-drifted, hash-drifted or
  empty-identity artifacts;
- the sampler selects two distinct identities with strictly positive soft
  early-frame preference, preserves later reachability, expands exact-M rows
  and resumes the same RNG stream;
- task correction applies current-only full-6D world-frame Delta SE(3) without
  XY scaling or dz clamp and adapts rotation to the unchanged MOSAIC host
  right-multiplied buffer;
- active v017 evaluation and diagnostics project sealed Contact, phase-ZMP,
  survival, lean, unplanned-contact, Gain, group-mass and exact-one facts
  without recomputation or training feedback;
- perturbation, canonical route-start reset and 158D policy cards now contain
  the missing hand-built and metamorphic cases.

Executed evidence:

- all 18 unchanged human-confirmed Module Test Atlas cards pass their focused
  independent oracle: `18 passed / 0 partial / 0 blocked`;
- active aggregate:
  `contract_count=49 failed_count=0 total_marker_count=49`;
- changed Python sources and contracts pass `python -m py_compile` and targeted
  `git diff --check`;
- final FRS-ENG-v001 review found and closed two in-scope P1 facts: real layout
  resolution initially sat outside the lifecycle pinch point, and Cache did
  not explicitly reject empty x_t identity. Final verdict is APPROVE with
  P0=0/P1=0;
- the retired `frontres_segment_live_sampler_contract.py` remains explicit
  historical v002/composite characterization and is excluded from the active
  aggregate rather than weakening v017 single-`local_rp` rejection.

Verdict:

- Module Readiness Closure is complete;
- E-FI-105 admits the next `formal-runtime-audit` only. It does not prove
  official connectivity or authorize simulator, training, live execution,
  policy-quality evaluation, deployment composition, or Step 2.

## E-FI-106: Phase A Legacy-Updater Isolation And Governance Correction

Date: 2026-08-03

Tier: Phase A source alignment plus deterministic S0/S1/S2 contract evidence.
No simulator, training, live run, optimizer update, checkpoint IO,
policy-quality evaluation or deployment composition was executed.

Implemented evidence:

- `run/run_frontres_stage3_segment_hrl.sh` no longer constructs commands for
  `single_update` or `update_loop`; both modes fail closed before `TRAIN_CMD`;
- `scripts/rsl_rl/train.py::_apply_frontres_stage_preset()` rejects both legacy
  flags before runner construction and installs both legacy config fields as
  false on the active Stage-3 route;
- the active dispatch no longer calls the legacy update-loop owner. Historical
  implementation files remain explicit compatibility code, not an active
  Composition Root;
- AGENTS, the v017 plan/checklist/task canvas, Test Atlas and Repo Architecture
  now describe the v017/v007/v005/v012 scalar route and no longer report the
  affected Phase A/curriculum rows as complete.

Executed evidence:

- shell and direct-preset negative contracts prove pre-construction rejection;
- Stage-3 entrypoint, launcher, live-sentinel and failure-preflight focused
  contracts pass;
- interface-refactor contract passes under the project torch-capable Python;
- the unchanged aggregate reports
  `contract_count=49 failed_count=0 total_marker_count=49`;
- changed Python files pass `py_compile`, the launcher passes `bash -n`, and
  final `git diff --check` passes.

Open blocker:

- TRAIN-v012 names independent frozen `g_8/g_16/g_32`, but the active Contract
  does not define which frozen-GMT Noisy-baseline statistic is calibrated, the
  required observation budget, or the freeze/acceptance rule. Reusing the old
  global episode-length threshold controller would silently restore retired
  semantics. No `g_K` implementation or persistence claim is made.

Verdict:

- the legacy updater-selection mismatch is closed offline;
- Phase A remains blocked on the `g_K` Contract Proposal and Design Inspector
  decision. Phase B, simulator, training and the bounded official transaction
  remain unauthorized.

## E-FI-107: K-DR Coupling Design Rebase

Date: 2026-08-03

Tier: document-only semantic rebase. No source code, test, checkpoint IO,
simulator, training, optimizer update, policy-quality evaluation or deployment
composition was executed.

Observed mismatch:

- active TRAIN-v012 assumes each K first calibrates and freezes an independent
  frozen-GMT Noisy frontier `g_K`;
- E-FI-106 correctly stopped because v012 does not define the statistic,
  observation budget or freeze rule;
- current design review establishes that this is not merely a missing estimator:
  frozen-GMT survival measures baseline difficulty, while the DR strength at
  which K8 stops preserving K16/K32 Repair-attempt ordering depends on the
  learned FEMR policy and cannot be known before training.

Proposal evidence:

- `FRS-TRAIN-v013-empirical-k-dr-coupling-proposal.md` records 2.381 only as a
  provisional first-campaign DR reference ceiling, not a mastery threshold or
  completed per-K frontier;
- the candidate first campaign preserves the fixed K8/M2 -> K16/M3 -> K32/M4
  schedule, keeps the DR distribution unchanged at a K transition, freezes
  Actor/std and recalibrates the same Critic for the new horizon;
- cross-horizon preservation of same-scenario Repair ordering is diagnostic-only
  during the first campaign and supplies evidence for a later frozen K-DR rule;
- the candidate does not revive the global episode-length controller, feed Gain
  back into the sampler, or add an online adaptive curriculum controller.

Governance result:

- active TRAIN-v012 is not silently modified or superseded;
- the Concept Figure parent points remain unchanged;
- the affected Perturbation Data, K-step Curriculum and Actor & Critic Warmup
  Design Inspector cards now project the TRAIN-v013 candidate for human review;
- plan, checklist, task canvas, registry and Test Atlas mark Phase A as blocked
  at this semantic gate.

Verdict:

- E-FI-107 is note-confirmed only and supplies no implementation or runtime
  evidence;
- next action is human review of the three affected Inspector cards. Only after
  confirmation may TRAIN-v013 be activated and an Engineering Plan authorize
  source changes or renewed Phase A work;
- Phase B, simulator, training, live execution and Step 2 remain unauthorized.

## E-FI-108: TRAIN-v013 Contract Activation And Engineering Rebase

Date: 2026-08-03

Tier: human-confirmed contract and document-only engineering-planning evidence.
No production/training source, test, checkpoint IO, simulator, optimizer,
training, live run, policy-quality evaluation or deployment composition was
executed.

Human-confirmed semantic result:

- TRAIN-v012's pre-training calibrated/frozen `g_K` requirement is retired;
- K8/M2 -> K16/M3 -> K32/M4 remains the outer curriculum;
- every K owns an inner lower-to-higher DR curriculum and returns to a lower
  informative distribution at the committed K transition;
- Actor/std freeze while the same Critic recalibrates; Critic weights and
  previously learned Actor behavior are not reset;
- the four perturbation-strength classes remain Easy/Medium/Hard/Broken tail at
  20/30/40/10 relative to the current explicit stage-local `d_cap`;
- 2.381 is the first-campaign reference ceiling, not mastery, graduation or a
  per-K estimated frontier;
- cross-horizon attempt ordering is diagnostic-only and cannot feed Gain, PPO,
  sampler or an online curriculum controller.

Governance changes:

- `FRS-TRAIN-v013-nested-k-dr-curriculum.md` is active and supersedes v012;
- v012 moved to training history with `status: superseded`;
- registry, AGENTS, Design Inspector/register, Architecture README, current
  plan, checklist, task canvas, Test Atlas/register, test control board and
  inventory were rebased;
- the Concept Figure was not changed because the parent design points and their
  interactions are unchanged;
- proposal wording that previously said DR stayed unchanged at K transition is
  superseded by the confirmed per-K DR restart;
- TEST-02 Training Config, TEST-06 Perturbation Data and TEST-16 Checkpointing
  are marked blocked/stale; the other 15 E-FI-105 cards remain valid for
  unchanged behavior;
- current production still carries TRAIN-v012/checkpoint-v7 and is explicitly
  contract-mismatched rather than reported complete.

Engineering review:

- the active FRS-ENG-v001 discipline was applied to the rebased plan;
- existing schedule, transaction, materialization, checkpoint, telemetry and
  Composition Root owners are retained;
- no new runner, Service Layer, wrapper, registry, second schedule owner or
  online controller is admitted;
- independent engineering-plan review verdict is `READY`, with implementation,
  formal connectivity, checkpoint-v8 and S4 facts explicitly unconfirmed.

Verdict:

- contract activation and document rebase are complete;
- source implementation is not authorized by this evidence record;
- next gate is human review of the three revised Module Test Cards, followed by
  one offline TRAIN-v013 engineering closure and renewed formal-runtime Phase A;
- simulator, training and live remain a separate Step 2 authority boundary.

## E-FI-109: TRAIN-v013 Offline Engineering Closure

Date: 2026-08-03

Tier: S1/S2/S3 deterministic offline implementation, Module Test regression,
Formal Runtime Audit Phase A and FRS-ENG-v001 construction/final review. No
simulator, training, optimizer live run, policy-quality evaluation or
deployment composition was executed.

Implemented facts:

- the formal Stage3 identity is `FRS-TRAIN-v013` with
  `frontres-v015-checkpoint-v8` and
  `nested-k-dr-four-class-v1`;
- the existing `frontres_segment_warmup.py` owner resolves K8/M2, K16/M3 and
  K32/M4 plus an explicit per-K DRStageSpec. A committed K transition resets
  the stage-local DR cap, freezes Actor/std during critic-only recalibration and
  retains the same Critic;
- current `d_cap` defines deterministic Easy/Medium/Hard/Broken sampling at
  20/30/40/10. One Segment source samples once and every reset/attempt reuses
  that class, strength and scenario identity;
- formal class selection depends only on the sealed stage identity and sample
  key. Gain, PPO, evaluation and diagnostics cannot feed it;
- only an exact-one committed transaction advances the absolute iteration and
  therefore K/M/DR progress. Abort/retry reuses the last committed identity;
- checkpoint-v8 atomically saves/restores the full ten-field schedule, current
  DR identity/progress/cap, model/Critic/std, optimizer, sampler, 928/158/770
  normalizers, Python/NumPy/Torch/CUDA RNG and committed receipt. v7, `g_K`,
  mixed, partial or tampered payloads reject before mutable restore;
- to keep the Broken tail nonempty at the 2.381 outer reference ceiling, the
  terminal Hard `d_cap` is `2.381 / 1.10`; this is the direct consequence of
  the active `(d_cap, min(1.10*d_cap, ceiling)]` support, not a new controller.

Executed evidence:

- `frontres_segment_warmup_contract.py`: passed;
- `frontres_segment_stage3_curriculum_distribution_contract.py`: passed;
- `frontres_v015_transaction_route_contract.py` with canonical `PYTHONPATH`:
  passed, two Segment sources / four attempts / exact-one update;
- `frontres_v015_checkpoint_resume_contract.py`: passed strict v8 atomic
  save/resume and pre-mutation negatives;
- `frontres_formal_runtime_audit_contract.py`: passed;
- `frontres_segment_stage3_launch_command_contract.py`: passed;
- active aggregate:
  `contract_count=49 failed_count=0 total_marker_count=49`;
- changed Python owners passed `python -m py_compile`; JSON syntax, Atlas
  structural check and `git diff --check` passed at final verification.

Formal Runtime Audit Phase A:

- official dispatch selects `learn_frontres_segment_live()` and returns before
  generic `runner.learn()`;
- formal sampler resolves TRAIN-v013 before the historical adaptive branch;
  the retired episode-length/frontier curriculum is imported only inside the
  explicitly non-formal branch;
- the formal transaction seals K/M/DR/class/strength identity, grouped PPO owns
  the single update, and the training loop increments iteration only after a
  committed result;
- telemetry and checkpoint consumers validate the same DR fingerprint,
  progress and `d_cap`; no active formal `g_K` or checkpoint-v7 fallback was
  found.

Engineering review:

- P0=0, P1=0. The curriculum policy remains in the existing warmup owner;
  runner/entrypoint remain composition only; no new runner, Service Layer,
  wrapper, second curriculum owner or MOSAIC host edit was added;
- retained historical v011/v012 helpers are compatibility-only and do not own
  or feed the formal TRAIN-v013 path.

Verdict:

- Step 1 / 2 is complete offline;
- TEST-02, TEST-06 and TEST-16 are refreshed from the active Contract without
  lowering their independent answers; all 18 cards pass;
- Step 2 / 2, Formal Runtime Audit Phase B, simulator, training and live remain
  separately unauthorized.

## E-FI-110: Human-Confirmed TRAIN-v013 Module Test Closure

Date: 2026-08-03

Tier: module-level S0/S1 and module-owned S3 persistence evidence. No Formal
Runtime Audit Phase A/B, simulator, training, optimizer live update,
policy-quality evaluation or deployment composition was executed.

Authority and scope:

- the user confirmed all 18 Module Test Atlas cards before this execution;
- card semantics, independent answers and failure standards remained frozen;
- this closure answers only whether each module's own computation and state
  behavior match its confirmed card. It does not prove official Stage3
  connectivity;
- E-FI-109's earlier Phase A evidence remains a separate record and was not
  used to upgrade any module result.

Executed facts:

- TEST-01 through TEST-18 each passed their focused deterministic fixture;
- ordinary, boundary, invalid, metamorphic/permutation and stateful cases pass
  where required by the card;
- TEST-16 additionally passes checkpoint-v8 S3 atomic roundtrip, resume
  continuation, tamper, partial receipt, v7 and `g_K` pre-mutation rejection;
- Test Atlas structural validation reports
  `cards=18 pseudo_cases>=5 passed=18 partial=0 blocked=0`;
- active TRAIN-v013 owners and the corrected test pass `python -m py_compile`;
- code-quality Atlas generation and source-link validation pass with 18
  modules, 1441 unique functions and 296 unique annotated blocks;
- targeted `git diff --check` passes.

Failure classification and repair:

- the first TEST-10 run failed because three executable assertions in
  `frontres_hsl_v007_s1_contract.py` still expected the superseded error-text
  identity `FRS-TRAIN-v012`;
- production correctly failed closed under `FRS-TRAIN-v013`; the accepted
  design and rejection behavior were not contradicted;
- classification: executable test did not faithfully translate the confirmed
  card;
- only the three expected identity strings were changed to v013. The complete
  TEST-10 fixture then passed, including actor-only HSL restore, 928/158/770
  visibility, finite 6D proposal, Stage-3 supervision rejection and
  pre-mutation legacy checkpoint rejection.

Code-discipline result:

- repository discipline: active (`FRS-ENG-v001`);
- final module-readiness verdict: APPROVE, P0=0 and P1=0;
- no confirmed card required a production behavior change, no semantic owner
  was duplicated, and no silent fallback/zero-fill was admitted by this
  closure;
- three existing P2 maintainability risks remain explicit: active/historical
  save and sampler branches still share large owners; trusted cache artifacts
  still use `torch.load(..., weights_only=False)`; active v017 behavior still
  retains substantial v015 file/symbol naming. None is reported as fixed or as
  module-semantic evidence.

Verdict:

- `18 passed / 0 partial / 0 blocked` at module scope;
- the current module prerequisite for Formal Runtime Audit Phase A human review
  is satisfied;
- official config/owner/consumer connectivity, simulator behavior, policy
  quality and Demo quality remain outside this evidence and unconfirmed by it.

## E-FI-111: Perturbation Boundary And Optional Probing Semantic Correction

Date: 2026-08-03

Tier: human-confirmed document and Atlas correction during Formal Runtime Audit
Phase A. No production source, test, checkpoint IO, simulator, training, live
run, policy-quality evaluation or deployment composition was executed.

Confirmed semantics:

- `2.381` is the experimentally measured maximum reliable perturbation boundary
  for the current frozen GMT, robot and perturbation definition, not merely an
  arbitrary reference value;
- the current campaign may configure this known value directly and let every
  per-K inner DR curriculum progress from lower strength toward it;
- after changing GMT, robot or perturbation scale, an optional offline
  Perturbation Probing pass may remeasure the boundary using frozen-GMT
  Noisy-only survival evidence;
- Probing only acquires and freezes the scalar outer boundary. It does not train
  FrontRES, decide K transitions, create a per-K `g_K`, adapt DR online, or
  receive Gain/PPO/evaluation feedback;
- direct configuration and offline Probing therefore have identical downstream
  curriculum authority.

Governance changes:

- TRAIN-v013 now contains `Optional GMT Boundary Acquisition` and maps the
  existing Concept Figure block `M-12` to supporting design ID `FRS-DP-01P`;
- the Perturbation Data Inspector card and design register project the same
  meaning, while the ten canonical parent tabs remain unchanged;
- the Concept Figure keeps its existing Perturbation Probing node but now names
  boundary measurement and frozen training support rather than generic hard-case
  discovery;
- the current repository Architecture points the active curriculum owner to
  `frontres_segment_warmup.py`; retired adaptive curriculum remains legacy only;
- no current implementation change is required because direct configuration of
  `2.381` is one accepted acquisition route. Optional automatic Probing remains
  unimplemented and is not required for the current campaign.

Verdict:

- FRS-DP-01 semantic correction is document-confirmed;
- Formal Runtime Audit Phase A remains in human review and must continue with
  the next canonical design point; Phase B/live remain unauthorized.

## E-FI-112: Segment Replay Phase A Human Review And Audit Projection Correction

Date: 2026-08-03

Tier: human-confirmed Formal Runtime Audit Phase A semantics plus deterministic
S1/S2 audit-projection evidence. No simulator, training, live run,
policy-quality evaluation or deployment composition was executed.

Human-facing correction:

- the phrase `motion -> Segment -> attempt equal mass` is now explained as a
  voting rule: first give each motion equal influence, then each Segment equal
  influence, then each valid attempt inside that Segment equal influence;
- equal influence does not mean equal score. Each attempt keeps its own
  `G_total` and advantage, so it may support or oppose its sampled action;
- a group cannot dominate merely because it contains more policy rows;
- all valid attempts remain in PPO. Winner-only, argmax and best-of-M weighting
  remain forbidden.

Phase A source facts:

- `FrontRESStage3Engine.run_training_transaction()` is the formal orchestration
  owner; it delegates the complete request to the existing formal transaction
  owner and verifies that collection performed zero optimizer steps;
- one transaction freezes one old policy, selects two distinct Segment sources,
  seals one scenario per Segment and collects exact M Repair rows per Segment;
- Clean and zero-action Noisy are executed once per Segment as shared evidence;
  only Repair attempts produce policy rows;
- `FrontRESFormalTransactionAccumulator` rejects partial/mixed collection and
  optimizer mutation before seal;
- the grouped adapter carries every Repair row to scalar PPO-v005, and the
  formal transaction performs exactly one committed optimizer update;
- the historical update loop, flat legacy probe, v002 Gain and PPO-v004
  projection do not own the active transaction route.

Governance and diagnostic correction:

- Registry, Design Inspector, Test Atlas wording and Repo Architecture now
  project TRAIN-v013/checkpoint-v8 and the v017/v007/v005 scalar route;
- the active formal transaction now calls the existing
  `frontres_formal_runtime_audit.py` owner through a read-only
  `AUDIT-SEGMENT-REPLAY-01` projection;
- when the explicit formal-audit flag is enabled, that projection fails closed
  unless it sees two Segments, exact M rows per
  Segment, all rows valid, positive equal motion/Segment/attempt voting
  weights, one update invocation, one optimizer step and the exact active
  contract identities;
- the retained 870D/v002 flat-probe audit is explicitly legacy and is no longer
  used by the active Phase B fixture.

Executed evidence:

- `frontres_formal_runtime_audit_contract.py`: passed current transaction,
  checkpoint-v8, old-PPO identity rejection and incomplete attempt-weight
  rejection;
- `frontres_v015_transaction_route_contract.py`: passed two Segment sources,
  four exact-M Repair attempts, four valid rows and exactly one update;
- `frontres_segment_grouped_ppo_contract.py`: passed scalar PPO-v005 grouped
  sign/mask/permutation and voting-weight behavior;
- changed Python files passed `python -m py_compile`.

Code-discipline result:

- repository discipline: active (`FRS-ENG-v001`);
- no new runner, Service Layer, wrapper, transaction owner or training-state
  mutation was introduced. The audit function is a read-only consumer of the
  immutable committed result; with the audit flag disabled it performs no
  validation, output or control-flow change;
- the active transaction owner gained one diagnostics call and no method,
  Gain, PPO, checkpoint or simulator responsibility;
- P0=0 and P1=0 for this correction. Retained legacy audit functions remain
  compatibility-only and cannot be cited as active Phase B evidence.

Verdict:

- FRS-DP-02 Segment Replay is human-confirmed and `integrated-offline` for
  Phase A;
- this evidence does not complete Phase A as a whole and does not authorize
  Phase B or live execution;
- the next canonical Phase A design point is FRS-DP-03 K-step Curriculum.

## E-FI-113: K-step Curriculum Phase A Human Review And Legacy-Owner Isolation

Date: 2026-08-03

Tier: human-confirmed Formal Runtime Audit Phase A semantics plus deterministic
S1/S2/S3 owner, connectivity and persistence evidence. No simulator, training,
live run, policy-quality evaluation or deployment composition was executed.

Human-confirmed semantics:

- K is only the executable-evidence horizon. Each Repair attempt samples one
  FEMR action at `t`, stores one PPO policy row, freezes FEMR, and lets frozen
  GMT execute exactly K evidence steps;
- the active outer schedule is K8/M2 -> K16/M3 -> K32/M4; K64 is inactive;
- each committed K transition restarts the stage-local DR curriculum at a lower
  informative distribution, freezes Actor/std and recalibrates the same Critic;
- only a committed transaction may advance K/M/DR/iteration/receipt state.

First-invalid Phase A fact and correction:

- the formal sampler supplied exact active K/M but
  `plan_frozen_policy_transaction()` still called the retired state-driven
  rollout-budget owner before overwriting those values. The stale decision did
  not reach PPO, but remained reachable and emitted misleading active K probes;
- the same existing sampler owner now constructs the sealed exact-K/exact-M
  budget directly when both values are present. Compatibility calls retain the
  characterized legacy planner, whose probes are explicitly labeled legacy;
- no new runner, wrapper, Service Layer, transaction owner, schedule owner or
  public payload was introduced.

Executed evidence:

- `frontres_segment_sampler_contract.py` proves K8/M2, K16/M3 and K32/M4 exact
  plans while replacing the legacy budget method with a raising sentinel; call
  count remains zero;
- `frontres_formal_runtime_audit_contract.py` proves active K audit identity is
  owned by TRAIN-v013 and cannot be supplied by the legacy sampler projection;
- interface, warmup, formal transaction, one-action-K and checkpoint-v8
  deterministic contracts pass;
- changed Python files pass `python -m py_compile`; JSON parsing and targeted
  `git diff --check` pass.

Verdict:

- FRS-DP-03 K-step Curriculum is user-reviewed and integrated offline for
  Phase A; live K-frame counts, real Actor/std freeze and production
  checkpoint-v8 identity remain Phase B facts;
- Phase A as a whole remains open. The next canonical design point is
  FRS-DP-04 FrontRES 6D Repair; Phase B and live execution remain unauthorized.

## E-FI-114: FrontRES Direct Full-6D Action Closure

Date: 2026-08-03

Tier: human-confirmed Formal Runtime Audit Phase A semantics plus deterministic
S1/S2/S3 implementation, connectivity and persistence evidence. No simulator,
training, live run, policy-quality evaluation or deployment composition was
executed.

Confirmed action contract:

- the task-space Actor emits one finite direct `[B,6]` world-frame
  `Delta SE(3)` value;
- HSL prediction/target, Stage-3 Gaussian, rollout storage, old/new log-prob,
  one-action-K evidence and command application use that same coordinate;
- active code applies no action mask, per-axis action scale, `tanh`, `clip` or
  `clamp`, and rejects mixed-width/legacy 12D actions instead of slicing them;
- HSL identity is `frontres-v017-hsl-proposal-v2`; Stage-3 identity is
  `frontres-v017-checkpoint-v9`; HSL-v1 and checkpoint-v8 reject before state
  mutation.

Executed evidence:

- proposal-only HSL S1/S2, task-space policy, rollout action-stat, one-action-K,
  live storage/PPO, single-update, policy-quality owner and strict checkpoint
  contracts passed;
- the aggregate `frontres_segment_all_contract_suite.py` passed 49/49;
- changed Python owners passed `python -m py_compile`;
- legacy MOSAIC host and joint-space `Delta q/Delta z` transforms remain outside
  the active task-space owner. PPO clipping and numerical quaternion/index/
  denominator guards retain their independent, non-action-authority semantics.

Code-discipline result:

- `FrontRESActorCritic` remains the single action-coordinate owner; existing
  HSL, rollout, storage, PPO and checkpoint modules consume its exact public
  `[B,6]` boundary;
- no new runner, service, wrapper, second action owner or MOSAIC host change was
  introduced; old `FrontRESActionCone` ownership was removed;
- P0=0 and P1=0 for the direct-action closure.

Verdict:

- FRS-DP-04 FrontRES 6D Repair is integrated offline and confirmed for Phase A;
- Phase A remains open. The next canonical design point is FRS-DP-05 Frozen
  GMT; Phase B and live execution remain unauthorized.

## E-FI-115: DP05 Frozen GMT Offline Readiness Closure

Date: 2026-08-04

Phase A facts recorded before implementation:

- `OnPolicyRunner.train_mode()` calls `policy.train()` recursively, while the
  composed FrontRES policy has no public invariant that restores the internal
  GMT policy, GMT normalizer and optional estimator to `eval`; parameters are
  no-grad and optimizer-excluded, but the module training flags are not closed;
- checkpoint-v9 validates the 770D layout and combined normalizer shape but
  does not persist the configured GMT checkpoint SHA256, so a shape-compatible
  different GMT could pass strict resume;
- both defects are offline-fixable inside existing FrontRES policy, runner-mode
  connector and checkpoint owners. No METHOD/GAIN/PPO/HSL, command,
  one-action-K or MOSAIC-host change is authorized.

Implementation:

- `FrontRESActorCritic.train()` now preserves normal `nn.Module` train/eval
  behavior for the residual Actor and scalar Critic, then enforces one public
  invariant over the GMT policy, GMT normalizer and optional reference-velocity
  estimator: `eval`, `requires_grad=False`, frozen normalizer statistics;
- the unchanged `OnPolicyRunner.train_mode()` continues to call the public
  `policy.train()` boundary; the override closes the GMT mode lifecycle without
  exposing GMT private state to the runner;
- the existing checkpoint owner now uses one shared frozen-GMT identity for
  HSL and Stage-3 persistence. Checkpoint-v9 stores the configured GMT artifact
  SHA256, the required 770D suffix dimension and the exact GMT normalizer-state
  fingerprint;
- strict Stage-3 resume recomputes that identity before actor, Critic,
  optimizer, sampler, normalizer, iteration or receipt restoration. A different
  same-shape GMT artifact therefore rejects pre-mutation.

Deterministic evidence:

- changed Python owners and focused contracts pass `python -m py_compile`;
- `frontres_frozen_gmt_contract.py` proves a parent `train()` call leaves the
  residual Actor/Critic in train mode while all three GMT-family modules remain
  eval/no-grad and optimizer-excluded; a real optimizer step changes
  Actor/Critic but not GMT;
- `frontres_v015_checkpoint_resume_contract.py` proves checkpoint-v9 records
  the expected SHA256/770D/normalizer identity, round-trips the matching GMT,
  and rejects a different same-shape GMT before actor, optimizer, sampler or
  loaded-path mutation;
- HSL S1/S2, 928/158/770 observation-authority and formal-runtime audit
  regressions pass;
- `frontres_segment_all_contract_suite.py` reports
  `contract_count=49 failed_count=0 total_marker_count=49`.

Code-discipline result:

- no runner, wrapper, second identity owner or MOSAIC-host modification was
  introduced. The public policy invariant reduces facts known by the runner;
  the checkpoint pre-mutation gate remains the persistence pinch point;
- METHOD-v017, GAIN-v007, PPO-v005, TRAIN-v014, HSL-v2, command routing,
  one-action-K and GMT parameters are unchanged. P0=0 and P1=0 for this scope.

Verdict: DP05 Frozen GMT offline readiness is complete and confirmed for Formal
Runtime Audit Phase A. DP06, Phase B, simulator, training and live remain
unauthorized.

## E-FI-116: DP07 Repair Gain Read-Only Projection Closure

Date: 2026-08-04

Phase A first-invalid fact:

- the formal v017 learning chain already consumed the unique
  FRS-GAIN-v007 `G_total`, but the immutable local report and final transaction
  telemetry exposed only a subset of the owner-produced decomposition;
- therefore an operator could see `G_I`, `G_P`, recovery pressure and final
  scalar, but could not trace the normalized channel values, `I_N/I_R/P_N/P_R`,
  support-foot drift, fixed scales, cost-free score and beta through the same
  row identity. This was a diagnostics contract mismatch, not a Gain or PPO
  formula defect.

Implementation:

- `FrontRESRecoveryAwareGainResult` now carries the complete already-computed
  decomposition plus immutable fixed scale/beta identity. Raw support-foot
  drift is emitted inside the existing Gain owner from its own normalized
  Physics channel;
- `build_frontres_v017_local_evaluation_report()` projects those values into
  the existing frozen report. It does not call Gain again. Required
  Contact/Intent channels remain finite; only phase-ZMP may be explicit N/A;
- `build_frontres_transaction_telemetry()` forwards every row field through
  the existing report-to-PPO permutation and rejects mixed scale/beta identity.
  It does not inspect evidence internals, access runner private fields, fill
  missing values, or mutate training state;
- legacy report types and mismatched v006/projection identities remain rejected.

Deterministic evidence:

- the focused TEST-18 formal transaction fixture assigns distinct sentinels to
  every new field and proves exact forwarding, permutation, exact-one counts,
  group mass and read-only behavior;
- the v017 step1 fixture proves report values equal the owner result, explicit
  phase-ZMP N/A is retained, missing Contact and non-finite beta reject;
- `frontres_gain_v007_contract.py`, `frontres_v017_step1_contract.py`, and
  `frontres_v015_transaction_route_contract.py` pass;
- changed Python files pass `python -m py_compile`;
- `frontres_segment_all_contract_suite.py` exits successfully with all 49
  expected markers observed.

Code-discipline result:

- the existing result/report/serializer pinch point was extended in place;
  no wrapper, Service Layer, second Gain owner, runner or feedback path was
  introduced;
- construction and final review report P0=0 and P1=0 for this scope.

Verdict: DP07 Repair Gain is integrated offline and Phase A reviewed. Its
formal producer, consumer, row identity and legacy isolation agree with
FRS-GAIN-v007/EVAL-v004. Real simulator evidence and policy quality remain
unconfirmed; Phase B, Step 2, training and live remain unauthorized.

## E-FI-117: DP08 HSL Direct Full-6D Target Offline Readiness Closure

Date: 2026-08-04

Phase A first-invalid fact:

- TRAIN-v014 and TEST-10 require one direct full-6D HSL target without an
  action mask, per-axis scale, clip or clamp;
- `get_supervision_target_task_space()` and
  `validate_frontres_hsl_current_frame_target()` nevertheless replaced every
  positive anti-DR `dz` with zero, and the old fixture repeated that answer;
- therefore the former passing test and implementation shared one retired
  asymmetric-Z assumption. This was an HSL target contract mismatch, not a
  Stage-3 action, Gain, PPO or checkpoint defect.

Implementation:

- the existing Stage-1 target owner now emits exact
  `target_pos = -anchor_dr_delta_pos` for all three world-frame translation
  axes; quaternion-to-RPY numerical-domain protection remains unchanged;
- the existing warmup validator independently reconstructs and checks that
  complete translation instead of applying a second axis policy;
- the confirmed TEST-10 oracle now executes the real target-owner body with a
  semantic two-row fixture and proves positive and negative anti-DR `dz` are
  both preserved. No Clean/future value enters the fixture or actor path;
- no runner, wrapper, target type, checkpoint field, optimizer, transaction or
  persistence owner was added.

Deterministic evidence:

- `python -m py_compile` passes for the producer, validator and two HSL
  contracts;
- `frontres_hsl_v007_s1_contract.py` prints
  `current anti-DR [B,6] preserves both dz signs without Clean input or axis clamp`;
- `frontres_hsl_v007_s2_connectivity_contract.py` proves the q29-normalizer-
  actor-current-target edge and Stage-3 zero HSL writer/loss;
- task-space proposal, actual policy distribution, 928/158/770 authority,
  checkpoint-v9/HSL-v2, legacy rollout-label and formal Stage-1 entrypoint
  contracts all exit zero;
- the complete `frontres_segment_all_contract_suite.py` exits zero after the
  change. This is deterministic regression evidence, not simulator/live or
  policy-quality evidence;
- `check_module_test_atlas.mjs` reports 18 cards, 18 passed, 0 partial and 0
  blocked; the regenerated Code Quality Atlas validator reports 18 modules,
  1417 unique functions and 296 unique annotated blocks.

Code-discipline result:

- the producer remains the single semantic owner and the validator remains a
  fail-closed independent check at the existing Pinch Point;
- the patch removes a hidden per-axis authority from both sides without adding
  caller knowledge, a Service Layer, wrapper, private dependency or new state;
- final review finds P0=0 and P1=0 for the bounded scope. Existing unrelated
  worktree changes and live-only facts were not reviewed as part of this unit.
  The bounded review is stored at
  `note/frontres_core/reviews/FRS-v017-dp08-hsl-target-final-review.md`.

Verdict: DP08 HSL Warmup is integrated offline and Phase A reviewed. HSL-v2
remains proposal-only, actor-only, deployable-158D and cold-start-only. Phase B,
simulator, training, live, policy-quality and DP09 remain outside this evidence.

## E-FI-118: DP09 Actor And Critic Warmup Offline Readiness Closure

Date: 2026-08-04

Phase A first-invalid fact:

- TRAIN-v014 defines the formal phases as `critic_only`, `actor_ramp` and
  `joint`, but the schedule owner, typed request validators and checkpoint-v9
  still emitted or accepted the retired public label `actor_warmup`;
- existing tests proved critic-only Actor parameter freeze at K8 but did not
  prove preservation of non-empty Actor/std optimizer state or continuity of
  the same Critic across both K transitions.

Implementation:

- `frontres_segment_warmup.py` now emits `actor_ramp`; the phase duration and
  linear actor-loss weight are unchanged;
- both public transaction request validators accept only
  `critic_only`, `actor_ramp`, `joint` and reject the old label;
- transaction, diagnostics/telemetry and checkpoint-v9 consume the resolver's
  immutable phase identity without recomputation;
- no HSL, Gain, PPO formula, schedule duration, K/M/DR rule, checkpoint schema,
  runner, simulator or MOSAIC host behavior changed.

Deterministic evidence:

- the red phase-oracle test first failed against `actor_warmup`, then
  `frontres_segment_warmup_contract.py` passed the exact actor-ramp sequence;
- `frontres_v015_transaction_route_contract.py` executed K16/M3 and K32/M4
  critic-only formal transactions through exact-one updates. Both reused one
  Critic object and changed its parameters, while pre-seeded Actor/std
  parameters and their Adam state remained bitwise unchanged;
- the same formal transaction contract proved `actor_ramp` reaches final
  telemetry with its unchanged fractional actor-loss weight;
- `frontres_v015_checkpoint_resume_contract.py` round-tripped checkpoint-v9 at
  actor-ramp and rejected a tampered `actor_warmup` phase before mutation;
- interface, legacy single-update regression and changed-file pycompile pass;
- `frontres_segment_all_contract_suite.py` exited zero with all 49 contract
  targets observing their required marker.

Code-discipline result:

- the existing pure resolver, typed request, Unit-of-Work, telemetry and
  checkpoint Gateway boundaries remain intact; no duplicated phase owner,
  private-state traversal, wrapper or fallback was added;
- final review reports P0=0, P1=0 and no in-scope P2. The retained internal
  `actor_warmup_iterations` duration field is compatibility naming only, not a
  formal identity;
- review: `note/frontres_core/reviews/FRS-v017-dp09-warmup-final-review.md`.

Verdict: DP09 Actor & Critic Warmup is integrated offline and Phase A reviewed.
The next canonical design point is DP10 Future Motion Context. Phase B,
simulator, training, live and policy quality remain unauthorized and
unconfirmed.

## E-FI-119: DP10 Future Motion Context Offline Readiness Closure

Date: 2026-08-04

Phase A first-invalid fact:

- the active contract, Design Inspector and checkpoint-v9 required exactly two
  deployment/Noisy q29 future frames at offsets `(1,2)`, but the layout owner
  still accepted arbitrary positive sorted offsets such as `(1,3)`;
- this allowed config and actor-context tests to describe a layout that could
  never satisfy the formal checkpoint identity. The mismatch was semantic
  acceptance drift, not a q29 value, normalizer or GMT defect.

Implementation:

- `FrontRESFutureIntentLayout` now owns the exact `(1,2)` identity;
- Stage-1/Stage-3 config parsing validates that identity before mutating config;
- formal transaction and checkpoint-v9 consumers independently reject layout
  drift before transaction collection or checkpoint mutation;
- focused fixtures use `[B,3,29]` role-aligned deployment/Noisy q29 rows and
  make `(1,3)` an explicit invalid case;
- the unmocked observation contract was reduced to its actual DP10 boundary:
  the production observation reader, normalizer and actor/GMT authority split.
  The current exact-one transaction remains independently proved by its formal
  transaction contract instead of being reconstructed inside the observation
  fixture.

Deterministic evidence:

- TEST-04 role alignment, TEST-05 observation authority/layout, TEST-10 actual
  policy distribution and TEST-16 checkpoint-v9 contracts pass;
- HSL S1/S2 future-intent connectivity and Stage-3 config fail-closed contracts
  pass with `(1,2)` and reject `(1,3)`;
- `frontres_v015_unmocked_observation_connectivity_contract.py` uses the real
  `_read_live_observations()` path and reports
  `870 + 58 -> 928 -> FEMR 158 / frozen GMT 770; actor_calls=1`;
- `frontres_v015_transaction_route_contract.py` independently passes the
  current v017 grouped exact-one transaction;
- changed-file `py_compile`, `git diff --check` and the complete deterministic
  aggregate pass; the aggregate reports `49/49` required markers.

Code-discipline result:

- Characterization showed that the former generic layout accepted `(1,3)` and
  failed only later at persistence. The Pinch Point is now the existing layout
  validator, with config/formal/checkpoint consumers as narrow fail-closed
  seams;
- the formal transaction keeps a small independent public identity assertion
  rather than importing policy-layout implementation into historical stub
  seams. No wrapper, Service Layer, second owner, fallback or private-state
  traversal was added;
- final bounded review finds P0=0 and P1=0. Existing unrelated worktree changes
  were preserved and remain outside this closure.

Verdict: DP10 Future Motion Context is integrated offline and Phase A reviewed.
All ten design points now have offline Phase A review evidence. Phase B,
simulator, training, live and policy quality remain unauthorized and
unconfirmed.

## E-FI-120: Phase B Runtime Audit Human Control Surface

Date: 2026-08-04

Scope:

- Phase A is already complete for DP01-DP10. This record does not reopen method
  semantics or module tests;
- the remaining live-only relationships are projected into one Runtime Audit
  Atlas before any source probe or official run is authorized.

Control surface:

- `note/architecture/runtime/06_frontres_runtime_audit_atlas.data.json` contains
  eight runtime-ordered cards: launch identity, sealed scenario/transaction,
  reset/observation authority, one-action-K/frozen GMT, paired v007 Gain,
  storage/return, grouped exact-one update, and committed checkpoint/telemetry;
- each card names its parent Design IDs, formal owner, upstream/probe/downstream
  route, expected runtime shape/count/value and `PENDING_LIVE` status;
- `note/architecture/06_frontres_runtime_audit_atlas.html` is linked from the
  repository Atlas index. The existing Module Test Atlas stage card now points
  to this separate Phase B surface instead of mixing connectivity into module
  correctness cards.

Run boundary:

- the intended cost is one fresh K8/M2 transaction with 8 envs, one optimizer
  update and one checkpoint-v9 save through the official `train` branch;
- the command template fixes the active schedule, `(1,2)` future offsets and
  checkpoint interval 1, but intentionally retains
  `<ABSOLUTE_HSL_V2_CHECKPOINT>` until a real
  `frontres-v017-hsl-proposal-v2` artifact is confirmed;
- the historical HSL-v1 `model_warmup.pt` path is not assumed compatible.

Validation:

- the runtime Atlas JSON parses and all referenced repository source paths
  exist;
- the actual local viewer loads all eight runtime cards and the run-contract
  support card. A browser screenshot confirms readable default-zoom layout and
  one source link resolves through the repository open-source endpoint;
- no training source, probe, checkpoint, simulator, optimizer or live state was
  changed or executed.

Verdict: Phase B instrumentation is ready for human review but not authorized.
The only unresolved command input is the absolute HSL-v2 artifact path.

## E-FI-121: Formal Runtime Audit Phase B Instrumentation Closure

Date: 2026-08-04

Scope and authority:

- `AUDIT-B01` through `AUDIT-B08` are installed as read-only, fail-closed
  assertions in the existing formal owners. No new runner, evaluator, Gain
  owner, PPO owner or checkpoint format was introduced;
- launch identity is checked at the formal route boundary; observation and
  one-action-K facts are checked from the collector-owned immutable evidence;
  final Gain/return/group/update facts are checked from the existing sealed
  telemetry projection; checkpoint-v9 facts are checked from the committed
  payload before reporting;
- diagnostics do not recompute Gain and do not write sampler, optimizer,
  transaction or curriculum state.

Offline evidence:

- `frontres_formal_runtime_audit_contract.py` emits B01-B08 exactly once for a
  valid K8/M2 fixture and rejects later FEMR action, mixed scenario identity,
  missing curriculum and mixed K/M payloads;
- `frontres_segment_stage3_launch_command_contract.py`,
  `frontres_v015_one_action_k_contract.py`,
  `frontres_v015_transaction_route_contract.py` and
  `frontres_v015_checkpoint_resume_contract.py` pass unchanged;
- Python compilation and `git diff --check` pass. No simulator, training or
  live transaction was started.

Final engineering-discipline review:

- verdict `APPROVE`, with P0=0 and P1=0. The added responsibility remains one
  CCP-coherent read-only runtime-audit owner; the formal collector and final
  serializer only call that owner at their existing immutable output seams;
- no wrapper, service, duplicate Gain/PPO calculation, private simulator
  traversal, silent fallback/zero-fill, persistence mutation or diagnostic
  feedback path was added. Runtime truth and policy quality remain explicitly
  outside this offline verdict.

Artifact and remaining boundary:

- the runtime-confirmed `frontres-v017-hsl-proposal-v2` artifact is
  `/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt`;
- the only remaining Phase B evidence is one official 8-env, K8/M2,
  one-transaction, one-update live execution and committed checkpoint-v9.

Verdict: Phase B instrumentation is offline-closed. Runtime observation remains
pending; no runtime or policy-quality claim is made.

## E-FI-122: Phase B First-Invalid Reset Lifecycle Closure

Date: 2026-08-04

Observed runtime fact:

- the first official GPU7 Phase B run emitted `AUDIT-B01` with the active
  v017/v007/v005/v014, HSL-v2, K8/M2 and `(1,2)` identities, then stopped before
  `AUDIT-B02`;
- the first invalid operation was the Clean-baseline phase calling
  `set_frontres_local_scenario_execution_mode("clean_baseline")` before the
  reset hook had installed an active sealed local scenario. The command owner
  correctly rejected this ordering. This was not a GPU, HSL, sampler-identity
  or audit-assertion failure.

Offline repair and evidence:

- the formal transaction now carries the requested Clean/Noisy/Repair phase
  through the existing reset request. The environment hook installs the sealed
  scenario first, selects the command-owned execution mode second, and only
  then refreshes the current command/cache and writes the robot reference;
- the command fail-closed check is unchanged. Legacy and non-local reset paths
  reject a supplied v017 execution mode, while all three formal phase names are
  validated before the environment hook runs;
- `frontres_v015_two_role_reset_contract.py` proves first-install Clean mode,
  same-identity Repair reuse and correct Clean/artifact cache selection;
  `frontres_v015_transaction_route_contract.py` proves the formal owner routes
  `clean_baseline -> noisy_baseline -> repair_attempts` through the reset seam;
  one-action-K and formal runtime-audit regressions also pass, together with
  Python compilation and `git diff --check`.

Verdict: the first-invalid owner lifecycle is closed offline. `AUDIT-B01` is
runtime-confirmed; `AUDIT-B02` through `AUDIT-B08` remain live-pending until the
same bounded official transaction is rerun. No policy-quality claim is made.

## E-FI-123: Phase B Repair-Trajectory Row-Alignment Closure

Date: 2026-08-05

Observed runtime fact:

- the rerun crossed the E-FI-122 reset repair: Clean, Noisy and Repair resets
  completed, the real `870D + 58D -> 928D -> 158D/770D` observation route ran,
  one finite full-6D action was applied and frozen GMT entered K8 execution;
- the first invalid operation was trajectory sealing. Repair frames had already
  selected four policy rows from eight role rows, while survival and valid-mask
  vectors still retained all eight role rows. The resulting trajectory mixed
  `[K,4,...]` state with `[K,8]` survival and correctly failed validation.

Offline repair and evidence:

- the one-action-K owner now selects global survival/valid vectors by the same
  `repair_rows` exactly once, then seals every field as `[K,M,...]`; it no longer
  selects an already selected Repair trajectory a second time;
- the deterministic regression uses non-contiguous global role rows and
  hand-written survival/valid values, proves the expected row permutation and
  rejects a missing requested role row;
- one-action-K, v007 Gain, formal transaction and formal runtime-audit focused
  contracts pass, together with Python compilation and `git diff --check`.

Verdict: Repair trajectory row alignment is owner-lifecycle-contract-confirmed
offline. The rerun provided real evidence through one-action-K collection, but
`AUDIT-B02` through `AUDIT-B08` still require another bounded official run. No
Gain, PPO, checkpoint or policy-quality result was produced by the failed run.

## E-FI-124: Phase B Formal Grouped-Batch Dependency Closure

Date: 2026-08-05

Observed runtime fact:

- the next GPU7 run crossed the reset and Repair row-alignment closures, emitted
  the active route, observation, action, application and frozen-GMT sentinels,
  and completed the real one-action-K evidence collection;
- request assembly then failed at the grouped storage adapter because
  `frontres_segment_formal_transaction.py` referenced
  `FrontRESSegmentPPOBatch` without importing its production definition. This
  was the first invalid operation; Gain/PPO update, optimizer commit and
  checkpoint save were not reached.

Offline repair and evidence:

- the formal transaction owner now imports the grouped PPO batch type directly
  from the algorithm owner. No test injection, fallback or duplicate batch
  schema is used;
- `frontres_v015_transaction_route_contract.py` now rejects a request owner
  whose grouped batch dependency is missing or foreign, then proves 2-Segment
  x exact-M scalar collection and exactly one update;
- the active downstream owner chain was checked for unresolved global symbols
  from request assembly through Stage-3 engine, update loop, PPO, diagnostics,
  telemetry and checkpointing; no other unresolved production symbol remains;
- focused transaction, v017 Step-1, formal runtime-audit, checkpoint-v9 and
  one-action-K contracts pass, Python compilation passes, and the active
  aggregate contract suite reports `49/49` with `failed_count=0`.

Verdict: the grouped-batch dependency is contract-confirmed offline. The next
bounded official transaction is still required for `AUDIT-B02` through
`AUDIT-B08`; no policy-quality claim is made. The historical
`frontres_v015_local_sentinel_connectivity_contract.py` is outside the active
49-contract suite and still depends on a removed legacy fixture loader; it is
recorded as legacy test debt rather than used as production evidence.
