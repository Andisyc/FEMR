# FrontRES Segment Replay HRL Method Design

Date: 2026-06-27

This note records the current next-method design for FrontRES/FEMR.  It
supersedes the acceptance-only HSL+HRL concept as a research direction, but it
is not yet an implementation contract.  Before coding, this note should be
converted into an engineering plan and checked against the live runner,
storage, algorithm, and config paths.

## 1. Core Judgment

The current understanding is:

- The old HSL+acceptance design is too narrow.
- HSL should not be the final repair; it should initialize repair learning.
- HRL should not learn acceptance; it should learn the full 6D Delta SE repair.
- The old probing idea was useful, but hand-designed linear probing is too weak
  for a 6D repair space.
- The right replacement is segment-level RL: each segment is a short repair
  task, and PPO explores 6D repair actions inside that task.

The method should therefore become:

- slice long motions into dynamic segments;
- use HSL to warm-start the repair policy;
- train HRL with short segment rollouts;
- coordinate global coverage and repeated local rollout through prioritized
  segment replay;
- deploy a single repair policy unless diagnostics later justify an expert or
  distillation stage.

## 2. Problem Being Solved

The original rollout design mixed two needs:

- global coverage over a large motion dataset;
- repeated local trial on the same difficult segment.

If training only samples globally, each segment may be seen once or twice.  That
is not enough for RL to learn a good 6D repair for that local dynamics state.

If training only repeats one segment, the policy may overfit and lose motion
coverage.

The design goal is to keep both:

- global sampling discovers diverse repair cases;
- repeated rollout lets the policy actually learn difficult repair cases.

The clean resolution is to treat a motion segment as a replayable RL level.

## 3. Sequence Slicing

Long motion sequences should be split into short dynamic segments.

Each segment should contain:

- motion id;
- start phase / start frame;
- segment length or K-step horizon;
- clean dynamic reset state;
- reference window for the segment;
- perturbation family and strength;
- metadata for contact, phase, and difficulty diagnostics.

The reset state must be dynamic, not a static pose.  It should include root and
joint velocities, angular velocity, contact-relevant state, and any controller
history needed by the frozen GMT tracker.

For unstable states such as single-foot phases, the reset should use either:

- full clean simulator state from cached rollout; or
- a short clean pre-roll before the segment start.

The purpose is to avoid asking the robot to start from a physically impossible
static pose.

## 4. HSL Initialization

HSL remains useful, but its role changes.

Old role:

- HSL gives a proposal;
- HRL decides whether to accept it.

New role:

- HSL gives an initial repair direction;
- HRL starts from this policy and learns full 6D dynamic repair.

HSL can be trained from Clean-Noisy geometric targets.  It reduces early RL
exploration difficulty because the policy does not begin from random 6D Delta
SE outputs.

HSL should not constrain the final HRL policy to scale the HSL direction.  HRL
must be allowed to:

- weaken HSL;
- strengthen HSL;
- change individual dimensions;
- oppose HSL in a dimension if rollout reward shows HSL is dynamically harmful.

Therefore the HRL action should be full 6D Delta SE, or a full 6D residual on
top of HSL.  It should not be a scalar write strength.

## 5. HRL Training Objective

HRL is trained as a segment-level repair policy.

For each sampled segment:

- reset the robot to the segment's clean dynamic state;
- corrupt the reference to form the Noisy segment;
- let HRL output a 6D Delta SE repair;
- apply the repair before the frozen GMT tracker;
- run GMT for K steps;
- compute reward from repaired performance relative to Noisy and Clean.

The central reward question is:

Does this repair make frozen GMT execute the corrupted segment better than doing
nothing?

The reward should emphasize:

- improvement over Noisy baseline;
- closeness to Clean execution;
- survival / no fall;
- contact consistency;
- smooth and bounded Delta SE;
- no unsafe upward dz unless intentionally enabled.

The reward should not turn HRL into a new tracker.  It should measure repair
improvement, not absolute task performance alone.

## 6. From Probing To RL

The earlier probing idea was:

- manually sample several Delta SE candidates around a segment;
- rollout each candidate;
- pick the best;
- use it as a supervised target.

The new RL version is:

- policy samples 6D Delta SE actions through its stochastic action distribution;
- rollout evaluates those actions;
- PPO reinforces actions that improve the segment and suppresses actions that
  damage it.

Segment PPO advantage scaling must preserve this no-regret meaning.  A positive
gain means Repaired improved over Noisy, so the default Segment HRL update must
not turn positive gain rows into negative training weights merely because they
are below the mini-batch mean.  Standard PPO mean-centering is allowed only as an
explicit ablation.  The default is sign-preserving scale-only advantage scaling,
recorded in `note/frontres_segment_replay/contracts/advantage_scaling_contract.md`.

So probing is not removed.  It is absorbed into PPO exploration.

The difference is:

- old probing had a hand-written search distribution;
- RL probing learns the search distribution;
- old probing solved each segment locally;
- HRL learns a reusable repair policy across segments.

## 7. Global Sampling And Repeated Rollout

The clean formulation is:

- global sampling discovers segments;
- replay sampling repeats valuable segments.

Do not train by sampling uniformly from the full dataset forever.  In a large
dataset, that makes most segments appear too rarely for local repair learning.

Do not train by fixing a small set of segments forever.  That loses global
coverage.

Use Prioritized Segment Replay.

Each segment is treated as a level.  A segment first enters training through
global random sampling.  If it has learning value, it enters a replay pool and
is sampled repeatedly.

This makes single-segment multiple rollout a scheduler outcome, not a manually
fixed inner loop.

When the scheduler chooses multi-trial replay for one segment, the policy
parameters should stay fixed during those trials.  The trials estimate local
repair evidence for that segment; they should not trigger an optimizer update
after every sampled action.  The policy update should still happen from a batch
containing multiple segments, so the actor learns a reusable repair rule rather
than overfitting one segment visit.

## 8. Prioritized Segment Replay

Each PPO batch should mix three sources.

Global samples:

- sampled from the full motion segment pool;
- maintain coverage;
- discover new repair cases.

Replay samples:

- sampled from segments with current learning value;
- provide repeated rollout on difficult but learnable cases;
- are the main place where local repair skill improves.

Review samples:

- sampled from already solved segments at low rate;
- prevent forgetting.

A practical schedule can start with more global sampling and gradually increase
replay:

- early: mostly global, some replay;
- middle: balanced global and replay, small review;
- late: stable global coverage, strong replay, more review.

The exact ratios are engineering knobs.  The method concept is the mixture, not
one fixed ratio.

## 9. What Makes A Segment Worth Replay

The replay priority should not be based only on low reward.

Low reward can mean:

- difficult but learnable;
- impossible or outside the repair authority;
- noisy or unstable reward.

The replay priority should estimate learning value.

High replay priority:

- Noisy is damaged but not hopeless;
- HRL has positive but incomplete repair gain;
- recent repaired gain is improving;
- reward is stable enough to learn from;
- the segment is near the repair frontier.

Low replay priority:

- Noisy is already fine;
- no repair action improves it;
- the segment is consistently unrecoverable;
- reward is unstable or contradictory;
- the segment is already solved.

This follows the same idea as prioritized level replay and reducible-loss
sampling: repeat samples that the policy can still learn from, not merely
samples with bad outcomes.

### 9.1 Mature-work alignment

The replay evaluator should adapt mature replay and local-search ideas rather
than use raw reward as a shortcut.

- Prioritized Level Replay / ACCEL: replay a level because it has future
  learning utility, not because it is simply bad.  For FEMR, the level is a
  motion segment.
- CEM / MPPI / PI2-style local search: repeated trials on the same local
  problem reveal whether good actions exist and whether the current policy can
  find them reliably.  For FEMR, this becomes multi-trial Delta SE evidence
  under a fixed policy snapshot.
- AWR / AWAC-style advantage weighting: high-value trial actions can later
  become supervised or weighted-regression evidence, but they must not be
  silently treated as PPO on-policy samples unless they were actually sampled
  from the stored old policy distribution with matching log probabilities.

The important transfer is not a literal algorithm copy.  The transferable rule
is: replay priority should estimate learning potential at the segment frontier.

### 9.2 Multi-trial replay evidence

A segment should be evaluated by a small set of local trials when its current
state suggests that one rollout is not enough to judge repairability.

Core multi-trial fields:

- `trial_count`: number of rollout attempts collected for this segment visit.
- `policy_gain`: gain from the ordinary policy-sampled action.
- `best_gain`: best gain observed among trials.
- `mean_gain`: average gain over trials.
- `success_frac`: fraction of trials with positive no-regret gain and no fall.
- `fall_frac`: fraction of trials that fall or become invalid.
- `oracle_gap`: `best_gain - mean_gain` or `best_gain - policy_gain`.

Interpretation:

- high `best_gain` and low `success_frac`: the segment is learnable but the
  current policy is unreliable, so replay priority should increase;
- high `best_gain` and high `success_frac`: the segment is near solved, so it
  should decay into review sampling;
- low `best_gain` and high `fall_frac`: the segment is likely hopeless or
  outside the current repair authority, so active replay should decrease;
- high `oracle_gap`: a good local repair exists but the policy does not find it
  consistently, so the segment has high learning value.

The replay score should therefore be learning-potential based:

```text
base_priority = repair_need * solvability * learning_gap * confidence
priority = base_priority + freshness_or_diversity_bonus
```

Where:

- `repair_need` measures whether Noisy is actually damaged;
- `solvability` measures whether at least one trial can improve the segment
  without falling;
- `learning_gap` is driven by `oracle_gap` and incomplete `success_frac`;
- `confidence` downweights a single lucky trial and increases after repeated
  consistent evidence;
- `freshness_or_diversity_bonus` prevents the pool from collapsing onto a few
  segments or one perturbation family.

Implementation checkpoint:

- Step 2 implements the fixed-policy multi-trial evidence aggregator in
  `frontres_segment_sampler.py`.
- The implemented S1 boundary groups multiple rollout rows for the same
  segment visit into `trial_count`, `policy_gain`, `best_gain`, `mean_gain`,
  `success_frac`, `fall_frac`, `oracle_gap`, and `confidence`.
- Until the live scheduler stores explicit trial roles, `policy_gain` is the
  first rollout row for that segment visit.
- Step 3 implements the sampler-owned rollout budget planner:
  `segment_state -> trial_count / horizon_k`.
- Step 3 also implements the pure trial-row expansion contract: each selected
  segment expands into one `policy` trial followed by zero or more `search`
  trials. This makes the Step 2 `policy_gain = first row` convention explicit.
- Step 4 wires trial-row expansion into the live sampler connector under a
  fixed env-row budget. The live sampler now builds executable rows such as
  `[segment 0 policy, segment 0 search]` before dataset batch construction and
  env probe, while keeping PPO storage/update semantics unchanged.
- In paired B1 rollout layouts, Step 4 budgets only the scorable FrontRES repair
  rows (`n_train`), not the total vectorized env count. Candidate, Noisy, and
  Clean rows are paired counterfactual baselines for those repair rows, so they
  must not be sampled as independent policy/search trial evidence rows.
- Step 4 intentionally does not enlarge `num_envs`, does not run optimizer
  updates between local trials, and does not feed best-trial actions into PPO.
  It only changes which segment rows the next live probe executes and how the
  sampler receives policy/search evidence.
- Step 5 wires the live probe interface for trial metadata. The current sampled
  batch now carries `policy/search`, `source_index`, `trial_index`, and
  `budget_horizon_k` through reset requests, storage-side priority evidence,
  probe summaries, and human-readable `trial.*` log lines.
- For paired quartet probes, Step 5 expands `n_train` scorable trial metadata to
  the full env capture batch by marking the paired candidate, Noisy, and Clean
  rows as `baseline`. These baseline rows are visible for role readability, but
  they are not policy/search evidence rows and cannot become PPO-valid rows.
- Step 5 still does not change PPO semantics: trial metadata is not added to
  the PPO batch, no optimizer update is run between local trials, and no
  best-trial regression is introduced.
- Step 6 makes the PPO boundary explicit. `policy` trial rows are eligible for
  the on-policy PPO tuple, while `search` rows remain rollout evidence for
  replay priority, local comparison, and future non-PPO branches.
- Step 6 intentionally does not change the PPO loss, optimizer, trust-region
  rule, or direct 6D Delta SE action semantics. It only prevents non-policy
  multi-trial evidence from silently becoming PPO update rows.
- Step 7 adds diagnostics for the Step 4-6 boundary. Live probe, live sampler,
  live update-loop, and normal live-training logs should expose trial roles,
  evidence rows, PPO-eligible rows, search-only evidence rows, policy-invalid
  rows, and sampler oracle quality (`oracle_gap`, `confidence`,
  `delayed_regret`) in human-readable form.
- Step 7 intentionally does not change PPO loss, sampler priority, rollout
  allocation, env stepping, reward, or long-horizon curriculum. It is a
  white-box visibility layer for auditing whether multi-trial evidence and
  on-policy PPO rows remain separated.
- Long-horizon execution remains a later S4 boundary: Step 5 lets index-reset
  probe requests prefer the planned `budget_horizon_k`, but it does not yet
  prove real IsaacLab long-window execution quality.

Suggested rollout-budget states:

- `unknown`: one probe rollout;
- `promising`: three local trials;
- `frontier`: up to six local trials;
- `diagnostic`: up to eight local trials, used rarely for debugging or method
  validation;
- `solved`: mostly review sampling;
- `hopeless`: low-rate recheck only.

Forbidden shortcuts:

- do not use low reward alone as priority;
- do not use positive gain alone as priority;
- do not update network parameters after each single segment trial;
- do not mix heuristic or best-of-trial actions into PPO as if the actor caused
  them, unless the action source, old log probability, and distribution stats
  are policy-consistent.

### 9.3 Horizon curriculum and delayed no-regret

The default short segment horizon, such as `k=8`, should be treated as a local
repair probe, not as proof that the repair remains valid over time.  A Delta SE
action can improve the first few control steps and still create delayed damage
through balance drift, contact mismatch, or accumulated tracker error.

The long-horizon variable is temporal repair validity:

```text
Does a locally useful repair remain no-regret when the frozen tracker executes
the repaired reference for a longer horizon?
```

The curriculum should combine two mechanisms:

- global horizon unlock: the training run gradually allows longer evaluation
  horizons;
- segment-specific horizon allocation: each segment receives only the horizons
  justified by its current evidence state.

Do not run every segment through every horizon from `k=8` to `k=64`.  That
would waste rollout budget on already fine, solved, or hopeless segments.  Also
do not use a purely global schedule where all segments move from `k=8` to
`k=64` at the same time.  Delayed-regret cases can appear early and should be
detected as soon as short-horizon success becomes suspicious.

Recommended global unlock schedule:

- Phase 0: `max_horizon = 8`, learn and diagnose immediate local repair.
- Phase 1: `max_horizon = 16`, check short persistence for promising segments.
- Phase 2: `max_horizon = 32`, detect delayed regret and early drift.
- Phase 3: `max_horizon = 64`, validate stability for frontier and solved
  segments.
- Phase 4: `horizon = 120+`, use offline sequence validation or low-rate review,
  not default dense training.

Recommended segment-specific horizon allocation:

- `unknown`: run `k=8`;
- `promising`: run `k=8` plus occasional `k=16`;
- `frontier`: run `k=8` multi-trial plus selected `k=16` or `k=32`;
- `delayed_regret`: run `k=32` or `k=64` because short gain is not trusted;
- `solved`: run low-rate `k=64` or sequence review to prevent forgetting;
- `hopeless`: run low-rate `k=8` recheck and avoid long rollout budget.

Core horizon fields:

- `gain_k8`: immediate repair gain;
- `gain_k16`: short persistence;
- `gain_k32`: delayed-regret check;
- `survival_k64`: longer stability check;
- `fall_time`: first fall or invalid step;
- `drift_slope`: long-window error growth rate;
- `delayed_regret`: true when short-horizon gain is positive but longer-horizon
  gain becomes negative or the rollout falls;
- `horizon_tag`: the horizon that produced a reward, priority update, or
  diagnostic row.

The horizon curriculum should use short horizons and long horizons differently:

- `k=8` and selected `k=16` are the primary PPO training signal because they are
  cheaper and have clearer credit assignment;
- `k=32` and `k=64` should first act as temporal verifiers that update replay
  priority, segment state, delayed-regret diagnostics, and review selection;
- if longer horizons are later used directly in PPO, the batch must carry
  `horizon_tag`, use horizon-aware advantage scaling, and avoid mixing reward
  scales as if `k=8` and `k=64` were the same objective.

State transitions:

- `unknown -> promising`: high repair need, positive `best_gain_k8`, and low
  `fall_frac_k8`;
- `promising -> frontier`: high `oracle_gap_k8` or low `success_frac_k8`;
- `promising/frontier -> delayed_regret`: positive `gain_k8` but negative
  `gain_k16` or `gain_k32`, or early `fall_time`;
- `frontier -> solved`: stable positive gain across unlocked horizons and low
  fall rate;
- `any -> hopeless`: repeated non-positive `best_gain` with high fall rate.

Forbidden shortcuts:

- do not promote all segments to long horizons just because training iteration
  increased;
- do not treat `gain_k8 > 0` as solved unless persistence has been checked;
- do not mix `k=8`, `k=32`, and `k=64` rewards into PPO without horizon tags and
  scale control;
- do not continue scoring long-horizon reward after a fall as if it were useful
  repair evidence.

## 10. Relationship To Grouping And Experts

Explicit group-wise experts are not the first choice.

Grouping can reduce training difficulty, but it creates reviewer questions:

- Why these groups?
- Are the groups physically meaningful?
- Is the group label available at test time?
- Is the grouping only an engineering trick?

Therefore the first method should use one HRL policy with prioritized segment
replay.  This internalizes part of the hierarchy into the sampler.

Explicit experts should be added only if diagnostics show that one policy
cannot resolve conflicts across repair regimes.

A possible future upgrade is:

- identify dynamic repair regimes from data;
- train one expert per regime;
- distill the experts into a single deployable FEMR policy.

This is justified only if the diagnostics show real conflict, not merely
because experts are convenient.

## 11. Privileged Information And Distillation

Distillation is not automatically needed.

If the HRL actor uses the same observation and same model family as deployment,
then the trained HRL policy can be deployed directly.

Distillation becomes meaningful if the teacher has something the deployed model
does not have, such as:

- privileged critic inputs;
- full simulator state;
- true perturbation type or strength;
- Clean future window;
- larger MoE teacher architecture;
- explicit group label;
- more expensive online computation.

For the first version, the cleaner route is:

- actor uses deployment-visible observations;
- critic may use privileged training information;
- reward and sampler may use Clean/Noisy/cache diagnostics;
- deploy the actor directly if it works.

This is an asymmetric actor-critic design rather than a full teacher-student
pipeline.

Distillation should be introduced only if:

- the direct actor cannot learn;
- a privileged or expert teacher can learn;
- the difference between teacher and deployable student is clear.

## 12. Required Diagnostics

The method cannot be judged by final videos alone.  The training log should
prove the sampler and repair path are active.

Segment sampling diagnostics:

- global / replay / review sample fractions;
- replay pool size;
- new segment discovery rate;
- replay priority mean and distribution;
- solved / active / hopeless segment counts.

Repair diagnostics:

- Noisy score;
- Repaired score;
- Clean score or Clean-relative score;
- repaired gain over Noisy;
- fall rate;
- contact consistency;
- Delta SE magnitude by dimension;
- unsafe dz fraction.

Learning-value diagnostics:

- recent gain trend per replay segment;
- number of repeats before solved;
- hard segment success rate;
- replay segments retired as solved;
- replay segments retired as hopeless.

Multi-trial replay diagnostics:

- trial_count by segment state;
- best_gain / mean_gain / policy_gain;
- success_frac and fall_frac;
- oracle_gap;
- confidence score;
- freshness / diversity bonus;
- segment state counts: unknown / promising / frontier / solved / hopeless.

Horizon curriculum diagnostics:

- current global max_horizon;
- horizon sample fractions for k8 / k16 / k32 / k64 / sequence review;
- gain_k8 / gain_k16 / gain_k32;
- survival_k64 and fall_time;
- drift_slope;
- delayed_regret count and fraction;
- segment state transitions into and out of delayed_regret;
- PPO batch horizon_tag distribution if long horizons enter policy updates.

HSL/HRL diagnostics:

- HSL supervised loss / proposal magnitude;
- HRL action magnitude;
- HRL residual from HSL if residual parameterization is used;
- HRL reward gain over HSL and Noisy;
- fraction of segments where HRL improves beyond HSL.

## 13. Implementation Boundary

This note is not yet code.

The implementation must be modular from the first commit.  The runner should
remain an orchestrator only.  It may call modules, pass tensors, and write
storage, but it must not own segment sampling math, motion-cache logic, reward
construction, replay priority updates, HSL/HRL loss math, or diagnostic
formatting.

Expected module ownership:

- `frontres_segment_dataset.py`: segment metadata, clean dynamic state payload,
  and reference-window indexing.
- `frontres_segment_sampler.py`: global / replay / review sampling mixture and
  prioritized segment replay state.
- `frontres_segment_reset.py`: dynamic reset or clean pre-roll adapter.
- `frontres_segment_reward.py`: Noisy-relative K-step executable reward and
  Clean-relative diagnostics.
- `frontres_hrl_action.py`: 6D Delta SE action construction, masks, bounds, and
  HSL initialization / residual mode.
- `frontres_segment_diagnostics.py`: console and scalar diagnostic formatting.

Each module should have its own small test with constructed toy data before it
is connected to the live training runner.  The minimum test ladder is:

- pure data-shape test for segment metadata and cache payload;
- deterministic sampler test for global / replay / review proportions;
- priority-update test showing solved segments decay and learnable segments
  replay more often;
- multi-trial evidence test showing best_gain, mean_gain, success_frac,
  oracle_gap, confidence, and state transitions are computed from a fixed-policy
  segment visit before any optimizer update;
- horizon-curriculum test showing segments receive k8 / k16 / k32 / k64 only
  according to global unlock and segment state, not by dense all-horizon sweep;
- delayed-regret test showing positive short gain plus negative long gain or
  early fall promotes the segment to delayed_regret and raises verifier priority;
- dynamic-reset payload test using a fabricated clean state;
- reward-construction test comparing Noisy, Repaired, and Clean scores;
- HRL action-bound test for 6D Delta SE masks and unsafe dz handling;
- runner integration sentinel only after the above tests pass.

The engineering failure to avoid is a second large `on_policy_runner.py`
implementation where every concept is embedded inline.  If a change requires
more than a thin runner adapter, first create or extend the owning FrontRES
module and test that module directly.

Before implementation, the live path must be audited in this order:

- config flag and default;
- segment dataset and motion cache;
- dynamic reset / pre-roll;
- perturbation construction;
- HSL warmup path;
- HRL action parameterization;
- frozen GMT execution path;
- K-step reward construction;
- prioritized segment replay scheduler;
- PPO storage and update;
- diagnostics.

Do not silently reuse acceptance-label storage fields for the new HRL action
unless their meaning is renamed or explicitly documented.

## 14. External Design Anchors

Useful mature references:

- DeepMimic: reference state initialization and short-horizon motion imitation
  from dynamic motion states.  https://arxiv.org/abs/1804.02717
- Prioritized Level Replay: replay environment instances that remain useful for
  learning.  https://arxiv.org/abs/2010.03934
- ACCEL: use edit/replay curriculum around the agent's capability frontier
  rather than only the hardest levels.  https://arxiv.org/abs/2203.01302
- Prioritized Experience Replay: non-uniform replay based on learning utility.
  https://arxiv.org/abs/1511.05952
- Reducible loss sampling: prefer samples whose loss can still be reduced, not
  merely samples with high irreducible error.  https://arxiv.org/abs/2208.10483
- AWR: advantage-weighted policy learning from high-value actions; useful as a
  future bridge from multi-trial evidence to supervised actor updates, but not a
  PPO on-policy substitute by default.  https://arxiv.org/abs/1910.00177
- MPPI / PI2-family local rollout optimization: repeated candidate rollouts can
  estimate local action quality, but FEMR should use it as budgeted segment
  evidence rather than exhaustive per-segment control.
  https://arxiv.org/abs/1206.4621

Code references for segment reset and motion cache are recorded in
`note/frontres_segment_replay/references/segment_rollout_code_references.md`.
