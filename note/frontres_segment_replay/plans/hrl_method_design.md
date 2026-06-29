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
- Prioritized Experience Replay: non-uniform replay based on learning utility.
  https://arxiv.org/abs/1511.05952
- Reducible loss sampling: prefer samples whose loss can still be reduced, not
  merely samples with high irreducible error.  https://arxiv.org/abs/2208.10483

Code references for segment reset and motion cache are recorded in
`note/frontres_segment_replay/references/segment_rollout_code_references.md`.
