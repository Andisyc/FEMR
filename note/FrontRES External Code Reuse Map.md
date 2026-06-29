# FrontRES External Code Reuse Map
Date: 2026-06-27

This note is Step 2 after `FrontRES Segment Replay Engineering Intake.md`.
Its purpose is to decide what we should copy or adapt before writing FEMR
Segment Replay HRL code.

The goal is not to copy a whole repository.  The goal is to reuse mature
building blocks for motion clips, dynamic reset, reference-state sampling,
segment replay, and prioritized sampling, then connect them to FEMR through
small modules that can be tested independently.

## 1. Main Decision

The first implementation should copy infrastructure ideas, not training logic.

Copy or adapt:

- motion clip library;
- dynamic reference-state reset;
- motion phase / reference-state sampling;
- segment metadata and clean-state cache;
- prioritized segment sampler;
- IsaacLab-style command/reset integration if the interface matches FEMR.

Keep FEMR-specific:

- Delta SE(3) action cone;
- Noisy/Repaired/Clean executable score;
- K-step repair reward;
- HSL initialization;
- HRL action semantics;
- recovery diagnostics.

Do not copy:

- a full external policy network;
- a full external runner;
- AMP discriminator training;
- diffusion teacher/student training;
- old HSL acceptance semantics.

## 2. Copy Priority

Priority A:

- IsaacGymEnvs AMP `MotionLib`
  - best source for batched motion state sampling and dynamic state payload;
  - use it to design `frontres_segment_dataset.py` and
    `frontres_segment_reset.py`.
- facebookresearch `level-replay` `LevelSampler`
  - best source for segment replay scheduling;
  - use it to design `frontres_segment_sampler.py`.

Priority B:

- BeyondMimic / whole_body_tracking
  - likely closest to FEMR if its IsaacLab command/reset path matches;
  - use it for command manager, reset adapter, and motion reference plumbing.
- PHC `MotionLibBase`
  - useful for large motion datasets, GPU cache, velocity cache, and
    success/statistics based sampling.

Priority C:

- DeepMimic
  - use for concept and reset rules, especially reference-state
    initialization and early termination;
  - do not copy its old C++ training stack.
- MimicKit
  - inspect for a cleaner modern motion abstraction before writing our own.
- ASE
  - secondary source for AMP-style motion datasets and history windows.

## 3. FEMR Module Map

### `frontres_segment_dataset.py`

Role:

- own motion segment metadata;
- own clean dynamic-state cache;
- provide the state payload needed by reset and reward modules.

External sources to copy/adapt:

- IsaacGymEnvs AMP `MotionLib`
  - `sample_motions`;
  - `sample_time`;
  - `get_motion_state`;
  - batched tensor output style;
  - root/joint velocity payload.
- PHC `MotionLibBase`
  - large dataset cache style;
  - motion-level statistics;
  - GPU/device cache patterns.
- MimicKit
  - inspect if it provides a simpler current motion dataset abstraction.

Required FEMR payload:

- `segment_id`;
- `motion_id`;
- `start_frame` or `start_time`;
- `phase`;
- `horizon_k`;
- `root_pos`;
- `root_quat`;
- `root_lin_vel`;
- `root_ang_vel`;
- `dof_pos`;
- `dof_vel`;
- key body states when available;
- GMT reference window;
- perturbation family;
- perturbation strength.

FEMR additions:

- `clean_score_cache`;
- `noisy_baseline_cache`;
- `repaired_score_cache`;
- replay priority fields;
- reset validity flags.

Do not do:

- do not assume static pose reset is enough;
- do not copy joint ordering or retargeting assumptions blindly;
- do not hide simulator-specific tensors inside the dataset API.

Test shape:

- create a tiny fake motion with root pose, joint pose, and nonzero velocities;
- sample multiple segment starts;
- verify returned segment ids, dynamic state tensors, and reference windows.

### `frontres_segment_reset.py`

Role:

- reset the simulator to a replayable segment state;
- choose direct dynamic reset or clean pre-roll;
- validate that the reset state is physically usable.

External sources to copy/adapt:

- DeepMimic
  - reference-state initialization;
  - motion phase sampling;
  - early termination for unrecoverable states.
- IsaacGymEnvs AMP
  - reset-to-motion-state payload;
  - root and joint velocity restore.
- BeyondMimic / whole_body_tracking
  - IsaacLab command/reset integration if interface-compatible.

Required behavior:

- support direct reset from cached clean dynamic state;
- support clean pre-roll for states that cannot be reconstructed from pose
  alone;
- preserve root velocity, root angular velocity, and dof velocity;
- return reset diagnostics, not just success/failure.

Reset diagnostics:

- direct reset success;
- pre-roll used;
- invalid static reset;
- fall at reset;
- contact mismatch;
- velocity mismatch.

Test shape:

- use fake segment states with nonzero velocities;
- verify reset request contains position and velocity fields;
- verify static-pose-only reset is rejected for dynamic segments.

### `frontres_segment_sampler.py`

Role:

- decide which segment to train on next;
- balance global coverage and repeated learning on hard/useful segments;
- update priority from rollout evidence.

External sources to copy/adapt:

- facebookresearch `level-replay`
  - unseen/seen level handling;
  - score transform;
  - staleness;
  - replay schedule;
  - update from rollout result.
- PHC
  - optional motion-level success statistics.

FEMR interpretation:

- a `level` is a motion segment;
- priority means learning value, not just difficulty.

High priority:

- Noisy is harmful;
- repaired rollout improves over Noisy;
- segment is not solved yet;
- segment is near the recoverable frontier;
- K-step score is stable enough to learn from.

Low priority:

- Noisy is already fine;
- segment is solved repeatedly;
- segment is hopeless at current perturbation strength;
- reset is unstable;
- reward evidence is noisy or invalid.

Required sampling mixture:

- global sampling for coverage;
- replay sampling for useful hard segments;
- review sampling for solved segments so they do not regress.

Test shape:

- create toy segment scores;
- verify unseen segments are sampled;
- verify high-gain unsolved segments get replayed;
- verify stale segments return occasionally;
- verify hopeless segments do not dominate training.

### `frontres_segment_reward.py`

Role:

- compute the K-step recovery reward for a segment;
- measure improvement over Noisy;
- keep Clean as reference/diagnostic, not as the direct HRL target.

External sources:

- no external reward should be copied wholesale;
- AMP/BeyondMimic reward code can be inspected only for feature organization
  and termination handling.

FEMR sources to reuse:

- `frontres_executability.py`;
- `frontres_reward_window.py`;
- `frontres_post_step_connector.py`.

Reward meaning:

- HRL is trained by executable gain over the Noisy baseline;
- the action is useful if it makes GMT more executable over K steps;
- full environment reward should not be used directly as the main signal.

Required outputs:

- `score_noisy`;
- `score_repaired`;
- `score_clean`;
- `gain_over_noisy`;
- `fall_flag`;
- `contact_consistency`;
- `valid_mask`;
- `priority_evidence`.

Test shape:

- feed synthetic Noisy/Repaired/Clean score windows;
- verify positive gain, negative gain, solved, and hopeless cases.

### `frontres_hrl_action.py`

Role:

- define the HRL action object;
- convert policy output to bounded Delta SE(3);
- support HSL initialization without retaining acceptance semantics.

External sources:

- none should be copied directly.

FEMR sources to reuse:

- `frontres_action_cone.py`;
- current 6D task correction projection;
- current upward `dz` safety rules.

Action meaning:

- HRL outputs a 6D repair action:
  - `dx`;
  - `dy`;
  - `dz`;
  - `droll`;
  - `dpitch`;
  - `dyaw`.

Allowed modes:

- full repair policy initialized from HSL;
- residual-on-HSL policy if later needed for ablation.

Forbidden mode:

- do not treat HRL as acceptance over HSL proposal.

Test shape:

- verify active dimension mask;
- verify action bounds;
- verify upward `dz` constraint;
- verify HSL weights can initialize the repair actor without loading an
  acceptance actor.

### `frontres_segment_diagnostics.py`

Role:

- log whether segment replay is doing the right work;
- keep new HRL diagnostics separate from old acceptance diagnostics.

External sources:

- use FEMR logging style;
- do not copy external logging systems.

Required metrics:

- global/replay/review sample ratio;
- replay pool size;
- priority distribution;
- solved / active / hopeless counts;
- K horizon;
- Noisy score;
- Repaired score;
- Clean score;
- gain over Noisy;
- fall rate;
- contact consistency;
- action norm per dimension;
- perturbation family / strength.

Test shape:

- build a fake batch of segment rollout evidence;
- verify aggregation names and values;
- verify no old acceptance keys are emitted as primary HRL diagnostics.

### Optional `frontres_segment_storage.py`

Role:

- only needed if existing `RolloutStorage` becomes too overloaded.

Required fields if created:

- `segment_id`;
- `segment_phase`;
- `perturbation_family`;
- `perturbation_strength`;
- 6D repair action;
- log-prob;
- value;
- return;
- advantage;
- valid mask;
- priority evidence.

Do not do:

- do not add another large set of unrelated fields to old acceptance storage
  before this tuple is fixed.

## 4. Repository-by-Repository Use

### IsaacGymEnvs AMP

Repository:

```text
https://github.com/isaac-sim/IsaacGymEnvs
```

Useful file:

```text
isaacgymenvs/tasks/amp/utils_amp/motion_lib.py
```

Use for:

- motion state sampling;
- batched dynamic state payload;
- velocity cache;
- state interpolation.

Map to FEMR:

- `frontres_segment_dataset.py`;
- `frontres_segment_reset.py`.

### PHC

Repository:

```text
https://github.com/ZhengyiLuo/PHC
```

Useful file:

```text
phc/utils/motion_lib_base.py
```

Use for:

- large motion library cache;
- motion-level statistics;
- success-history based sampling;
- GPU cache design.

Map to FEMR:

- `frontres_segment_dataset.py`;
- `frontres_segment_sampler.py`.

Local status:

- the local `RoboJudo_Real/third_party/phc` checkout appears incomplete;
- inspect the GitHub source before copying code.

### DeepMimic

Repository:

```text
https://github.com/xbpeng/DeepMimic
```

Paper:

```text
https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf
```

Use for:

- reference-state initialization;
- phase sampling;
- reset to intermediate dynamic states;
- early termination;
- classifying unrecoverable segments.

Map to FEMR:

- `frontres_segment_reset.py`;
- segment validity rules.

### MimicKit

Repository:

```text
https://github.com/xbpeng/MimicKit
```

Use for:

- modern motion-imitation organization;
- possible cleaner motion library abstraction;
- simulator-backend separation.

Map to FEMR:

- `frontres_segment_dataset.py`;
- `frontres_segment_reset.py`.

### BeyondMimic / whole_body_tracking

Project page:

```text
https://beyondmimic.github.io/
```

Use for:

- IsaacLab-style whole-body tracking task organization;
- command/reset integration;
- teacher-training workflow ideas.

Map to FEMR:

- reset adapter;
- command manager integration;
- reference state sampling path.

Do not copy yet:

- diffusion policy;
- distillation pipeline;
- full teacher/student training stack.

### Level Replay

Repository:

```text
https://github.com/facebookresearch/level-replay
```

Use for:

- prioritized replay over levels;
- unseen/seen scheduling;
- staleness;
- score transforms;
- update-from-result API.

Map to FEMR:

- `frontres_segment_sampler.py`.

FEMR translation:

- level = motion segment;
- level score = learning value from K-step repair evidence.

### ASE

Repository:

```text
https://github.com/nv-tlabs/ASE
```

Use for:

- AMP-style large motion dataset handling;
- history-window feature organization;
- secondary reference for scalable sampling.

Map to FEMR:

- secondary reference for `frontres_segment_dataset.py`;
- possible future history features.

## 5. Integration Rule

The runner should remain thin.

The intended live flow is:

```text
segment_sampler
 -> segment_dataset
 -> segment_reset
 -> HRL repair action
 -> K-step rollout
 -> segment_reward
 -> PPO update
 -> sampler priority update
 -> segment diagnostics
```

The runner may call this flow, but it should not own the logic.

## 6. Engineering Order

Next implementation order:

1. inspect Priority A source files and copy the minimal interfaces into a
   FEMR contract note;
2. define exact FEMR APIs for dataset, sampler, reset, reward, action, and
   diagnostics;
3. implement pure Python dataset/sampler modules with toy tests;
4. implement reset adapter behind a fake-env test first;
5. implement reward/action modules with synthetic score tests;
6. add a thin runner connector;
7. add PPO integration only after segment tuple and diagnostics are stable.

## 7. Stop Conditions

Stop before training if any of these happen:

- segment reset only restores pose and not velocity;
- segment sampler cannot explain why a segment is replayed;
- reward uses full environment reward as the main signal;
- HRL action becomes acceptance over HSL proposal;
- new fields are added to old acceptance storage without a segment storage
  contract;
- diagnostics cannot separate global/replay/review segments;
- old `stage2_acceptance` runs silently when the new Segment Replay stage is
  requested.

## 8. Step 2 Result

Step 2 fixes the external reuse plan:

- copy motion-state infrastructure from AMP/PHC/MimicKit style code;
- copy dynamic reset and RSI concepts from DeepMimic/AMP/BeyondMimic style
  code;
- copy prioritized replay structure from Level Replay;
- keep reward, action cone, HSL initialization, and recovery diagnostics as
  FEMR-specific modules;
- add new modules first, then connect them to the runner through a thin adapter.

Recommended next step:

- Step 3: write the FEMR Segment Replay Engineering Contract with exact module
  APIs, test data, and acceptance criteria before editing training code.
