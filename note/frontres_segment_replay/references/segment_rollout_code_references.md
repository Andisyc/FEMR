# FrontRES Segment Rollout Code References

Date: 2026-06-26

This note records external codebases to copy from before implementing segment
rollout, reference-state initialization, motion caches, or local rollout
probing.  Do not implement these systems from scratch unless the referenced
code cannot be adapted.

## Why This Note Exists

The next FrontRES design needs mature motion-imitation infrastructure:

- sample motion segments;
- reset to dynamic reference states, not static poses;
- store root and joint velocities;
- handle unstable phases with reference-state initialization or pre-roll;
- cache clean/noisy baselines;
- run short local rollout probing around a Delta SE repair.

These are standard motion-imitation engineering problems.  The first
implementation pass should copy/adapt a proven structure.

## Repositories To Inspect First

### 1. MimicKit

Repository:

```text
https://github.com/xbpeng/MimicKit
```

Why useful:

- current lightweight successor-style codebase from the DeepMimic author;
- explicitly supports motion imitation methods;
- supports multiple simulator backends, including Isaac Gym and Isaac Lab;
- contains motion data abstractions and modern training entrypoints.

What to look for:

- motion clip representation;
- motion dataset sampling;
- reset from motion time / reference state;
- environment config for DeepMimic-style humanoid imitation;
- clean separation between motion library, environment reset, and reward.

Use for:

```text
FrontRESSegmentDataset
FrontRESMotionSegmentSampler
Clean dynamic-state cache structure
```

### 2. DeepMimic

Repository:

```text
https://github.com/xbpeng/DeepMimic
```

Paper / concept:

```text
https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf
```

Why useful:

- canonical reference-state initialization design;
- handles dynamic motion imitation from intermediate motion states;
- uses early termination to avoid wasting training on unrecoverable failures.

What to look for:

- reference state initialization;
- motion phase sampling;
- reset to motion state;
- early termination conditions;
- dynamic motion handling for flips/kicks/highly unstable phases.

Use for:

```text
RSI concept
dynamic segment reset
unstable-segment pre-roll policy
termination/filtering rules
```

### 3. IsaacGymEnvs AMP

Repository:

```text
https://github.com/isaac-sim/IsaacGymEnvs
```

Useful file:

```text
isaacgymenvs/tasks/amp/utils_amp/motion_lib.py
```

Why useful:

- Isaac Gym-compatible motion library;
- `get_motion_state()` returns root position, root rotation, dof position,
  root velocity, root angular velocity, dof velocity, and key body positions;
- this is close to the state payload needed for segment reset/cache.

What to look for:

- `MotionLib.sample_motions`;
- `MotionLib.sample_time`;
- `MotionLib.get_motion_state`;
- velocity computation from adjacent frames;
- batching many motion states on GPU.

Use for:

```text
Clean cache payload:
  root_pos
  root_quat
  dof_pos
  root_lin_vel
  root_ang_vel
  dof_vel
  key_body_pos
```

### 4. ASE

Repository:

```text
https://github.com/nv-tlabs/ASE
```

Why useful:

- large-scale reusable skill embedding code built on AMP-style motion
  infrastructure;
- useful for motion dataset handling and scalable humanoid imitation training.

What to look for:

- AMP/ASE humanoid task reset;
- motion library reuse;
- motion sampling over large unstructured clips;
- discriminator observation windows if FrontRES later needs segment-history
  features.

Use for:

```text
large motion dataset sampling
history-window feature structure
segment sampling at scale
```

### 5. PHC

Repository:

```text
https://github.com/ZhengyiLuo/PHC
```

Useful file:

```text
phc/utils/motion_lib_base.py
```

Why useful:

- modern humanoid motion control codebase;
- has a motion library base with motion loading, GPU cache ideas, dof velocity
  computation, sampling history, and motion-level success statistics;
- useful for large-scale motion dataset management.

What to look for:

- `MotionLibBase`;
- motion data loading from pkl/joblib;
- dof velocity computation;
- GPU/device cache patterns;
- sampling probability and success-rate tracking.

Use for:

```text
motion cache implementation
segment sampling statistics
hard/easy motion sampling based on success history
```

## Implementation Rule

Before writing FrontRES segment rollout code, inspect the above codebases and
copy the closest mature structure for:

```text
motion library
dynamic state reset
reference-state initialization
motion phase sampling
velocity cache
segment metadata
```

Only write new FrontRES-specific code for:

```text
Noisy/Clean cache schema
Delta SE perturbation/probing
FrontRES executable score aggregation
Recovery-Aware Repair diagnostics
```

## Design Boundary

The method should not insert many arbitrary rollouts between Noisy and Clean.
The planned direction is:

```text
motion segment
  -> dynamic reset or clean pre-roll
  -> fixed noisy baseline cache
  -> local Delta SE probing
  -> short K-step executable score
  -> direct Delta SE improvement signal
```

This changes rollout from branch evaluation to local action-space probing.

