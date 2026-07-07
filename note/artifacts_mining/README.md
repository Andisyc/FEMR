# Artifacts Mining Notes

## Purpose

This folder records the working discussion, checklist, and experiment notes for
mining real reference artifacts for FrontRES/FEMR.

The goal is to show that the Repair problem exists in practical reference
generation pipelines, not only in synthetic perturbation families.

## Core Claim

Practical humanoid tracking often depends on upstream reference sources such as
video-derived motion recovery, retargeted MoCap, teleoperation, and motion
synthesis. These sources can produce local robot-domain reference artifacts:
root drift, root height jumps, contact inconsistency, foot skating, penetration,
temporal jitter, and clip-boundary discontinuities.

FrontRES targets the repairable subset:

```text
real reference artifact
  -> frozen GMT execution degrades
  -> task-space repair can reduce the artifact
  -> repaired reference becomes more executable
```

This is different from Recovery. Recovery starts after the robot has fallen or
entered an unstable state. Repair acts before execution, by improving the
reference frame given to the frozen tracker.

## Working Taxonomy

| Class | Meaning | Use |
| --- | --- | --- |
| `clean` | Reference is continuous, contact-consistent, and executable by GMT. | Control set. |
| `hard_clean` | Motion is difficult but the reference itself is not visibly corrupted. | Robust tracking boundary. |
| `noisy_repairable` | Local artifact exists, GMT degrades, and correction is plausibly inside the FrontRES action cone. | Main mining target. |
| `broken_unrepairable` | Reference is too corrupted or physically impossible. | Boundary diagnostic, not main training data. |
| `recovery` | Robot has already fallen or needs stand-up behavior. | Out of scope for FrontRES repair. |

## Artifact Sources

- Video-derived motion recovery: camera/depth errors, temporal jitter, foot
  skating, root drift.
- Retargeted AMASS/MoCap: joint limits, penetration, root height mismatch,
  contact inconsistency after robot-domain retargeting.
- Teleoperation logs: sensor drift, latency, operator inconsistency.
- Motion synthesis or clip stitching: abrupt transitions, phase mismatch,
  velocity discontinuity.

## Scoring Axes

Use two scores, not one:

1. `artifact_score`: how corrupted the reference is before execution.
2. `exec_damage_score`: how much the frozen tracker suffers when executing it.

The main sample condition is:

```text
artifact_score high
exec_damage_score high
sample not fully broken
artifact family inside FrontRES action cone
```

## RGMT Gap

RGMT acknowledges imperfect references and evaluates robustness under noisy
commands, but it does not provide a reproducible artifact mining or artifact
scoring pipeline. This folder records the missing mining layer for FrontRES.

## Files

- `checklist.md`: operational checklist for collecting, scoring, and reviewing
  candidate artifact windows.
