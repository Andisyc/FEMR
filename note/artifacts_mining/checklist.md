# Artifacts Mining Checklist

## Scope Check

- [ ] Source is a practical reference pipeline, not only synthetic noise.
- [ ] Window has a plausible underlying human motion intent.
- [ ] Failure is caused by reference artifact, not only by motion difficulty.
- [ ] Artifact is plausibly repairable by FrontRES task-space action.
- [ ] Sample is not a pure Recovery case.

## Candidate Sources

- [ ] Video-derived motion recovery outputs.
- [ ] Retargeted AMASS/MoCap outputs.
- [ ] Teleoperation logs.
- [ ] Motion synthesis or clip-stitching outputs.

## Window Extraction

- [ ] Slice references into short windows, e.g. 1-3 seconds.
- [ ] Use overlap, e.g. 0.25-0.5 second stride.
- [ ] Keep source id, motion id, window start, and window end.
- [ ] Preserve enough metadata to replay the same window through GMT.

## Artifact Score

- [ ] Root height sink or jump.
- [ ] Root planar drift.
- [ ] Yaw or orientation discontinuity.
- [ ] Root velocity, acceleration, or jerk spike.
- [ ] Joint position or velocity spike.
- [ ] Foot skating during inferred contact.
- [ ] Foot or body penetration.
- [ ] Contact inconsistency.
- [ ] Joint limit violation after retargeting.
- [ ] Clip-boundary discontinuity or phase mismatch.

## Execution Damage Score

- [ ] Frozen GMT survival time.
- [ ] Fall flag.
- [ ] MPJPE or keypoint position error.
- [ ] Keypoint velocity error.
- [ ] Keypoint acceleration error.
- [ ] Contact quality degradation.
- [ ] Repaired-vs-noisy gain, if a repair candidate exists.

## Labeling

- [ ] `clean`
- [ ] `hard_clean`
- [ ] `noisy_repairable`
- [ ] `broken_unrepairable`
- [ ] `recovery`
- [ ] `uncertain`

## First Mining Pass

- [ ] Collect a small mixed source set.
- [ ] Score every candidate window.
- [ ] Keep bottom-score clean controls.
- [ ] Keep high-score artifact candidates by artifact family.
- [ ] Manually review boundary cases.
- [ ] Record which artifact families are actually common.
- [ ] Calibrate synthetic perturbation ranges from observed artifact statistics.
