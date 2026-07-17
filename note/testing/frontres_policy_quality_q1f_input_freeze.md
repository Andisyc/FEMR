# FrontRES Policy Quality Q1-F Input Freeze

Status: `prepared-awaiting-human-review`
Date: 2026-07-17

## Objective

Freeze one minimal real-simulator comparison question before Q1-F. This file
authorizes no live execution and makes no policy-quality claim.

## Frozen Inputs

| Object | Frozen value | Evidence / rationale |
| --- | --- | --- |
| Manifest | `note/testing/manifests/frontres_policy_quality_q1f_single_v1.json` | Immutable Q1 schema; one item only. |
| HSL baseline | `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_02-59-32_FEMR_FORMAL_RUNTIME_AUDIT_ACTOR_SENTINEL_20260717/model_200.pt` | Actor-update-free HSL inheritance after critic warmup; loaded as iter 200 in the formal log. |
| Tested policy | `/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_11-12-46_FEMR_FORMAL_RUNTIME_AUDIT_JOINT_RESUME_20260717/model_701.pt` | Same resumed lineage after full-weight actor updates; complete persistence recorded at iter 701. |
| Motion | `KIT/572/amass_g1_wave_right02_poses_reflect.npz` | Successfully reached the canonical index-reset owner in the formal run. |
| Start frame | `163` | Same formal reset evidence row as the selected motion. |
| Perturbation | `local_rp`, `dr_scale=1.25` | Active specialist family and formal-run DR scale. |
| Horizon | `K=8` | Smallest active K; minimizes the identity sentinel cost. |
| Seed | `42` | Formal run seed. |
| Result | `/hdd1/cyx/FEMR/policy_quality_q1f_single_v1_result.json` | Dedicated artifact outside old eval outputs. |

Frozen signatures:

- item: `206e4c1bd7aec5e987049fa9697b755cef826ed093c3683fb7f057f38e29d2eb`;
- manifest: `4c7122e5278c2371d2917659e0ac5944ac1dd8579de94cc99811bdf95dd5eee0`.

## Required Server Preflight

Before Q1-F may run, verify both files exist and record SHA-256 hashes. The
local workspace does not contain these large checkpoints, so their hashes are
currently `UNCONFIRMED`.

```bash
test -f /hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_02-59-32_FEMR_FORMAL_RUNTIME_AUDIT_ACTOR_SENTINEL_20260717/model_200.pt
test -f /hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_11-12-46_FEMR_FORMAL_RUNTIME_AUDIT_JOINT_RESUME_20260717/model_701.pt
sha256sum /hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_02-59-32_FEMR_FORMAL_RUNTIME_AUDIT_ACTOR_SENTINEL_20260717/model_200.pt
sha256sum /hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-17_11-12-46_FEMR_FORMAL_RUNTIME_AUDIT_JOINT_RESUME_20260717/model_701.pt
```

## Q1-F Acceptance

The live sentinel may establish only:

- one comparison signature for zero/HSL/policy;
- identical initial-state hashes before all three routes;
- explicit checkpoint identities;
- unchanged optimizer, Segment sampler, and warmup state;
- finite owner outputs and one atomic result artifact.

It must not be interpreted as Q2 policy superiority, Q3 checkpoint trajectory,
or long-training admission.

## Blocker

Q1-F remains blocked until Dr. Cheng reviews this freeze and the server
checkpoint existence/hash preflight is available.
