---
contract_id: FRS-GAIN-v006
status: active
effective_date: 2026-07-27
updated_date: 2026-07-27
supersedes: FRS-GAIN-v005
scope: scalar paired root-invariant Intent improvement minus full-6D repair cost, with independent Contact, loaded-support phase-ZMP, and survival actor constraints
---

# Loaded-Support ZMP Applicability And Contact Failure

## Design Delta

FRS-GAIN-v005 correctly separated the scalar Intent objective from Contact,
phase-ZMP, and survival constraints. Its evidence rule nevertheless conflated
two different states:

```text
corrupt or unavailable ContactSensor payload
valid ContactSensor payload with zero actual loaded support
```

The first state invalidates the transaction. The second is an observed Physics
failure: Contact records the missed expected support, while ZMP is N/A because
no physical support resultant exists.

FRS-GAIN-v006 changes no actor, Critic, PPO projection, replay, HSL, observation,
or scalar objective. It changes only Physics evidence applicability and its
versioned identity.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Evidence Authority |
| `FRS-DP-07` | Repair Gain | `Q-01` | Loaded-Support Phase-ZMP |

## Evidence Authority

Noisy and every Repair attempt retain the same scenario, `x_t`, artifact, q29
Intent, Clean continuation, expected support, K, valid-step clock, frozen GMT,
and hash. Expected support and its immutable oriented envelope remain
Clean-continuation Physics-only evidence and never enter actor observations.

Actual left/right support comes from the existing force-threshold ContactSensor.
Raw contact points, normals, and normal-force magnitudes come from the separate
filtered foot-to-ground views used by `contact-wrench-zmp-v1`.

Evidence integrity and physical outcome are distinct:

```text
invalid payload = missing API, malformed shape/count/start, non-finite valid
                  contact value, or disagreement where actual loaded support
                  exists but no finite raw resultant can be constructed

physical no-load = valid payload and no actual loaded support
```

Invalid payload fails closed. Physical no-load remains a scored row and must not
be dropped, resampled, zero-filled, or converted into a transaction failure.

## Contact Constraint

For valid K step `k` and foot `f`, expected and actual Contact retain the fixed
early/late tolerance. Extra contact, missed expected contact, illegal switch,
or dragging contributes one foot-step violation:

```text
E_contact = dt * sum_valid_k,f mismatch[k,f]
c_contact = E_contact - B_contact
```

An expected-support step with zero actual loaded support is therefore an
explicit Contact violation.

## Loaded-Support Phase-ZMP

ZMP is evaluated only when all three conditions hold:

```text
zmp_applicable[k,role]
  = valid_step[k]
  AND expected_support[k].any()
  AND actual_loaded_support[k,role].any()
  AND NOT transition_recovery_exemption[k]
```

Consequences:

- expected flight: ZMP N/A, including illegal actual contact;
- expected support plus actual load: finite contact-wrench ZMP is required;
- expected support plus zero actual load: Contact violation and ZMP N/A;
- actual load reported by ContactSensor but raw wrench cannot produce finite
  ZMP: evidence disagreement and transaction failure.

Repair and Noisy may have different ZMP applicability because their actual
loaded support may differ. Diagnostics must preserve both masks. Noisy remains
a paired Intent/no-op reference and does not become the Repair safety threshold.

For applicable steps:

```text
zmp_depth[k] = relu(-margin[k])
E_zmp = dt * sum_applicable_k zmp_depth[k]
c_zmp = E_zmp - B_zmp
```

If a role has no applicable step, its ZMP trajectory and aggregate diagnostic
remain explicit N/A. The Repair ZMP constraint is inactive with an explicit
false applicability identity; it is never silently filled with zero evidence.

## Scalar Objective And Optimization

The scalar target remains unchanged:

```text
y_I = paired Intent improvement - full-6D repair cost
return_K = y_I
V_I(o_critic) ~= E[y_I | o_critic, active global K stage]
```

Contact, ZMP, and survival remain separate nonnegative actor constraints. Their
grouped first-order projection remains owned by FRS-PPO-v004. No Physics term
enters the scalar Critic target.

## Required Identity

```text
gain_contract_id = FRS-GAIN-v006
scalar_target_id = paired-intent-minus-repair-v1
constraint_schema_id = contact-loaded-phase_zmp-survival-physical-v2
zmp_estimator_id = contact-wrench-zmp-v1
support_envelope_id = clean-foot-pose-oriented-box-v1
actual_contact_id = contact-sensor-net-normal-force-threshold-v1
expected_phase_id = clean-foot-height-phase-v1
```

Checkpoint format remains `frontres-v015-checkpoint-v5`, but strict full resume
must reject v005/schema-v1 identities before mutating actor, Critic, optimizer,
sampler, or normalizer state.
Atomic held-out reports use `frontres-v015-heldout-quality-report-v2`; v1 cannot
represent separate Repair/Noisy ZMP applicability and is historical evidence.

## Forbidden Behavior

- aborting a transaction merely because a valid physical rollout lost support;
- evaluating ZMP without actual loaded support;
- treating contact-point count as loaded support;
- zero-filling an N/A ZMP margin or diagnostic;
- using Repair-created support polygons or root/capture-point proxies;
- combining Physics into the scalar objective/Critic;
- leaking expected support, actual Contact, or ZMP into actor input.

## Acceptance And Stop Conditions

S1 must prove expected-supported/actual-unloaded becomes Contact violation plus
role-specific ZMP N/A, while malformed/non-finite payload and loaded-support/raw-
wrench disagreement fail closed. It must also prove flight, transition recovery,
row/foot permutation, and unequal per-foot contact capacities.

S2 must carry separate Repair/Noisy applicability through one-action-K, Gain,
return evidence, formal transaction diagnostics, and atomic quality reporting
without changing exact-one update or scalar return.

S3 must preserve checkpoint-v5 layout while accepting only GAIN-v006/schema-v2
identity. S4 remains one bounded official sensor-authority sentinel. Stop on
silent row deletion, shared Repair/Noisy applicability, fabricated finite ZMP,
legacy v005 acceptance, actor-input leakage, or any PPO/Critic formula change.
