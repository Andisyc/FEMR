---
contract_id: FRS-GAIN-v005
status: superseded
effective_date: 2026-07-23
updated_date: 2026-07-27
supersedes: FRS-GAIN-v004
scope: scalar paired root-invariant Intent improvement minus full-6D repair cost, with unsaturated physical-unit Contact, phase-ZMP, and survival residuals retained as separate actor constraints
---

Superseded by `FRS-GAIN-v006`. Runtime evidence showed that this version
incorrectly invalidated a transaction when an expected-support phase contained
valid sensor evidence of zero actual load. That state is a Contact violation,
not a corrupt evidence payload.

# Vector Physics Constraints And Scalar Intent Objective

## Design Delta

FRS-GAIN-v004 normalized and saturated Physics violations, reduced them with
`max`, and joined Physics and Intent inside one scalar role utility. E-FI-72
shows that distinct severe ZMP states can therefore produce identical return.

FRS-GAIN-v005 removes Physics from scalar Gain. It produces one scalar objective
for the Critic and three independent actor-constraint families:

```text
scalar: paired Intent improvement - repair cost
vector: Contact residual, phase-ZMP residual, survival residual
```

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Paired Evidence Authority |
| `FRS-DP-07` | Repair Gain | `Q-01` | Scalar Objective And Physics Constraints |

## Paired Evidence Authority

Noisy and every Repair attempt share scenario, `x_t`, artifact, q29 Intent,
Clean continuation, expected support, K, valid-step clock, frozen GMT, and hash.
Noisy supplies paired Intent/no-op evidence only. Physics residuals are absolute
properties of each Repair execution relative to expected support and declared
phase-specific feasibility; they are never `Repair - Noisy` residuals.

Expected support comes from the sealed Clean continuation for GMT/Physics use
only. Actual contact comes from ContactSensor. ZMP uses the expected Contact
phase and may not use a Repair-created illegal support polygon. Flight ZMP is
semantic N/A.

The formal ZMP evidence identity is `contact-wrench-zmp-v1`. Separate filtered
left/right foot-to-ground ContactSensors retain each raw contact point, normal
and normal-force magnitude. Their vertical ground resultant defines the world-
frame ZMP/CoP. Root/capture-point proxies, foot-centre net-force surrogates and
Repair-created support polygons are forbidden as formal Physics evidence.

The sealed Clean continuation also derives
`clean-foot-pose-oriented-box-v1`: one `[center_x, center_y, cos(yaw),
sin(yaw), half_x, half_y]` support/recovery envelope per K step, together with
the existing expected left/right Contact phase. It is immutable, hashed,
Physics-only evidence. It cannot enter actor observation, Intent target or GMT
command. A supported phase without a finite contact resultant invalidates the
transaction; flight remains explicit N/A.

## Scalar Intent Objective

For Repair row `i=(g,s,m)`:

```text
DeltaI_i = IntentQuality(Repair_i, deployment_q29)
         - IntentQuality(Noisy_s, deployment_q29)

C_repair_i = existing full-6D magnitude + temporal-change cost

y_I_i = DeltaI_i - C_repair_i
```

IntentQuality retains the root-invariant q29/qvel/qacc definition and fixed
versioned scales from v003/v004. Clean global/root similarity and direct
Repair-vs-Noisy trajectory similarity remain forbidden. `C_repair` is the only
regularizer in the scalar objective. Contact, ZMP, survival, admissibility,
constraint status, or projection outcome may not enter `y_I`.

The Critic target and return are exactly:

```text
return_K = y_I
V_I(o_critic) ~= E[y_I | o_critic, active global K stage]
A_I = y_I - V_I_old
```

## Physical-Unit Constraint Residuals

For valid Repair evidence with simulator step time `dt`, define raw residuals
before any optimizer normalization.

### Contact

Expected and actual left/right contact are aligned with the fixed permitted
early/late switch tolerance. Let `m_contact[k,f]` be one iff foot `f` has an
unmatched extra/missed contact, illegal switch, or declared dragging event at
valid step `k`. The union indicator prevents double-counting one foot-step.

```text
E_contact = dt * sum_valid_k,f m_contact[k,f]       # foot-seconds
c_contact = E_contact - B_contact                  # foot-seconds
```

`B_contact` is the fixed versioned post-alignment allowance. It is not inferred
from the current batch or Noisy rollout.

### Phase-Conditioned ZMP

For every support-applicable step, `margin[k]` is signed distance in metres to
the expected phase support/recovery envelope; positive is feasible. Planned
transition excursion is encoded in that immutable envelope. Flight steps are
N/A and excluded.

```text
zmp_depth[k] = relu(-margin[k])
E_zmp = dt * sum_applicable_k zmp_depth[k]          # metre-seconds
c_zmp = E_zmp - B_zmp                              # metre-seconds
```

No temporal/channel maximum is permitted. If K contains no applicable support
step, ZMP is N/A and the ZMP constraint is inactive with explicit provenance;
missing required margin evidence invalidates the transaction.

### Survival

```text
T_required = valid_planned_steps * dt
T_survived = executed_valid_steps * dt
c_survival = T_required - T_survived                # seconds
```

Early terminal events remain explicit diagnostics. Full K survival gives zero;
there is no clipping to `[0,1]` and survival cannot cancel Contact or ZMP.

## Fixed Scale Normalization Without Scalarization

For numerical conditioning only:

```text
z_j = c_j / S_j,  j in {contact, zmp, survival}
```

`S_j` has the same physical unit as `c_j`, is positive and versioned in the
constraint schema, and is never batch-derived. `z_j` is not clipped. The three
values remain separate fields and separate constraint surrogates; scales do not
become reward weights.

For constraint activation, positive violation cannot be offset by safe rows:

```text
q_ij = relu(z_ij)
C_j = grouped_equal_mass_mean(q_ij)
```

All valid Repair rows remain present. This positive-part statistic is not a row
mask and does not saturate severity.

## Constraint Policy-Gradient Signals

Within each sealed scenario, use the detached M-attempt mean as a variance-only
baseline:

```text
A^c_ij = q_ij - stop_gradient(mean_m q_gsmj)
```

The Noisy residual is not this baseline. FRS-PPO-v004 constructs one grouped
score-function constraint gradient per family from `A^c_ij`. If all M attempts
have identical evidence, the corresponding empirical gradient is zero and must
be reported; it may not be fabricated by noise, zero-fill, or a scalar fallback.

## Required Identity

Every row and checkpoint binds:

```text
gain_contract_id = FRS-GAIN-v005
scalar_target_id = paired-intent-minus-repair-v1
constraint_schema_id = contact-phase_zmp-survival-physical-v1
dt, B_contact, B_zmp, S_contact, S_zmp, S_survival
phase/support schema and N/A-mask identity
zmp_estimator_id = contact-wrench-zmp-v1
support_envelope_id = clean-foot-pose-oriented-box-v1
actual_contact_id = contact-sensor-net-normal-force-threshold-v1
expected_phase_id = clean-foot-height-phase-v1
```

## Forbidden Behavior

- v004 deficit, admissible/unsafe utility, or scalar `physics_gain` as return;
- `[0,1]` severity clipping, `max/amax`, or silent valid-step masking;
- relative Noisy Physics as Repair feasibility;
- weighted Contact/ZMP/survival sum in objective or Critic target;
- adverse-row deletion or priority-weighted actor loss;
- Clean/global actor target or any constraint evidence in actor observation.

## Acceptance And Stop Conditions

P2 S1 must prove physical units, ordered time aggregation, flight N/A, no
saturation collision, row permutation, missing/nonfinite fail-closed, scalar
target purity, and distinct per-family constraint outputs. S2 must prove the
formal transaction carries these fields unchanged to FRS-PPO-v004 and rejects
all v004/v003 consumers.

Stop if raw evidence cannot remain ordered until the owner, physical budgets or
scales are batch-derived, one safe row cancels another row's violation, any
constraint contaminates return/value, or Noisy becomes the safety threshold.
