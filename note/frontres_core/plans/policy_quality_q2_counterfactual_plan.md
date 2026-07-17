# Policy Quality Q2 Counterfactual Oracle Plan

Status: `manifest-accepted-offline-report-complete`
Date: 2026-07-17

## Objective

Test whether HSL and policy repairs beat an explicit zero action across a fixed
multi-motion, multi-seed bank, and whether PPO model_701 produces a resolvable
improvement or regression relative to HSL model_200.

## Scope

- one immutable 8-motion x 2-seed manifest;
- zero, frozen HSL model_200, and policy model_701 on every item;
- fixed start frame, local_rp, DR scale 1.25, K=8, and evaluator revision;
- per-item Gain and route differences before any aggregation;
- per-item conservative zero noise floor `epsilon_i = abs(Gain_zero_i)`;
- one independent offline result validator/reporter before live execution.

## Non-Scope

- no reset, Gain, PPO, perturbation-mask, training, or old-evaluator changes;
- no checkpoint trajectory, Q4 distribution audit, or long training;
- no claim from an aggregate that hides a failed motion/seed item.

## Owners

- manifest identity: `frontres_policy_quality_manifest.py`;
- collection: dedicated `frontres_policy_quality_eval` route;
- canonical label: `frontres_gain.py::compute_segment_gain`;
- report owner: `frontres/frontres_policy_quality_q2_report.py`, an independent
  offline reader that does not enter the live evaluator control flow;
- report contract: `tests/frontres_policy_quality_q2_report_contract.py`.

## Item Classification

For each item `i`:

```text
epsilon_i     = abs(Gain_zero_i)
HSL-Zero      = Gain_hsl_i - Gain_zero_i
Policy-Zero   = Gain_policy_i - Gain_zero_i
Policy-HSL    = Gain_policy_i - Gain_hsl_i
```

- `HSL-Zero > epsilon_i`: HSL repair is resolved above paired-env noise.
- `Policy-Zero > epsilon_i`: policy repair is resolved above zero.
- `Policy-HSL > epsilon_i`: PPO improvement over HSL is resolved.
- `Policy-HSL < -epsilon_i`: PPO regression relative to HSL is resolved.
- otherwise: policy versus HSL is unresolved on that item.

Motion-level consistency requires both seeds to have the same resolved class.

## Acceptance And Stop Conditions

Technical PASS requires all 16 items to have matching manifest/state identity,
finite scalar Gain for all three routes, complete role identity, and explicit
per-item differences.

Oracle-valid PASS requires HSL and policy each beat zero beyond `epsilon_i` on
both seeds for at least 6 of 8 motions. This establishes a usable positive
control and prevents one or two motions from carrying the result.

The PPO-improvement hypothesis is supported only if `Policy-HSL > epsilon_i`
on both seeds for at least 5 of 8 motions and the per-item median is positive.
It is rejected as a stable improvement if fewer than 5 motions satisfy this;
resolved negative motions must be reported separately.

Stop immediately before aggregation if any identity mismatch, missing route,
non-finite/non-scalar Gain, evaluator revision mismatch, or role corruption
oracle failure occurs. Stop and return to method/Gain review if HSL fails to
beat zero on both seeds for 3 or more motions. Do not modify PPO from Q2 alone.

## Cost

The bank contains 16 items x 3 routes x 8 steps = 384 simulator steps, 16
canonical resets, and 48 route executions with the same 4-env quartet and two
checkpoint loads as Q1-F. Memory demand should remain Q1-F-sized; scoring work
is approximately 16x the single-item run after startup. No trustworthy wall
clock was recorded for Q1-F, so 5-15 minutes on the same server is a planning
estimate, not evidence.

## Pre-Live Gap

The existing evaluator is sufficient to collect every required raw fact:
per-item identity, role identity, route actions, canonical Gain components, and
execution. It does not emit the three route differences, conservative zero
noise floor, per-motion two-seed classification, or fail-fast Q2 summary.

The independent offline validator/reporter is now implemented. It rejects
identity/schema/role/Gain corruption, preserves all 16 item rows, and reports a
negative scientific outcome as a verdict instead of raising a technical error.
The live evaluator schema and control flow remain unchanged.

## Next

The motion bank, thresholds, and cost are accepted under the governed default:
technical validity is fail-closed, while scientific pass/fail remains visible
and does not crash collection. Run focused and aggregate offline verification,
then present the frozen manifest and one bounded Q2 live command for control.
