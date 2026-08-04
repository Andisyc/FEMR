# FRS-v017 DP08 HSL Target Final Gate Review

Date: 2026-08-04  
Mode: `code-review-expert::final_gate_review`  
Verdict: `APPROVE`  
Repository discipline: active `FRS-ENG-v001`  
Method-completeness evidence consumed: DP08 `integrated-offline`, E-FI-117

## Accepted Boundary

The Stage-1 HSL target is the exact detached finite current anti-DR `[B,6]`.
Translation is `-anchor_dr_delta_pos` in world coordinates. No axis-specific
mask, scale, clip or clamp may change it. HSL remains 158D, actor-only,
proposal-only, HSL-v2-only and isolated from Stage-3 supervised learning.

## Reviewed Change

- target producer: `get_supervision_target_task_space()`;
- independent validator and HSL consumer: `frontres_warmup.py`;
- semantic S1/S2 regressions: the existing HSL contracts;
- current plan, checklist, task canvas, Test Atlas, test inventory and evidence
  projection.

The patch removes the same retired asymmetric-Z policy from the producer and
validator. It adds no owner, wrapper, Protocol, state, persistence field or
caller dependency. The test Seam executes the real target-owner body with a
two-row hand-calculated fixture; the expected answer does not call an internal
helper.

## Findings

- P0: none.
- P1: none.
- P2: none in the bounded change.
- P3: none.

## Discipline Gates

- Ownership / CCP: the existing target producer remains the sole semantic
  owner; the validator is a fail-closed check at the existing Pinch Point.
- Characterization / effect boundary: the removed `dz clamp(max=0)` and its old
  fixture are the characterized defect; effects end at the HSL target/loss.
- Dependency direction / CRP / SDP: command fields flow into a detached tensor;
  no simulator object enters Stage-3, Gain, PPO or checkpoint logic.
- Pattern admission: no new pattern or abstraction was admitted.
- State / reliability: no mutable lifecycle or persistence boundary changed;
  malformed/non-finite target rejection remains intact.
- Removal delta: one duplicated hard-axis policy was removed; no compatibility
  fallback remains on the active target edge.

## Evidence Consumed

- changed Python files compile;
- HSL S1 and S2 contracts pass with both anti-DR `dz` signs;
- direct task-space proposal, distribution, observation authority,
  HSL-v2/checkpoint-v9, legacy-label isolation and Stage-1 entrypoint contracts
  pass;
- the complete deterministic contract suite exits zero;
- Module Test Atlas and Code Quality Atlas validators pass.

## Limitation

This review does not claim simulator behavior, live HSL convergence, Stage-3
policy quality or Phase B closure. Those boundaries remain unchanged and
unauthorized.
