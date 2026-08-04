# FrontRES Module Test Execution

Date: 2026-08-03

Scope: execute the 18 active Module Test Atlas cards at S0/S1 and
module-owned S3. This asks whether each module implements its accepted
semantics. It does not prove official simulator connectivity, training
stability, policy quality, or Demo quality.

Result after the human-confirmed TRAIN-v013 Module Test Closure:
**18 passed / 0 partial / 0 blocked**. This execution did not run Formal
Runtime Audit Phase A or B. E-FI-109's earlier Phase A result remains separate
evidence and is not inferred from these module results.

## Card Results

| Card | Result | Independent observed fact |
| --- | --- | --- |
| TEST-01 Launch Entry | passed | A call-recorder proves fresh/resume/evaluation mode exclusivity, ordered layout/load/dispatch, one dispatch, zero evaluation updates, and failure-stop. |
| TEST-02 Training Config | passed | K8/M2, K16/M3, K32/M4, `num_envs=4*M`, explicit per-K DR stage, committed-only progress, K-transition DR restart and no-hidden-default rejection match boundary cases. |
| TEST-03 OnPolicyRunner | passed | The observable startup lifecycle wraps the real layout/load/dispatch pinch points; duplicate/failed/partial paths cannot continue. |
| TEST-04 Motion Command | passed | Current/H q29 rows come from one sealed deployment reference; permutation, Clean exclusion, cursor bounds, and clock hold pass. |
| TEST-05 Observation Layout | passed | `870+58=928`, FEMR 158D/GMT 770D authority, permutation, malformed tail, and zero FrontRES width rejection pass. |
| TEST-06 Perturbation Data | passed | Sealed single-`local_rp` scenarios use deterministic Easy/Medium/Hard/Broken support at 20/30/40/10 from current `d_cap`; attempts/reset cannot resample and Gain/PPO/evaluation facts cannot affect the draw. |
| TEST-07 Segment Cache | passed | Cache owns immutable x_t identity, Clean continuation, expected support/envelope and hash; exact slice, source isolation, malformed evidence and roundtrip pass. |
| TEST-08 Segment Sampler | passed | Two distinct Segment identities and exact-M rows pass; early preference is soft, later rows remain reachable, identity weighting and RNG resume are stable. |
| TEST-09 Trial Plan & Reset | passed | One canonical snapshot restores Clean, Noisy and all Repair starts, including physical/command/perturber and Python/NumPy/Torch RNG state. |
| TEST-10 FrontRES Policy | passed | A fixed-weight 158D-to-6D forward matches hand calculation; suffix/row isolation and 157/159D/NaN/Inf rejection pass; HSL mutates Actor only. |
| TEST-11 Task-Space Correction | passed | Translation is unscaled, world-left rotation is represented in the host right buffer, only current rows change, and non-finite actions reject before mutation. |
| TEST-12 Stage 3 Live Loop | passed | Two-Segment x exact-M collection seals before one grouped commit; partial/mixed transaction and later FEMR actions reject. |
| TEST-13 Repair Gain | passed | Clean anchor, Noisy-to-Repair ordering, recovery pressure, beta cost, permutation, missing evidence, and baseline reuse match independent arithmetic. |
| TEST-14 Segment Storage | passed | Each Repair contributes one immutable policy row; baseline remains evidence-only and mixed identity rejects. |
| TEST-15 Segment PPO | passed | Scalar clipped PPO, equal hierarchical mass, sign/mask/permutation, retired projection isolation, and exact-one optimizer invocation pass. |
| TEST-16 Checkpointing | passed | Checkpoint-v8 atomically roundtrips full DRStageSpec/cursor/RNG/optimizer/normalizer/receipt state; v7, `g_K`, partial, mixed and tampered payloads reject before mutation. |
| TEST-17 Evaluation | passed | Artificial trajectories close Contact/load/flight, phase-ZMP applicability/violation/recovery, survival, lean, unplanned contact, atomicity, and zero training writes. |
| TEST-18 Runtime Diagnostics | passed; DP07 rerun E-FI-116 | Owner-produced normalized Intent/Physics channels, `I_N/I_R/P_N/P_R`, support-foot drift, `G_I/G_P/lambda_RA`, weighted Physics Gain, full-6D cost, cost-free score, beta/penalty/`G_total`, identities, masks, group mass and exact-one counts are copied without recomputation, zero-fill, row drift, or feedback. Phase-ZMP N/A remains explicit. |

## Closure Repairs

- Added one observable `FrontRESStartupLifecycle` at the existing composition
  seam; the real authority resolver, HSL/resume load, and dispatch operations
  execute inside it.
- Made Cache own the immutable Clean K-step artifact and reject missing, short,
  source-drifted, hash-drifted, or empty replay-state identity.
- Added identity-bearing two-Segment selection with a strictly positive soft
  early-frame preference and persisted RNG continuation.
- Replaced historical task-correction scaling/clamping with current-only
  full-6D world-frame semantics without changing the MOSAIC command owner.
- Added v017 read-only evaluation and telemetry projections for Contact,
  phase-ZMP, survival, lean, unplanned contact, Gain, group mass, and update
  identity.
- Completed the confirmed artificial cases for perturbation, canonical reset,
  policy input, and diagnostics.

## Executed Evidence

- Every one of the 18 human-confirmed cards was rerun through its focused
  deterministic contract and independent fake/hand-calculated oracle.
- Ordinary, boundary, invalid, row-permutation/metamorphic and stateful cases
  passed where required by the confirmed card. Checkpoint-v8 additionally
  passed its module-owned S3 roundtrip, tamper and pre-mutation rejection cases.
- TEST-10 first failed because its executable assertion still expected the
  superseded text `FRS-TRAIN-v012`. Production correctly rejected nonzero
  Stage-3 HSL supervision under `FRS-TRAIN-v013`; the test translation was
  corrected without changing the card or weakening the rejection, then the
  complete TEST-10 fixture passed.
- The active 49-target aggregate was not rerun or used as evidence for this
  closure. Its prior E-FI-109 result remains secondary regression evidence.
- `frontres_segment_live_sampler_contract.py` remains historical: it asserts
  FRS-GAIN-v002 adaptive search roles and composite `planar+global_z`, so it is
  explicitly excluded from the active v017 aggregate rather than weakening
  the single-`local_rp` production guard.
- The corrected HSL contract and the active TRAIN-v013 owner set pass
  `python -m py_compile`; Test Atlas structure, code-quality Atlas generation,
  code-quality source-link validation and targeted `git diff --check` pass.

## Engineering Discipline Gate

Verdict: **APPROVE** for module readiness, with P0=0 and P1=0.

The earlier TRAIN-v013 engineering closure had already closed two in-scope P1
issues: the production layout resolver initially sat outside the observable
lifecycle pinch point, and Cache deserialization did not explicitly reject an
empty x_t identity. This fresh module review found no new P0/P1. No new runner,
service, evaluator, semantic owner, stable open dictionary, MOSAIC host change,
or method parameter was introduced by this closure.

Three existing P2 maintainability risks remain explicit and are not module
correctness failures: generic checkpoint/save code still co-locates active and
historical branches before v8 strips legacy fields; trusted cache artifacts
still use `torch.load(..., weights_only=False)`; and active v017 behavior still
retains substantial v015 symbol/file naming. They do not weaken any confirmed
card and were not silently reported as fixed.

No simulator, training, live run, policy-quality run, or deployment composition
was executed. The module prerequisite for a human Formal Runtime Audit Phase A
review is satisfied; that review must independently inspect official
connectivity and must not reuse this module result as proof of it.
