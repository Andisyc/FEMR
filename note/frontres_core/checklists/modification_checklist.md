# FrontRES Current Change Checklist

Updated: 2026-07-17
Scope: `FRS-DP-09` Stage 3 Actor/Critic warmup, `FRS-DP-05` Frozen GMT evidence,
and `FRS-DP-07` normalized survival Gain alignment.

## Step Status

| Step | Scope | Status | Evidence / blocker |
| --- | --- | --- | --- |
| 1 | Segment warmup phase owner | completed | `frontres_segment_warmup_contract.py` |
| 2 | Formal Stage 3 integration | completed | entrypoint + single-update contracts |
| 3 | Persistence and Frozen GMT | completed offline | checkpoint config guard + Frozen GMT contract |
| 4 | Cross-file acceptance | completed offline | aggregate suite 44/44 after pre-fall Style and zero-valid audit regressions; live rerun3 remains separate |

## DP-09 Acceptance Matrix

| Boundary | Owner | Required S/T | Status | Evidence |
| --- | --- | --- | --- | --- |
| Phase schedule | `frontres_segment_warmup.py` | S1 `T-value`, `T-meta` | implemented | warmup contract |
| Weighted Segment objective | `frontres_segment_ppo.py` | S1 `T-grad`, `T-value` | implemented | warmup contract |
| Official Stage 3 preset | `scripts/rsl_rl/train.py` | S0/S2 `T-connect`, `T-oracle` | confirmed offline | Stage 3 entrypoint contract |
| Live loop propagation | live training/update loop | S2 `T-connect`, `T-state` | confirmed offline | Stage 3 pseudo suite |
| Update gradient boundary | `frontres_segment_live_probe.py` | S1/S2 `T-grad`, `T-update-order` | confirmed offline | single-update contract |
| Diagnostics | update loop/live formatter | S1/S2 `T-oracle`, `T-connect` | confirmed offline | pseudo suite logs phase/weight |
| Checkpoint/resume phase | `frontres_checkpointing.py` | S3 `T-persist`, `T-state` | confirmed offline | iteration persistence + warmup config mismatch guard |

## Frozen GMT Test Profile

Module type: algorithm/optimizer plus frozen downstream policy dependency.

Required proof:
- [x] S1 `T-grad`: GMT parameters have `requires_grad=False`.
- [x] S2 `T-connect`: formal FrontRES optimizer excludes every GMT parameter.
- [x] S2 `T-grad/T-diff`: one Segment-style update changes permitted actor/critic
  parameters but leaves all GMT parameters bitwise unchanged.

No S4 test is required merely to prove optimizer isolation. Phase B may later
confirm that frozen GMT executes the repaired reference, but that is a separate
runtime reachability fact.

## Training Gate

- [x] Critic-only changes critic parameters and not actor/std parameters.
- [x] Actor warmup weight is monotonic and bounded in `[0, 1]`.
- [x] Joint phase uses actor weight `1`.
- [x] Formal Stage 3 entry reaches the same phase reported by diagnostics.
- [x] Resume selects phase from persisted iteration and rejects warmup-config drift.
- [x] Frozen GMT optimizer isolation is contract-confirmed.
- [x] Aggregate S0-S3 suite passes: 44/44 on 2026-07-15.
- [x] Architecture and task evidence are current for this change.
- [x] Offline gate passed; Phase B Gain consumer alignment remains the active
      formal boundary. Long-run actor quality is a post-training observation,
      not a pre-training gate.

## Phase B Formal Runtime Audit

| Boundary | S/T | Status | Evidence |
| --- | --- | --- | --- |
| Formal route | S4 `T-connect` | runtime-observed | `AUDIT-ROUTE-01`, `E37`: official train route, alternate_modes=0 |
| Perturbation config/application | S2/S4 `T-config/T-value/T-source` | runtime-observed | `AUDIT-PERTURB-01`, `AUDIT-PERTURB-02`, E41: rp, max K=64, local_rp=8, finite strength distribution |
| Segment data/sampler transaction | S4 `T-source/T-state` | runtime-observed | `AUDIT-SEGDATA-01`, `AUDIT-SAMPLER-01`, `E37`: 8 source rows, priority update observed |
| K plan/executed horizon | S4 `T-shape/T-forward` | runtime-observed | `AUDIT-KPLAN-01`, `AUDIT-KROLLOUT-01`, `E37`: all quartet rows survive K=8; policy valid=8 |
| Quartet reset lifecycle | S4 `T-role/T-state/T-timeout` | live-confirmed-aligned | episode=0, root max<=1.91e-6, joint max=0 for all roles; downstream step-0 termination remains |
| Quartet reset repair | S1/S2/S4 `T-role/T-state/T-forward/T-timeout` | live-confirmed | 32 role rows reached adapter and robot/lifecycle state aligned; perturbation remains policy-owned |
| Termination term localization | S2/S4 `T-role/T-source/T-value` | runtime-observed | `AUDIT-RESET-LIFECYCLE-01`, `E37`: all active terms remain zero for every role through K=8 |
| Anchor-position value localization | S2/S4 `T-source/T-value/T-frame/T-role` | runtime-observed | `AUDIT-ANCHOR-Z-01`, `E37`: first raw/clean/robot z align, max abs error=0.020011m, all role masks zero |
| Sampled-frame command-cache initialization | S1/S2/S4 `T-frame/T-role/T-state/T-forward` | integrated-live | `E36/E37`: one no-advance refresh offline; first-step cache/termination and K=8 survival live-confirmed |
| Observation/full-6D repair/application | S4 `T-shape/T-source/T-value` | runtime-observed | `AUDIT-OBS-01`, `AUDIT-ACTION-01`, `AUDIT-APPLY-01`, E39: finite full-6D action and delta norm |
| Frozen GMT | S4 `T-grad/T-state` | runtime-observed | `AUDIT-GMT-01`, `E37`: gmt_training=False, trainable=0, in_optimizer=0 |
| Paired roles/execution evidence | S4 `T-role/T-source` | runtime-observed | `AUDIT-PAIR-01`, `AUDIT-PAIR-EVIDENCE-01`, E39: policy=8/baseline=24, valid=7 |
| Gain/returns | S4 `T-value/T-forward` | identity propagation offline-confirmed; formal numeric consumer comparison open; actor quality deferred to post-training | `AUDIT-GAIN-01`, `AUDIT-RETURN-01`, E62/E63/E64: shared capture route, transaction/batch identity, local K-normalized survival trace, canonical gain_total forwarding, returns, and advantages |
| Formal v002 audit instrumentation | S1/S2/S4 `T-connect/T-unit/T-K/T-step-sum/T-live` | runtime path reached; Gain consumer alignment open | `frontres_formal_runtime_audit_contract.py`, Runtime Atlas GAIN/RETURN cards, E58 |
| Survival Gain unit alignment | S1/S2 `T-unit/T-K` | implementation/offline complete; formal consumer comparison open | `FRS-GAIN-v002`, `plans/survival_gain_unit_alignment_20260716.md`, E55 |
| Survival units/K aggregation | S1/S2 `T-value/T-forward` | contract-confirmed | `frontres_gain_components_contract.py:test_survival_unit_and_k_aggregation_probe`, E55: K=1/4/8 and per-step sum match final K=4 Gain |
| HSL Stage2-to-Stage3 load | S4 `T-persist/T-source` | runtime-observed | `AUDIT-HSL-LOAD-01`, `E37`: model_warmup actor and EmpiricalNormalization loaded |
| Warmup/PPO/trust/diagnostics | S4 `T-grad/T-update-order/T-state` | runtime-observed | `AUDIT-WARMUP-01`, `AUDIT-PPO-01`, `AUDIT-DIAG-01`, `E39`: actor_warmup weight=0.5, valid=7, post KL=0.005442, trust accepted |
| Checkpoint payload identity | S4 `T-persist` | runtime-observed | `AUDIT-PERSIST-01`, `E41`: model_2.pt includes model/optimizer/normalizer/sampler/Gain/warmup |
