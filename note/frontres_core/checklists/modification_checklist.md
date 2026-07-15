# FrontRES Current Change Checklist

Updated: 2026-07-15
Scope: `FRS-DP-09` Stage 3 Actor/Critic warmup and `FRS-DP-05` Frozen GMT evidence.

## Step Status

| Step | Scope | Status | Evidence / blocker |
| --- | --- | --- | --- |
| 1 | Segment warmup phase owner | completed | `frontres_segment_warmup_contract.py` |
| 2 | Formal Stage 3 integration | completed | entrypoint + single-update contracts |
| 3 | Persistence and Frozen GMT | completed offline | checkpoint config guard + Frozen GMT contract |
| 4 | Cross-file acceptance | completed offline | aggregate suite 44/44 after pre-fall Style mask regression; live rerun2 remains separate |

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
- [x] Offline gate passed; Phase B tiny formal live run is now the remaining evidence boundary.

## Phase B Formal Runtime Audit

| Boundary | S/T | Status | Evidence |
| --- | --- | --- | --- |
| Formal route | S4 `T-connect` | stale-rerun-required | `AUDIT-ROUTE-01` |
| Perturbation config/application | S4 `T-config/T-value` | stale-rerun-required | `AUDIT-PERTURB-01`, `AUDIT-PERTURB-02` |
| Segment data/sampler transaction | S4 `T-source/T-state` | stale-rerun-required | `AUDIT-SEGDATA-01`, `AUDIT-SAMPLER-01` |
| K plan/executed horizon | S4 `T-shape/T-forward` | stale-rerun-required | `AUDIT-KPLAN-01`, `AUDIT-KROLLOUT-01` |
| Observation/full-6D repair/application | S4 `T-shape/T-source/T-value` | stale-rerun-required | `AUDIT-OBS-01`, `AUDIT-ACTION-01`, `AUDIT-APPLY-01` |
| Frozen GMT | S4 `T-grad/T-state` | stale-rerun-required | `AUDIT-GMT-01` |
| Paired roles/execution evidence | S4 `T-role/T-source` | stale-rerun-required | `AUDIT-PAIR-01`, `AUDIT-PAIR-EVIDENCE-01` |
| Gain/returns | S4 `T-value/T-forward` | stale-rerun-required | `AUDIT-GAIN-01`, `AUDIT-RETURN-01`; second live attempt found terminal `done_any` erased pre-fall Style; owner fixed offline; rerun2 required |
| HSL Stage2-to-Stage3 load | S4 `T-persist/T-source` | stale-rerun-required | `AUDIT-HSL-LOAD-01` |
| Warmup/PPO/trust/diagnostics | S4 `T-grad/T-update-order/T-state` | stale-rerun-required | `AUDIT-WARMUP-01`, `AUDIT-PPO-01`, `AUDIT-DIAG-01` |
| Checkpoint payload identity | S4 `T-persist` | stale-rerun-required | `AUDIT-PERSIST-01` |
