# TRAIN-v017 Output-Preserving Value Scale Module Test Execution

Date: 2026-08-08
Status: `18 passed / 0 partial / 0 blocked`; Formal Runtime Audit Phase A passed

## Scope

The accepted change addresses Critic numerical conditioning only. It preserves
FRS-GAIN-v007, raw `G_total`, raw `V(s)`, Actor loss/advantages, the 347D
state-value Critic, networks, fixed Actor/Critic LR, K/M/DR, GMT and simulator.
This is offline module/formal-route evidence, not simulator or policy-quality
evidence.

## Affected Cards

| Card | Executable evidence | Observed fact | Limitation |
| --- | --- | --- | --- |
| TEST-02 Training Config | `frontres_segment_stage3_entrypoint_pseudo_contract.py`, `frontres_v015_real_optimizer_counter_contract.py` | Official composition fixes normalization id, decay `0.9`, floor `1.0`; drifted/partial identities reject | No live config dispatch yet |
| TEST-15 Segment PPO | `frontres_v017_adaptive_value_scale_contract.py`, `frontres_v016_state_value_ppo_contract.py` | Hand-computed EMA moments and `raw_loss/scale^2`; scale is finite, permutation-invariant and at least one; raw target/value and Actor facts are identical | No convergence claim |
| TEST-16 Checkpointing | `frontres_v016_checkpoint_contract.py` | checkpoint-v12 round-trips exact moments/count and active read-only quality identity; v11/missing/non-finite/count-mismatched state rejects before restoration | No server filesystem evidence |
| TEST-18 Runtime Diagnostics | `frontres_v016_runtime_telemetry_contract.py`, `frontres_formal_runtime_audit_contract.py` | Owner-produced raw/scaled loss, scale, moments and count transition reach final telemetry and B01/B07/B08 checks | Offline route only |

## Construction Review

- The pure immutable normalizer owner lives in
  `frontres/frontres_value_normalization.py`; PPO previews it, the formal
  transaction commits it, and checkpointing persists it.
- The transaction rejects a normalizer count that differs from committed
  iteration before gradients or optimizer mutation.
- Failed/partial transactions leave optimizer and normalizer state unchanged.
- The change adds no wrapper/service/optimizer/network or cross-layer private
  access. Evaluation and sampling receive no feedback from the scale.

## Verification

- Python compilation, JSON parsing, shell syntax and `git diff --check`: pass.
- Focused TEST-02/15/16/18 plus observation/interface/quality regressions: pass.
- Stage-3 pseudo suite: `13/13` pass, `147` expected probe occurrences.
- Formal Runtime Audit Phase A: config -> PPO scale -> exact-one commit ->
  telemetry -> checkpoint-v12/read-only inspection is connected and fail closed.

## Next Gate

Run exactly one fresh K8/M2 bounded official transaction from HSL-v2. Require
normalizer count `0 -> 1`, finite scale `>=1`, invariant raw target/Actor facts,
one Critic update, zero Actor/std delta in critic-only, and checkpoint-v12
atomic readback. Do not start long training or resume v016/checkpoint-v11 until
that evidence passes.
