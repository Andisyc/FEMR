# TRAIN-v020 Outer Scenario Replay Evidence Ledger

Date: 2026-08-10  
Checkout base: `ce89dab` plus the scoped working-tree changes listed by Git  
Authority: `FRS-METHOD-v021` / `FRS-TRAIN-v020` / `FRS-GAIN-v008` / `FRS-PPO-v008`

## Scope

This ledger covers offline engineering closure for the new cross-Transaction
outer prioritized sealed-Scenario replay. It does not claim simulator execution
or policy quality.

## Verified Evidence

| Boundary | Command / artifact | Result |
| --- | --- | --- |
| Scenario identity and isolated RNG | `frontres_v020_outer_scenario_replay_contract.py` | PASS |
| Stable physical noisy hash across visit IDs | `frontres_local_scenario_kernel_contract.py` | PASS |
| Exact-one PPO plus exact-one replay commit | `frontres_v015_transaction_route_contract.py` | PASS |
| Final telemetry and outer replay evidence | `frontres_v016_runtime_telemetry_contract.py` | PASS |
| checkpoint-v15 strict replay round-trip | `frontres_v016_checkpoint_contract.py` | PASS |
| Historical checkpoint-v14 read-only Evaluation | `frontres_v018_policy_quality_eval_contract.py`; `frontres_v018_policy_quality_compatibility_contract.py` | PASS |
| Formal runtime audit projection | `frontres_formal_runtime_audit_contract.py` | PASS (offline fixtures) |
| Active Contract registry | `frontres_design_contract_sentinel.py` | PASS |
| Full FrontRES contract suite | `frontres_segment_all_contract_suite.py` | 57 passed / 0 failed |
| Atlas static validation | `npm run check` in `note/architecture/auxiliary/atlas_app` | PASS; 22 Module Test Cards and all Atlas contracts valid |

## Proven Facts

- The replay item contains motion/frame/x_t, perturbation seed and artifact
  identity, K, future Intent and planned support, rather than a bare Segment ID.
- Replaying a seed does not advance the external CPU Actor/global RNG state.
- Learning value is the current-policy exact-M mean absolute utility error and
  remains scheduler-only; it does not alter Gain, target, advantage or PPO mass.
- Failed, mismatched and duplicate commits do not mutate replay records or RNG.
- A matching committed exact-one receipt produces exactly one replay state delta.
- checkpoint-v15 saves and restores records, per-K scores, visit/staleness,
  owner RNG and last receipt before training resume.
- checkpoint-v14 remains accepted only by the read-only Evaluation boundary.

## Unverified Live Facts

- The official IsaacLab/MOSAIC composition root constructs the owner on the
  server and materializes the same ScenarioKey under the real simulator.
- One bounded official transaction reports `optimizer_step_delta=1`,
  `outer_replay_state_delta=1`, two key digests, finite learning values and a
  successful atomic checkpoint-v15 readback.
- GPU memory, wall-clock cost and simulator lifecycle remain within the prior
  K8/M4 bounded envelope.

These facts require one server bounded sentinel after code synchronization.
Long training remains blocked until that sentinel log is reviewed.
