# TRAIN-v021 Low-DR Coupled Replay Evidence Ledger

Date: 2026-08-11  
Checkout base: `ca7c3a737c6f46c38885b1e3ae313392f6c2266d` plus the authorized scoped working-tree changes  
Authority: `FRS-METHOD-v022` / `FRS-GAIN-v008` / `FRS-PPO-v009` / `FRS-TRAIN-v021`

## Scope

This ledger covers offline engineering closure for low-DR joint Actor/Critic
adaptation and phase-aware, current-DR-compatible outer Scenario replay. It
does not claim simulator execution, calibration quality or policy quality.

## Verified Evidence

| Boundary | Command / artifact | Result |
| --- | --- | --- |
| Coupled phase, Actor weight and DR schedule | `frontres_v021_coupled_replay_contract.py`; `frontres_segment_warmup_contract.py` | PASS |
| Phase-aware `E_V/E_A` and DR-compatible replay | `frontres_v021_coupled_replay_contract.py`; `frontres_v020_outer_scenario_replay_contract.py` | PASS |
| First-transaction joint gradients and exact-one commit | `frontres_v015_transaction_route_contract.py` | PASS |
| checkpoint-v16 strict replay-v2 round-trip | `frontres_v016_checkpoint_contract.py` | PASS |
| Official offline Stage-3 consumer route | `frontres_segment_all_contract_suite.py` | 58 passed / 0 failed |
| Module alignment | `frontres_train_v021_module_alignment.json` | ADMITTED-OFFLINE |
| Formal runtime projection | `frontres_train_v021_formal_phase_b.json` | PHASE_B_READY |
| Construction and final review | `FRS-TRAIN-v021-coupled-replay-*-review.json` | PASS; no open P0/P1 |

## Proven Facts

- `low_dr_joint_init`, `coupled_ramp` and `joint` all update Actor and Critic;
  the first Actor weight is positive and each K transition lowers DR without a
  Critic-only phase.
- Warmup phases schedule replay with Scenario mean calibration error `E_V`;
  joint optimization schedules with within-Scenario Repair spread `E_A`.
- Replay eligibility is filtered by current K and current absolute DR-class
  interval before source selection; empty replay/review falls back to the
  global pool in the same class.
- Actor 158D, Critic 449D, M4, full-6D Repair, Gain-v008, per-attempt symlog,
  split LR, separate clipping, exact-one Adam and frozen GMT remain unchanged.
- checkpoint-v16 persists replay-v2 dual scores and rejects the v15 training
  identity before mutation.

## Unverified Live Facts

- The official IsaacLab/MOSAIC composition root must complete one real K8/M4
  bounded transaction with nonzero Actor and Critic deltas.
- The same transaction must report the selected replay score kind, current DR
  compatibility, exact-one optimizer/replay transitions and atomic
  checkpoint-v16 readback.
- GPU memory, simulator lifecycle and wall-clock cost must remain finite and
  within the bounded sentinel envelope.

These facts require one server bounded sentinel after user-controlled code
synchronization. Long training remains blocked until that log is reviewed.
