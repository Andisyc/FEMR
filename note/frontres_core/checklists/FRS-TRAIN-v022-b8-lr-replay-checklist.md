# FRS-TRAIN-v022 B8/LR/Replay Checklist

Plan: `../plans/FRS-TRAIN-v022-b8-lr-replay-global-formal-test-plan.md`

Status: CPU alternate-path Global Simplified Formal Test and the 57-target
offline contract suite PASS. The first official bounded attempt started Isaac
Sim but exited before a transaction while CUDA was reported in a bad state;
fresh official K8/B8/M4 simulator evidence remains pending.

- [x] DP02/DP03/DP09 semantic delta confirmed.
- [x] METHOD-v023 / PPO-v010 / TRAIN-v022 activated.
- [x] TEST-23A actual Actor-LR assertions executed inside TEST-23E.
- [x] TEST-23B B8/M4 assertions executed inside TEST-23E.
- [x] TEST-23C Replay consumption and capacity-bound assertions executed inside TEST-23E; expansion/replacement remains unproved.
- [x] TEST-23D simplified persistence roundtrip executed inside TEST-23E; official checkpoint-v17 readback remains pending.
- [x] TEST-23E Global Simplified Formal Test passed on the current aligned CPU alternate path.
- [x] Formal launcher contract runs before TEST-23E and enforces v022 B8/LR identity.
- [x] Full active offline contract suite passed: 57 targets, zero failures.
- [x] B2-to-B8 transaction, telemetry, checkpoint-v17 and K-stage handoff regressions repaired.
- [x] Held-out evaluation retains two-Segment x M4 identity without borrowing the B8 training Plan.
- [ ] Module manifest validated as MODULE-CORRECT.
- [ ] Formal manifest validated through R2.
- [ ] Exactly one successful B8/M4/K8 bounded live fact, after code sync and healthy CUDA startup.
- [ ] Long training separately authorized.

Evidence note: the pulled server `log.txt` is raw PASS evidence but predates the
current `linear-coupled-v1` correction. The aligned local rerun passed with
`transactions=32 K=8 B=8 M=4`. Neither result is IsaacLab official-path proof.
The pulled official log
`FRS_TRAIN_V022_K8_B8_M4_BOUNDED_OFFICIAL_20260811.log` proves preset/config
startup only; repeated `CUDA being in bad state` messages precede termination,
and no formal transaction or optimizer receipt appears.
