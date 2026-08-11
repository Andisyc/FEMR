# FRS-TRAIN-v022 B8/LR/Replay Checklist

Plan: `../plans/FRS-TRAIN-v022-b8-lr-replay-global-formal-test-plan.md`

Status: CPU alternate-path Global Simplified Formal Test PASS; official bounded
K8/B8/M4 simulator evidence pending.

- [x] DP02/DP03/DP09 semantic delta confirmed.
- [x] METHOD-v023 / PPO-v010 / TRAIN-v022 activated.
- [x] TEST-23A actual Actor-LR assertions executed inside TEST-23E.
- [x] TEST-23B B8/M4 assertions executed inside TEST-23E.
- [x] TEST-23C Replay consumption and capacity-bound assertions executed inside TEST-23E; expansion/replacement remains unproved.
- [x] TEST-23D simplified persistence roundtrip executed inside TEST-23E; official checkpoint-v17 readback remains pending.
- [x] TEST-23E Global Simplified Formal Test passed on the current aligned CPU alternate path.
- [ ] Module manifest validated as MODULE-CORRECT.
- [ ] Formal manifest validated through R2.
- [ ] Exactly one B8/M4/K8 bounded live fact, if still required.
- [ ] Long training separately authorized.

Evidence note: the pulled server `log.txt` is raw PASS evidence but predates the
current `linear-coupled-v1` correction. The aligned local rerun passed with
`transactions=32 K=8 B=8 M=4`. Neither result is IsaacLab official-path proof.
