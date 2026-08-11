# FRS-METHOD-v022 Proposal: Low-DR Coupled Adaptation And Phase-Aware Replay

Status: confirmed and activated as FRS-METHOD-v022 / FRS-PPO-v009 / FRS-TRAIN-v021
Date: 2026-08-11
Affected design points: FRS-DP-02, FRS-DP-03, FRS-DP-09

## Problem

TRAIN-v020 freezes Actor during the first phase and delays DR growth until
joint optimization. The Critic therefore learns `V(s | pi_frozen)` under a
narrow policy/distribution and immediately becomes stale when Actor and DR are
released. Its outer replay score also conflates value calibration with
within-Scenario Repair variation and ignores current absolute DR compatibility.

## Confirmed Design

- Start Actor and Critic together on the first low-DR transaction.
- Increase Actor loss weight and `d_cap` together; enter joint optimization
  only when Actor weight and current-K DR coverage are full.
- Draw all four relative DR classes from transaction one, bounded by current
  `d_cap`.
- During joint initiation/ramp rank replay by
  `E_V=|mean U - V_old|`; during joint optimize rank by
  `E_A=mean|U-mean U|`.
- Draw current DR class before source, then replay only same-K Scenarios whose
  absolute strength lies in that class's current interval; empty compatible
  pools fall back to global discovery without changing class.
- Persist both score maps, phase/DR identity and sampler state atomically.

All Actor/Critic inputs, Gain, M4 target, grouped PPO, LRs, K values, full-6D
action and frozen GMT remain unchanged.

## Human Receipt

The user confirmed the complete Design Inspector and authorized one-shot local
implementation through the training-readiness boundary on 2026-08-11. No long
training, Git publication or remote operation is included.
