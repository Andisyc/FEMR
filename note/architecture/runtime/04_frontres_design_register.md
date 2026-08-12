# FrontRES Design Inspector Register

Status: aligned to FRS-METHOD-v024 / FRS-GAIN-v008 / FRS-PPO-v011 /
FRS-TRAIN-v023 / FRS-EVAL-v005 on 2026-08-12.

Interactive page: `../02_frontres_design_inspector.html`

This file is the concise reading register for Atlas 04. It does not own method
semantics; the active Contract registry does. The complete card text and
highlight mapping live in `04_frontres_design_inspector.data.json`.

## Current Transaction

```text
HSL-v2 Actor initialization
-> resolve K/M, low-DR coupled phase and Actor LR
-> select B8 sealed ScenarioKeys from the K-local active Replay pool/global admission
-> freeze pi_old and collect exact M4 current Repair attempts per Scenario
-> form one raw G_total and one symlog utility per attempt
-> Replay preview: append/reset compatible utility window, estimate robust mean,
   outcome variance, standard error and excess Critic error
-> PPO-v011: use 8 robust means for Critic/normalizer; use all 32 current
   U(G)-V_old values for Actor
-> clip Actor and Critic separately; execute exactly one grouped Adam step
-> atomically commit optimizer receipt, Replay-v4, curriculum and checkpoint-v18
```

## Design Points

| ID | Design point | Current decision |
| --- | --- | --- |
| FRS-DP-01 | Perturbation Data | One sealed `local_rp` artifact per Scenario; four relative DR classes under current `d_cap`; full-6D action remains available. |
| FRS-DP-01P | Perturbation Probing | `2.381` is the measured frozen-GMT ceiling for this setup, not an online controller. |
| FRS-DP-02 | Segment Replay | Inner M4 supplies current evidence. Outer Replay owns a 32-visit policy-compatible window per Scenario/K, a 20% winsorized expected-utility estimate, uncertainty and bounded selection. |
| FRS-DP-03 | K-step Curriculum | `K8/M4 -> K16/M4 -> K32/M4`; each K restarts lower DR and Actor LR while retaining learned parameters. |
| FRS-DP-04 | FrontRES 6D Repair | The 158D Actor emits one unclamped full-6D world-frame Delta SE(3) at `t`. |
| FRS-DP-05 | Frozen GMT | Frozen 770D GMT executes the repaired continuation for K steps. |
| FRS-DP-06 | Paired Rollouts | One Clean and one fixed Noisy baseline are read-only anchors for all M4 current Repair attempts. |
| FRS-DP-07 | Repair Gain | `G_total=G_I+lambda_RA*G_P-beta*C_repair`; utility is per-attempt symmetric log. |
| FRS-DP-08 | HSL Warmup | HSL-v2 initializes Actor/std and Actor-prefix normalizer only. |
| FRS-DP-09 | Actor & Critic Warmup | Actor and Critic update together from low DR; Actor LR `3e-7 -> 1e-6`, Critic LR `1e-5`, B8/M4. Critic target is Replay's compatible robust Scenario mean. |
| FRS-DP-10 | Future Motion Context | Actor 158D, Critic 449D action-pre support-conditioned state, GMT 770D; no action-conditioned or variance-head Critic. |

## Replay Statistics

For each Scenario/K window:

```text
mu_hat   = 20% symmetric winsorized mean of compatible utility samples
sigma2   = within-Scenario rollout outcome variance
SE       = sqrt(sigma2 / N)
h95      = 1.96 * SE
E_V      = max(abs(V_old - mu_hat) - h95, 0)
E_A      = current-M4 centered absolute utility spread
```

The policy anchor is the 6D diagonal-Gaussian `pi_old` mean and sigma at
window creation. A new visit appends only when symmetric Gaussian KL to that
fixed anchor is at most `0.02`; otherwise it starts a new M4 window. High
outcome variance alone does not create calibration priority. No historical
action, log probability or Actor advantage is replayed.

The active pool is K-local and grows `64 -> 128 -> 256` only after every active
Scenario has four visits in its current policy-compatible window and the K
stage reaches full joint optimization. A policy reset returns that Scenario's
capacity maturity to one compatible visit; lifetime visits remain diagnostic
only. Warmup slots are
`1 admission + 6 E_V + 1 stale`; joint slots are
`1 admission + 4 E_A + 2 E_V + 1 stale`. Four DR quotas remain
Easy/Medium/Hard/Broken-tail `20/30/40/10`.

## State And Failure Boundary

- Replay preview includes the current valid M4 but mutates no owner.
- Only the matching exact-one receipt commits windows, scores, membership,
  capacity, staleness and RNG.
- Checkpoint-v18 stores Replay-v4 windows and rejects checkpoint-v17/replay-v3
  before restore; no migration or zero-fill exists.
- Evaluation is read-only and does not build or mutate a training Replay window.
- Current offline evidence proves construction and transaction connectivity,
  not simulator policy quality. One bounded official K8 transaction is the
  next live boundary; long training remains unauthorized.

## Acceptance

- Atlas shows B8/M4, not two Segments.
- Critic predicts expected symlog utility; robust estimation does not turn it
  into max/min/median/quantile or Q(s,a).
- Critic targets may use compatible history; Actor advantages may not.
- Aleatoric variance, mean confidence, reducible value error and policy drift
  remain distinct diagnostics.
- Gain, K/DR, LR, optimizer count, Actor/Critic architecture and simulator are
  unchanged.
