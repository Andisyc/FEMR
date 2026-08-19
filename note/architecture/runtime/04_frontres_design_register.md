# FrontRES Design Inspector Register

Status: aligned to FRS-METHOD-v026 / FRS-GAIN-v009 / FRS-PPO-v014 /
FRS-TRAIN-v025 on 2026-08-19. FRS-DP-09 remains a retired historical card;
FRS-DP-09R is the active-pre-training Actor-only Curriculum card.

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
-> Replay preview: compute 8 current-M4 means, current outcome variance,
   standard error and latest excess Critic error
-> PPO-v012: use 8 current means for Critic/normalizer; use all 32 current
   U(G)-V_old values for Actor
-> clip Actor and Critic separately; execute exactly one grouped Adam step
-> atomically commit optimizer receipt, Replay-v5, curriculum and checkpoint-v19
```

## Design Points

| ID | Design point | Current decision |
| --- | --- | --- |
| FRS-DP-01 | Perturbation Data | One sealed `local_rp` artifact per Scenario; four relative DR classes under current `d_cap`; full-6D action remains available. |
| FRS-DP-01P | Perturbation Probing | `2.381` is the measured frozen-GMT ceiling for this setup, not an online controller. |
| FRS-DP-02 | Segment Replay | Outer Replay selects sealed Scenarios for fresh current-Actor M4 execution; it stores latest priorities and visits, never historical utility targets. |
| FRS-DP-03 | K-step Curriculum | `K8/M4 -> K16/M4 -> K32/M4`; each K changes consequence horizon and DR coverage without resetting Actor or Adam state. |
| FRS-DP-04 | FrontRES 6D Repair | The 158D Actor emits one unclamped full-6D world-frame Delta SE(3) at `t`. |
| FRS-DP-05 | Frozen GMT | Frozen 770D GMT executes the repaired continuation for K steps. |
| FRS-DP-06 | Paired Rollouts | One Clean and one fixed Noisy baseline are read-only anchors for all M4 current Repair attempts. |
| FRS-DP-07 | Repair Gain | (退役历史卡) Historical scalar `G_total=G_I+lambda_RA*G_P-beta*C_repair`; retained for provenance, not consumed by the current relational route. |
| FRS-DP-07R | Relational Gain | Active-pre-training `BETTER/WORSE/SAME/INCOMPARABLE`; v014 uses `L_{pref}=|E|^{-1}sum softplus(-(log pi_w-log pi_l))`; v013 remains retired-compatible only. |
| FRS-DP-08 | HSL Warmup | HSL-v2 initializes Actor/std and Actor-prefix normalizer only. |
| FRS-DP-09 | Actor & Critic Warmup (退役) | Historical Actor/Critic, scalar target and K-local LR reset card; retained for provenance only. |
| FRS-DP-09R | Actor-only Curriculum | Initial `3e-7 -> 1e-6` ramp once; K transitions preserve Actor/Adam state and keep `1e-6`; DR restarts without becoming an LR controller. The candidate Loss and LR are pre-training decisions, not a live activation claim. |
| FRS-DP-10 | Future Motion Context | Actor 158D remains; retired 449D action-pre Critic context is not consumed by TRAIN-v025. |

## Replay Statistics

For each current Scenario visit:

```text
target   = arithmetic mean of the current four utility samples
sigma2   = current-M4 sample outcome variance
SE       = sqrt(sigma2 / 4)
h95      = 1.96 * SE
E_V      = max(abs(V_old - target) - h95, 0)
E_A      = current-M4 centered absolute utility spread
```

The M4 target is never combined with an earlier visit. Policy mean/sigma, KL,
anchor, reset count and utility windows are absent from Replay state. High
current outcome variance only widens priority uncertainty. No historical
utility, action, log probability or Actor advantage is replayed.

The active pool is K-local and grows `64 -> 128 -> 256` only after every active
Scenario has four committed fresh-M4 visits at that K and the K stage reaches
full joint optimization. Every visit is a real current-policy rerun. Warmup slots are
`1 admission + 6 E_V + 1 stale`; joint slots are
`1 admission + 4 E_A + 2 E_V + 1 stale`. Four DR quotas remain
Easy/Medium/Hard/Broken-tail `20/30/40/10`.

## State And Failure Boundary

- Replay preview includes the current valid M4 but mutates no owner.
- Only the matching exact-one receipt commits latest scores, visits, membership,
  capacity, staleness and RNG.
- Checkpoint-v19 stores Replay-v5 without utility outcomes and rejects checkpoint-v18/replay-v4
  before restore; no migration or zero-fill exists.
- Evaluation is read-only and does not build or mutate a training Replay window.
- Current offline evidence proves construction and transaction connectivity,
  not simulator policy quality. One bounded official K8 transaction is the
  next live boundary; long training remains unauthorized.

## Acceptance

- Atlas shows B8/M4, not two Segments.
- Critic predicts expected symlog utility; current M4 estimation does not turn it
  into max/min/median/quantile or Q(s,a).
- Critic targets and Actor advantages both use only the current transaction; only their aggregation differs.
- Aleatoric variance, mean confidence, reducible value error and policy drift
  remain distinct diagnostics.
- Gain, K/DR, LR, optimizer count, Actor/Critic architecture and simulator are
  unchanged.

## Pending Semantic Transition: Relational Gain

Status: `proposal-not-active` / `TRANSITION-BLOCKED` on 2026-08-16.

The candidate hierarchical Gain preserves the confirmed behavioral partial
order and returns `BETTER`, `WORSE`, `SAME`, or `INCOMPARABLE`. It is not the
active training objective. `FRS-GAIN-v008` remains the sole production Gain
contract, and the existing PPO consumer still receives scalar `G_total`,
symlog utility, M4 target, and scalar Actor advantage.

The candidate masked pairwise credit defines the Actor-side learning signal.
The scalar state-value Critic is retired on this route: a fully ordered M4 has
zero-sum edge credit, and the current state-value mean would add no information.
Activation requires a coordinated Contract migration, a public module-alignment
replay, and a formal checkpoint/route review. Until then, this proposal must not be
connected to the live training path or used to start a new training campaign.
This engineering unit ends at `PRE-TRAINING-READY`: Codex may prepare and audit
the code and exact command, but the user owns synchronization, checkpoint
generation, live execution, and training.
