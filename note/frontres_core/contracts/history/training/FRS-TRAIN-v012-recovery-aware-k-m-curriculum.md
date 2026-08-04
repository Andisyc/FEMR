---
contract_id: FRS-TRAIN-v012
status: superseded
effective_date: 2026-08-01
updated_date: 2026-08-03
supersedes: FRS-TRAIN-v011
superseded_by: FRS-TRAIN-v013
scope: HSL-to-HRL Recovery-Aware Critic initialization, coordinated K x exact-M and perturbation-frontier curriculum, per-K Critic recalibration, actor ramp, grouped scalar PPO, and strict checkpoint-v7 persistence
---

# Recovery-Aware K x M Training Curriculum

## Design Delta

FRS-TRAIN-v011 coordinated K8/M2, K16/M3, and K32/M4 around the old
FRS-GAIN-v006 scalar Intent Critic and FRS-PPO-v004 Physics projection.
FRS-TRAIN-v012 preserves the proven transaction and curriculum structure while
changing the learned scalar target to FRS-GAIN-v007 `G_total` and the optimizer
to FRS-PPO-v005 grouped scalar PPO.

Because the Critic target meaning changes, no old Stage-3 Critic or optimizer
state may initialize the new campaign. HSL-v1 remains the only cold-start actor
source. At every later K increase, the same new Critic retains its weights but
re-enters critic-only recalibration before actor learning resumes.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `M-02` | K-Conditioned Perturbation Frontier |
| `FRS-DP-02` | Segment Replay | `SR-01` | Exact-M Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Coordinated K x M Schedule |
| `FRS-DP-08` | HSL Warmup | `M-03` | First Entry From HSL |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Critic Recalibration And Actor Ramp |

## Preserved Method Authority

- actor input remains the deployable 158D prefix;
- actor output remains one world-frame full-6D `Delta SE(3)` at `t`;
- one attempt remains one policy row regardless of K;
- one Segment executes one Clean, one fixed zero-action Noisy, and exact M
  Repair rollouts;
- each transaction selects exactly two Segments and performs exactly one grouped
  optimizer update after complete sealing;
- H=2 remains actor context and K remains executable-evidence horizon;
- Clean continuation remains frozen-GMT/evaluator-only evidence.

No rho, second actor/Critic/optimizer, Clean actor future, K actor input, Noisy
physical prefix, noise label, perturbation timing, independent Physics
projection, or KKT actor gate is introduced.

## First Entry From HSL

A fresh FRS-TRAIN-v012 campaign may initialize only from strict
`frontres-v015-hsl-proposal-v1`:

```text
restore proposal actor parameters
restore full-6D distribution/std
restore 158D actor-prefix normalizer
fresh-initialize Recovery-Aware scalar Critic and its normalizer/state
fresh-initialize Stage-3 optimizer and sampler
resolve and freeze curriculum plus metric identities
enter K8/M2 critic_only with actor_loss_weight=0
```

HSL never supplies a Stage-3 supervised target. Critic, Critic normalizer,
optimizer, sampler, transaction state, or Clean-global labels may not enter the
HSL payload. HSL initialization and strict Stage-3 resume are mutually
exclusive.

Checkpoint-v6 and earlier reject before mutating actor, std, Critic,
normalizers, optimizer, sampler, curriculum, or absolute iteration. An old
Stage-3 actor is not migrated around this boundary; the accepted cold-start
authority is HSL-v1.

## Scalar Critic Authority

At active K stage `j`:

\[
V^{RA}_j(o_{critic,t})
\approx
\mathbb E[G_{\mathrm{total},K_j}\mid o_{critic,t}],
\qquad
A^{RA}_{K_j}=G_{\mathrm{total},K_j}-V^{RA}_{old,j}(o_{critic,t}).
\]

There is one scalar Critic, no K input, and no K-specific head. It predicts the
complete FRS-GAIN-v007 return, including Intent improvement, pressure-weighted
Physics improvement, and repair cost. Raw channel evidence, applicability,
sampler priority, beta calibration statistics, and old constraint/KKT state may
not create an additional Critic target or value loss.

## Coordinated K x M Schedule

The immutable campaign schedule is:

```text
C_KM = [(K_j, M_j, N_c_j, N_a_j, N_joint-review_j)]

stage 0 = (8,  2, 200, 500, 1300)
stage 1 = (16, 3, 300, 300,  900)
stage 2 = (32, 4, 400, 300,  625)
```

K is positive and strictly increasing. M is at least two and non-decreasing.
K64 is not active. Every selected Segment contributes exact M Repair policy
rows and paired Noisy role rows. The fixed transaction identities are:

```text
selected_segment_count = 2
maximum_absolute_iteration = 8000
checkpoint_review_boundaries = (2000, 3500, 4825, 6500, 8000)
policy_rows_per_transaction = 2 * M_j
role_rows_per_transaction = 2 * policy_rows_per_transaction
required_num_envs = role_rows_per_transaction
```

Formal widths are therefore 8 env at K8/M2, 12 env at K16/M3, and 16 env at
K32/M4. Replay priority selects Segment identity only; it cannot change M,
transaction width, grouped mass, or the opened transaction.

## K-Conditioned Perturbation Frontier

Each K owns an independently calibrated frozen-GMT Noisy executability frontier
`g_K`. The method does not require `g_8`, `g_16`, and `g_32` to be equal or
monotonic. Before Critic recalibration at a new K, actor/std remain frozen while
the Noisy baseline calibrates and freezes the stage frontier.

For active K:

\[
\begin{aligned}
p(d\mid K)={}&0.20\,\mathcal U(0,0.25g_K)
+0.30\,\mathcal U(0.25g_K,0.70g_K)\\
&+0.40\,\mathcal U(0.70g_K,g_K)
+0.10\,\mathcal U(g_K,\min(1.10g_K,d_{max})).
\end{aligned}
\]

Easy cases remain present, the largest mass stays near the repair frontier, and
the hard region is a capped tail. One Segment samples one strength and one
single `local_rp` artifact, then reuses them across Noisy and all Repair
attempts. Clean remains uncorrupted. Gain/PPO may not adapt this distribution.

The stage order is:

```text
new K
-> freeze actor/std
-> calibrate and freeze g_K
-> Critic-only recalibration
-> actor ramp
-> joint optimization
```

## Critic Recalibration And Actor Ramp

For stage-local committed iteration `l_j`:

```text
0 <= l_j < N_c_j:
    phase = critic_only
    actor_loss_weight = 0

N_c_j <= l_j < N_c_j + N_a_j:
    phase = actor_ramp
    actor_loss_weight = (l_j - N_c_j + 1) / N_a_j

l_j >= N_c_j + N_a_j:
    phase = joint
    actor_loss_weight = 1
```

At a K transition, the same Recovery-Aware Critic and compatible optimizer
state continue; the Critic is not reinitialized. Actor/std freeze while the
Critic adapts to the longer consequence and changed return distribution. The
Critic continues updating during actor ramp and joint training.

M changes the number of same-scenario candidates but does not define a new
Critic target and does not independently trigger another warmup. K changes the
meaning of the executable consequence and therefore owns recalibration.

## Exact-M Frozen-Policy Transaction

Before opening a transaction, resolve and seal:

```text
contract/checkpoint/curriculum identity
stage, K, M, perturbation frontier and beta
two Segment scenarios and fixed Clean/Noisy lifecycle
frozen pi_old and transaction identity
```

Collection performs zero optimizer steps. Every Segment must produce exact M
eligible policy rows, and every row must use the same active K/M/beta/metric
identity. Only complete sealing may call FRS-PPO-v005 once. A failed or partial
transaction cannot advance phase, K, M, `g_K`, beta, absolute iteration,
optimizer count, sampler state, or checkpoint receipt.

Stage advancement happens only after a committed receipt. The next transaction
observes the new K/M and critic-only phase at local iteration zero.

## Beta Calibration Boundary

The first bounded calibration uses `beta_init=0.02`. Training telemetry retains
the cost-free Recovery-Aware score, repair cost, applied cost, final `G_total`,
and within-Segment order changes. The run cannot mutate beta. Human review may
change the single global value between bounded calibration runs. Once accepted,
the value and its identity remain fixed across every Segment, attempt, and K
stage of the formal campaign.

## Checkpoint-v7 Identity

```text
checkpoint_schema = frontres-v015-checkpoint-v7
method_contract_id = FRS-METHOD-v017
gain_contract_id = FRS-GAIN-v007
optimization_contract_id = FRS-PPO-v005
training_contract_id = FRS-TRAIN-v012
scalar_target_id = clean-anchored-recovery-aware-gain-v1
physics_evidence_schema_id = clean-anchored-contact-zmp-survival-v1
grouped_update_schema_id = grouped-all-attempt-scalar-v1
```

The payload binds:

- 928D combined / 158D FrontRES / 770D GMT observation identity;
- actor, std, Critic, their normalizers, and optimizer state;
- complete K x M schedule, active stage/K/M/local phase/actor weight;
- `g_8/g_16/g_32` identities and active frozen frontier;
- exact fixed `S_j`, repair units, beta, and Gain formula identity;
- selected Segment count, role/policy row counts, sampler state, and maximum
  absolute iteration;
- absolute committed update and matching committed transaction receipt.

Full resume requires exact v7 equality before mutable restoration. Different
metric scales, beta, K/M schedule, frontier identity, actor/GMT layout,
transaction width, target identity, partial receipt, HSL-as-resume, checkpoint
v6, or an active projection schema rejects pre-mutation.

## Required Telemetry

- stage/K/M, `g_K`, beta, phase, actor weight, absolute/local iteration;
- transaction/Segment/attempt counts, role/policy row widths, exact-one update,
  and committed receipt;
- `G_I`, `G_P`, `lambda_RA`, repair cost, `G_total`, return, value, raw/scaled
  advantage, and value calibration;
- raw Contact, support-foot drift, phase-ZMP, survival, sustained lean, and
  unplanned support changes;
- action magnitude/non-collapse and actor/std/Critic parameter deltas;
- checkpoint-v7 and all four contract identities.

## Required Evidence And Stop Conditions

Deterministic evidence must cover schedule/fingerprint resolution, exact M at
all stages, environment widths, cold-start and strict-resume rejection,
critic-only isolation, actor-ramp arithmetic, K transition with the same Critic,
frontier freezing, beta freezing, grouped exact-one update, and checkpoint-v7
save/reload identity. Formal and live evidence remain separate later gates.

Stop if actor/std drift during critic-only, the Critic is reinitialized at a K
transition, old constraint/projection state reaches learning, M becomes
state-driven, a transaction mixes K/M/frontier/beta/scales, live telemetry
mutates beta, checkpoint-v6 changes state, exact-one update fails, or a
sufficiently informative quality block shows systematic no-op, sustained lean,
unplanned support change, or inverted raw-evidence ordering.
