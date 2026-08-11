---
contract_id: FRS-METHOD-v022
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FRS-METHOD-v021
superseded_by: FRS-METHOD-v023
scope: Historical Recovery-Aware exact-M training with DR-compatible phase-aware outer sealed-Scenario replay
---

# DR-Compatible Phase-Aware Scenario Replay

## Design Delta

FRS-METHOD-v021 established stable sealed-Scenario identity, fresh current-policy
M4 recollection, 40/50/10 global/replay/review selection and committed-only
outer replay mutation. Those boundaries remain. FRS-METHOD-v022 corrects two
scheduler mistakes: one score cannot serve both Critic calibration and later
Repair learning, and replay from a foreign absolute DR interval cannot calibrate
the current curriculum state.

Outer replay remains a Scenario scheduler only. It never reuses old policy rows,
changes Gain/PPO mass, adds an optimizer step or enters the Actor/Critic loss.

## Stable Scenario Identity

The validated ScenarioKey remains:

```text
motion_id, start_frame, segment_id, x_t_identity
perturbation_family, perturbation_strength, perturbation_seed
noisy_segment_hash, K
future_intent_identity, planned_support_identity
```

Replaying a key rematerializes the same Scenario and samples fresh M=4 Repair
actions from the transaction's frozen current `pi_old`.

## Two Committed Scores

For one Scenario with exact-M utility advantages
`A_m = U(G_m) - V_old(s)`, store two detached scheduler values per K:

```text
E_V(s) = abs(mean_m A_m)
       = abs(mean_m U(G_m) - V_old(s))

E_A(s) = mean_m abs(A_m - mean_m A_m)
       = mean_m abs(U(G_m) - mean_m U(G_m))
```

`E_V` measures state-value calibration error. `E_A` measures the current
policy's within-Scenario Repair differentiation. Each has its own committed EMA
map per K. Neither changes the exact-M Critic target or Actor advantage.

During `low_dr_joint_init` and `coupled_ramp`, replay/review ranking uses `E_V`.
During `joint`, ranking uses `E_A`. A K transition returns to `E_V` for the new
K; a prior-K score never ranks the current pool.

## DR-Compatible Selection

Each of the two Scenario slots first draws one current relative DR class using
the fixed weights:

```text
Easy / Medium / Hard / Broken-tail = 0.20 / 0.30 / 0.40 / 0.10
```

For current `d_cap=d` and frozen ceiling `D`, the absolute intervals are:

```text
Easy        [0, 0.25d)
Medium      [0.25d, 0.70d)
Hard        [0.70d, d]
Broken-tail (d, min(1.10d, D)]
```

Only same-K records whose stored absolute perturbation strength lies in the
slot's current interval are eligible for replay/review. Stored historical class
labels are diagnostic and do not override the current interval. Source is then
drawn with global/replay/review = 0.40/0.50/0.10. Empty compatible replay or
review pools fall back to global discovery in the already drawn class, rather
than resampling another class.

Replay uses rank of the phase-selected score plus committed staleness. Review
uses low phase-selected score plus staleness. The two selected Segment identities
remain distinct.

## Transaction And Persistence

Admission requires a complete finite exact-M visit. The candidate contains both
score updates, current phase/score identity, DR class/interval, records and RNG
transition. It mutates state only after one matching exact-one optimizer receipt.
Failed, partial, mixed or duplicate transactions change no record, score,
membership, staleness or RNG.

Checkpoint-v16 strictly stores replay schema v2, both per-K score maps, stable
keys, visit/staleness state, sampler RNG and last committed receipt. Checkpoint-
v15 and earlier are incompatible with the active training route.

## Preserved Boundaries

- Actor remains the deployable 158D full-6D direct Delta SE(3) policy.
- Critic remains the 449D support-conditioned scalar state value `V(s)`.
- Raw FRS-GAIN-v008, per-attempt symlog utility and exact-M mean target remain.
- M=4; K8 -> K16 -> K32; two Scenarios and one grouped Adam step remain.
- Actor LR=3e-6, Critic LR=1e-5, separate clipping and frozen GMT remain.
- No Q(s,a), stale PPO buffer, winner weighting, second Critic/optimizer or
  MOSAIC host modification is permitted.

## Falsifiers

- Warmup ranking uses `E_A` or joint ranking uses `E_V`.
- `E_V` is implemented as mean absolute per-attempt advantage.
- A Scenario outside the current absolute DR interval is replayed.
- Empty compatible replay changes the already drawn DR class.
- Failed commit changes either score map, staleness or RNG.
- Save/resume drops either score map or accepts checkpoint-v15.
