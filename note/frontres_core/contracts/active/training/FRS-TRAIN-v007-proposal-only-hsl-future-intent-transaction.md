---
contract_id: FRS-TRAIN-v007
status: active
effective_date: 2026-07-20
updated_date: 2026-07-21
supersedes: FRS-TRAIN-v006
scope: proposal-only Stage-1 HSL initialization on the deployable future-intent actor interface, plus formal Stage-3 routing for local repair, one policy action per attempt, and frozen-FEMR Clean-continuation K evidence
---

# Proposal-Only HSL Future-Intent / Single-Action K Training Contract

## Design Delta

`FRS-TRAIN-v006` separated actor information from the evaluator continuation,
but left HSL target authority open pending H0. H0-A fixes that boundary:

```text
HSL: current artifact -> proposal-direction initialization only
Stage 3: Noisy -> Executable objective from local K evidence only
```

The active actor/evaluator separation remains:

```text
actor at t:
  current Noisy root/anchor artifact + future 29DoF intent

GMT at t:
  Noisy root/anchor artifact, optionally written by Delta SE(3)_t

GMT at t+1 ... t+K:
  common full Clean continuation, with FEMR frozen
```

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-08` | HSL Warmup | `M-03` | HSL Proposal-Only Initialization |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Formal Transaction Route |

`M-03` continues to mean supervised initialization of the full-6D actor. This
contract does not change the human-facing Concept Figure.

## HSL Proposal-Only Initialization

Stage 1 must consume the same deployable actor interface as Stage 3:

```text
existing robot, balance, and tracking observation
+ current Noisy root/anchor artifact
+ ordered future deployment/Noisy internal-intent window I[t:t+H]
-> one full-6D actor distribution
```

The future carrier is root-invariant articulated motion, not a legacy `[H,65]`
raw-reference prefix. The active minimum is 29DoF joint intent; extra derivative
or phase fields require explicit deployment availability and provenance.

The only allowed Stage-1 target is a current-frame simulation-oracle anti-DR
Delta SE(3): the delta that undoes the selected current root artifact. It is
privileged training evidence only. It may initialize repair direction, but it
is not actor input, a deployment signal, a future reference, or an
executable-return target.

HSL authority ends at initialization. It must not define a Stage-3 return,
advantage, priority, PPO loss, or continuing online supervised anchor.

## HSL Forbidden Inputs And Targets

The Stage-1 actor input must reject:

- Clean current or future reference provenance;
- future root translation, anchor pose, or global orientation from Noisy or
  Clean reference;
- a raw 65D future tape, a Noisy physical prefix, a perturbation label, or a
  perturbation-time/truth field.

The Stage-1 target must reject:

- Clean future q29 or any full Clean reference sequence;
- a full-Clean rollout target, Clean-global Style target, or post-rollout
  Clean root/global residual;
- a target whose meaning requires a Stage-3 quartet, search, oracle, or Clean
  scored role.

## Stage-3 HSL Isolation

`build_frontres_hsl_rollout_target()` is a legacy quartet/Clean-global route.
It must be disabled and fail closed under v007. It may not write
`transition.supervised_target`, `supervised_weight`, or
`supervised_harm_weight` into active Stage-3 storage, loss, PPO, or formal
route.

The two scored Stage-3 roles remain Noisy and Repair only. Clean continuation
is a shared GMT evaluator condition, not an HSL label source, PPO role, actor
observation, or actor target.

## HSL Layout And Checkpoint Boundary

Stage-1 HSL must build the q29 actor tail through the same versioned layout and
deployment-provenance validator as the v015 actor bridge before normalization
and before the residual actor consumes its input. Its command-owned proposal
snapshot is deliberately smaller than a Stage-3 local scenario:

```text
current root-artifact identity
+ deployment/Noisy q29 I[t:t+H]
+ immutable proposal-context identity
```

It contains no Clean `x_t`, Clean continuation, K, Segment role, attempt,
return, priority, or PPO state. This is an implementation ownership
clarification, not a new actor observation: Stage 1 and Stage 3 still expose
the same current-artifact plus q29 future-intent interface to the actor.

Legacy HSL checkpoints are not migration inputs:

- no automatic first-layer reshape or partial compatibility load;
- no acceptance of legacy or unversioned prefix-normalizer statistics;
- no cold-start or resume behavior that treats an old HSL checkpoint as v007.

G2-S2 defines the only accepted Stage-1 migration identity:
`frontres-v015-hsl-proposal-v1`. Its exact payload is residual actor,
`std` or `log_std`, and the complete 158D prefix-normalizer state. The identity
binds the `870/928/158/770` layout, `(1,2)` q29 offsets, full-6D action, frozen
GMT artifact hash, and frozen 770D GMT-normalizer fingerprint. Critic,
critic-normalizer, optimizer, sampler, transaction, Gain/PPO state, and generic
warmup markers are forbidden. Legacy HSL checkpoint load remains rejected
rather than adapted.

The Step 4C v015 Stage-3 envelope is not a new Stage-1 HSL checkpoint format.
It may retain `frontres_warmup_complete` only as historical Stage-3 state after
the complete v015 layout/normalizer/transaction identity has validated. A
payload without that envelope remains a legacy HSL checkpoint and rejects before
state restoration.

## Local Scenario And Reset Route

For one selected scenario:

```text
select local source
-> bind x_t, root_artifact_t, I[t:t+H], C = Clean[t+1:t+K], K
-> seal one scenario identity/hash
-> reset Clean x_t for each M attempt
-> install the same current artifact, I window, and C
-> run one actor action at t
-> freeze FEMR and execute GMT through C
```

Reset restores dynamics only. It must not resample an artifact, modify the
intent window, replace the Clean continuation, or make `Clean x_t` actor
reference.

## One Policy Row And Formal Transaction

Each policy-sampled attempt stores exactly one policy tuple:

```text
observation_t, action_t, old mean/sigma/log_prob/value_t
+ K-step frozen-FEMR GMT evidence
-> one return_K, one advantage_K, one policy_row_valid
```

The formal route verifies q29 layout/provenance, freezes one `pi_old`, collects
all M attempts from their same `x_t`/artifact/I/C, aggregates their single-
action K evidence, then runs grouped v003 PPO with exactly one optimizer step.

H is deployment-visible future internal-motion intent. K is the duration of the
first action's frozen-FEMR GMT evidence. Neither H nor K creates future actor
actions, extra PPO rows, or actor-loss mass.

## Required Diagnostics

The formal route must prove or print current-artifact/H-intent/K-continuation
provenance, q29 invariant, reset identity, one action then frozen FEMR, two
scored role counts, zero active Stage-3 HSL-target writes, one policy row per
attempt, grouped mass, advantage sign/scale, and exact-one update.

## Acceptance Gates

| Gate | Required proof | Status |
| --- | --- | --- |
| H0-A | proposal-only HSL authority and rollout-label prohibition | confirmed 2026-07-20; no code migration |
| H1 S1a | q29 HSL input, current-frame target provenance, direct/legacy Stage-3 rejection, zero/rejected loss, legacy checkpoint rejection | completed 2026-07-20; `E-FI-6` deterministic evidence |
| H1 S2 | fake Stage-1 connectivity and fake Stage-3 isolation | completed 2026-07-20; `E-FI-7` CPU-only connectivity evidence |
| G2-S1a | command-owned minimal proposal carrier and shared q29 bridge | completed 2026-07-21; `E-FI-35` deterministic evidence |
| G2-S1b | formal `928/158/770` Stage-1 route and actor-only critic invariance | completed 2026-07-21; `E-FI-36` deterministic evidence |
| G2-S2 | strict proposal-only HSL identity/save/reload and pre-mutation rejection | completed 2026-07-21; `E-FI-37` deterministic S3 evidence |
| G2-S3 | offline fresh-runner q29/normalizer/actor proposal equality | completed 2026-07-21; `E-FI-38` deterministic S2/S3 evidence |
| G2-S4 | bounded formal Stage-1 q29/target/actor-only update/HSL-v1 save and cross-device reload | completed 2026-07-21; `E-FI-42` live S4 evidence |
| G3-S1A | explicit actor-only HSL-v1 migration into fresh q29/grouped/formal Stage 3, with legacy training dispatch blocked | completed 2026-07-21; `E-FI-43` deterministic S1/S3 evidence |
| G3-S1B | ordinary Stage-3 complete multi-Segment x M provider -> sealed grouped exact-one update -> matching committed-only save trigger | completed 2026-07-21; `E-FI-44` deterministic S2/S3 evidence |
| G3-S2 | same semantic policy exact-one update -> actual v015 save -> strict fresh inference reload with exact q29/158D/6D identity | completed 2026-07-21; `E-FI-45` deterministic S3 evidence |
| S1 | root-only perturbation / q29 invariant and actor H provenance | partially implemented outside HSL |
| S2A | two-role local reset and sealed command layout | completed 2026-07-20; `E-FI-8` deterministic fake-reset evidence only |
| S2B | one action -> frozen FEMR -> Clean GMT continuation | completed 2026-07-20 at candidate-only deterministic fake S1/S2 (`E-FI-9`); legacy formal collector rejects active v015 local scenarios |
| S3 | intent/physics Gain storage and consumer connectivity | partially completed at candidate-only deterministic S1 (`E-FI-11`--`E-FI-13`): post-`t` q29 and sealed `I[t]` reach v003 return/priority evidence, local diagnostics, and a v015 metadata-bearing grouped candidate batch; legacy v002 evaluators and adapters reject v015. Sampler state, real evaluation, and runner connectivity remain pending |
| S4 | transaction/grouped-PPO formal route and persistence | R5 offline S2 (`E-FI-23`) now connects actual command/observation/q29/normalizer/actor-GMT/K evidence for 2 Segment x 2 attempts to the unchanged grouped v003 exact-one owner; R4 S3 (`E-FI-22`) binds exact v2 persistence. Actual checkpoint cadence/resume, simulator, and live evidence remain blocked. |
| S5 | bounded live identity sentinel | R6-S0 snapshot/preflight and R6-F1 deterministic clock isolation completed (`E-FI-24`--`E-FI-25`); the repaired single SUST_Main_2 transaction has not been rerun |

H1 changes only the local Stage-1/reject boundaries. `E-FI-42` runtime-confirms
the bounded proposal-only HSL route and its strict v1 artifact; `E-FI-43`
deterministically confirms explicit Stage-3 actor/distribution/158D-prefix
migration with zero critic/optimizer/sampler/transaction state leak. It does
not confirm a trained v015 policy checkpoint. `E-FI-44` separately confirms
ordinary formal provider/update/commit/save-trigger connectivity and legacy
isolation. `E-FI-45` closes the actual save-to-fresh-inference persistence
chain at offline S3, but it does not claim live training or policy quality. R5 and R4
retain offline-S2 observation/update
connectivity and exact v2 persistence for v015 Stage 3. The existing full-65D
Noisy tape and legacy immediate-update routes remain contract mismatches outside
the v015 route.
