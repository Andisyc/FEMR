# FRS-v015 Engineering Plan: Future-Intent, Single-Action K Replay

Status: active, volatile engineering plan. Steps 1A--5A and R0--R6 are complete
through bounded S4 at `E-FI-27`: the dedicated v015 route preserves local
identity, `928D` actor / `289D` critic / `770D` GMT authority, one-action K=8,
equal grouped mass, and exactly one update. `E-FI-32` rebases the remaining
test path after the observation migration: no compatible trained checkpoint or
defined external Noisy `.npz` exists, so direct Step 5B-S4 execution is blocked.
`E-FI-33` closes the read-only G1 audit with four formal training gaps.
`E-FI-34` audits G2-S0 and blocks implementation on the Stage-1 q29 carrier
decision plus removal of the legacy HSL energy-critic path.

Active contracts:

- `FRS-METHOD-v015-future-intent-single-action-k-replay.md`
- `FRS-TRAIN-v007-proposal-only-hsl-future-intent-transaction.md`
- `FRS-GAIN-v003-intent-physics-local-repair.md`
- `FRS-PPO-v003-single-policy-row-k-evidence-grouped-reduction.md`
- `FRS-EVAL-v003-local-repair-composition-evaluation.md`

Supersedes as the active planning surface:

- the earlier coarse five-step version of this same volatile plan;
- `FRS-v013-future-context-double-segment-replay-engineering-plan.md`.

## Objective

Migrate Stage 3 to the accepted local first-action experiment:

```text
current root artifact + future deployment-provenance q29 intent
-> one Delta SE(3)_t
-> FEMR frozen
-> GMT on the common Clean continuation for K
-> two-role intent / physics / cost Gain
-> M-attempt one-row grouped PPO transaction
```

The implementation route must make each semantic object independently
testable. It must not bring back a full-65D future actor input, a K-step FEMR
policy sequence, a third scored Clean role, or a Clean-global Style target.

## Planning Vocabulary

The active training contract's `S0`--`S5` rows are method acceptance gates.
The repository test board's `S0`--`S4` rows are evidence tiers. To avoid
overloading `S`, this plan calls its ordered migration gates `G0`--`G5` and
uses `S/T` only for the repository evidence tier and test kind.

```text
G0 documentation and owner audit
G1 local scenario and actor-information boundary
G2 two-role single-action K lifecycle
G3 v003 Gain and all active consumers
G4 transaction, formal route, and persistence
G5 user-gated live local evidence and composition evaluation
```

## Human Design-Point Map

| Design ID / Figure block | Canonical human name | Active semantic owner | Current code/evidence gap |
| --- | --- | --- | --- |
| `FRS-DP-01` / `M-02` | Perturbation Data | Method v015, Local Root-Artifact Scenario | Current materializer produces a 65D fixed tape rather than a separated root artifact, q29 intent, and Clean continuation. |
| `FRS-DP-02` / `SR-01` | Segment Replay | Method v015, Frozen-Policy Transaction | Candidate/offline accumulator exists, while the formal route still needs lifecycle and metadata rebase. |
| `FRS-DP-03` / `M-06` | K-step Curriculum | Method v015, Single-Action K-step Evidence | Current execution must be isolated from later FEMR actions and later root perturbations. |
| `FRS-DP-04` / `M-04` | FrontRES 6D Repair | Method v015, Actor Observation And Action | Full 6D output remains; its new q29 future-intent input layout is absent. |
| `FRS-DP-06` / `Q-PAIR` | Paired Rollouts | Gain v003, Two-Role Pairing | Current command/reset plumbing retains quartet roles. |
| `FRS-DP-07` / `Q-01` | Repair Gain | Gain v003, Core Decision | Current Style code compares to global Clean motion rather than q29 intent. |
| `FRS-DP-08` / `M-03` | HSL Warmup | Training v007, HSL Proposal-Only Initialization | H0-A permits only current-frame anti-DR proposal initialization after q29 migration; Clean-quartet rollout supervision is forbidden. |
| `FRS-DP-10` / `M-11` | Future Motion Context | Method v015, Future Intent Context | Runtime currently prepends `[H,65]` raw tape fields. |

The detailed register remains the contract registry:
`../contracts/README.md#concept-figure-design-point-register`. The human
Concept Figure remains
`../../architecture/concept/03_frontres_concept_tabs.data.json`; this planning
revision does not change it.

## White-Box Starting Facts (E-FI-1) And Closed Isolation Targets

| Semantic object | Starting owner at E-FI-1 | Starting mismatch / legacy route | Active-route target | Planned gate |
| --- | --- | --- | --- | --- |
| Scenario materialization | `frontres_segment_stage1_env_hooks.py::materialize_frontres_fixed_noisy_tape` | Produces one `[L,65]` tape | Immutable local object containing `x_t`, current root artifact, q29 intent window, Clean continuation, K, and identity | 1A |
| Actor H context | `frontres_runtime.py::append_frontres_fixed_noisy_future_context` | Requires `[B, |H|*65]`, then prepends it to actor input | Ordered deployment-provenance future q29 tail only | 1B |
| Pair and reset layout | `frontres_training_setup.py`, `commands.py`, `frontres_segment_stage1_env_hooks.py` | Quartet/projected/candidate/base/clean roles remain available in active setup | Two scored roles: Noisy and Repair; Clean only supplies continuation | 2A |
| K collector | `frontres_rollout_step.py` and `frontres_segment_live_probe.py` | Existing route is not yet proven to authorize one action then freeze FEMR | One policy tuple at `t`, GMT-only execution through `C[t+1:t+K]` | 2B |
| Gain | `frontres_gain.py` and capture in `frontres_segment_live_probe.py` | Style uses Clean global body/root comparisons | `fidelity_internal(executed_q29, I)` paired across Noisy/Repair | 3A--3B |
| PPO adapter | `frontres_segment_storage.py` and `frontres_segment_ppo.py` | `to_ppo_batch()` intentionally drops transaction metadata; grouped candidate adapter is offline-only | Metadata-bearing candidate path becomes the only grouped formal path | 4A--4B |
| HSL | `frontres_warmup.py`, `frontres_hsl_rollout_target.py`, and checkpoint loader | Raw-observation warmup bypasses q29; rollout label uses a Clean quartet | Proposal-only current-frame HSL after q29 bridge; legacy rollout label and old checkpoints reject | H0-A -> H1 |

`E-FI-1` records this pre-implementation baseline. `E-FI-2`--`E-FI-27` close
the listed v015 targets through the required deterministic, integration,
persistence, and bounded-live tiers; the legacy routes remain isolated.

## Why This Plan Has Twelve Stage-3 Steps And One Conditional HSL Step

The previous plan combined materialization with actor wiring, reset with
rollout control, formula with every consumer, and transaction with
checkpointing. Those objects have different owners and failure modes:

- a provenance leak is not detectable by a Gain formula test;
- a third role or later FEMR action is not detectable by a storage test;
- a correct grouped-loss formula does not prove the formal runner avoided
  `to_ppo_batch()`;
- checkpoint/resume can violate transaction atomicity after all local tests
  pass.

Therefore each Stage-3 step below changes one owner boundary and has its own
S/T evidence and stop condition. H1 is a separately user-gated HSL migration
step: it does not silently activate and it does not block a later Step 2A that
explicitly keeps HSL disabled.

## Step Map

### Step 0 / 12: Contract, Concept-Figure, and Plan Closure

Status: completed, documentation only.

Objective:
- activate v015/v007/v003/v003/v003 semantics;
- synchronize the Concept Figure, contract registry, Architecture, evidence
  ledger, and the initial acceptance surface;
- re-plan after a white-box audit without changing active semantics.

Scope:
- governed Markdown/JSON and read-only code-owner inspection.

Non-scope:
- Python, config, test execution, optimizer, simulator, checkpoint, or live
  execution.

Expected evidence:
- S0 T-doc/T-version/T-map/T-plan/T-matrix; `E-FI-0`, `E-FI-1`.

Stop condition:
- a Concept Figure block, registry entry, active contract, or current code
  owner contradicts the accepted H, K, pairing, or Gain meaning.

### Step 1A / 12: Immutable Local-Scenario Kernel

Gate: G1. Status: completed at S1 deterministic-module evidence (`E-FI-2`).

Objective:
- replace the semantic use of a complete fixed Noisy tape with one immutable
  local scenario whose pieces have separate provenance and consumers.

Canonical data contract to implement:

```text
scenario = {
  x_t_identity,
  current_root_artifact_t,
  intent_q29[t:t+H],          # deployment / Noisy provenance
  clean_continuation[t+1:t+K],# GMT-only full reference
  horizon_k,
  scenario_id / noisy_segment_hash
}
```

`intent_q29` may numerically equal the Clean calibration track, but it must be
materialized and labeled through the deployment/Noisy carrier. The exact
current-artifact representation remains the command owner's existing root
artifact representation; this step does not change the 6D repair action.

Scope:
- local-scenario schema, deterministic materializer, immutable identity/hash,
  q29 projection, and sampler-to-command carrier;
- explicit separation between the H intent array and the K continuation array.

Non-scope:
- actor observation concatenation, normalizer/checkpoint migration, role
  layout/reset behavior, K rollout, Gain, PPO, HSL, formal runner, or live run.

Owner modules:
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py`;
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`.

Expected evidence:
- S1 T-schema/T-invariant/T-hash/T-provenance/T-metamorphic;
- unit fixture proves `Pi_internal(Noisy) == Pi_internal(Clean)` while only the
  current root artifact differs;
- mutation or attempted resampling changes/rejects the sealed identity;
- a scenario reports q29 intent and Clean continuation as different fields,
  never as one ambiguous 65D tape.

Stop condition:
- the owner cannot provide q29 without future root/global fields; any M-trial
  reset can rematerialize or mutate the scenario; or the hash fails to cover
  `x_t`, artifact, intent source/window, continuation, and K.

### Step 1B / 12: Future-Intent Actor Bridge

Gate: G1. Depends on: 1A. Status: completed. `E-FI-3` proves the isolated S1
tail builder, `E-FI-20`/`E-FI-23` close role-aligned formal consumption, and
`E-FI-27` confirms the bounded-live actor/GMT/critic authority split.

Objective:
- make the actor consume exactly the future q29 intent offsets from the sealed
  scenario, while retaining the existing current robot/balance/current-artifact
  observation boundary.

Scope:
- replace the actor-only `[B, |H|*65]` future tail with ordered
  `[B, |H|*29]` q29 future offsets;
- freeze one explicit layout/version and normalizer compatibility check;
- reject a selected scenario that lacks the q29 carrier.

Non-scope:
- changing the GMT observation suffix; putting the Clean continuation into the
  actor; changing HSL target semantics; accepting old normalizer statistics;
  reset/K/Gain/PPO/checkpoint/live execution.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/frontres_runtime.py`;
- `source/rsl_rl/rsl_rl/modules/frontres_observation_layout.py`;
- the FrontRES observation-layout config owner discovered by the step.

Expected evidence:
- S1 T-shape/T-offset/T-provenance/T-clean-isolation/T-legacy-reject;
- actor prefix contains only ordered q29 values for declared positive H offsets;
- altering a future root/global field cannot alter the actor tail;
- numeric equality with Clean calibration does not change the tail's declared
  deployment/Noisy source;
- a legacy `[H,65]` layout or incompatible normalizer is rejected fail-closed.

Stop condition:
- a future root/global field, Clean provenance, perturbation metadata, or a
  raw full tape reaches the actor; or actor/normalizer dimensions have no
  versioned compatible layout.

R0 evidence correction:
- the live command carrier has `B=8` Repair/Noisy rows, while
  `_future_intent_context_batch()` reads the `B=4` policy-attempt batch;
- the configured `num_frontres_obs=0` makes the residual actor consume the full
  `928D` tensor even though the frozen contract is FEMR `158D` / GMT `770D`;
- therefore `E-FI-3` does not prove formal role alignment, actor visibility, or
  live normalizer consumption.

### Gate H0: HSL Interface and Target Audit

Status: completed at S0 source/layout/target evidence (`E-FI-4`). H0-A is
user-confirmed and recorded in `FRS-TRAIN-v007`; this gate did not modify code.

H0-A decision:

- retain HSL only as a Stage-1 proposal-direction initializer;
- permit only a current-frame anti-DR oracle target as privileged training
  evidence, never as actor input or Stage-3 objective;
- require the actor to consume the v015 current-artifact plus deployment/Noisy
  q29 future-intent interface;
- forbid the quartet/Clean rollout label, all Stage-3 HSL storage/loss/PPO
  consumers, and direct legacy-HSL checkpoint migration.

H0 blocks until H1 is accepted:

- HSL/warmup reactivation under v015;
- formal runs that can invoke the legacy rollout label;
- HSL checkpoint migration or compatibility loading.

H0 does not block offline local-scenario, two-role, K-lifecycle, or Gain-owner
work that explicitly keeps HSL disabled.

### HSL Migration Step H1: Proposal-Only Future-Intent Initialization

Status: H1-S1a completed at deterministic S1 evidence (`E-FI-6`) and H1-S2
completed at CPU-only deterministic S2 evidence (`E-FI-7`) on 2026-07-20. It
added the v007 direct-writer and loss-side rejection discovered during the
H1-S1 preflight, then proved the two owner-to-owner fake connectors without
entering any formal/live route.
H1 is outside the twelve Stage-3 lifecycle steps because it is optional
initialization work and may be retired without changing the local K method.

Objective:

- make Stage-1 HSL initialize the v015 deployable actor using current Noisy
  root artifact plus ordered deployment/Noisy q29 future intent;
- retain only the current-frame anti-DR privileged Delta SE(3) target;
- make legacy quartet/Clean rollout supervision and old HSL checkpoint loading
  fail closed.

Primary owner:

- `source/rsl_rl/rsl_rl/runners/frontres_warmup.py::run_frontres_joint_warmup`
  owns the Stage-1 actor-input -> current-frame target -> warmup-loss path.

Required connectors and isolated owners:

- reuse the v015 actor-tail builder in `frontres_runtime.py` /
  `frontres_observation_layout.py`; no second H layout is permitted;
- `observations.py::get_supervision_target_task_space` remains the current-frame
  target provider and may not gain future/clean fields;
- `frontres_hsl_rollout_target.py` owns the forbidden legacy rollout label and
  must be rejected before storage;
- `frontres_checkpointing.py::load_runner` is a reject-only boundary, not a
  migration owner, until a separately authorized persistence step.

Scope:

- Stage-1 q29 actor-input route before normalizer and residual-actor use;
- current-frame target provenance assertion and full-6D target shape;
- hard Stage-3 rejection before any HSL transition/storage/loss write;
- explicit legacy HSL checkpoint/layout rejection.
- H1-S1a additionally rejects the pre-step anti-DR transition writer and any
  nonzero v015 Stage-3 supervised-loss configuration, so zero-valued storage
  defaults cannot become a continuing HSL optimizer anchor.

Non-scope:

- changing the anti-DR target into executable reward/Gain evidence;
- Clean actor input, future Clean target, full-Clean rollout target, HSL as a
  Stage-3 auxiliary loss, PPO, reset/K/Gain/grouped-PPO/formal runner work;
- any new checkpoint format, checkpoint conversion, resume, simulator,
  training, or live run.

Expected evidence:

- S1 T-HSL-layout/T-HSL-provenance/T-HSL-target/T-HSL-stage3-reject/
  T-HSL-direct-write-reject/T-HSL-loss-reject/T-HSL-legacy-checkpoint-reject
  with a semantically valid sealed local-scenario fixture;
- S2 T-HSL-connect proves fake Stage-1 q29 -> normalizer -> actor -> current
  target connectivity and fake Stage-3 zero HSL-write/loss connectivity
  (`E-FI-7`, completed);
- no S3 persistence or live evidence is claimed by H1.

Stop condition:

- q29 is unavailable without future root/global data; Clean or perturbation
  truth reaches actor input; the target reads Clean future/full rollout state;
  a Stage-3 HSL field reaches storage/loss/PPO; or a legacy checkpoint is
  reshaped, partially accepted, or silently normalized.

### Step 2A / 12: Two-Role Local Reset and Command Layout

Gate: G2. Depends on: 1A.

Implementation status: completed 2026-07-20 at deterministic S1 only
(`E-FI-8`). The command carrier is intentionally sealed-but-unrouted for actor
H and GMT K; Step 2B remains separately gated.

Objective:
- install exactly two scored physical branches from the same sealed local
  scenario: Noisy and Repair.

Scope:
- training layout selection, command/reference installation, and dynamic reset
  binding for `x_t`, current artifact, q29 intent, Clean continuation, K, and
  scenario identity;
- active-route rejection of projected/candidate/search/Clean scored roles.

Non-scope:
- actor sampling, later K execution, Gain, return construction, grouped PPO,
  checkpointing, HSL, or live run.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/frontres_training_setup.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py`;
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`.

Expected evidence:
- S1 T-role/T-state/T-scenario-identity/T-legacy-reject;
- deterministic fake reset shows Noisy and every Repair attempt share the same
  `x_t`, artifact, q29 intent, continuation, K, and hash;
- `Clean` is absent from the scored role set and cannot receive a policy row.

Stop condition:
- quartet/triplet role setup remains reachable from active v015 configuration;
- reset samples/mixes an artifact, intent, or continuation; or a Clean state is
  exposed as an actor reference.

### Step 2B / 12: One-Action Frozen-FEMR K Collector

Gate: G2. Depends on: 1B and 2A.

Implementation status: completed 2026-07-20 at deterministic S1/S2
candidate-only fake connectivity (`E-FI-9`). The formal runner remains
explicitly isolated: its legacy repeated-actor collector rejects an active v015
local scenario rather than adopting this candidate route.

Objective:
- collect one policy action at `t`, then measure its K-step consequence with
  FEMR frozen and GMT executing the common Clean continuation.

Scope:
- rollout action authorization, command cursor/reference route, K masks,
  capture lifecycle, and one policy-tuple carrier.

Non-scope:
- Gain formula replacement, priority, grouped loss, formal runner, checkpoint,
  HSL, and live simulator evidence.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`;
- command reference methods in `commands.py`.

Expected evidence:
- S1 T-action-count/T-frozen/T-continuation/T-row/T-K-metamorphic;
- S2 T-connect through reset -> actor -> one repair write -> GMT continuation
  -> capture/storage using a fake runner/environment;
- each eligible attempt has exactly one action/log-prob/value tuple regardless
  of K, while continuation carries all GMT-required reference fields.

Stop condition:
- a later FEMR action/write or later root perturbation occurs; GMT reads the
  H intent instead of the Clean continuation; or K creates additional PPO rows.

### Step 3A / 12: Root-Invariant Intent Gain Core

Gate: G3. Depends on: 2B for the final capture shape; deterministic unit work
may begin from an explicit fixture earlier.

Implementation status: completed 2026-07-20 at deterministic S1 (`E-FI-10`).
`frontres_gain.py` now exposes a typed, pure v003 owner that accepts only
deployment/Noisy-provenance q29 intent plus paired execution/physics and
full-6D action evidence. It preserves unobserved qvel/qacc/one-action temporal
terms as `NaN`; it does not route capture, returns, priority, diagnostics,
evaluation, PPO, checkpointing, formal execution, or live work.

Objective:
- make `frontres_gain.py` own v003 intent/physics/cost calculation with q29
  intent as its sole Style target.

Scope:
- typed gain input/output fields and pure q29/relative-articulation fidelity;
- paired intent gain, existing paired physics decomposition, and full-6D repair
  cost; explicit diagnostic alias from `style_gain` to intent realization only.

Non-scope:
- rollout capture, returns, priority, diagnostics, evaluator, PPO, HSL,
  checkpoint, or live run.

Owner module:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`.

Expected evidence:
- S1 T-value/T-sign/T-noop/T-root-exclusion/T-invariant;
- q29 improvements change intent gain in the expected direction;
- root translation/orientation changes alone cannot change intent fidelity;
- equal Noisy/Repair execution produces zero intent gain rather than a no-op
  bonus; physics and repair cost retain their independently named meaning.

Stop condition:
- a full Clean global body/root term remains in the active formula; direct
  Repair-vs-Noisy similarity becomes intent fidelity; or a missing component is
  silently zero-filled.

### Step 3B / 12: Gain-to-Return and Priority Connectivity

Gate: G3. Depends on: 2B and 3A.

Objective:
- route the sole v003 Gain owner through local capture, stored return evidence,
  and replay-priority evidence.

Scope:
- q29 execution/intention capture; Gain invocation; return/advantage input;
  priority evidence; source/provenance diagnostics necessary to fail closed.
- the candidate-only q29 fidelity anchor is post-`t` robot q29 against
  `I_s[t]`; H remains actor context and C/K remain GMT executable evidence,
  not a substitute q29 intent target.

Non-scope:
- periodic or sequence evaluation UI/reporting, grouped reduction, formal
  runner, checkpointing, HSL, or live run.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py`.

Expected evidence:
- S1 T-consumer/T-provenance/T-no-v002-fallback/T-priority-isolation;
- a deterministic capture proves Noisy and Repair are both compared to the same
  q29 intent, not to each other or to global Clean motion;
- stored return and priority use the same decomposition and do not add actor
  loss mass.

Stop condition:
- any active return, difficulty, or priority fallback calls the former
  Clean-global/RP score; or q29 intent provenance is absent at the consumer.

Bounded implementation status:

- completed at candidate-only deterministic S1 (`E-FI-11`): post-`t` robot q29
  and sealed deployment/Noisy `I_s[t]` reach the sole v003 owner, which emits
  one return/advantage carrier and immutable scenario-keyed priority evidence;
- legacy `compute_segment_gain()`, Clean-global capture, `to_ppo_batch()`, sampler-state
  mutation, PPO loss, optimizer, checkpoint, and formal-route access remain excluded.

### Step 3C / 12: Diagnostics and Evaluation Isolation

Gate: G3. Depends on: 3B.

Objective:
- expose the v003 decomposition in diagnostics and make local-K evaluation and
  full-sequence composition evaluation explicitly separate consumers.

Scope:
- local evaluator/periodic evaluator, diagnostic fields, and composition
  evaluator protocol/isolation assertions.

Non-scope:
- changing local return/priority after Step 3B; grouped PPO, formal runner,
  checkpointing, HSL, or live sequence evaluation.

Owner modules:
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py` as the sealed
  candidate-evidence upstream, not an evaluator owner.

Expected evidence:
- S1 T-diagnostic/T-evaluator/T-no-zero-fill/T-composition-isolation;
- local reports contain q29 intent/physics/cost/source fields; composition
  records a separate protocol and cannot mutate return or replay priority.

Stop condition:
- a local evaluator reports global Clean Style as active Gain; a composition
  result flows into PPO/priority; or diagnostics silently substitute zeros.

White-box implementation split:

- `frontres_segment_diagnostics.py` owns the pure candidate-only local-K report
  and its v003 formatter. It reads the sealed Step 3B carrier only and cannot
  create or mutate return/priority/PPO state.
- `frontres_segment_live_training.py` keeps the existing periodic/offline/
  sequence evaluators explicitly legacy: each rejects the v015 layout before
  sampling, reset, legacy rollout, or `FRS-GAIN-v002` capture.
- The same diagnostics owner exposes a distinct composition protocol object.
  It records deployment-stream identity and frame/action counts but contains no
  local return, replay-priority, or PPO feedback field. A deterministic
  two-role carrier test proves both consumer isolation boundaries.

Bounded implementation status:

- completed at candidate-only deterministic S1 (`E-FI-12`): a v003 local-K
  report projects sealed Step 3B carrier facts without recomputation or
  feedback; legacy v002 periodic/offline/sequence evaluators fail closed for
  v015 layouts; and the composition protocol is typed separately.
- no formal periodic evaluator, sequence executor, simulator, checkpoint, PPO,
  sampler mutation, or live evaluation route was invoked or connected.

### Step 4A / 12: Sealed Transaction Metadata and Grouped Candidate Adapter

Gate: G4. Depends on: 2B and 3B.

Objective:
- rebase the existing one-row v003 candidate path onto the v015 local scenario
  identity without changing grouped-loss mathematics or actor mass.

Scope:
- row-aligned sealed metadata, one-row storage validation, scenario/hash
  meaning, candidate adapter, and grouped-loss preconditions.

Non-scope:
- formal `on_policy_runner` hookup, optimizer invocation, checkpoint/resume,
  HSL, alternative PPO objective, best-of-M selection, or live run.

Owner modules:
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`;
- `source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py`;
- transaction metadata/accumulator owner in
  `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`.

Expected evidence:
- S1 T-schema/T-row/T-metadata/T-permute/T-scale/T-legacy-reject;
- each ordinary-valid policy attempt carries one row and a single scenario
  identity; K/evidence steps never alter represented mass;
- `to_ppo_batch()` is explicitly legacy for grouped mode, while the metadata
  candidate adapter fails closed on missing/mixed local identities.

Stop condition:
- `noisy_segment_hash` still means a whole Noisy K tape; metadata is dropped
  before grouped loss; or K/M/best-of-M/priority multiplies actor mass.

White-box implementation split:

- `frontres_segment_storage.py` will own the immutable v015 candidate metadata
  schema and the one-row storage construction. Its local hash meaning is
  scenario/hash/x_t/q29 provenance/Coverage, not a whole Noisy K tape.
- `frontres_segment_live_probe.py` will own only the candidate connector from
  sealed Step 3B evidence into that storage schema and then the grouped
  candidate adapter. It must not select, update, or re-sample a policy.
- `frontres_segment_ppo.py` remains the grouped reduction owner. Step 4A may
  exercise it with the new candidate batch but must not alter its formula or
  actor mass.
- `to_ppo_batch()` remains legacy. It must reject a v015 metadata carrier
  rather than silently dropping it; only `to_grouped_ppo_candidate_batch()` may
  materialize the v015 candidate batch.

Bounded implementation status:

- completed at candidate-only deterministic S1 (`E-FI-13`): v015 metadata
  binds one Repair policy row to transaction/snapshot/motion/start/Segment/
  source/trial plus scenario/hash/x_t/q29/K/evidence-step identity; the grouped
  candidate adapter requires a complete transaction and the legacy adapter
  rejects it.
- grouped-loss mathematics and actor mass are unchanged; row permutation and
  K/evidence-step mutation are tested as mass-invariant.
- no formal runner, optimizer, persistence, simulator, training, or live
  boundary was connected or invoked.

### Step 4B / 12: Formal Stage-3 Route and Exact-One Update

Gate: G4. Depends on: 4A and H0 for any route that could invoke warmup.

Objective:
- connect a complete v015 transaction through a dedicated formal boundary once,
  isolating the legacy storage adapter and proving one optimizer update only
  after all selected Segments and M attempts are sealed. Step 4B-S2 is a
  CPU-only fake formal-route proof, not permission to launch the generic
  training entrypoint or a simulator.

Scope:
- pure sealed-transaction plan -> public `OnPolicyRunner` connector -> dedicated
  v015 update-loop connector -> live-probe formal-update owner -> storage ->
  grouped candidate adapter -> unchanged `FrontRESSegmentPPO` loss -> exactly
  one optimizer step;
- explicit q29 actor-tail routing before normalizer, v015 config isolation,
  transaction/update diagnostics, and legacy-route rejection;
- a fake injected request provider only. `scripts/rsl_rl/train.py` and the
  legacy live-training loop remain non-dispatching in this step.

Non-scope:
- changing grouped formula, HSL target migration, checkpoint/resume, simulator
  execution, long training, or composition evaluation.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_update_loop.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`;
- `source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py`;
- `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py` and
  `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py` as fail-closed connectors.

Expected evidence:
- S2 T-connect/T-order/T-exact-one-update/T-no-legacy-route/T-diagnostic;
- a fake formal runner proves a complete multi-Segment × M transaction produces
  one grouped update, while immediate/partial updates and `to_ppo_batch()` use
  are rejected;
- output identifies transaction, snapshot, motion, Segment, attempt, K,
  valid-row count, group mass, scale, scenario hash, q29 provenance, and update
  delta;
- the fake route proves q29 tail construction occurs before normalizer use and
  that HSL warmup, non-grouped normalization, and any partial/mixed transaction
  fail before an optimizer step.

Stop condition:
- legacy flat-batch adapter reaches grouped mode; old warmup runs implicitly;
  optimizer steps during collection; or the formal route admits partial/mixed
  transactions;
- a generic training entrypoint silently dispatches this fake-only connector,
  Clean continuation reaches the actor tail, or the path needs simulator/reset
  execution to establish its S2 claim.

Bounded implementation status:

- completed 2026-07-20 at CPU-only fake S2 (`E-FI-14`):
  `FrontRESV015FormalTransactionPlan` and its accumulator seal every planned
  multi-Segment x M candidate shard, the public runner connector reaches a
  dedicated fake-only update-loop/probe owner, the unchanged v003 grouped loss
  runs once, and one explicit optimizer counter increment is required;
- q29 actor-tail routing is selected before normalization and cannot concatenate
  the old 65D fixed tape in v015 mode; HSL/warmup, non-grouped normalization,
  partial/mixed metadata, and legacy `to_ppo_batch()` reject before a step;
- generic `train.py` / live-training dispatch, checkpoint/resume, simulator,
  real training, and live evidence remain outside this completed S2 step.

### Step 4C / 12: Layout and Transaction Persistence

Gate: G4. Depends on: 4B. Stage-3 persistence is complete; a new Stage-1 HSL
checkpoint identity remains a separate, undefined boundary.

Objective:
- version the v015 future-intent layout and transaction boundary so resume
  cannot silently reinterpret an old actor prefix or resume a partial frozen
  transaction.

Scope:
- checkpoint metadata/version checks, normalizer layout identity, transaction
  atomicity at save/resume, and diagnostic persistence fields.

Non-scope:
- a new HSL supervision target, backward-compatible automatic conversion of
  old 65D statistics, optimizer/objective changes, simulator, or live run.

Owner modules:
- `source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py`;
- `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_runtime.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_update_loop.py`, which
  opens the in-flight persistence barrier before the injected provider;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`, which binds,
  seals, and commits the metadata-only receipt around the sole update.

Expected evidence:
- S3 T-checkpoint/T-resume/T-layout/T-atomicity;
- checkpoint declares future-intent layout/version and grouped-loss identity;
- incompatible old normalizer/layout rejects rather than adapts silently;
- save/resume at a partial transaction fails closed or is explicitly forbidden.

Stop condition:
- an old `[H,65]` checkpoint loads as v015; a resume changes transaction
  identity; or checkpointing permits a partial collection to update later.

Bounded implementation status:

- completed 2026-07-20 at CPU-only fake S3 (`E-FI-15`): the checkpoint owner
  persists the exact q29 H layout, prefix-normalizer fingerprint, grouped-loss
  identity, and only `idle` or committed exact-one-update receipt;
- an in-flight `collecting`/`sealed`/`failed` transaction fails closed on save
  and resume, and a valid committed resume returns to `idle` without recreating
  a provider request, candidate batch, or raw local scenario;
- generic training entry, real checkpoint cadence/resume, simulator, training,
  and live runtime remain out of scope and unproven.

### Step 5A / 12: User-Gated Local Live Identity Sentinel

Gate: G5. Depends on: 4C and explicit user authorization.

Status: completed at bounded S4 (`E-FI-27`). This closes the dedicated local
identity/formal-update route only; it does not authorize long training or
deployment-composition evaluation.

#### Step 5A-S0: Pre-Live Formal Sentinel Connectivity

Status: completed. `E-FI-16` proves config/entrypoint isolation and the sealed
fake transaction; `E-FI-23` closes the unmocked offline observation connector;
`E-FI-24` adds the structured live snapshot; `E-FI-27` confirms the same
dedicated owner chain in the bounded live sentinel.

Objective:
- connect the already sealed v015 objects through one dedicated, opt-in
  sentinel path: selected local scenario -> two-role Clean reset -> q29 actor
  input -> one action/frozen-GMT K capture -> v003 candidate adapter -> sealed
  grouped transaction -> existing exact-one update owner -> identity telemetry.

Scope:
- an explicit v015 sentinel config/entrypoint which is mutually exclusive with
  legacy sentinel/probe/update modes and requires explicit q29 H offsets;
- selection-time local-scenario materialization instead of the legacy 65D tape
  for that route, plus local-carrier fields on the index-reset request;
- a dedicated provider which reuses the existing frozen snapshot, one-action K,
  Gain, grouped-candidate, and Step 4B/4C transaction owners without altering
  their formulas or persistence rules;
- deterministic fake-route tests and structured telemetry for scenario, x_t,
  root-artifact, q29, continuation, role, K, policy-row, group-mass, and update
  identities.

Non-scope:
- invoking the new route against IsaacLab or any real environment; long
  training, sampler-state learning claims, HSL changes, grouped-PPO/Gain
  formula changes, checkpoint/resume changes, deployment composition, or any
  change to the Concept Figure.

Owner split and evidence:
- S0a config/dispatch: `rsl_rl_cfg.py`, `frontres_unified.py`,
  `frontres_segment_runner_boundary.py`, `on_policy_runner.py`, and
  `scripts/rsl_rl/train.py`; S2 T-config/T-entrypoint/T-legacy-isolation.
- S0b local transaction provider: `frontres_segment_live_sampler.py`,
  `frontres_segment_live_probe.py`, and `frontres_segment_storage.py`; S2
  T-connect/T-state/T-provenance/T-frozen/T-order/T-mass using a semantic fake
  environment only.
- S0c formal observation route: completed by `E-FI-23` and runtime-confirmed by
  `E-FI-27`; it executes the real command property, `870D` policy observation,
  role-aligned q29 append, normalization, and FEMR/GMT split without replacing
  `_read_live_observations()`.

Stop condition:
- any legacy fixed tape, legacy `to_ppo_batch()`, immediate-update path, HSL
  writer, Clean actor input, later FEMR action, mixed scenario identity, or
  update count other than one becomes reachable; or the required explicit H
  offsets cannot be represented without a new method decision.

#### Step 5A-S1: Bounded Local Runtime Trace

Gate: explicit user confirmation after the actual command and preflight are
reported. Depends on: completed Step 5A-S0.

Status: completed on SUST_Main_2 at S4 (`E-FI-27`). The successful log identity
is `v015_r6_live_sentinel_gpu3.log` with SHA-256
`d67ed9327d8166ef7617b61f1cd746ee1f4b94710277b28cff6a825b6483f15b`.
It records `4 Repair + 4 Noisy` physical rows, two Segment sources with M=2,
one actor action, eight valid Clean-C GMT steps, four grouped policy rows, and
one optimizer update.

Objective:
- obtain the smallest real-environment trace proving one v015 local scenario
  reaches the active formal path without a provenance or lifecycle violation.

Scope:
- one bounded local transaction/sentinel and its raw log/evidence extraction.

Non-scope:
- long training, policy-quality claims, reward tuning, HSL redesign, or full
  sequence composition evaluation.

Expected evidence:
- S4 T-live/T-state/T-provenance/T-frozen/T-order/T-mass;
- log records `x_t`, current artifact, q29 intent source/invariant, Clean
  continuation identity, role counts, action count, K, scenario hash, grouped
  mass, and exactly one optimizer update.

Stop condition:
- any required identity is absent; a later FEMR action occurs; a Clean actor
  input appears; or the update count differs from one.

## R0--R6 Formal Observation Remediation

This sequence supersedes the earlier assumption that Step 1B and Step 5A-S0
already covered the formal observation route. It does not change v015 method
semantics or the Concept Figure.

### R0 / 7: Formal Observation Contract Freeze

Status: completed at read-only S0 evidence (`E-FI-18`).

Objective:
- freeze `870D raw + 58D H-tail = 928D`, with FEMR consuming the first `158D`
  and frozen GMT consuming the original final `770D`.

Scope:
- source/log audit and plan/checklist/evidence correction only.

Non-scope:
- Concept Figure, active-contract semantics, training source, tests, simulator,
  checkpoint execution, or live rerun.

Stop condition:
- current GMT command cannot be sourced from the selected deployment carrier at
  `t` without Clean C, future root/global data, or a new privileged field.

### R1 / 7: Current-Frame GMT Command Route

Status: completed at deterministic S1 evidence (`E-FI-19`).

Objective:
- let an active local scenario construct exactly one current GMT command at `t`
  before the one FEMR action.

Scope:
- `MultiMotionCommand.command` / `_gather_future_by_motion()` current-frame
  branch and its deterministic command-owner contract test.

Non-scope:
- q29 H-tail append, actor visibility, normalizer, K continuation, Gain, PPO,
  checkpoint, formal runner, simulator, or live run.

Owner and shape:
- unique owner: `commands.py::MultiMotionCommand`;
- input: role-aligned `env_motion_indices[B]`, `time_steps[B]`, local scenario
  identity, and `motion_horizon=1`;
- output: deployment-carrier `q29_t + dq29_t` as `[B,1,58]`, flattened by the
  command property to `[B,58]`; IsaacLab history remains a later integration
  consumer that produces the existing `[B,290]` policy term.

Expected evidence:
- S1 T-current-command/T-shape/T-provenance/T-role-identity/
  T-current-only/T-continuation-isolation/T-legacy-reject.

Stop condition:
- `motion_horizon != 1`; any row reads `C[t+1:t+K]`, future root/global, or
  Clean actor data; same-scenario role rows disagree; or the fix weakens the
  explicit K-execution gate.

### R2 / 7: Role-Aligned q29 H Bridge

Status: completed at deterministic S1 evidence (`E-FI-20`).

Objective:
- read the sealed command carrier as `[B,H+1,29]` and construct the positive
  offsets as `[B,58]` for all Repair/Noisy rows.

Scope:
- command snapshot accessor plus `frontres_runtime.py` actor-tail connector.

Non-scope:
- actor/GMT visibility split, checkpoint, formal runner, simulator, or live.

Expected evidence:
- S1 T-role-expand/T-offset/T-permute/T-no-root/T-no-Clean/T-no-C.

Stop condition:
- the bridge still reads the policy-attempt batch or cannot preserve one sealed
  scenario across all role rows.

### R3 / 7: FEMR 158D / GMT 770D Authority Split

Status: completed at deterministic S1 evidence (`E-FI-21`).

Objective:
- make the combined `[B,928]` observation expose only `[B,158]` to FEMR while
  preserving the frozen GMT `[B,770]` suffix and checkpoint.

Scope:
- v015 config, runner layout resolution, actor prefix assertion, and frozen-GMT
  consumer isolation.

Non-scope:
- command provenance, K execution, loss formulas, checkpoint persistence, or
  live run.

Expected evidence:
- S1 T-928-layout/T-158-actor/T-770-GMT/T-num-frontres-zero-reject/
  T-frozen-GMT-isolation.

Stop condition:
- FEMR can consume any GMT-only suffix field, GMT input changes from `770D`, or
  the old GMT checkpoint requires reshaping.

### R4 / 7: Layout Persistence Revalidation

Status: completed at deterministic S3 evidence (`E-FI-22`).

Objective:
- revalidate v015 checkpoint/normalizer identity against `H=(1,2)`, prefix
  `158D`, suffix `770D`, and the exact prefix-stat fingerprint.

Scope:
- deterministic S3 checkpoint/resume contracts only.

Non-scope:
- real checkpoint cadence/resume, simulator, training, or live run.

Expected evidence:
- S3 T-layout/T-prefix-stats/T-legacy-zero-reject/T-atomicity-regression.

Stop condition:
- a full-`928D`, `num_frontres_obs=0`, legacy 65D, or unversioned layout loads
  as v015.

### R5 / 7: Unmocked Offline Formal Observation Connectivity

Status: completed at deterministic offline S2 evidence (`E-FI-23`).

Objective:
- execute command -> 870D observation -> q29 append -> normalization ->
  FEMR/GMT split -> one-action K -> grouped exact-one update without stubbing
  `_read_live_observations()`.

Scope:
- semantic CPU fake with the real command/observation connector and structured
  boundary trace.

Non-scope:
- simulator physics, long training, policy quality, or deployment composition.

Expected evidence:
- S2 T-command-connect/T-history-layout/T-role-tail/T-normalizer/T-consumer/
  T-one-action/T-exact-one-update.

Stop condition:
- any observation owner is bypassed, a weak shape-only fake substitutes for the
  current command, or update count differs from one.

### R6 / 7: Bounded Live Identity Sentinel

Status: completed at bounded S4 (`E-FI-27`). The final rerun preserves the
`[4,928]` actor observation and row-aligned `[4,289]` critic observation,
executes all K=8 evidence steps, seals four equal-mass policy attempts, and
publishes `optimizer_step_delta=1` / `exact_one_update=true`.

R6-F1 command-clock repair contract:

- Status: completed at deterministic S1 (`E-FI-25`).

- Objective: isolate the sealed local-scenario clock from the legacy automatic
  `time_steps/reference/cache` clock during both the unique t transition and
  every explicit Clean-C K transition.
- Scope: `MultiMotionCommand` clock dispatch plus one deterministic lifecycle
  regression in `frontres_v015_current_gmt_command_contract.py`.
- Non-scope: reset semantics, q29/FEMR/GMT authority, K cursor definition,
  Gain, storage, PPO, checkpoint, HSL, formal runner, simulator, or live run.
- Expected evidence: S1 T-t-clock-hold/T-K-clock-hold/T-legacy-clock/
  T-duplicate-refresh-reject.
- Stop: local `_update_command` changes `time_steps`, current artifact, or C
  cursor; legacy rows stop advancing; or the direct duplicate-refresh guard is
  removed/bypassed.

R6-F2 critic-observation route contract:

- Status: completed at deterministic S1/S2 (`E-FI-26`) and runtime-confirmed
  at bounded S4 (`E-FI-27`).
- Objective: preserve the t policy tuple's real privileged/critic observation
  from role selection through one-action evidence and candidate storage, then
  feed it to the frozen critic during the sealed formal evaluation.
- Scope: `frontres_segment_live_probe.py` collection/request/evaluator boundary,
  `frontres_segment_storage.py` evidence/storage carrier, and deterministic
  regression contracts.
- Non-scope: actor/GMT observation layout, q29 values or provenance, K/Gain,
  grouped PPO formula, checkpoint, HSL, simulator, training, or live execution.
- Core parameter path: t `transition.privileged_observations [8,289]` -> Repair
  role selection `[4,289]` -> immutable one-action evidence -> candidate
  storage -> sealed formal request -> critic; actor evaluation remains
  `[4,928]`.
- Expected evidence: S1/S2 T-critic-route/T-role-order/T-missing-reject/
  T-shape-reject/T-exact-one-update.
- Stop: the t transition lacks a real critic observation; critic rows cannot be
  ordered identically to the sealed candidate rows; actor observations still
  reach the critic; or any optimizer step occurs before the full transaction.

Objective:
- run exactly one SUST_Main_2 transaction with a structured observation and
  identity snapshot.

Scope:
- one `2 Segment x 2 attempt` local transaction and evidence extraction.

Non-scope:
- long training, tuning, checkpoint migration, or deployment composition.

Expected evidence:
- S4 trace contains `870/58/928/158/770`, role identities, q29 provenance,
  scenario hash, one action, K, group mass, and update delta one.

Stop condition:
- any dimension/identity is absent, FEMR sees GMT-only fields, GMT is not
  `770D`, a later FEMR action appears, or update delta is not one.

Acceptance:
- no stop condition triggered in `E-FI-27`; R6 and Step 5A are closed.

### Step 5B / 12: User-Gated Deployment Composition Evaluation

Gate: G5. Depends on: 3C and 5A, with separate explicit user authorization.

Objective:
- evaluate whether repeated deployment-mode repairs compose over a full
  artifact-bearing reference stream without changing the local first-action
  credit assignment.

Scope:
- separately named sequence evaluator, persistent-artifact protocol, and
  non-feedback evidence report.

Non-scope:
- feeding later artifacts into local K return, PPO eligibility, replay priority,
  or changing the local Gain definition.

Expected evidence:
- S4 T-composition/T-isolation/T-protocol;
- report lists per-frame action use, corruption protocol, q29 intent and
  physics results, and accumulated failure statistics separately from local K
  evidence.

Stop condition:
- composition metrics enter training feedback; the report is used to claim
  local-return validity; or the protocol lacks deployment reference provenance.

Bounded substeps:
- `5B-S0` Formal Route Audit: freeze the owner, shapes, legacy isolation, and
  S1/S2/S4 evidence gates without modifying code or running the simulator.
- `5B-S1` Immutable Request/Report Kernel: in
  `frontres_segment_sequence_eval.py`, validate an explicit deployment `.npz`,
  seal its file identity and persistent-corruption protocol identity, define a
  per-frame composition report with no training-feedback fields, and reject
  legacy/mixed configuration. This is deterministic CPU-only schema evidence.
- `5B-S2A` Deployment Carrier And H Snapshot: install one immutable request in
  the command owner and expose only the current deployment command plus
  `[B,H+1,29]` q29 intent to the runtime bridge. It does not sample FEMR, step
  GMT, aggregate metrics, or touch the formal runner.
- `5B-S2B` Formal Composition Executor: connect the verified S2A carrier
  through per-frame FEMR, frozen GMT, sequence metrics, immutable report, and
  the dedicated config/runner entry. It must prove zero sampler/storage/PPO/
  optimizer mutation with an offline semantic fixture before any live run.
- `5B-S4-S0` Dedicated Live Composition Entrypoint: expose one v015-only
  server CLI that binds the registered IsaacLab task, explicit frozen-GMT and
  v015 FEMR checkpoints, one pre-materialized deployment `.npz`, CUDA-visible
  device identity, and one absolute report path to the S2B owner. It must prove
  config/dispatch isolation without launching IsaacLab.
- `5B-S4` Bounded Live Composition: run one explicitly identified deployment
  stream only after S2 passes, then record per-frame action use, corruption
  protocol, intent/physics metrics, and accumulated failures.

`5B-S1` non-scope:
- command/reset/actor/GMT execution, local scenario or Clean continuation;
- sampler, return, priority, storage, PPO, optimizer, checkpoint, simulator,
  training, or live evaluation;
- legacy sequence evaluator migration or Concept Figure changes.

`5B-S1` evidence:
- S1 `T-npz-schema/T-identity/T-corruption-protocol/T-report/T-no-feedback/T-config-fail-closed/T-legacy-reject`.

`5B-S1` stop condition:
- the request accepts a non-`.npz` or malformed deployment stream; corruption
  identity depends on mutable parameter order; report lengths/counts disagree;
  or any local return, priority, sampler, PPO, optimizer, Clean continuation,
  or local-scenario payload can enter the S1 owner.

`5B-S1` acceptance:
- completed at `E-FI-28`; the dedicated S1 owner and its semantic CPU contract
  pass, while the legacy v002 sequence contract and v015 legacy-isolation
  contract remain unchanged.

`5B-S2A` scope:
- `commands.py::MultiMotionCommand` installs one validated E-FI-28 request as
  an immutable command-owned q29/dq29 sequence and explicit frame cursor;
- `frontres_runtime.py` reads a cloned current `[B,58]` q29+dq29 command and
  dense `[B,H+1,29]` deployment q29 intent with row-aligned reference/protocol
  identity and provenance.

`5B-S2A` non-scope:
- local scenario, Clean continuation, Segment sampler, actor/GMT execution,
  metrics/report production, formal config/runner, storage, return, priority,
  PPO, optimizer, simulator, training, or live evaluation.

`5B-S2A` evidence and stop condition:
- S1/S2 semantic CPU
  `T-install/T-current/T-H/T-frame-order/T-cursor/T-boundary/T-row-alignment/T-identity/T-provenance/T-mixed-reference/T-no-execution/T-no-training-state`;
- stop if installation accepts a changed file hash or mixed carrier, a read
  advances the cursor, an H read clamps at sequence end, or the carrier reaches
  actor/GMT/training state before S2B.

`5B-S2A` acceptance:
- completed at `E-FI-29`; the current/H carrier and read-only runtime connector
  pass deterministic tests, while actor/GMT/formal execution remains absent.

`5B-S2B` scope:
- `frontres_segment_sequence_eval.py` owns one typed run config and the formal
  per-frame executor; the input `.npz` must identify itself as the already
  materialized deployment stream, so S2B never draws or invents corruption;
- for each unclamped frame `t in [0, T-max(H))`, command current q29/dq29/body
  reference and q29 H produce `870+58=928`, one deterministic full-6D FEMR
  correction, one frozen-GMT motor action, one environment step, and one
  deployable intent/physics metric row;
- the immutable report is written atomically to an explicit JSON identity and
  a before/after fingerprint proves optimizer, sampler, storage, and transition
  state did not change.

`5B-S2B` non-scope:
- corruption generation/resampling, local scenario, Clean continuation, Gain,
  return, priority, PPO, checkpoint, CLI, simulator, training, live evaluation,
  or Concept Figure changes.

`5B-S2B` evidence and stop condition:
- S2 semantic CPU
  `T-connect/T-per-frame/T-frozen-GMT/T-report/T-zero-write/T-formal-entry/T-legacy-isolation`;
- stop if tail frames are clamped or fabricated, actor H includes future
  root/global data, GMT is trainable, action/cursor/report counts disagree,
  legacy sequence evaluation is called, or any training fingerprint changes.

`5B-S2B` acceptance:
- completed at `E-FI-30`; `T=6,Hmax=2` yields exactly four FEMR actions, four
  frozen-GMT reads and four report rows, with atomic JSON and zero forbidden
  writes. Physical metric ownership and simulator timing remain S4-only.

`5B-S4-S0` scope and non-scope:
- `scripts/rsl_rl/frontres_v015_deployment_composition.py` is the only live
  CLI. It accepts only `FrontRES-Unified-Tracking-Flat-G1-v0`, absolute
  checkpoint/reference/report identities, explicit persistent-corruption
  metadata, and a CUDA device selected through `CUDA_VISIBLE_DEVICES`;
- it configures the existing v015 H/checkpoint identity while every Segment
  replay/live-train/HSL/legacy-eval flag remains false, constructs the formal
  IsaacLab runner, installs GMT before construction, loads FEMR with
  `load_optimizer=False, load_critic=False`, and dispatches only S2B;
- no evaluator formula, observation authority, sampler, PPO, optimizer,
  checkpoint format, simulator execution, training, live run, or Concept
  Figure change belongs to this substep.

`5B-S4-S0` evidence, stop condition, and acceptance:
- S2 deterministic `T-path/T-gpu/T-protocol/T-config/T-dispatch/T-zero-update/
  T-owner/T-formal-entry/T-no-training` plus S2B, observation-authority,
  checkpoint, local-sentinel-config, and runner-boundary regressions;
- stop if the registered task/checkpoint owner is ambiguous, the config
  requests Segment Replay and creates a sampler, checkpoint loading restores
  optimizer state, or the CLI can call learn/update/legacy sequence evaluation;
- completed at `E-FI-31` as config/dispatch isolation evidence only. No
  simulator was imported or launched. `E-FI-32` reclassifies the CLI as
  implemented-not-runnable: the compatible v015 checkpoint is a later training
  output, and an external pre-materialized Noisy `.npz` is not a user input.

### Post-Observation-Change Test-Path Rebase: G0--G7

Confirmed blockers:
- no trained FEMR checkpoint exists for the new `928D actor / 158D FEMR /
  770D GMT` layout;
- no external pre-materialized `Noisy.npz` exists or belongs to the accepted
  user-facing workflow;
- the S4-S0 CLI is implemented but cannot yet produce scientific composition
  evidence;
- the current S2B/CLI pre-materialized-file assumption is a code/contract
  alignment gap, not an instruction for the user to invent a file.

Dependency rule:
- a compatible `FRONTRES_CKPT` is produced only after new-layout HSL,
  Stage-3 training, checkpoint save, and fresh-runner reload succeed;
- controlled evaluation starts from an ordinary Clean/reference `.npz` and
  materializes one immutable persistent-artifact carrier at selection time;
- full-sequence usefulness is paired No-FEMR/GMT versus FEMR/GMT under the
  same carrier and initial conditions.

#### G0 / 7: Test-Path And Document Rebase

Objective: remove the false ready-to-run S4 dependency and freeze the corrected
training/evaluation order.

Scope: active evaluation contract, plan, checklist, canvas, evidence,
Architecture, and testing views only.

Non-scope: source code, Concept Figure, simulator, training, checkpoint IO, or
artifact generation.

Evidence: S0 T-doc/T-dependency/T-status at `E-FI-32`.

Stop: any document still calls an external Noisy `.npz` a required user input,
calls an untrained/missing checkpoint available, or marks S4 runnable.

#### G1 / 7: v015 Training Readiness Audit

Objective: white-box the complete new-layout route:
`q29 H -> proposal-only HSL -> HSL persistence -> Stage-3 load -> grouped PPO
-> v015 save -> fresh inference reload`.

Scope: read-only config, runner, HSL target/input, normalizer, storage/loss,
optimizer, checkpoint, and formal training entrypoint ownership.

Evidence: S0 T-owner/T-layout/T-checkpoint/T-train-dispatch/T-stop.

Stop: new HSL persistence is undefined, old observation/layout can enter, the
formal train branch cannot save/reload exact v015 identity, or any claimed
owner remains unconfirmed.

Status: completed as a stopped S0 audit at `E-FI-33`. The audit confirmed all
four code gaps below; it did not authorize source changes or training.

| Formal boundary | Confirmed current gap | Required retirement / connection |
| --- | --- | --- |
| Stage-1 observation | the formal HSL preset does not request the v015 q29 layout, so the legacy `870D` route remains reachable | G2 must make `870D + 58D -> 928D`, FEMR `158D`, and GMT `770D` mandatory for proposal-only HSL |
| HSL persistence | `model_warmup.pt` uses generic persistence and no accepted proposal-only HSL identity exists | G2 must define a separate versioned HSL envelope and reject legacy/unversioned layout or normalizer state before mutation |
| Stage-3 update dispatch | ordinary Stage-3 training calls the legacy sampler/update loop rather than the sealed grouped transaction owner | G3 must migrate actor-side initialization and make the complete sealed transaction the only formal grouped-update dispatch |
| Stage-3 checkpoint production | ordinary training cannot save and fresh-reload an exact v015 identity; the sentinel identity path is not a training checkpoint producer | G3 must connect committed transaction state to the exact v015 save owner and a fresh inference reload |

#### G2 / 7: New-Layout HSL Persistence And Smoke

Objective: define the new proposal-only HSL checkpoint identity, connect the
formal Stage-1 preset to the mandatory q29 v015 layout, save/reload the HSL
actor and layout-aware normalizer state, then run one bounded Stage-1 smoke
that proves current anti-DR proposal identity.

Scope: `scripts/rsl_rl/train.py` Stage-1 preset, runner layout resolution,
proposal-only HSL input/target owner, HSL-specific persistence identity and
pre-mutation validation, actor/prefix normalizer save/reload, and the smallest
deterministic plus bounded-smoke entry required to prove the route.

Non-scope: Stage-3 actor migration, Segment sampling, grouped PPO/update,
Stage-3 checkpoint production, long HSL training, carrier materialization,
composition evaluation, and Concept Figure changes.

Evidence: S0/S1/S3/S4 T-owner/T-formal-layout/T-HSL-input/T-current-target/
T-identity/T-save/T-reload/T-legacy-reject/T-live-smoke.

Stop: the formal Stage-1 route can still construct a legacy `870D` actor
observation; q29 lacks deployment/Noisy provenance; Clean future/label reaches
the actor or target; the new HSL identity is not unique; an old/unversioned
checkpoint loads; prefix-normalizer identity is incomplete; or reload changes
the proposal output/layout.

##### G2-S0: Proposal-Only HSL Persistence Contract Freeze

Status: carrier decision confirmed by the user. The minimal Stage-1-only
carrier is implemented in existing owners at deterministic S1 (`E-FI-35`);
no source or test module was created. The formal layout and actor-only warmup
are completed at deterministic S1 (`E-FI-36`); strict HSL persistence is
completed at deterministic S3 (`E-FI-37`); fresh-runner connectivity is
completed at deterministic S2/S3 (`E-FI-38`); the bounded telemetry/reload
connector is completed at deterministic S1/S3 (`E-FI-39`). After the
diagnostic and cross-device verifier repairs at `E-FI-40` and `E-FI-41`, the
bounded live smoke closed G2 at S4 (`E-FI-42`). G3-S0 is complete, G3-S1A
closes actor-only Stage-3 migration at `E-FI-43`, G3-S1B closes ordinary formal
transaction/commit-only-save dispatch at `E-FI-44`, and G3-S2 closes the exact
save-to-fresh-inference chain at `E-FI-45`. G3 is complete at offline S2/S3;
G4 closes the controlled carrier materializer at `E-FI-46`; G5 is next and
separately user-gated.

The white-box audit confirmed two additional pre-implementation gaps beyond
the four G1 findings:

- `prepare_frontres_hsl_actor_observation()` can read q29 only through
  `MultiMotionCommand.frontres_local_scenario_intent_snapshot()`, but the
  formal Stage-1 HSL route never materializes or installs an active local
  scenario;
- `run_frontres_joint_warmup()` currently builds an executable-energy target,
  optimizes both `residual_actor` and `critic`, and saves both through the
  generic runner checkpoint. This conflicts with v007's actor-only current
  anti-DR proposal boundary.

The user selected the Stage-1-only immutable proposal carrier:

```text
proposal_context = {
  current_root_artifact_identity,
  intent_q29[t:t+H],
  deployment_noisy_provenance,
  proposal_context_id
}
```

It is command-owned and sealed for one HSL sample, but contains no `x_t`, Clean
continuation, K, Segment role, attempt, return, priority, or PPO state. The
rejected alternative was to reuse the complete Stage-3 local-scenario carrier
while leaving its Clean continuation and K fields unused. That would make
proposal initialization depend on Segment Replay objects HSL does not consume.
The clarification is recorded in v007 without creating another code module or
changing the Concept Figure.

Proposed persistence owner and schema, pending that decision:

```text
owner: frontres_checkpointing.py
key: frontres_v015_hsl_checkpoint_identity
format: frontres-v015-hsl-proposal-v1

identity:
  FRS-METHOD-v015 / FRS-TRAIN-v007
  objective = proposal_only_current_antidr_delta_se3
  H = (1, 2), intent = 29D, tail = 58D
  raw / actor / FEMR / GMT = 870 / 928 / 158 / 770
  action = full 6D Delta SE(3)
  GMT checkpoint artifact hash + frozen 770D normalizer identity
  complete 158D FEMR-prefix normalizer fingerprint

payload allowed for migration:
  residual_actor state
  action-distribution std or log_std
  158D prefix-normalizer state

payload forbidden:
  critic, critic normalizer, HSL optimizer, Segment sampler, transaction,
  Gain/return/priority/PPO state, Clean continuation, or rollout labels
```

The loader must validate the complete identity and exact payload key set before
mutating actor or normalizer state. Generic `load_runner()` compatibility,
first-layer reshaping, missing-prefix identity normalization, and legacy
`frontres_warmup_complete` fallback are forbidden.

Implementation is split to keep each parameter path independently testable:

1. `G2-S1a`: completed at `E-FI-35`. The existing command-owned HSL
   proposal carrier and q29 snapshot. Core path:
   `motion/frame + current artifact -> [B,H+1,29] + provenance -> immutable
   proposal_context_id`. Evidence: S1 T-carrier/T-shape/T-provenance/
   T-immutability/T-no-C-K. Clean/K/Segment mixing and future-frame clamping
   reject; the existing local-scenario route remains unchanged.
2. `G2-S1b`: completed at `E-FI-36`. Stage-1 preset -> runner
   `928/158/770` layout -> q29 normalizer -> actor-only current anti-DR HSL.
   The executable-energy target, critic optimizer membership, critic forward,
   and energy diagnostics were removed from the HSL owner. Evidence: S1
   T-config/T-layout/T-target/T-actor-only/T-critic-unchanged/T-legacy-reject.
3. `G2-S2`: completed at `E-FI-37`. The dedicated HSL
   identity/save/pre-mutation reload lives in
   `frontres_checkpointing.py`, with `frontres_warmup.py` as the only save
   connector. The strict three-field payload is actor/distribution/prefix-only;
   exact identity and all fingerprints validate before the first state write.
   Evidence: S3 T-schema/T-save/T-reload/T-tamper/T-GMT-identity/T-normalizer/
   T-forbidden-payload/T-unmutated-reject.
4. `G2-S3`: completed at `E-FI-38`. One offline fresh-runner fixture proves
   exact q29 -> combined 928D -> normalized 158D -> bounded 6D proposal
   equality after strict HSL reload. It uses two independently initialized
   runner objects and no legacy prefix padding, critic, optimizer, simulator,
   or training. Evidence: S2/S3 T-fresh-runner/T-output/T-layout/
   T-zero-state-leak.
5. `G2-S4-S0`: completed at `E-FI-39`. The existing Stage-1 owners now expose
   one explicit bounded-smoke flag, structured real-input/target/gradient/
   critic/checkpoint telemetry, and an independent pre-warmup CPU shadow that
   must differ before strict HSL-v1 reload and match normalized 158D input plus
   bounded 6D proposal exactly after reload. No new source module was created.
6. `G2-S4-S1`: completed at `E-FI-42`. The bounded Stage-1 live smoke used
   eight envs, one warmup iteration, one environment step, three actor epochs,
   and zero PPO iterations. S4 T-live-input/T-current-target/T-save/
   T-fresh-reload passed with no legacy fallback, shape drift, critic/optimizer
   payload, critic delta, reload mismatch, or PPO entry.
7. `G2-S4-S0a`: completed at `E-FI-40` after the first S4-S1 attempt exposed
   a stale diagnostics-only `_sup_mask` read. The obsolete partial-dimension
   masking fragment was deleted rather than restored; the existing HSL S1
   contract now rejects `_sup_mask` and `frontres_active_task_dims` anywhere in
   the proposal-only warmup owner. No live retry was executed in this repair.
8. `G2-S4-S0b`: completed at `E-FI-41` after the next S4-S1 attempt proved the
   normalized 158D input was bitwise equal but the live CUDA and CPU-shadow 6D
   proposals were not bitwise equal. Strict checkpoint fingerprints, strict
   actor load, and exact normalized input remain unchanged. Only the
   cross-device float32 forward comparison now uses `rtol=1e-5, atol=1e-6`,
   records bitwise status and maximum absolute error, and rejects errors beyond
   that bound. No live retry was executed in this repair.
9. `G2-S4-S1`: completed at live S4 `E-FI-42`. One Main-2 run observed real
   artifact/q29 provenance, exact 928/158/770 authority, current anti-DR target,
   nonzero actor-only gradient, zero critic gradient/delta, strict HSL-v1
   payload, CUDA/CPU reload error `2.79396772e-09`, `ppo_entered=0`, and the
   warmup-only exit before PPO. G2 is complete.

Exact Main-2 command after checkout synchronization:

```bash
cd /hdd1/cyx/FEMR
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 \
FEMR_LOG_ROOT=/hdd1/cyx/FEMR \
/hdd1/cyx/miniconda3/envs/mosaic/bin/python scripts/rsl_rl/train.py \
  --task=FrontRES-Unified-Tracking-Flat-G1-v0 --device=cuda:0 --num_envs=8 \
  --motion=/hdd1/cyx/AMASS_G1NPZ_Final --headless --logger=tensorboard \
  --experiment_name=g1_flat_frontres_stage1_hsl --run_name=G2_S4_BOUND_HSL \
  --max_iterations=0 --supervised_warmup_iterations=1 \
  --supervised_warmup_steps_per_iter=1 --supervised_warmup_max_envs_per_step=8 \
  --frontres_v015_future_offsets=1,2 --frontres_specialist_mode=rp \
  --frontres_hsl_live_smoke --frontres_stage=stage1_hsl \
  2>&1 | tee /hdd1/cyx/FEMR/v015_g2_s4_hsl_smoke_gpu3.log
```

Planned deterministic commands after implementation:

```text
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_hsl_formal_route_contract.py
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_v015_hsl_checkpoint_contract.py
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s1_contract.py
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_hsl_v007_s2_connectivity_contract.py
```

No command above is authorized or executed by G2-S0.

#### G3 / 7: Stage-3 Training And Checkpoint Smoke

Objective: migrate only the G2 actor-side initializer into Stage 3, replace the
ordinary legacy sampler/update dispatch with the sealed grouped transaction,
execute exactly one complete transaction/update, produce the first exact v015
Stage-3 checkpoint, and reload it in a fresh inference runner.

Scope: actor-only HSL migration, formal Stage-3 config/runner dispatch,
transaction provider/accumulator connection, existing grouped PPO consumer,
exact-one-update boundary, committed v015 save producer, and fresh inference
reload. The grouped PPO formula and HSL target semantics remain unchanged.

Evidence: S0/S2/S3/S4 T-actor-migration/T-no-HSL-state-leak/T-formal-dispatch/
T-one-transaction/T-grouped/T-exact-one-update/T-save/T-fresh-reload.

Stop: critic/optimizer/HSL target state migrates from the G2 initializer;
legacy sampler/update remains reachable from the formal v015 train mode; a
partial or mixed transaction reaches loss; update count is not exactly one;
checkpoint lacks the exact v015 identity or committed receipt; or fresh
inference changes actor/layout/normalizer identity.

##### G3-S0: Formal Stage-3 Migration And Save-Producer Audit

Objective: read-only freeze the exact actor-only HSL migration owner, formal
sealed grouped dispatch, committed Stage-3 v015 save producer, and fresh
inference reload boundary before modifying Stage-3 training.

Scope: Stage-3 config/entrypoint, HSL-v1 load, actor/prefix-normalizer migration,
legacy sampler/update isolation, sealed transaction provider/accumulator,
grouped PPO dispatch, exact-one update, committed checkpoint trigger, and fresh
inference load owner.

Non-scope: code or document changes, simulator/training/live run, policy-quality
claims, carrier materialization, deployment composition, HSL target/formula,
grouped PPO formula, or checkpoint format changes.

Evidence: S0 T-owner/T-load-boundary/T-formal-dispatch/T-legacy-isolation/
T-save-producer/T-fresh-reload/T-stop.

Stop: any critic/optimizer/HSL target can migrate; the ordinary formal route
can still choose legacy sampling/immediate update; the checkpoint trigger can
save partial transaction state; no unique committed v015 producer exists; or
fresh inference requires padding, fallback, or non-actor HSL state.

##### G3-S1A: Actor-Only HSL Migration And Formal Config

Objective: consume one explicit `frontres-v015-hsl-proposal-v1` artifact as a
Stage-3 initializer without treating it as resume state, then leave the runner
in the q29/grouped/formal configuration required by the later transaction
dispatch.

Scope: explicit Stage-3 initializer argument, strict pre-mutation HSL-v1
validation in the existing checkpoint owner, thin runner connector, actor/6D
distribution/158D prefix-normalizer restore, and a fail-closed guard before any
ordinary Stage-3 training dispatch.

Non-scope: transaction provider, candidate collection, grouped loss changes,
optimizer state, critic or critic-normalizer migration, sampler state, Stage-3
checkpoint save, simulator, training, or live execution.

Owner files/modules: `frontres_checkpointing.py` is the unique migration owner;
`on_policy_runner.py` is the thin runner connector; `train.py` owns the explicit
argument, v015 formal config, and the S1A-time pre-G3-S1B dispatch stop.

Evidence: S1/S3 T-explicit/T-layout/T-actor-only/T-prefix/T-zero-state-leak/
T-legacy-reject/T-pre-mutation/T-dispatch-stop.

Stop: validation requires legacy fallback or identity padding; critic,
privileged normalizer, optimizer, sampler, or transaction state changes; HSL
remains active after migration; or ordinary training can start before G3-S1B.

Status: completed at deterministic S1/S3 `E-FI-43`. The explicit HSL-v1 path
restores only residual actor, 6D distribution, and 158D prefix-normalizer state;
ordinary Stage-3 now selects `(1,2)` q29, grouped reduction, and formal identity,
and at the S1A boundary stopped before G3-S1B. `E-FI-44` subsequently released
that stop through the formal transaction owner without reopening HSL migration.

##### G3-S1B: Formal Sealed Transaction Dispatch And Committed Save

Objective: replace the ordinary Stage-3 legacy sampler/update call with the
existing complete v015 transaction provider/accumulator, grouped exact-one
update owner, and committed checkpoint trigger.

Scope: formal training-loop dispatch, one complete multi-Segment x M request,
existing grouped loss consumer, exact-one update accounting, committed receipt,
and save trigger after commit.

Non-scope: HSL migration changes, grouped PPO formula, checkpoint format,
fresh-inference reload, simulator, real training, live run, policy quality,
carrier materialization, or deployment composition.

Owner files/modules: `frontres_segment_live_training.py` owns iteration and save
order; `frontres_segment_live_update_loop.py` owns formal dispatch;
`frontres_segment_live_sampler.py` owns plan/accumulator; the existing
`frontres_segment_live_probe.py` update owner remains unchanged.

Evidence: S2/S3 T-provider/T-complete-transaction/T-grouped/
T-exact-one-update/T-legacy-isolation/T-commit/T-save.

Stop: legacy `run_frontres_segment_sampler_step()` or `to_ppo_batch()` remains
reachable; collection steps the optimizer; partial/mixed rows reach loss;
update delta is not one; or save occurs before a committed receipt.

Status: completed at deterministic S2/S3 `E-FI-44`. Ordinary Stage-3 now
selects a whole-row multi-Segment x M plan, routes it through the existing
sealed accumulator/grouped exact-one owner, increments iteration only after the
matching committed receipt, and triggers save only from that committed state.
The legacy immediate-update loop and legacy `to_ppo_batch()` remain outside the
formal branch. No simulator, training, fresh inference, or live run was used.

##### G3-S2: Exact Stage-3 Save Producer And Fresh Inference Reload

Objective: connect one offline committed ordinary transaction to the actual
v015 `save_runner()` producer, then load the artifact into a fresh inference
runner and prove exact actor/layout/normalizer identity and proposal equality.

Scope: semantic CPU fixture, actual v015 checkpoint write/read owner, committed
receipt, fresh inference construction, and identical normalized 158D input plus
6D proposal before save and after reload.

Non-scope: HSL or grouped-PPO formula changes, checkpoint format changes,
simulator, real training, live run, policy quality, carrier materialization, or
deployment composition.

Owner files/modules: `frontres_checkpointing.py` remains the unique persistence
owner; `on_policy_runner.py` remains a thin save/load/inference connector; the
existing transaction and training owners may only provide the committed input.

Evidence: S3 T-save-producer/T-v015-identity/T-commit-receipt/T-fresh-runner/
T-prefix-normalizer/T-proposal-equality/T-legacy-reject.

Stop: the test substitutes a fake save, requires legacy fallback/padding,
restores a partial transaction, loses the exact 928/158/770 or q29 identity, or
fresh inference changes the normalized 158D actor input or 6D proposal.

Status: completed at deterministic S3 `E-FI-45`. One semantic 158D/6D policy
was frozen for a complete two-Segment x two-attempt request, updated by the
existing grouped exact-one owner with a real Adam step counter, saved by the
actual `save_runner()`, and strictly loaded into an independently initialized
fresh runner. The committed receipt, 928/158/770 layout, q29 identity, full
158D prefix statistics, normalized actor input, and bounded 6D proposal all
round-tripped exactly. No fake save, fallback, padding, partial transaction,
simulator, training loop, or live run was used. G3 engineering readiness is
closed; policy quality and bounded training remain G5 responsibilities.

#### G4 / 7: Controlled Artifact Carrier Materializer

Objective: define the smallest evaluation preparation owner:
`ordinary Clean/reference .npz + fixed protocol -> one immutable carrier`.

Scope: selection-time materialization, source/protocol/carrier hashes,
deployment-only current/H output, and no-resample lifecycle.

Non-scope: actor input metadata, framewise resampling, training, or live eval.

Evidence: S1/S2 T-materialize/T-hash/T-determinism/T-no-label/T-no-resample.

Stop: user must manually supply an unexplained Noisy file, metadata enters
actor input, or baseline/repair branches receive different carriers.

Status: completed at deterministic S1/S2 `E-FI-46`. The existing sequence-eval
owner now transforms one ordinary `.npz` plus a canonical fixed protocol into
one deterministic atomic carrier archive. It seals source/protocol/carrier/q29
and materialization hashes, requires explicit `root_body_index`, preserves
q29/dq29 bit-for-bit, stores no label/truth metadata in the archive, and rejects
second materialization in the same lifecycle. The existing strict request and
command carrier consume the generated artifact as current `[B,58]` plus dense
H `[B,H+1,29]`. No actor/GMT execution, report, training, simulator, live run,
or paired composition occurred.

#### G5 / 7: Formal Training And Policy-Quality Gate

Objective: train the v015 policy after G2/G3 closure and produce the checkpoint
that later evaluation consumes.

Evidence: bounded training logs and checkpoint identity plus nondegenerate
action, executable Gain, harmful-repair, and reload-consistency diagnostics.

Stop: route evidence is mistaken for policy quality, action collapses, harmful
repair dominates, or checkpoint/reload identity fails.

G5 is split because transaction telemetry, checkpoint identity, held-out
evaluation, offline persistence connectivity, and live policy quality have
different semantic owners and evidence tiers. No later G5 step may start before
the preceding Step End Report exists.

##### G5-S0: Formal Training And Policy-Quality Preflight

Objective: read-only trace the formal Stage-3 config, explicit HSL-v1
initializer, sealed grouped exact-one update, committed save, fresh reload, and
policy-quality route before authorizing bounded training.

Scope: config/entrypoint, formal training owner, transaction/update owner,
checkpoint producer/loader, current policy-quality evaluator, manifests, and
required S4 telemetry.

Non-scope: code or document changes during the audit, simulator, training,
checkpoint IO, live run, Gain/PPO/HSL formula changes, or Concept Figure edits.

Evidence: S0 T-owner/T-shape/T-HSL-artifact/T-transaction/T-save/T-reload/
T-quality-route/T-stop.

Stop: fresh reload is not connected to ordinary training; the active quality
route requires quartet/Clean roles, legacy HSL rollout targets, v011/v002
manifest identity, or a checkpoint schema incompatible with strict HSL-v1 and
Stage-3 v015.

Status: completed as a stopped read-only audit at `E-FI-47`. The ordinary
Stage-3 route reaches explicit HSL-v1 migration, complete multi-Segment x M
collection, grouped exact-one update, committed receipt, and actual v015 save.
It does not perform a post-save fresh-runner verification or emit an atomic
v015 policy-quality report. The existing quality evaluator remains a
quartet/Clean, repeated-action, v011/v002 route and cannot consume the strict
HSL-v1 prefix-normalizer key. Bounded training remains prohibited.

##### G5-S1: Transaction-Side v003 Action/Gain/Harm Telemetry

Objective: expose policy-quality facts already present in each sealed v015
candidate without adding another rollout or changing any training signal.

Scope: extend the existing read-only
`frontres_segment_diagnostics.py::build_frontres_v015_local_evaluation_report`
owner with full-6D action distribution and negative-Gain/harm facts, then let
`frontres_segment_live_probe.py` carry one immutable transaction projection in
the formal update diagnostics. Preserve transaction/scenario/hash/row identity.

Non-scope: checkpoint loading, held-out evaluator, fresh runner, optimizer,
PPO/return/priority/sampler mutation, Gain formula, HSL, simulator, training, or
live execution. Do not create a new diagnostics module.

Evidence: S1/S2 T-action-shape/T-finite/T-nondegenerate-visible/T-v003-source/
T-component/T-positive-negative/T-row-mask/T-identity/T-no-feedback/
T-legacy-reject using the existing diagnostics and transaction contracts.

Stop: telemetry recomputes Gain, reads v002/Clean-global fields, loses the
one-action policy-row mask or scenario identity, silently zero-fills unavailable
components, or can affect PPO, return, priority, sampler, or optimizer state.

Status: completed at `E-FI-48`. The sealed v003 action/component rows are now
published as immutable post-update diagnostics with no training feedback.

##### G5-S2A: Strict Quality Checkpoint And Manifest Identity

Objective: make policy-quality inputs explicitly distinguish the strict
proposal-only HSL artifact from the strict trained Stage-3 v015 artifact and
bind a held-out manifest to the active v015/v003 layout.

Scope: existing checkpoint validation helpers, policy-quality request/manifest
owners, pre-mutation identity checks, exact `928/158/770`, q29 offsets `(1,2)`,
full-6D action identity, and immutable manifest/checkpoint fingerprints.

Non-scope: rollout/evaluator execution, training, optimizer, checkpoint-format
migration, padding, legacy compatibility, simulator, or live run.

Evidence: S1/S3 T-HSL-v1/T-Stage3-v015/T-manifest/T-layout/T-prefix/
T-pre-mutation/T-tamper/T-legacy-reject.

Stop: generic `obs_norm_state_dict` substitutes for the HSL-v1 prefix key;
v011/v002, unversioned, padded, partial, or mixed checkpoint/manifest identity
can load; or validation mutates runner state.

Status: completed at deterministic S1/S3 at `E-FI-49`. The v015-only request
now binds a strict manifest fingerprint to separate HSL-v1 and Stage3-v015-v2
checkpoint receipts before any runner mutation. Evaluator execution remains
outside this step.

##### G5-S2B: Repair/Noisy One-Action-K Held-Out Evaluator

Objective: evaluate zero, HSL, and trained-policy actor routes on the same fixed
held-out v015 scenarios while each route internally uses only Repair and Noisy
scored roles and one first action with frozen-FEMR K evidence.

Scope: reuse the existing quality state-isolation/atomic-report utilities, but
route reset, observation, action, K execution, and Gain through the active v015
local scenario and v003 owners. HSL is an inference baseline only.

Non-scope: quartet/Clean scored roles, `build_frontres_hsl_rollout_target`,
repeated FEMR actions within K, Clean actor input, PPO/return/priority/sampler,
training, checkpoint changes, simulator, or live run.

Evidence: S1/S2 T-two-role/T-same-scenario/T-one-action/T-frozen-K/
T-v003/T-zero-HSL-policy/T-state-isolation/T-atomic-report/T-legacy-reject.

Stop: any route uses candidate/Clean scored rows, legacy HSL targets, v002 Gain,
later FEMR actions, mismatched scenario identity, or writes training state.

Status: completed at deterministic S1/S2 at `E-FI-50`. The v015-only owner
consumes route/checkpoint/item-bound one-action-K evidence, computes only v003
Gain, checks matched scenarios and unchanged training state, and emits one
atomic report. Fresh-runner actor loading and live physics remain later gates.

##### G5-S3: Actual Save To Fresh Reload To Atomic Quality Report

Objective: connect one offline committed ordinary v015 transaction through the
actual save producer, an independently initialized strict fresh runner, and the
fixed held-out quality evaluator.

Scope: actual `save_runner()`, strict Stage-3 v015 load, exact prefix normalizer
and 6D proposal equality, immutable manifest, atomic quality report, and zero
training-state mutation during evaluation.

Non-scope: HSL/Gain/PPO formula changes, checkpoint-format changes, simulator,
training, or live run.

Evidence: S2/S3 T-commit/T-save/T-fresh-runner/T-checkpoint-identity/
T-normalizer/T-proposal-equality/T-quality-report/T-isolation.

Stop: fake save, fallback, padding, partial transaction, identity drift,
proposal mismatch, non-atomic report, or evaluator mutation.

Status: completed at deterministic S2/S3 at `E-FI-51`. One real committed
ordinary transaction now reaches the actual Stage3-v015 save producer and an
independently initialized strict fresh runner with exact q29, `928/158/770`,
158D prefix-normalizer, and 6D proposal equality. A separately saved/reloaded
strict HSL-v1 baseline and the fresh Stage3 proposal then reach the G5-S2B
atomic report without changing optimizer, sampler, transaction, or warmup
state. This is offline semantic connectivity, not physical policy quality.

##### G5-S4: Bounded Live Training And Policy-Quality Gate

Objective: run one user-authorized bounded Stage-3 transaction/update/save,
strictly reload the produced checkpoint, then evaluate it on the fixed v015
held-out manifest.

G5-S4 is rebased at `E-FI-52` because launcher authority, live transaction
telemetry, held-out evaluator construction, and the final simulator run have
independent owners and evidence tiers. No later substep may start before the
preceding Step End Report exists.

Confirmed readiness blockers:

1. The Stage-3 launcher still maps the HSL artifact through
   `--resume_student_checkpoint`, which sets `resume=True`, while ordinary v015
   requires explicit `--frontres_v015_hsl_initializer_checkpoint`, offsets
   `(1,2)`, and no resume path.
2. The sealed transaction result owns immutable action/v003 Gain/harm reports,
   but `_v015_formal_update_summary()` drops them before bounded live logging.
3. The formal runner never installs the v015 Repair/Noisy one-action-K quality
   owner bundle, so strict artifacts stop before evaluation.
4. No fixed `frontres-v015-policy-quality-manifest-v1` artifact or formal
   actual-save -> fresh-reload -> atomic-report dispatch exists.

###### G5-S4-S1A: Explicit Training Launch And Transaction Telemetry

Objective: make one bounded formal training command select only the active
v015 initializer and expose the already sealed transaction diagnostics.

Scope: update the existing Stage-3 launchers to pass explicit HSL-v1,
`--frontres_v015_future_offsets 1,2`, no resume/student-checkpoint route, eight
envs, one formal iteration, one update, checkpoint interval one, and disabled
legacy periodic evaluation. Extend only the existing formal live summary/log
projection so the immutable `policy_actions [4,6]`, valid mask, v003
`intent_gain`, `physics_gain`, `repair_cost`, `gain_total`, sign fractions,
scenario/noisy hash, grouped mass, and exact-one counts remain visible after
commit.

Owner files/modules: `run/run_frontres_stage3_segment_hrl.sh`, `run_stage3.sh`,
`frontres_segment_live_training.py`, existing Stage-3 launch contracts, and
existing v015 transaction/diagnostics contracts.

Non-scope: held-out owner-bundle construction, manifest creation, evaluator
execution, fresh runner, checkpoint-format/Gain/PPO/HSL changes, simulator,
training, or live run.

Evidence: S1/S2 T-explicit-HSL/T-offsets/T-no-resume/T-no-periodic-legacy/
T-one-iteration/T-telemetry-shape/T-v003-source/T-identity/T-exact-one/
T-no-feedback/T-command.

Stop: launcher can still set resume or legacy periodic evaluation; required
v015 arguments are missing; telemetry recomputes Gain, silently fills missing
values, loses transaction/scenario/hash identity, or reaches training inputs.

Status: completed at deterministic S1/S2 evidence (`E-FI-53`). The bounded
launcher now selects the strict HSL-v1 initializer and offsets `(1,2)`, fixes
the 8-env/one-iteration/one-update/checkpoint-interval-one contract, and rejects
resume plus legacy periodic evaluation. The formal summary publishes only the
sealed transaction's immutable v003 action/Gain/identity projection after the
exact-one update; invalid rows remain unavailable rather than zero-filled.

###### G5-S4-S1B: Formal Held-Out Owner And Fresh-Report Dispatch

Objective: make the strict G5-S2B evaluator constructible from the formal
runner after an actual committed save.

Scope: install one v015-only Repair/Noisy one-action-K owner bundle using the
existing reset/observation/K/v003 owners; add one fixed
`frontres-v015-policy-quality-manifest-v1`; bind actual HSL-v1 and committed
Stage3-v015-v2 file identities; create an independent fresh inference runner;
and dispatch source/fresh proposal equality into the existing atomic quality
report.

Owner files/modules: `train.py`, `on_policy_runner.py`,
`frontres_policy_quality_eval.py`, existing live local-scenario/reset/K owners,
`frontres_checkpointing.py`, one v015 manifest under `note/testing/manifests/`,
and focused formal quality contracts.

Non-scope: launcher/transaction telemetry changes, quartet/Clean scored roles,
legacy quality-owner adaptation, repeated FEMR action, checkpoint-format,
Gain/PPO/HSL formula, simulator, training, or live run.

Evidence: S1/S2/S3 T-owner-install/T-manifest/T-two-role/T-one-action-K/
T-actual-save/T-independent-fresh/T-928-158-770/T-q29/T-normalizer/
T-proposal-equality/T-route-hash/T-atomic-report/T-state-isolation/
T-legacy-reject.

Stop: owner construction requires a legacy quartet/Clean path, fake save,
fallback, padding, partial transaction, mixed identity, repeated FEMR action,
non-atomic report, or any evaluator mutation.

Status: completed at deterministic S1/S2/S3 evidence (`E-FI-54`). The formal
v015 evaluator now resolves one fixed held-out manifest item to one immutable
4-Repair/4-Noisy scenario, temporarily installs strict HSL-v1 or committed
Stage3-v015 actor/prefix state, collects one deterministic proposal followed by
frozen-GMT K evidence, and restores all training state before the existing
v003 atomic report is committed. Legacy quality execution remains isolated.

###### G5-S4-S1C: Held-Out Index Identity / Execution-K Resolver Repair

Objective: keep the Stage-1 cache window and held-out executable-evidence K as
separate objects when resolving a fixed manifest item.

Scope: resolve one Stage-1 Segment by unique `(motion_id, start_frame)` identity,
then carry manifest `effective_horizon_k` unchanged into the sample and local-
scenario materializer. A K4 cache index may therefore materialize the active K8
continuation, exactly as the ordinary training route already does.

Non-scope: manifest semantics, cache rebuilding, K selection, HSL, Gain, PPO,
checkpoint format, simulator, training, or live evaluation.

Evidence: S1/S2 T-K4-index/T-K8-budget/T-K8-continuation/T-unique-identity/
T-heldout-owner/T-save-fresh/T-atomic-report.

Stop: motion/start resolves zero or multiple index rows; execution K is replaced
by cache K; or the materialized Clean continuation is shorter than execution K.

Status: completed at deterministic evidence `E-FI-55`. The resolver no longer
uses cache `spec.horizon_k` as scenario identity, preserves manifest K8 through
the materializer, and rejects duplicate motion/start identities. Live held-out
quality remains unconfirmed until the corrected server command completes.

###### G5-S4-S1D: Quality Inference-Mode Isolation

Objective: prevent zero/HSL/policy held-out inference from updating any live
policy or observation-normalizer state before the atomic quality report.

Scope: wrap the complete held-out route set in one reversible inference-mode
boundary; freeze policy, 158D prefix, frozen-GMT, privileged, and teacher
normalizer module modes before the first observation read; restore every
submodule's original mixed mode on success or exception; include all four
normalizer states in the mutation signature.

Non-scope: checkpoint payload/restore semantics, K, Gain, PPO, manifest,
sampler, simulator, training, or live command.

Evidence: S1/S2 T-train-mode-write/T-zero-write/T-mixed-mode-restore/
T-exception-restore/T-heldout/T-save-fresh/T-observation-authority.

Stop: any normalizer running state changes; the mutation guard is weakened;
mixed source modes are flattened on restore; or save/reload/observation
contracts regress.

Status: completed at deterministic evidence `E-FI-56`. A live-style updating
normalizer reproduces the prior mutation without the guard; all held-out routes
now run with zero normalizer writes and restore exact mixed module modes after
success and intentional failure. Live quality remains unconfirmed.

###### G5-S4-S1E: Manifest Item Lifecycle Isolation

Objective: close one held-out manifest item's sealed local scenario only after
its matched zero/HSL/policy counterfactual routes finish, before the next item
installs a different scenario/hash.

Scope: add an explicit evaluator-owned item-close callback that clears the
command carrier and closes the corresponding immutable batch lifecycle on
success or exception. Preserve one sealed scenario throughout all three routes
inside the item and reject any training-state mutation caused by close.

Non-scope: command sealed-carrier guards, scenario materialization, K, Gain,
checkpoint, PPO, sampler policy, simulator, training, or live command.

Evidence: S1/S2 T-route-order/T-item-close/T-next-item/T-exception-close/
T-command-close/T-batch-close/T-no-feedback/T-save-fresh.

Stop: scenario closes between counterfactual routes; the next item can replace
an active carrier; exception leaves either lifecycle active; command fail-
closed is weakened; or close changes training state.

Status: completed at deterministic evidence `E-FI-57`. The evaluator now owns
the exact `zero -> HSL -> policy -> item close` boundary, and the formal owner
closes both command and batch lifecycle before the next manifest item. The
original 16-item live quality run remains unconfirmed.

###### G5-S4-S2: Final Command Artifact And Threshold Preflight

Objective: perform a read-only preflight after S1A/S1B and freeze the exact
server command, current server artifacts, expected sentinels, and numeric gate.

Scope: verify HSL-v1 file, motion/cache roots, fixed manifest, output directory,
GPU selection, one-transaction command, post-save fresh command/dispatch, and
all required telemetry. Freeze the user-confirmed numeric thresholds below or
their explicit replacement.

Candidate thresholds pending user confirmation:

- transaction: two Segments, four policy attempts, `valid_rows=4/4`,
  `update_count=1`, and `optimizer_step_delta=1`;
- action: all 24 scalars finite, at least two rows with L2 norm greater than
  `1e-4`, and at least one dimension with cross-row std greater than `1e-5`;
- saturation: rows exceeding `0.285` position or `0.38` rotation magnitude at
  most `0.25`;
- quality: trained `gain_total_mean > 0`, positive fraction at least `0.50`
  and no lower than HSL, negative fraction at most `0.25` and no higher than
  HSL;
- harm: do not introduce a second semantic variable; use
  `gain_total < 0` fraction as harmful-repair fraction unless the user changes
  this boundary;
- reload: normalized 158D input equal and 6D proposal close with
  `rtol=1e-5`, `atol=1e-6`;
- identity/atomicity: committed receipt, manifest SHA, route checkpoint SHA,
  and final JSON artifact all match with no partial output.

Non-scope: code/document changes, checkpoint IO, simulator, training, or live
run.

Evidence: S0 T-artifact/T-command/T-telemetry/T-threshold/T-stop.

Stop: any required artifact is absent, the command still reaches a legacy
path, telemetry cannot evaluate every gate, or numeric thresholds remain
unconfirmed.

Current runtime status: partially confirmed at `E-FI-55`. The bounded training
half completed one exact update and saved `model_1.pt`; the first quality launch
failed before evaluation because its shell expression expanded an unset HSL
positional variable and selected the legacy default checkpoint. The corrected
command and numeric gates remain user-controlled before the quality rerun.

###### G5-S4-S4: One Bounded Live Training And Quality Run

Objective: execute exactly one user-confirmed Stage3-v015 transaction and the
matched post-save held-out quality evaluation.

Scope: eight envs, one complete 2-Segment x M transaction, one optimizer step,
one committed checkpoint, one independent fresh reload, and one atomic quality
report against the frozen thresholds.

Non-scope: long training, deployment composition, grouped-PPO/Gain/HSL changes,
checkpoint-format changes, or G6/G7 execution.

Evidence: S4 T-train/T-HSL-input/T-layout/T-transaction/T-exact-one-update/
T-action/T-gain/T-harm/T-save/T-fresh-reload/T-report.

Stop: any legacy fallback or role/layout drift appears; action is nonfinite or
collapses; harmful repair crosses the confirmed gate; update count is not one;
checkpoint/reload identity differs; or the evaluator mutates training state.

#### G6 / 7: Paired Composition Connectivity

Objective: connect the same fixed carrier and initial conditions to two
no-feedback branches: frozen-GMT baseline and per-frame FEMR plus frozen GMT.

Evidence: S1/S2 T-pair/T-identity/T-baseline/T-repair/T-no-feedback/T-report.

Stop: only the Repair branch exists, branches differ in carrier/reset/GMT, or
composition metrics enter PPO, sampler, return, priority, or optimizer.

#### G7 / 7: Bounded Live Composition

Objective: run one user-authorized paired simulator sequence with the trained
G5 checkpoint and G4 carrier, then seal hashes, telemetry, failures, and report.

Evidence: S4 T-composition/T-pair/T-protocol/T-isolation/T-checkpoint.

Stop: any dependency above is missing, the CLI uses an untrained checkpoint,
the carrier resamples, baseline is absent, or training state changes.

## Current Plan Cursor

R0--R6 / 7 and Step 5A are complete at `E-FI-18`--`E-FI-27`. The final bounded
S4 run confirms the complete dedicated local route through separate
`928D` actor / `289D` critic observations, K=8 frozen-GMT evidence, grouped
equal-mass reduction, and exactly one optimizer update. Step 5B deployment-
composition schema, carrier, CPU executor, and dedicated CLI have interface
evidence at `E-FI-28`--`E-FI-31`; they are not ready-to-run evaluation evidence.
G0 is completed at `E-FI-32`. G1 is completed as a stopped read-only audit at
`E-FI-33`: Stage-1 still admits the old `870D` route, no new HSL checkpoint
identity exists, ordinary Stage-3 bypasses the sealed grouped transaction, and
ordinary training cannot produce a fresh-reloadable exact v015 checkpoint.
The human carrier decision recorded at `E-FI-34` is implemented through G2,
which is complete at `E-FI-42`. G3 is complete at offline S2/S3: `E-FI-43`
closes explicit actor-only HSL migration, `E-FI-44` closes ordinary whole-M
formal dispatch/exact-one commit/commit-only save triggering, and `E-FI-45`
closes actual save to strict fresh-inference equality, and `E-FI-46` closes G4
ordinary-reference-to-fixed-carrier materialization. G5--G7 retain their
policy-quality, paired-connectivity, and bounded-composition duties in that
order. `E-FI-47` closes G5-S0 as a stopped preflight: formal training through
committed save is code-confirmed, while strict post-save fresh reload and the
v015 policy-quality route remain absent. G5 is locally rebased into S1, S2A,
S2B, S3, and S4; G5-S1 is the only ready code step. No Stage-3 training or
trained checkpoint is ready.
