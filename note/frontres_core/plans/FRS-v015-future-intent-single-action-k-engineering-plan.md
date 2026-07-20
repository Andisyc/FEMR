# FRS-v015 Engineering Plan: Future-Intent, Single-Action K Replay

Status: active, volatile engineering plan. Re-planned after a read-only
white-box owner audit on 2026-07-19 and H0-A contract closure on 2026-07-20.
Step 0, 1A, 1B, and H0 are complete at their recorded evidence tiers; no source
implementation, simulator, training, formal runner, checkpoint migration, or
live run is authorized by this document unless its named step is separately
confirmed.

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

## White-Box Starting Facts And Isolation Targets

| Semantic object | Confirmed current owner | Current mismatch / legacy route | New active-route target | Planned gate |
| --- | --- | --- | --- | --- |
| Scenario materialization | `frontres_segment_stage1_env_hooks.py::materialize_frontres_fixed_noisy_tape` | Produces one `[L,65]` tape | Immutable local object containing `x_t`, current root artifact, q29 intent window, Clean continuation, K, and identity | 1A |
| Actor H context | `frontres_runtime.py::append_frontres_fixed_noisy_future_context` | Requires `[B, |H|*65]`, then prepends it to actor input | Ordered deployment-provenance future q29 tail only | 1B |
| Pair and reset layout | `frontres_training_setup.py`, `commands.py`, `frontres_segment_stage1_env_hooks.py` | Quartet/projected/candidate/base/clean roles remain available in active setup | Two scored roles: Noisy and Repair; Clean only supplies continuation | 2A |
| K collector | `frontres_rollout_step.py` and `frontres_segment_live_probe.py` | Existing route is not yet proven to authorize one action then freeze FEMR | One policy tuple at `t`, GMT-only execution through `C[t+1:t+K]` | 2B |
| Gain | `frontres_gain.py` and capture in `frontres_segment_live_probe.py` | Style uses Clean global body/root comparisons | `fidelity_internal(executed_q29, I)` paired across Noisy/Repair | 3A--3B |
| PPO adapter | `frontres_segment_storage.py` and `frontres_segment_ppo.py` | `to_ppo_batch()` intentionally drops transaction metadata; grouped candidate adapter is offline-only | Metadata-bearing candidate path becomes the only grouped formal path | 4A--4B |
| HSL | `frontres_warmup.py`, `frontres_hsl_rollout_target.py`, and checkpoint loader | Raw-observation warmup bypasses q29; rollout label uses a Clean quartet | Proposal-only current-frame HSL after q29 bridge; legacy rollout label and old checkpoints reject | H0-A -> H1 |

`E-FI-1` in the evidence ledger records the source locations and limits of this
audit. These are code facts, not a claim that the v015 route already works.

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

Gate: G1. Depends on: 1A. Status: completed at S1 deterministic-module
evidence (`E-FI-3`).

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

Gate: G4. Depends on: 4B. HSL checkpoint work remains blocked by H0 and user
confirmation.

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

#### Step 5A-S0: Pre-Live Formal Sentinel Connectivity

Status: completed 2026-07-20 at deterministic CPU-only S2 evidence
(`E-FI-16`). This is a code-and-contract prerequisite for Step 5A, not its S4
live evidence and not permission to start an environment transaction.

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

Stop condition:
- any legacy fixed tape, legacy `to_ppo_batch()`, immediate-update path, HSL
  writer, Clean actor input, later FEMR action, mixed scenario identity, or
  update count other than one becomes reachable; or the required explicit H
  offsets cannot be represented without a new method decision.

#### Step 5A-S1: Bounded Local Runtime Trace

Gate: explicit user confirmation after the actual command and preflight are
reported. Depends on: completed Step 5A-S0.

Status: preflight completed 2026-07-20 at `E-FI-17`; the live transaction was
not started because the current Darwin arm64 host has no IsaacLab/NVIDIA
runtime and no `/hdd1/cyx` Stage-1 cache or motion data. Execution remains
pending on the synchronized SUST_Main_2 workspace.

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

## Current Plan Cursor

Step 0, 1A, 1B, H0, H1-S1a/H1-S2, 2A, 2B, 3A, 3B, 3C, 4A, 4B, 4C, and
5A-S0 are complete at their recorded evidence tiers. Step 5A-S1 preflight is
complete at `E-FI-17`; its sole live transaction is pending a synchronized
SUST_Main_2 shell with IsaacLab, GPU, Stage-1 cache, and motion data. Generic
training, checkpoint cadence/resume, long training, and deployment composition
evaluation remain outside the current authorization.
