---
contract_id: FRS-EVAL-v003
status: active
effective_date: 2026-07-19
updated_date: 2026-07-21
supersedes: FRS-EVAL-v002
scope: Stage 3 local first-action K evaluation and separate full-sequence deployment-composition evaluation
---

# Local Repair And Composition Evaluation Contract

## Evaluation Boundary

Evaluation distinguishes two questions that must not be merged:

```
Local K evaluation:
  Did one repair at t improve execution over the same Clean continuation?

Composition evaluation:
  Does repeatedly deployed FEMR remain useful on a full reference sequence
  whose later frames may also contain artifacts?
```

The first is the active paired-Gain / PPO evidence regime. The second is a
separate deployment-quality evaluation; it never changes a stored local return,
PPO eligibility, or replay priority.

Evaluation remains isolated from policy updates, optimizer state, and replay
priority unless an explicitly named review protocol says otherwise.

## Local First-Action K Evaluation

Each evaluation item independently:

```
select Segment and Clean dynamic x_t
-> build one root-only artifact at t and future internal intent I[t:t+H]
-> prepare one common Clean continuation C[t+1:t+K]
-> reset x_t for Noisy and Repair roles
-> apply no repair or one Delta SE(3)_t
-> freeze FEMR after t and execute frozen GMT through C
-> aggregate intent/physics/cost paired Gain
```

Only Noisy baseline and Repair are scored roles. Clean is not a third rollout:
it supplies the common continuation and optional q29-invariance calibration.

Output must identify motion, Segment/start frame, K, H offsets, local artifact
identity, future-intent provenance, continuation identity, role, and trial
identity.

## Full-Sequence Deployment Composition Evaluation

A separately named sequence evaluation feeds one ordinary deployment/reference
`.npz` stream to FEMR and GMT. A real extracted stream may already contain
artifacts. A controlled synthetic evaluation instead starts from an existing
Clean/reference `.npz` and materializes one fixed persistent-corruption carrier
at sequence-selection time. A separately user-managed `Noisy.npz` is not a
method input or prerequisite.

The materializer runs once before either branch executes, seals the Clean
source hash, corruption protocol/seed/scale, and materialized-carrier hash, and
then exposes only deployment-visible current reference and future q29 intent.
Corruption metadata is report-only and must not enter FEMR observation. No
frame, reset, or comparison branch may resample the carrier.

Composition usefulness requires a paired comparison under the same fixed
carrier and initial conditions:

```
Baseline: fixed artifact carrier -> frozen GMT
Repair:   same carrier -> per-frame FEMR -> frozen GMT
```

FEMR may act at each deployment frame under its normal inference interface.

This evaluation reports whether local repairs compose under temporally
persistent artifacts. It must report its own reference-corruption protocol,
per-frame FEMR action count, and accumulated failures. It is not evidence that
the single-action K return used a fully Noisy continuation.

## Metrics

Local evaluation reports the v003 decomposition:

```
Intent: root-invariant q29/relative-articulation fidelity and paired intent_gain
Physics: success, fall, survival, ZMP/support, contact, and paired physics_gain
Repair: Delta SE norm and temporal change
Summary: gain_total, K, H, current artifact, intent invariant, continuation identity
```

Do not report global Clean MPJPE/root-orientation as active Style or use it to
rank the actor. It may be an offline dataset-quality/calibration diagnostic,
clearly labeled non-deployment supervision.

Full-sequence composition evaluation reports the same deployable intent and
physics metrics across time, but keeps its aggregate result separate from a
local K Gain.

## Pairing And State Isolation

Noisy and Repair must share x_t, current root artifact, I, Clean continuation,
K, RNG/reset controls, and frozen GMT identity. The item must fail closed if any
of those identities differ.

No evaluation event may reuse a stale training capture. Sampler RNG/state and
replay priority remain unchanged.

## Bounded Implementation Evidence

`E-FI-12` (2026-07-20) completes candidate-only deterministic S1 evaluation
isolation:

- a local-K report consumes the sealed Step 3B v003 carrier and prints only
  q29 intent provenance/source, intent gain, physics gain, repair cost, total
  gain, local identity, and K; it has no return/priority/PPO feedback path;
- its deterministic contract exercises the all-invalid case: every diagnostic
  remains `NaN`/`UNCONFIRMED`, and a zero-filled replacement is rejected;
- the existing periodic, offline, and sequence evaluators are explicit legacy
  v002/quartet owners and reject a v015 future-intent layout before sampling,
  reset, rollout, or Gain capture;
- a separate deployment-composition protocol records reference-stream identity
  and frame/action counts, with immutable false return/priority/PPO feedback
  flags. It does not execute a sequence or claim composition metrics.

This proof does not establish the formal periodic evaluator, a real full-
sequence evaluator, simulator timing, evaluation checkpoint state, or live
deployment composition evidence.

`E-FI-28` (2026-07-21) adds the deterministic S1 deployment-composition
kernel without changing this evaluation boundary:

- `frontres_segment_sequence_eval.py` validates one explicit structured `.npz`
  containing q29/dq29 plus body pose/velocity arrays and seals its file hash,
  deployment provenance, H offsets, frame/body identity, and fps;
- a canonical `persistent_full_sequence` corruption protocol has an
  order-independent scalar-parameter hash and cannot carry per-frame actor
  labels or training state;
- the immutable per-frame report records FEMR action use, q29 intent error,
  physics success/fall, ZMP margin, contact consistency, and accumulated
  failures; return, priority, PPO, sampler, optimizer, Clean continuation, and
  local-scenario payloads are absent from its type;
- the dedicated config rejects disabled, malformed, non-`.npz`, invalid-H, and
  legacy mixed modes before any runner connection.

This remains S1 schema evidence. It does not connect command, actor, GMT,
simulator, report persistence, or the formal config/runner entrypoint.

`E-FI-29` (2026-07-21) completes only the S2A deployment carrier boundary:

- `MultiMotionCommand` verifies the sealed `.npz` hash again and copies one
  immutable q29/dq29 sequence with one explicit transaction-wide frame cursor;
- its read-only snapshot exposes current q29+dq29 `[B,58]`, dense deployment
  q29 intent `[B,H+1,29]`, row ids, cursor, file/stream/protocol identity, and
  deployment provenance;
- reads do not advance; explicit advance rejects before H would clamp; active
  local scenario, fixed tape, legacy window, reinstall, and changed file hash
  all fail closed;
- `frontres_runtime.py` validates and clones the snapshot but does not append an
  actor observation, execute GMT, create metrics/report rows, or write training
  state.

This is semantic CPU connectivity evidence, not composition execution. It does
not prove that the declared corruption protocol has been materialized into a
runtime stream, that FEMR/GMT consumed the carrier, or that a formal report or
live metric exists.

`E-FI-30` (2026-07-21) completes the S2 formal composition executor boundary:

- formal input must declare `source=pre_materialized_deployment_npz`; the
  evaluator verifies and consumes the artifact-bearing stream but never draws,
  resamples, or guesses a corruption from scalar protocol metadata;
- a reference with T frames and fixed H offsets evaluates exactly
  `T-max(H)` frames. Tail frames are neither clamped nor filled with fabricated
  future context;
- each evaluated frame follows current q29/dq29/body reference -> 928D
  deployment observation -> one deterministic full-6D FEMR correction -> one
  frozen-GMT motor action -> one environment step -> one q29/physics metric row;
- the dedicated runner method cannot call the legacy v002 sequence evaluator;
  the final JSON is written atomically and retains file, stream, protocol,
  evaluated-frame, action, metric, and no-feedback identities;
- optimizer, sampler, storage, and transition fingerprints are identical before
  and after the executor. Any change rejects the report.

This remains semantic CPU evidence. Default physical ZMP/contact readers,
Isaac command timing, auto-reset behavior, actual artifact severity, report
values, and bounded deployment composition require S4 simulator evidence.

`E-FI-31` (2026-07-21) completes the S4-S0 dedicated live-entry boundary:

- `frontres_v015_deployment_composition.py` accepts only the registered
  `FrontRES-Unified-Tracking-Flat-G1-v0` task and absolute v015 FEMR, frozen
  GMT, pre-materialized `.npz`, and new report identities;
- CUDA selection remains explicit through `CUDA_VISIBLE_DEVICES`; the CLI
  records the remapped runtime device and rejects missing/out-of-range mapping;
- the runner resolves the existing q29 H/checkpoint layout from formal v015
  identity without requesting Segment Replay, so no sampler is constructed;
- GMT is bound before runner construction, FEMR loads through the formal
  checkpoint owner with optimizer/critic restore disabled, and the only
  dispatch is the S2B evaluator. A persistent optimizer step counter must stay
  unchanged.

This is deterministic config/dispatch evidence only. It does not import or
launch IsaacLab in tests and does not supply the actual S4 checkpoint, stream,
physical metrics, timing, or report.

`E-FI-32` (2026-07-21) corrects the post-observation-change test dependency:

- the user confirmed that no trained checkpoint compatible with the new
  `928/158/770` observation layout exists. A v015 FEMR checkpoint is therefore
  an output of HSL/Stage-3 training and reload validation, not a current S4
  input;
- the user confirmed that no pre-materialized `Noisy.npz` exists and that this
  object had not been defined in the accepted workflow. Controlled corruption
  is now owned by a planned selection-time materializer from an ordinary
  Clean/reference `.npz`; an optional serialized carrier is evidence/cache,
  not a user-facing semantic input;
- E-FI-28--E-FI-31 remain valid interface/config evidence, but S2B and the CLI
  are `implemented-not-runnable` for scientific evaluation until training,
  materialization, and paired-baseline dependencies close;
- Step 5B-S4 is blocked. It cannot begin directly after CLI connectivity.

This correction does not change FEMR observation, action, local K Gain, PPO,
or the no-feedback evaluation boundary. The current source still requires a
pre-materialized file and a checkpoint, so source-code alignment remains a
future bounded implementation step rather than an accomplished fact.

## Acceptance Gates

1. S1: root-only scenario, q29 invariant, role identity, one-action/frozen-
   FEMR lifecycle, and intent-metric value tests.
2. S2: periodic/offline local evaluator calls the same v003 owner and rejects
   Clean-global or legacy-score fallback.
3. S3: full-sequence composition evaluator records its distinct protocol and
   cannot feed back into PPO or priority.
4. S4: bounded real simulator evidence for both local identity and sequence
   composition, with no claim that one substitutes for the other.

The current quartet evaluation layout and v002 Clean-global Style report are
legacy implementation paths until migrated or isolated.
