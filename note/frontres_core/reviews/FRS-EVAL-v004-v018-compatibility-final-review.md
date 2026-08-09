# FRS-EVAL-v004 v018 Compatibility Final Review

Date: 2026-08-10

```text
Review modes: construction_review + final_gate_review
Verdict: OFFLINE_READY
Discipline: active (FRS-ENG-v001)
Live status: SERVER_EVALUATION_NOT_RUN
```

## Reviewed Boundary

The active Held-out Policy Quality route now accepts only METHOD-v019,
TRAIN-v018, PPO-v007, GAIN-v007 and checkpoint-v13. It keeps the fixed eight
K16 Segments and evaluates M4 Repairs per Segment. The checkpoint Gateway
temporarily installs the tested 158D Actor, 449D state-value Critic, Actor
prefix statistics and saved 449D privileged-observation normalizer. Every
touched state is restored on success and exception.

The report retains per-attempt GAIN-v007 evidence and adds one raw Critic
calibration row per Segment: shared `V(s)`, exact-M arithmetic-mean target and
raw error. The value-normalizer identity is validated and fingerprinted but is
not applied to inference output.

## Findings And Repairs

Construction review found two in-scope P1 gaps:

1. the old evaluator restored Actor/Critic weights but omitted checkpoint-v13
   Critic observation coordinates;
2. the value-normalizer fingerprint path assumed a tensor-only state dict even
   though its validated schema contains string, float and integer fields.

Both were repaired in the existing checkpoint owners. A third compatibility
issue was prevented by limiting the 449D normalizer requirement to
checkpoint-v13, leaving explicit checkpoint-v10 legacy routing unchanged.
Final review finds no open P0/P1.

## Discipline Review

- Ownership: checkpoint identity and reversible installation remain in the
  checkpoint Gateway; Segment calibration stays in the evaluator projection.
- Dependency direction: manifest -> request -> checkpoint Gateway -> existing
  collector/Gain -> atomic report. No simulator or training owner depends on
  Evaluation.
- Preserved behavior: Actor full-6D action, frozen GMT, GAIN-v007, held-out
  Segment bank, optimizer, sampler, curriculum and deployment are unchanged.
- Failure behavior: mixed identity, non-M4 rows, missing/invalid evidence,
  non-shared same-Segment value and restoration drift fail closed.
- Legacy safety: old v017 artifacts remain historical; the explicit legacy
  entrypoint is not silently redirected to v018.

## Evidence

- `frontres_segment_all_contract_suite.py`: 55/55 passed.
- Focused manifest, compatibility, evaluator, entrypoint, checkpoint-v13,
  held-out batch adapter, state-value PPO, formal transaction, telemetry and
  formal-runtime contracts passed.
- Python compilation, shell preflight, and the K16/M4 server-script command
  construction passed.

Two archived standalone v015 fixture files are not members of the active
aggregate suite and remain stale against current checkpoint/HSL identities.
They were not used as v018 evidence and were not modified.

## Remaining Gate

Run the prepared read-only server script against the real HSL-v2 artifact,
model_2000 checkpoint, Stage-1 cache and IsaacLab environment. Admission
requires an atomic v018 JSON report, eight held-out Segments, four K16/M4
transactions, finite Segment calibration rows and unchanged training owners.
This offline review is not a policy-quality verdict.
