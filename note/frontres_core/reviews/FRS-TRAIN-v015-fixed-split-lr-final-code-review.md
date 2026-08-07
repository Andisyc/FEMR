# FRS-TRAIN-v015 Fixed Split-LR Final Code Review

Date: 2026-08-07

```text
Review modes: construction_review + final_gate_review
Verdict: OFFLINE_READY
Discipline: active (FRS-ENG-v001)
Live status: STALE_RERUN_REQUIRED
```

## Reviewed Boundary

The reviewed change keeps one `FrontRESTrackedAdam` and partitions it into
exactly two named, disjoint and exhaustive groups: residual Actor at `3e-6`
and scalar Critic at `1e-5`. Stage-3 composition is fixed-schedule only. The
existing grouped PPO loss, phase weights, exact-one transaction, full-6D
action, fixed std, frozen GMT, Gain and K/M/DR curriculum are unchanged.

Checkpoint-v10 persists and validates role membership, exact LRs, optimizer
moments and the shared step count before restoring mutable runner state.
Checkpoint-v9 remains historical evidence and is not migrated.

## Findings And Repairs

Construction review initially found two in-scope P1 contract gaps:

1. a non-convertible checkpoint LR could escape as a raw `TypeError` instead
   of the checkpoint owner's uniform fail-closed `RuntimeError`;
2. optimizer construction proved Actor/Critic disjointness but did not
   explicitly reject an additional unowned trainable policy parameter.

Both were repaired in their existing semantic owners. Artificial malformed-LR
and third-trainable-parameter cases now reject. Final review finds no remaining
P0 or P1. No removal is authorized or required in this unit.

## Discipline Review

- Ownership: optimizer partition stays in `FrontRESUnified`; checkpointing and
  telemetry consume it without repartitioning.
- Public boundary: callers provide two finite LRs and fixed schedule; policy
  internals do not leak into CLI or runner composition.
- Dependency direction: config -> algorithm -> existing update -> persistence
  and telemetry. No new wrapper, service, Protocol or second optimizer exists.
- Legacy safety: shared/adaptive Stage-3 inputs and checkpoint-v9 resume reject;
  HSL and read-only policy-quality routes retain their prior optimizer behavior.
- Reliability: critic-only freezes Actor parameters and Adam state; exact-one
  step count is shared; malformed persistence rejects before model, sampler or
  receipt mutation.
- Research boundary: offline closure proves construction and route contracts,
  not convergence, policy-quality improvement or physical efficacy.

## Evidence

- `frontres_segment_all_contract_suite.py`: 50/50 contracts passed after final
  repairs.
- focused optimizer and checkpoint contracts include role, LR, Actor freeze,
  moments/count, malformed identity and pre-mutation rejection cases.
- Python compilation, shell syntax, JSON parsing, `git diff --check`, Design
  Inspector, Module Test Atlas, Repository Reading Atlas and Code Quality
  Evidence Atlas passed.

Static hotspot triage used local `fuck-u-code 2.2.2` with
`fuck-u-code analyze . --format markdown --top 20`: 572 files were discovered,
570 analyzed, with no exclusions supplied by the tool invocation. Shell files
fell back to its regex parser and bundled KaTeX files introduced noisy rankings.
Among changed owners, `frontres_checkpointing.py`, `frontres_unified.py` and
`train.py` ranked as hotspots. White-box review classifies the added validation
as intentional fail-closed owner complexity: the change did not add a new
responsibility or dependency, and the malformed, exhaustive-membership,
composition and persistence cases have independent contract tests. Analyzer
scores are not treated as correctness or severity evidence.

## Remaining Gate

The prior v014 live evidence is stale for the v015 optimizer/checkpoint
identity. One official bounded K8/M2 critic-only transaction must still show
the exact two groups/LRs, Actor zero delta, Critic nonzero delta, shared step
delta one and checkpoint-v10 round-trip. Long training remains blocked until
that sentinel passes. Policy quality requires a separate fresh campaign.
