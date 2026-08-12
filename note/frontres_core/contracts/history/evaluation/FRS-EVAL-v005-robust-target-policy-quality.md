---
contract_id: FRS-EVAL-v005
status: superseded
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-EVAL-v004
scope: Clean-anchored local and composition evaluation for checkpoint-v18 robust-target policies
---
# Robust-Target Policy Evaluation

## Preserved evaluation questions

Local one-action-K evaluation and full-sequence deployment composition retain
the complete FRS-EVAL-v004 Clean/Noisy/Repair identity, Gain-v008 fields,
inference-only isolation and zero training-state mutation.

## Checkpoint-v18 interpretation

The tested policy remains a 158D Actor and 449D state-value Critic. Local
evaluation reports raw Gain, per-attempt symlog utility, current held-out M4
mean and `V(s)`. It additionally labels `V(s)` as the expected utility learned
from policy-compatible robust Scenario means. Evaluation does not construct or
mutate a training Replay window and does not pretend that one held-out M4 mean
is the stored training target.

Checkpoint-v18 installs the Actor, Critic, Actor-prefix statistics and 449D
privileged-observation normalizer reversibly. Its Replay-v4 state may be
inspected for identity but is never loaded into an evaluator-owned mutable
training object. Checkpoint-v17 and earlier routes remain historical and cannot
resume TRAIN-v023.

## Required evidence

All FRS-EVAL-v004 isolation, field, identity and atomic-report requirements
remain. New evidence must prove strict checkpoint-v18 identity, no Replay-v4
mutation, correct expected-value labeling, and restoration after success and
exception. Evaluation cannot authorize training or infer robust-target efficacy
from checkpoint loading alone.
