# FEMR Notes Index

This folder stores human-readable engineering memory for FEMR and FrontRES work.

## First-Level Documents

- `00_repository_architecture_map.md`: global module ownership and repository structure.
- `architecture/`: visual architecture atlas and its editable data sources.

## Task Folders

- `frontres_core/`: original FrontRES contracts, checklists, paper notes, and discussion logs.
- `frontres_segment_replay/`: Segment Replay method, cache, sampler, reset, HRL, and external-code references.

## Organization Rule

Global repository structure belongs in first-level note documents.
Task-specific contracts, plans, references, and logs belong in task folders.

Do not scatter new multi-step planning notes in the root of `note/`.
For a complex implementation, create or update:

```text
note/<task_name>/contracts/
note/<task_name>/plans/
note/<task_name>/logs/
```

When code ownership changes, update `00_repository_architecture_map.md`.
