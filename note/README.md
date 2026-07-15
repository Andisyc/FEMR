# FEMR Notes Index

This folder stores human-readable engineering memory for FEMR and FrontRES work.

## First-Level Documents

- `00_repository_architecture_map.md`: global module ownership and repository structure.
- `architecture/`: visual architecture atlas and its editable data sources.
- `testing/`: current test inventory, control board, semantic objects, impact
  rules, and dated evidence ledgers.

## Task Folders

- `frontres_core/`: FrontRES active contracts, contract history, checklists,
  paper notes, and discussion logs. Segment Replay is part of the active
  FrontRES contract set, not a separate task folder.

## Organization Rule

Global repository structure belongs in first-level note documents.
Current FrontRES method truth belongs in `frontres_core/contracts/active/`.
Superseded contracts and preserved source material belong under
`frontres_core/contracts/history/`.

Do not scatter new multi-step planning notes in the root of `note/`.
For a complex implementation, use the existing task owner and keep lifecycle
classes separate:

```text
note/<task_name>/contracts/
note/<task_name>/plans/
note/<task_name>/checklists/
note/<task_name>/logs/ or evidence/
```

When code ownership changes, update `00_repository_architecture_map.md`.
Plans and checklists are current replaceable views; do not append history to
them. Testing views are also refreshed current-state indexes, while dated
evidence ledgers remain immutable evidence.
