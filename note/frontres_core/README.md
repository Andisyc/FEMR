# FrontRES Core Notes

This folder contains the original FrontRES design and implementation notes.

## Layout

- `../architecture/concept/03_frontres_concept_tabs.data.json`: human-facing
  Concept Figure and canonical top-level design-point names.
- `contracts/README.md`: active contract registry and history access rule.
- `contracts/active/`: current method and subsystem contracts.
- `contracts/history/`: superseded, rejected, ablation, and migration records;
  never a default reading source.
- `contracts/design_contract.md`: compatibility pointer to the registry.
- `plans/engineering_plan.md`: replaceable current implementation plan.
- `checklists/modification_checklist.md`: replaceable current acceptance state.
- `paper/method_outline.md`: current paper-facing method view.
- `logs/`: raw discussion or decision history.

Segment Replay method truth belongs in the registered active contracts under
`contracts/active/`. The retired standalone folder is preserved as restricted
source material under `contracts/history/sources/segment_replay/`.

Read the Concept Figure first to recover human intent, then use the Design
Point Register in `contracts/README.md` to reach the detailed active contract
section for each visible block. Do not reconstruct top-level design points from
the engineering plan, code modules, or historical notes.

Pre-Segment paper drafts are preserved under
`contracts/history/sources/paper/` and are never default reading sources.
