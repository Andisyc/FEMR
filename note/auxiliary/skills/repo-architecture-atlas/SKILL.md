---
name: repo-architecture-atlas
description: Create or update repository architecture atlas maps that connect method design, module design, runtime flow, and concrete code locations. Use when the user asks for 仓库流程图, 架构图, architecture atlas, module ownership maps, method-to-code maps, FEMR/MOSAIC architecture notes, or diagrams that must explain how a research method becomes modules and code.
---

# Repo Architecture Atlas

## Core Contract

Show the chain from method design to module design to concrete code. Do not make a pretty diagram that hides ownership, interfaces, or implementation status.

For each important concept, try to record:

- Method design: what problem is solved, what signal/data object is used, and what assumption is being made.
- Module design: which module owns the behavior, what enters and leaves it, and what it must not decide.
- Concrete code: file path, class/function/block id, line hint when available, and test/probe evidence when the claim depends on runtime behavior.

## Workflow

1. Read the local architecture contract first, usually `note/architecture/README.md`.
2. Inspect the nearest existing atlas data file before creating a new format.
3. Read the design note or user discussion that defines the method.
4. Inspect code architecture with CodeGraph and UnderstandAnything first. Use targeted `rg` and file reads only after the graph tools identify likely owners or when the graph is stale/missing.
5. Update atlas data JSON first. Edit the HTML viewer only when the current layout cannot express the map.
6. Validate JSON parsing and report which claims are code-confirmed, runtime-confirmed, or still conceptual.

## Code Reading Priority

Use graph tools before manual reading when mapping architecture:

1. Use CodeGraph first for symbol ownership, call paths, callers/callees, impacted files, and precise file/function locations.
2. Use UnderstandAnything for whole-repo topology, cross-module relationships, and unfamiliar architecture areas.
3. Use targeted `rg` only for exact text search, config names, logs, or when graph coverage is missing.
4. Use direct file reads only to confirm final evidence, inspect edited files, or resolve stale graph results.

Do not reconstruct repository architecture by broad manual grep/read loops when CodeGraph or UnderstandAnything can answer the structural question directly.

## Atlas Files

Prefer the existing FEMR/MOSAIC atlas structure:

- `note/architecture/architecture/*.data.json`: repository and module ownership maps.
- `note/architecture/runtime/*.data.json`: runtime flow and boundary maps.
- `note/architecture/concept/*.data.json`: method concept maps.
- `note/architecture/auxiliary/atlas_app/architecture_atlas.html`: viewer.

Use the current atlas layout types unless there is a clear mismatch:

- `repo_tree`: file and module ownership.
- `flow_tree`: runtime path, input/output boundary, diagnostics.
- `tabs`: concept families or method taxonomy.
- Method-to-code chain: use `flow_tree` or `repo_tree` now; add a dedicated layout later only if needed.

## Design Rules

- Preserve the atlas rule: same code block id means same concept name, same color, and same code location.
- Mark unimplemented ideas explicitly as `planned`, `not implemented`, or `legacy`; do not imply they exist in code.
- Keep node names short and human-readable.
- Put long explanations in descriptions, not labels.
- Use code paths as evidence, not decoration.
- If a runtime claim depends on tensor shape, mask, gradient, reset behavior, or saved artifact content, pair this skill with `runtime-probing-debug`.

## Relation To Other Skills

- Use CodeGraph as the first code-intelligence layer for owner symbols, call flow, and concrete code locations.
- Use `understand-anything` as the first whole-repo graph layer when the repo is large or unfamiliar and a machine graph helps locate modules and relationships.
- Use `human-readable-html` when building or polishing the viewer page itself.
- Use `runtime-probing-debug` when the architecture map must reflect live training, rollout, storage, or checkpoint behavior.
- Use this skill as the bridge between research method intent and repository structure.

## Quality Gate

Before finishing, check:

- The map says what method it is explaining.
- Every central method concept has a module owner or an explicit missing-status label.
- Every central module has a concrete file path.
- Runtime edges are not guessed when a probe or test is required.
- The final answer names the modified atlas files and the remaining uncertain points.
