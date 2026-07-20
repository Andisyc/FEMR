# MOSAIC Architecture Atlas

This folder stores human-readable architecture maps for FrontRES/FEMR work.

The maps use one shared rule:

```text
same Code Block ID
  -> same concept name
  -> same color
  -> same code location
```

## Current Maps

- `architecture/01_repo_architecture.data.json`: editable source data for the ultra-wide Repository Reading Atlas.
- `runtime/02_frontres_flow.data.json`: editable source data for the Concept Figure-driven Method-to-Code Reading Atlas.
- `runtime/04_stage3_formal_runtime_audit.data.json`: permanent Runtime Audit Atlas for the Concept Figure-mapped Stage 3 Phase B probe route.
- `runtime/05_policy_quality_audit.data.json`: Policy Quality Audit Atlas for the eight `QUALITY-*` causal owners and their Q evidence status.
- `runtime/06_frontres_design_point_review.data.json`: human-facing grouped two-column table of FrontRES design questions and atomic design points; detailed implementation material remains in the active contracts and other atlas views.
- `concept/03_frontres_concept_tabs.data.json`: editable source data for the paper-style FrontRES method figure.
- `auxiliary/atlas_app/`: current helper viewer, local server, checks, and JS dependencies.
- `auxiliary/legacy/`: retired viewer/render helpers kept outside the active app.

## Folder Contract

```text
note/architecture/
  architecture/   repo/file/block mind map
  runtime/        module interface contract map
  concept/        FrontRES design concept tabs
  auxiliary/      helper app files kept out of the map folders
  index.html      clean entry page
```

## Map Lifecycle

Temporary maps are allowed, including in the main entry page, while they are
actively guiding a change. After the change lands, a temporary map must be
either deleted or integrated into one of the active maps.

The main entry should stay small: current repo map, method/runtime maps,
concept tabs, and explicitly named diagnostics pages.

## VSCode Workflow

```bash
cd note/architecture
node auxiliary/atlas_app/serve_architecture.mjs
```

Open one of these URLs on the right side of VSCode:

```text
http://127.0.0.1:8765/
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../architecture/01_repo_architecture.data.json
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../runtime/02_frontres_flow.data.json
http://127.0.0.1:8765/04_stage3_formal_runtime_audit.html
http://127.0.0.1:8765/06_frontres_design_point_review.html
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../concept/03_frontres_concept_tabs.data.json
```

Open the matching `*.data.json` on the left. Saving the JSON refreshes the graph
automatically. The atlas page also polls the current JSON file, so it still
updates even if an older server process is running.

Viewer controls:

- The built-in JSON editor is hidden by default so the graph uses the full page.
- `Show Editor` opens the built-in JSON editor when quick in-browser edits are useful.
- `+`, `-`, `Fit Width`, and `100%` control graph zoom.
- `Fit Width` also restores auto-fit behavior after manual zooming.
- Drag the graph canvas to pan. Trackpad horizontal scroll also works on large maps.
- `Ctrl`/`Cmd` + wheel zooms around the pointer.

## HTML Design Contract

The current atlas uses one reusable HTML viewer:

```text
auxiliary/atlas_app/architecture_atlas.html
  -> loads one *.data.json through ?data=...
  -> chooses renderer by data.layout
  -> draws rough SVG cards with shared colors, IDs, zoom, pan, editor, and live reload
```

The three main pages are data variants, not separate applications:

- Repository Architecture uses `layout: "repository_reading_atlas"`.
  - Source: `architecture/01_repo_architecture.data.json`.
  - Purpose: read the whole repository as runtime-ordered module-family cards.
  - Reading direction: module-family cards follow `runtimeOrder[]` directly
    from left to right. Inventory-style system containers are not rendered.
  - Every card shows responsibility, read-first files, key functions, core
    objects, and the module-internal formal main path.
  - `mainRoute[]` and `mainRouteTitles[]` define matching `B1/B2/...` steps.
    Each rendered step exposes a human title, owner, input, and output.
  - Cards omit eval/debug/legacy branches. Those belong in separate Runtime
    Atlas views when they are the subject of review.
  - Non-main-path repository context uses `supportOrder[]`; it is rendered in a
    separate Supporting Boundaries row and never inserted into the formal route.
  - Main schema: `systems[].modules[]`, with `files[]`, `objects[]`, and
    `mainRoute[]` as the human code-reading contract.

- Method to Code uses `layout: "repository_reading_atlas"`.
  - Source: `runtime/02_frontres_flow.data.json`.
  - Purpose: Concept Figure design point -> coherent owner module family -> internal formal route.
  - Main schema: `runtimeOrder[]`, `supportOrder[]`, `systems[].modules[]`, matching the 01 reading-card layout.
  - Each card names one Concept Figure design point and exposes responsibility, read-first files, functions, objects, and B1/B2/B3 route.

- Design Questions And Points uses `layout: "design_point_table"`.
  - Source: `runtime/06_frontres_design_point_review.data.json`.
  - Purpose: a minimal human index of the method, not a contract, code reader, or evidence audit.
  - Conceptual parent headings locate atomic decisions; every visible row still contains only `设计问题` and `设计点`.
  - No IDs, code paths, owner names, status, evidence or expandable technical detail are rendered.
  - Active contracts and the other architecture maps remain the detailed reference material for agents and implementation review.

- Concept uses `layout: "method_figure"`.
  - Source: `concept/03_frontres_concept_tabs.data.json`.
  - Purpose: show the active method as one causal paper figure before exposing code ownership.
  - Main schema: `title`, `subtitle`, `claim`, `zones[]`, `nodes[]`, `edges[]`, `callouts[]`, `acceptance[]`.
  - Nodes keep stable block IDs, concise method summaries, evidence status, and secondary `codeRefs` metadata.
  - Edges expose forward execution, paired comparison, PPO feedback, replay-priority feedback, and evidence boundaries.
  - Method-to-code and runtime maps remain separate engineering views that reuse the same IDs and concept colors.


## Reuse Contract

For another LLM Agent: this atlas is meant to be reused by copying the whole
folder, not by copying a single HTML file. The folder is a small self-contained
viewer plus JSON map sources.

Copy this directory into the new project:

```text
note/architecture/
```

The copied folder should keep this shape:

```text
note/architecture/
  index.html
  README.md
  architecture/
    *.data.json
  runtime/
    *.data.json
  concept/
    *.data.json
  auxiliary/atlas_app/
    architecture_atlas.html
    serve_architecture.mjs
    package.json
    package-lock.json
```

In the new project, start the viewer from the copied folder:

```bash
cd note/architecture
npm --prefix auxiliary/atlas_app install
node auxiliary/atlas_app/serve_architecture.mjs
```

Then open:

```text
http://127.0.0.1:8765/
```

To reuse the current HTML page for a specific map, create or edit a
`*.data.json` file and open:

```text
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../PATH/TO/MAP.data.json
```

Choose the `layout` field by the thinking task:

- Use `repository_reading_atlas` when the question is "how is the whole
  repository divided, and in what order should I read its code?".
- Use `method_figure` when the question is "what is the method, why does it work, and how does its feedback loop close?".
- Use `repo_tree` when the question is "which file owns which code block?".
- Use `flow_tree` when the question is "what enters a module, what does it own, what exits, and what is forbidden?".
- Omit `layout` or use `tabs` when the question is conceptual taxonomy rather than code ownership.

Reusable parts:

- Page shell: header, hidden editor, status, live reload, zoom, fit-width, pan.
- Drawing helpers: `drawHeader`, `drawLegend`, `drawCard`, `wrapText`, `conceptColor`.
- Shared visual grammar: Code Block IDs, concept color IDs, rough SVG cards, Chinese explanatory text with stable English names.
- Data-driven rendering: a new map should usually require only a new JSON file and an `index.html` link.

Non-reusable parts without refactoring:

- The renderer functions are currently embedded in `architecture_atlas.html`, not exported as a JS library.
- Adding a fourth layout still requires editing `architecture_atlas.html`.
- Cross-file automatic consistency checks are not built into the viewer; consistency is maintained by the JSON contract and review.

New-project adaptation checklist for another LLM Agent:

- Keep `auxiliary/atlas_app/architecture_atlas.html` unchanged at first.
- Replace the example JSON content with the new project's architecture data.
- Update `index.html` links so they point to the new JSON files.
- Keep stable English names in `title` / module labels when they identify code concepts.
- Put explanations, roles, risks, and diagnostics in Chinese if the project owner reads Chinese.
- Preserve Code Block IDs and concept color IDs across maps when the same concept appears in multiple diagrams.
- Do not split the HTML into a JS library unless the viewer itself becomes difficult to maintain.

If the atlas grows further, the next engineering step should be to split the
embedded script into:

```text
viewer_shell.js       shared loading, editor, status, zoom, pan
render_helpers.js    SVG text, cards, colors, wrapping
layouts/             repo_tree.js, flow_tree.js, tabs.js
```

Do not do this split merely because one map changes. Do it only when the HTML
itself becomes a maintenance bottleneck.

## ID Convention

- `P-*`: real problem layer.
- `C-*`: concept variable layer.
- `M-*`: engineering owner/module layer.
- `R-*`: runner code block.
- `A-*`: algorithm code block.
- `S-*`: storage contract block.
- `D-*`: diagnostics block.
- `DR-*`: DR curriculum / GMT frontier block.
- `F-*`: executable floor block.
- `SR-*`: Segment Replay block.
- `Q-*`: repair-quality / gain block.
- `G-*`: diagnostics block.
