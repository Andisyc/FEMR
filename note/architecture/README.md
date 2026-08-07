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

The human-facing page order is:

```text
01 Method Figure -> 02 Design Inspector -> 03 Module Inspector
-> 04 Code Quality Evidence -> 05 Module Test Atlas
```

The data filenames below keep their established registry names; page numbering
describes the human reading order rather than renaming semantic registries.

- `architecture/01_repo_architecture.data.json`: single owner registry for the default FEMR Module Inspector, including the ordered A-E Evaluation function chains, and the explicit ultra-wide Repository Reading fallback.
- `architecture/02_code_quality_evidence.data.json`: source-scanned function and B-block projection aligned with the Module Inspector order; Evaluation can switch between complete chain-ordered and file-ordered views from the same A-E registry. Exclusive, shared, and unassigned counts must conserve the full scanned function inventory.
- `runtime/04_frontres_design_inspector.data.json`: interactive Design Inspector: ten compact parent design points highlight where their fine-grained decisions change one shared Stage-3 Transaction spine. Perturbation Data, K-step Curriculum and Actor & Critic Warmup project the active TRAIN-v015 nested K-DR and fixed split-LR identity; E-FI-135 closes optimizer/config/telemetry/checkpoint-v10 behavior offline.
- `runtime/04_frontres_design_register.md`: approved interaction and language contract for the Design Inspector.
- `testing/05_frontres_module_test_atlas.data.json`: Module Test Atlas with eighteen completed module cards plus a compact Formal Runtime Audit stage-reading card. The module cards use human-readable `伪样本 | 正确结果 | 证明什么` cases; the stage card explains the Phase A method/code alignment gate and the Phase B official-route runtime gate without pretending to be a nineteenth module test.
- `testing/05_frontres_module_test_register.md`: source and lifecycle contract for the Module Test Atlas.
- `concept/03_frontres_concept_tabs.data.json`: editable source data for the paper-style FrontRES method figure. Its `Repair Gain -> FrontRES` interaction records the active Clean-anchored Recovery-Aware design. E-FI-101 closes deterministic Step-1 source alignment; E-FI-102 closes baseline-capture, transaction-Aggregate and active-import P1 maintenance findings. The bounded official transaction and policy quality remain unconfirmed runtime evidence.
- `concept/08_trajectory_conditioned_execution_alignment.data.json`: current candidate paper-style Concept Figure for the offline within-Intent calibration loop. It exposes Planner, trapezoidal Tracker Encoder/Decoder, an inverted Context Encoder above the bottleneck, one shared Rollout block for first-trajectory Context and second-execution evidence, and privileged supervision; deployment details remain in the concept note rather than the primary figure.
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
  open_atlas.command durable local-server launcher
```

## Map Lifecycle

Temporary maps are allowed, including in the main entry page, while they are
actively guiding a change. After the change lands, a temporary map must be
either deleted or integrated into one of the active maps.

The main entry should stay small: current repo map, method/runtime maps,
concept tabs, and explicitly named diagnostics pages.

## VSCode Workflow

Preferred one-command launcher on macOS:

```bash
./note/architecture/open_atlas.command
```

It starts or reuses the Atlas server and opens the detailed design-contract
review. Set `ATLAS_PAGE=/` to open the index or `PORT=...` to choose another
local port.

Manual server workflow:

```bash
cd note/architecture
node auxiliary/atlas_app/serve_architecture.mjs
```

Open one of these URLs on the right side of VSCode:

```text
http://127.0.0.1:8765/
http://127.0.0.1:8765/01_frontres_method_figure.html
http://127.0.0.1:8765/02_frontres_design_inspector.html
http://127.0.0.1:8765/03_femr_module_inspector.html
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../architecture/01_repo_architecture.data.json&view=repository_reading
http://127.0.0.1:8765/04_code_quality_evidence_atlas.html
http://127.0.0.1:8765/05_frontres_module_test_atlas.html
http://127.0.0.1:8765/auxiliary/atlas_app/architecture_atlas.html?data=../../concept/08_trajectory_conditioned_execution_alignment.data.json
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

The main Atlas pages are data variants, not separate applications:

- FEMR Module Inspector is the default view of `architecture/01_repo_architecture.data.json`.
  - `moduleInspector.stages[]` defines the unchanged seven-step Training Main Loop.
  - The top index contains every runtime and supporting module exactly once.
  - Selecting a module highlights its owning Training Main Loop stage without replacing or reordering the spine.
  - The bottom card exposes only that module's responsibility, owner files, and ordered `B-step -> file -> function` chain.
  - Chains longer than four steps wrap as a continued route rather than shrinking text or creating an ultra-wide canvas.
  - The Inspector and the fallback share `systems[].modules[]`; no second owner or function inventory exists.

- The explicit `view=repository_reading` fallback uses `layout: "repository_reading_atlas"`.
  - Source: `architecture/01_repo_architecture.data.json`.
  - Purpose: inspect every runtime-ordered module-family card simultaneously when a wide comparison is necessary.
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

- FrontRES Design Inspector uses `layout: "design_transaction_inspector"`.
  - Source: `runtime/04_frontres_design_inspector.data.json`.
  - Purpose: inspect how every accepted design point participates in the same formal Stage-3 Transaction.
  - Each compact button names one canonical parent design point. Selection changes the highlighted Transaction steps and one minimal reading card below the spine; every numbered card row is a fine-grained method decision.
  - The shared spine covers pre-Transaction initialization, Segment collection and K-step execution, paired evidence, grouped update, and committed curriculum state.
  - The detail card contains no field-category headings, implementation evidence, source links, risk panels, matrices, or review prose; those remain in their authoritative documents.

- FrontRES Module Test Atlas uses `layout: "module_test_inspector"`.
  - Source: `testing/05_frontres_module_test_atlas.data.json`.
 - Purpose: let the human inspect concrete design-driven pseudo-sample tests for every module before tests are implemented or run.
 - The shared spine is `确认设计规则 -> 构造简单伪样本 -> 手算正确结果 -> 确认测试题 -> 执行模块并逐项比较 -> 定位第一个错误 -> 记录通过与失败证据`.
 - Each selected card contains one plain-language rule and a `伪样本 | 正确结果 | 证明什么` table. Generic responsibility/interface metadata is deliberately absent from the primary reading card.
- The eighteen module cards cover the formal runtime module families and report `18 passed / 0 partial / 0 blocked`. Formal Runtime Audit Phase A reviewed DP01-DP10 offline; E-FI-128 runtime-confirms the complete B01-B08 official transaction. Policy quality remains separate: E-FI-129 closes the cross-environment support-foot coordinate blocker offline, while actual policy efficacy still requires evaluation.
 - A separate `Formal Runtime Audit` stage card explains the current Phase A -> Phase B progression. It has its own reading spine and does not count as a module card or claim policy quality.

- FrontRES Runtime Audit Atlas uses `layout: "repository_reading_atlas"`.
  - Source: `runtime/06_frontres_runtime_audit_atlas.data.json`.
  - Entrypoint: `06_frontres_runtime_audit_atlas.html`.
  - Purpose: let the human review the eight Phase B official-route edge cards before any probe is inserted or live run starts.
  - Every card maps to existing Concept Figure design IDs and shows the formal owner, upstream/probe/downstream route, expected runtime fact and E-FI-128 runtime result.
  - The control surface proves formal connectivity only. It does not promote the bounded critic-only transaction to policy-quality evidence.

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
  open_atlas.command
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
- Use `design_transaction_inspector` when the question is "where does each accepted design point act inside the same formal training Transaction?".
- Use `module_test_inspector` when the question is "what must this module calculate or mutate correctly, and what independent artificial cases prove it?".
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
