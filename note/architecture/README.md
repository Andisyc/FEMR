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

- `architecture/01_repo_architecture.data.json`: editable source data for the VSCode-style repo map.
- `runtime/02_frontres_flow.data.json`: editable source data for the Interface Contract Map.
- `concept/03_frontres_concept_tabs.data.json`: editable source data for the concept-tab map.
- `concept/03_frontres_concept_tabs.mmd`: Mermaid structural source.
- `concept/03_frontres_concept_tabs.svg`: generated static visual artifact.
- `auxiliary/atlas_app/`: helper viewer, local server, static renderer, checks, and JS dependencies.

## Folder Contract

```text
note/architecture/
  architecture/   repo/file/block mind map
  runtime/        module interface contract map
  concept/        FrontRES design concept tabs
  auxiliary/      helper app files kept out of the map folders
  index.html      clean entry page
```

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

## Static SVG

```bash
node note/architecture/auxiliary/atlas_app/render_rough_arch_svg.mjs
```

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
- `AL-*`: state alpha block.
- `RH-*`: structured rho block.
- `G-*`: diagnostics block.
