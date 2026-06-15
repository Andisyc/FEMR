# MOSAIC Architecture Atlas

This folder stores human-readable architecture maps for FrontRES work.

The maps use one shared rule:

```text
same Code Block ID
  -> same concept name
  -> same color
  -> same code location
```

## Current Prototype

- `03_frontres_concept_tabs.data.json`: editable source data for the concept-tab map.
- `03_frontres_concept_tabs.mmd`: Mermaid structural source.
- `frontres_concept_tabs.html`: interactive rough.js viewer.
- `serve_architecture.mjs`: local auto-refresh server for VSCode side-by-side editing.
- `render_rough_arch_svg.mjs`: static SVG renderer with a rough hand-drawn style.
- `03_frontres_concept_tabs.svg`: generated static visual artifact.

## VSCode Workflow

```bash
cd note/architecture
npm run serve
```

Open this URL on the right side of VSCode:

```text
http://127.0.0.1:8765/frontres_concept_tabs.html
```

Open `03_frontres_concept_tabs.data.json` on the left. Saving the JSON refreshes
the graph automatically.

## Static SVG

```bash
node note/architecture/render_rough_arch_svg.mjs
```

## ID Convention

- `P-*`: real problem layer.
- `C-*`: concept variable layer.
- `M-*`: engineering owner/module layer.
- `R-*`: runner code block.
- `A-*`: algorithm code block.
- `S-*`: storage contract block.
- `D-*`: diagnostics block.
