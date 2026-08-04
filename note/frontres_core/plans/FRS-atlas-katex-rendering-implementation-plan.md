# Atlas 07 KaTeX Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the three Repair Gain equations as offline typeset LaTeX while preserving every existing plain-text Atlas card.

**Architecture:** The existing `design_transaction_inspector` remains the only rendering owner. Detail rows may be either legacy strings or `{text, latex}` objects; one local KaTeX helper renders explicit formulas through SVG `foreignObject` and falls back to source text on errors.

**Tech Stack:** Existing HTML/SVG viewer, local KaTeX ESM/CSS, Node contract checks, in-app browser verification.

---

### Task 1: Add Backward-Compatible Formula Rows

**Files:**
- Modify: `note/architecture/auxiliary/atlas_app/package.json`
- Modify: `note/architecture/auxiliary/atlas_app/package-lock.json`
- Modify: `note/architecture/auxiliary/atlas_app/architecture_atlas.html`
- Modify: `note/architecture/runtime/07_frontres_design_contract_review.data.json`
- Modify: `note/architecture/auxiliary/atlas_app/check_design_contract_review.mjs`

- [x] **Step 1: Extend the focused contract before production code**

Require legacy string rows or exact `{text, latex}` formula rows, require the
three confirmed expressions, and reject arbitrary row fields.

- [x] **Step 2: Run the focused contract and verify it fails**

Run: `node check_design_contract_review.mjs` from the Atlas app directory.
Expected: FAIL because the viewer has no KaTeX import/helper and Atlas 07 still
stores the equations as ordinary strings.

- [x] **Step 3: Install local KaTeX and implement the existing-owner renderer**

Import `katex.mjs` beside the existing Rough.js import, load local KaTeX CSS,
add one `addLatex()` helper with source-text fallback, normalize detail rows,
and compute content-driven row/card/canvas heights. Do not create another
viewer or layout owner.

- [x] **Step 4: Store the three formulas explicitly**

Convert only the three Repair Gain rows to `{text, latex}`. Keep all other rows
as strings.

- [x] **Step 5: Run focused and complete static checks**

Run: `npm run check` from the Atlas app directory.
Expected: all viewer, figure, repository, Transaction, module, and evidence
checks pass.

- [x] **Step 6: Verify the served page**

Open Atlas 07, select `Repair Gain`, confirm three `.katex` expressions, confirm
ordinary cards still render, inspect console errors, and verify no text or
formula crosses its SVG/card boundary.

- [x] **Step 7: Close documentation state**

Mark this plan complete only after browser evidence exists. Keep the semantic
proposal and active contracts unchanged.
