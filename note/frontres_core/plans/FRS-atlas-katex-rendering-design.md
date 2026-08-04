# Atlas 07 KaTeX Rendering Design

Status: implemented; six-section layout pending final browser verification.
Date: 2026-07-31.

## Objective

Render mathematical expressions in the Atlas 04 Design Inspector as
typeset LaTeX instead of plain SVG text. Keep the shared Transaction layout
unchanged. Repair Gain uses a six-section vertical derivation so each physical
explanation remains adjacent to the formula it defines.

## Design

- Add local KaTeX as an Atlas app dependency. The page must not require a CDN
  or network access.
- Extend a detail row with an optional heading and one or more explicit LaTeX
  expressions. Existing string rows and a single LaTeX string remain valid.
- The existing `design_transaction_inspector` renderer remains the sole owner.
  It renders the formula inside an SVG `foreignObject`, using KaTeX only for
  the explicitly supplied expression.
- Do not infer formulas by parsing arbitrary text. Natural-language text and
  LaTeX remain separate data fields.
- Detail rows receive content-driven height from heading, wrapped explanation,
  and formula count. Every formula occupies its own line; Repair Gain does not
  use multi-line `aligned` blocks whose actual height is hidden from layout.
- Thin separators preserve the six-step reading order without nesting cards.
- Atlas 04 opens at 80% zoom with horizontal scrolling instead of shrinking the
  1680px canvas to fit a narrow viewport.
- If KaTeX cannot load or a formula is malformed, render the source expression
  as readable text and expose a console error. The Atlas must remain readable.

## Initial Scope

Only the Atlas 07 `Repair Gain` detail card receives the six-section derivation:

1. Intent, Physics, and Segment Replay responsibilities.
2. Recovery-Aware classification rules.
3. Clean anchor and fixed-scale normalization.
4. Intent Gain and per-constraint Physics Gain.
5. Recovery pressure and its weighted contribution.
6. Total Gain.

Its eight independent formula lines cover:

\[
r_j(X\mid Clean)=D_j(X,Clean)/S_j,
\]

\[
G_I^{(m)},\qquad G_{P,j}^{(m)}
\]

\[
P_{N,j},\qquad P_{R,j}^{(m)},\qquad
\lambda_{RA,j}^{(m)},\qquad \lambda_{RA,j}^{(m)}G_{P,j}^{(m)}
\]

\[
G_{total}^{(m)}=G_I^{(m)}+\sum_j\lambda_{RA,j}^{(m)}G_{P,j}^{(m)}
-\beta C_{repair}^{(m)}.
\]

The renderer helper is reusable by other Atlas layouts, but this change does
not migrate unrelated cards.

## Preserved Boundaries

- No Concept Figure semantic change.
- No active-contract activation or Gain implementation change.
- No training, evaluation, checkpoint, or deployment code change.
- No change to the ten design-point tabs or the thirteen-step Transaction
  spine.
- The current plain-text detail schema remains backward compatible.

## Verification

1. Parse Atlas 07 data and run `check_design_contract_review.mjs`.
2. Run the complete Atlas app static check suite.
3. Open Atlas 07, select `Repair Gain`, and verify all eight independent
   formulas render as typeset math at the default 80% zoom.
4. Verify ordinary text cards still render and no formula or text crosses its
   card or SVG boundary.
5. Verify malformed-formula fallback without changing the stored method
   semantics.

## Stop Condition

Stop without broadening scope if SVG `foreignObject` is unsupported in the
served Atlas, local KaTeX cannot be loaded without changing the server
authority, or formula layout requires rewriting other Atlas layouts.
