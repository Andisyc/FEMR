---
name: human-readable-html
desc: Create polished, human-readable HTML pages for research notes, engineering summaries, group-meeting reports, architecture explanations, method walkthroughs, and visual work updates. Use when the user asks for an HTML page, HTML note, HTML report, readable presentation page, visual research summary, or asks to convert Markdown/text notes into an attractive page for humans.
---

# Human Readable HTML

Use this skill to create HTML pages meant for people to read, discuss, and present. The goal is not a frontend product. The goal is a clear research or engineering artifact that looks good in a browser and can be opened during a meeting.

## Core Rule

Design for human scanning first:

- make the main argument visible in the first viewport;
- use sections, cards, timelines, tables, diagrams, and callouts to expose structure;
- avoid plain Markdown dumped into HTML;
- avoid decorative complexity that does not explain the work;
- keep text short enough for meeting reading.

## Page Types

Choose the page form from the task:

- Research method page: problem, bottleneck, idea, pipeline, training signal, risks.
- Engineering note page: module ownership, data flow, interfaces, tests, status.
- Reading or learning page: categories, reading order, why each item matters, checklist.
- Group-meeting page: motivation, current progress, evidence, next decisions.
- Architecture map: modules, arrows, ownership, runtime paths, contracts.

## FEMR / MOSAIC Note Contract

When writing inside `note/architecture/` or when the page is an architecture/map viewer, prefer the existing atlas pattern:

- put editable content in a `*.data.json` file;
- render it through `note/architecture/auxiliary/atlas_app/architecture_atlas.html`;
- do not create an isolated HTML viewer unless the interaction model is genuinely different;
- keep the shared rule: same Code Block ID -> same concept name -> same color -> same code location.

For non-architecture note pages, a standalone HTML file is acceptable, but it must still be presentation quality.

## Visual Standards

Use a restrained research-page style:

- one clear typography system;
- 2-4 section colors with semantic meaning;
- high contrast body text;
- enough whitespace, but not a marketing landing page;
- no nested cards;
- no vague gradient/orb decoration;
- no huge hero if the content is a note;
- responsive layout for laptop and projector widths;
- readable at 100% browser zoom.

Prefer these components:

- overview strip for the main takeaway;
- section cards for comparable items;
- timeline for staged plans;
- matrix/table for tradeoffs;
- callout for decisions or risks;
- checklist for review questions;
- simple SVG or Mermaid only when it clarifies a relation.

## Content Rules

Write in the user's preferred style:

- direct Chinese;
- short sentences;
- no inflated explanation;
- concrete labels over abstract slogans;
- method and engineering logic should be visible without reading every paragraph.

For research pages, keep this order unless the user asks otherwise:

1. What problem is being solved.
2. Why the old method is insufficient.
3. What the new object or mechanism is.
4. How the pipeline runs.
5. What evidence or tests support it.
6. What remains uncertain.

For engineering pages, keep this order:

1. Entry point.
2. Data object.
3. Module owner.
4. Runtime path.
5. Test/probe evidence.
6. Open risk.

## Implementation Rules

- Keep HTML self-contained unless the repo already provides a viewer.
- Do not depend on remote CDNs.
- Use semantic HTML: `header`, `main`, `section`, `article`, `footer`.
- Keep CSS in the file for portable notes unless the folder has a shared style system.
- Use stable widths and responsive grids.
- Avoid viewport-scaled body fonts.
- Use ASCII in source unless Chinese content is required.
- When replacing an ugly page, preserve the useful content and redesign the structure instead of merely changing colors.

## Quality Check

Before finishing, check:

- first viewport tells the reader what the page is about;
- headings form a useful outline;
- long lists are grouped into visible blocks;
- links are distinguishable;
- page still works without internet;
- mobile width does not overlap;
- the page can be used directly in a group meeting.

If the page belongs to an existing repo note system, also check whether a local design contract exists before choosing the file format.
