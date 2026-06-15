---
name: profleo
title: ProfLeo Paper Style
description: "Revise ML/robotics paper prose in ProfLeo style: short direct sentences, problem-first logic, explicit gaps, simple method statements, and low AI flavor. Especially useful for FrontRES/FEMR introduction, abstract, and contribution paragraphs."
version: 0.1.0
author: local
license: private
metadata:
  hermes:
    tags: [Research, Paper Writing, Robotics, FrontRES, FEMR, LaTeX, Introduction]
    category: research
    related_skills: [research-paper-writing]
    requires_toolsets: [terminal, files]
---

# ProfLeo Paper Style

Use this skill when the user asks for paper writing that should resemble the
style distilled from `scalemargin.tex`.

The goal is not literary polish. The goal is clear research writing with short
sentences, visible logic, and low AI flavor.

## Core Style

- One sentence should do one job.
- Prefer simple subject-verb-object structure.
- Put the technical object early in the sentence.
- Use short logical connectors: `However`, `As a result`, `In this work`,
  `To achieve this`, `In particular`, `In summary`.
- Explain a new term after the phenomenon is already clear.
- State the gap directly. Do not hide it inside a long subordinate clause.
- State the method directly. Do not over-motivate before naming it.

## Introduction Pattern

Follow this paragraph-level structure:

1. Background and importance.
2. Main problem in current setting.
3. Existing explanation or existing methods.
4. Missing part or under-studied issue.
5. Proposed method.
6. Contributions and evidence.

Each paragraph should have a clear role. Do not let one paragraph both define
the problem, justify the method, and preview experiments.

## Sentence Patterns

Use patterns like these:

```text
X plays a crucial role in Y.
However, X often struggles with Z.

Recent research identifies that A and B contribute to Z.
A occurs when ...
B means that ...

Despite significant progress in A, B has not been fully studied.
As shown in Fig. X, ...
As a result, ...

In this work, we aim to ...
To achieve this, we ...
In particular, we propose ...
This approach ensures that ...

In summary, our contributions are ...
```

For FrontRES/FEMR, adapt them as:

```text
Motion trackers can reproduce complex humanoid motions.
However, they often assume that the reference motion is dynamically executable.

This assumption is not always valid for video-reconstructed motions.
Such motions may look natural in pose space.
However, small errors in root height, body orientation, contact timing, or
horizontal displacement can make them dynamically fragile.

We refer to this mismatch as the kinematic-to-dynamic gap.
This gap means that tracking failure can come from the upstream reference,
not only from the tracker.

In this work, we propose FEMR, a front-end motion refiner placed before a
frozen motion tracker.
FEMR predicts task-space corrections for the reference motion.
It aims to improve executability rather than replace the tracker.
```

## Chinese Drafting Pattern

When drafting in Chinese first, use plain research Chinese:

```text
现有方法已经能够...
然而，这些方法通常假设...

这一假设在...中并不总是成立。
此类数据通常...
但是，它们仍然可能...

我们将这一问题称为...
该问题说明...

为了解决这一问题，我们提出...
该方法的目标不是...
具体来说，...
```

Avoid ornate Chinese clauses such as:

```text
虽然...但整体仍然...然而...
在...的情况下，由于...从而导致...
这意味着...并不一定...也可能...
```

Split them into two or three sentences.

## What To Avoid

- Long `although/while/where/which` chains.
- Sentences with more than two commas unless they list examples.
- Vague pronouns such as `it`, `they`, and `its` when the antecedent is not
  immediately obvious. Repeat the technical noun instead.
- Over-abstract words before examples.
- Method descriptions that sound like marketing.
- Phrases such as `It is worth noting that`, `This highlights the fact that`,
  `plays an important role in enabling`.
- Saying `may` repeatedly when the paper is making a claim.
- Verbs that sound dramatic but imprecise, such as `damage` for stability.
  Prefer `reduce`, `weaken`, or `decrease`.
- Terms that sound plausible but are not standard in the field. If unsure,
  replace them with a more explicit phrase.

## Revision Checklist

For each paragraph, check:

- Can the paragraph role be named in one phrase?
- Is the first sentence simple enough?
- Is the gap stated directly?
- Are examples placed after the claim, not before?
- Can any sentence be split without losing meaning?
- Does the paragraph end with the point readers should remember?

For each sentence, check:

- Is the subject close to the verb?
- Is there only one main action?
- Are modifiers necessary?
- Would a human advisor write this sentence on a first clean pass?

## User Comment-Derived Rules

These rules come from revision feedback on the FrontRES introduction.

- In a requirement paragraph, describe the requirement before revealing the
  exact method. Avoid sentences that make the reader feel the solution has
  already been announced.
- When introducing a method, clearly state what the method is not before
  describing what it does. Example: `FEMR is not a new motion tracker.`
- For method scope, use a positive target and a negative exclusion:
  `The method targets frame-level spatial artifacts. Contact timing mismatch
  requires sequence-level temporal alignment and is outside this work.`
- Do not write `learn a new controller` unless the method actually optimizes a
  controller. For FrontRES, say that FEMR is not a new tracker and does not
  modify the frozen tracker.
- Replace weak pronoun chains with repeated nouns:
  bad: `They may be subtle. However, they directly change...`
  good: `Reference-frame artifacts can be subtle. However, these artifacts
  directly change...`
- Avoid awkward abstractions such as `per-video trajectory solving` when a
  clearer phrase exists. Prefer `solve a new trajectory for each video`.
- Contribution bullets should not sound like a template. Use concrete objects:
  `reference-frame errors`, `robustness budget`, `frozen tracker`,
  `executability objectives`, and `disturbance recovery experiments`.

## FrontRES-Specific Guidance

Use consistent terms:

- `motion tracker`
- `reference motion`
- `video-reconstructed motion`
- `dynamically executable`
- `kinematic-to-dynamic gap`
- `front-end motion refiner`
- `frozen tracker`
- `task-space correction`
- `executability`

Keep the research claim aligned with FrontRES:

- FrontRES does not replace GMT.
- FrontRES corrects reference artifacts before tracking.
- The goal is executable correction, not visual similarity.
- The problem is upstream reference errors consuming the tracker robustness
  budget.
- The method should learn both when to remain unchanged and when to correct.
- The method targets frame-level spatial artifacts such as root height,
  orientation, and horizontal displacement errors.
- The method does not solve sequence-level temporal errors such as contact
  timing mismatch.
