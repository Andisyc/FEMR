# FrontRES Transaction Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the long Atlas 07 method cards with ten compact parent-design-point buttons controlling one shared Stage-3 Transaction spine and one minimal fine-detail reading card.

**Architecture:** Keep the existing Atlas shell, controls, and data-driven renderer. The dedicated Transaction inspector contains one canonical ordered spine plus per-design highlights and ordered atomic detail decisions. Active contracts remain the semantic authority; implementation evidence and source navigation stay outside Atlas 07.

**Tech Stack:** Static JSON, browser-native JavaScript/SVG, Rough.js, Node contract checks.

---

### Task 1: Freeze The Transaction-Only Consumer Contract

**Files:**
- Modify: `note/architecture/auxiliary/atlas_app/check_design_contract_review.mjs`
- Modify: `note/architecture/auxiliary/atlas_app/check_viewer_import.mjs`

- [x] Require `layout=design_transaction_inspector`, exactly ten index cards, one `transaction.steps[]` sequence, and a dedicated `renderDesignTransactionInspector` route.
- [x] Require the canonical 13-step order from HSL initialization through checkpoint/curriculum commit.
- [x] Require every card to contain only `designId`, `blockId`, `title`, `color`, `responsibility`, `highlightSteps`, `details`, and an always-empty compatibility `chips` array.
- [x] Reject implementation/source/evidence/risk/formula/matrix/authority/review-panel fields from the primary data.
- [x] Require exact Segment Replay, K/M, 928/158/770, H/K, exact-one-update, and K64-inactive facts in steps or chips.
- [x] Run `npm run check`; expected result: FAIL because Atlas 07 still uses `design_specification_review` and long-card fields.

### Task 2: Project The Active Method As One Transaction

**Files:**
- Modify: `note/architecture/runtime/07_frontres_design_contract_review.data.json`

- [x] Replace card-local sections with one ordered Transaction spine:

```text
HSL 初始化 Actor
-> 确定 K/M 与训练阶段
-> 选择两个 Segment
-> 封存 scenario
-> 恢复到同一 x_t
-> 从冻结 pi_old 采样 exact M 个修复动作
-> FrontRES 在 t 输出一次 Delta SE(3)
-> 冻结 FrontRES，由 GMT 执行 K 步
-> 构造 Repair/Noisy 配对证据
-> 生成 Intent 目标与 Physics 约束
-> 封存 2 x M 条 PPO row
-> 执行一次 grouped update
-> 提交 checkpoint 与 curriculum 状态
```

- [x] Keep every parent-design-point title canonical and concise, map it to highlighted step ids, add four to eight ordered atomic detail decisions, and keep parameter chips absent.
- [x] Use complete Chinese action sentences; preserve only established method nouns and symbols in English.

### Task 3: Render The Shared Spine

**Files:**
- Modify: `note/architecture/auxiliary/atlas_app/architecture_atlas.html`

- [x] Add `renderDesignTransactionInspector(data)` without changing other Atlas layouts.
- [x] Preserve the current compact 5 x 2 index buttons.
- [x] Render compact parent-design-point buttons, the same three-row Transaction spine, and one minimal selected-detail reading card.
- [x] Use color only on the selected index and its highlighted steps; render all other steps as quiet context.
- [x] Keep arrows continuous across pre-Transaction, collection/evidence, and commit/update phases.
- [x] Route `design_transaction_inspector` to the new renderer.

### Task 4: Synchronize Entrypoints And Verify The Real Page

**Files:**
- Modify: `note/architecture/README.md`
- Modify: `note/architecture/index.html`
- Modify: `note/architecture/07_frontres_design_contract_review.html`

- [x] Rename the visible page to `FrontRES Transaction Inspector` and describe the shared-spine interaction.
- [x] Run `npm run check`, JavaScript syntax validation, and `git diff --check`.
- [x] Open the actual wrapper in Firefox and select Segment Replay, K-step Curriculum, HSL Warmup, and Future Motion Context.
- [x] Confirm identical step order, correct highlights, canonical parent titles, atomic detail decisions, concise Chinese prose, active K/M schedule, K64 inactive status, and no implementation/evidence footer.
- [x] Complete a final in-scope code review; resolve P0/P1 findings and rerun the checks.
