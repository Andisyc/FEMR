# Evidence Ledger: DP-09 Warmup And Frozen GMT

Date: 2026-07-15
Scope: offline S0-S3 method-code alignment only; no IsaacLab live claim.

## Verified Facts

- `frontres_segment_warmup_contract.py`: critic-only, linear actor warmup, and joint phase values pass; critic-only blocks actor/std gradients.
- `frontres_segment_stage3_entrypoint_pseudo_contract.py`: the official Stage 3 preset sets critic warmup to 200 iterations and actor ramp to 500 iterations.
- `frontres_segment_live_single_update_contract.py`: the formal Segment update holds actor parameters while updating the critic in critic-only, then updates the actor at weight 0.25 on the first actor-warmup iteration.
- `frontres_checkpointing.py`: Stage 3 checkpoints persist warmup configuration; full resume rejects a different runtime warmup configuration and restores the persisted learning iteration.
- `frontres_frozen_gmt_contract.py`: GMT parameters are frozen, excluded from the optimizer, receive no gradient, and remain bitwise unchanged while residual actor and critic update.
- Aggregate command `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py` passed `42/42` with `failed_count=0`.

## Remaining Boundary

The first live sentinel must confirm the printed phase, phase iteration, actor weight, and observed actor/critic parameter deltas on the real IsaacLab route. No long training is authorized by this ledger alone.

## Phase B Probe Insertion

- Added explicit `--frontres_formal_runtime_audit`; the default remains disabled.
- Inserted `AUDIT-ROUTE-01`, `AUDIT-SAMPLER-01`, `AUDIT-DATAFLOW-01`, `AUDIT-PPO-01`, and `AUDIT-PERSIST-01` at official Stage 3 owner boundaries.
- `frontres_formal_runtime_audit_contract.py` proves silent-off behavior, stable labels, compact tensor summaries, Frozen GMT fields, checkpoint identity fields, and hook presence.
- Fresh aggregate result after insertion: `43/43`, `failed_count=0`.
- These are insertion and offline-connectivity facts only. Every audit row remains live-pending until an official `MODE=train` run produces the labels.

## Runtime Audit Atlas Synchronization

- Created the permanent `note/architecture/runtime/04_stage3_formal_runtime_audit.data.json` and its `04_stage3_formal_runtime_audit.html` entrypoint.
- The five reading blocks map stable `AUDIT-*` IDs to Concept Figure design IDs, formal owners, caller/callee boundaries, checked fields, invariants, and `PENDING_LIVE` state.
- Added matching `PENDING_LIVE` comments beside each real owner call in live training, sampler, rollout/storage, PPO, and checkpoint code.
- `frontres_formal_runtime_audit_contract.py` now rejects missing Atlas IDs, checklist IDs, owner comments, design mappings, or pending-live status.
- Viewer import/data contract passed with the bundled modern Node runtime; `43/43` aggregate contracts passed after synchronization.
- Visual screenshot inspection remains unconfirmed because no local Playwright Chromium or Chrome executable was available. The Atlas server is running at `http://127.0.0.1:8765/04_stage3_formal_runtime_audit.html` for human review.
- After human review found a style mismatch, the Runtime Audit Atlas was migrated from `flow_tree` to the same `repository_reading_atlas` schema as 01: flat left-to-right `runtimeOrder`, one owner-boundary reading card per probe, and B1/B2/B3/B4 internal routes. Both `repo-architecture-atlas` and `formal-runtime-audit` now enforce explicit 01-style precedence over generic runtime-to-flow-tree routing.
