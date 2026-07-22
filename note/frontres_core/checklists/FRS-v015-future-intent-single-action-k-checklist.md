# FRS-v015 One-Shot Acceptance Checklist

Status: active, volatile acceptance surface. Updated: 2026-07-22.

Plan: `../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md`

Rows are embedded acceptance assertions for one G5-P1 engineering closure, not
separate approval steps.

| Unit | Tier / kind | Acceptance assertion | Status | Evidence |
| --- | --- | --- | --- | --- |
| Preserved foundation | S2/S4 identity | v015 q29, 928/158/770, two roles, one-action K, sealed grouped exact-one update, HSL-v1, scalar Critic/value carrier, and persistence remain unchanged | completed / frozen | `E-FI-0--E-FI-65`; historical E68--E70 only for old Warmup route |
| Gain authority | document | FRS-GAIN-v004 is active; v003 is superseded; Q-PAIR/Q-01 retain the existing Concept Figure ownership | completed | `E-FI-66` |
| Training authority | document | FRS-TRAIN-v008 is active; HSL remains actor-only; M-05 owns v004 critic-only -> actor-ramp -> joint PPO | completed | `E-FI-67` |
| Superseded mismatch | source audit | foot-height/additive v003 and formal-v015 hardcoded joint/no-Warmup routes are isolated from the active route | completed | `E-FI-66--E-FI-68` |
| Expected support | S1 T-schema/T-provenance/T-hash | sealed Clean continuation deterministically yields `[K,2]` left/right `11/10/01/00`; all M attempts reuse one identity; actor cannot read it | completed | `E-FI-68`; focused local-scenario and one-action-K contracts |
| Actual Contact | S1 T-owner/T-value | actual left/right contact comes only from configured `contact_forces`; no robot-height fallback | completed | `E-FI-68`; S4 `actual_contact_*_steps` |
| Contact alignment | S1 T-temporal/T-metamorphic | planned steps pass; early/late tolerance is bounded; extra/missed/dragging/out-of-window switches fail; foot/row permutation preserves identity | completed | `E-FI-68`; v004 focused contracts |
| Phase ZMP | S1 T-phase/T-mask | single/double support uses matching domain; transitions permit planned transient then require recovery; flight masks ZMP as N/A | completed | `E-FI-68`; v004 focused contracts |
| Physics ordering | S1 T-order/T-sign/T-noop | safe outranks unsafe; both unsafe compare deficit only; both safe compare Intent; repair cost cannot invert tier; no-op is zero before cost | completed | `E-FI-68`; v004 golden fixtures |
| Missing evidence | S1 T-fail-closed | missing/non-finite Contact, phase, ZMP, survival, or identity is UNCONFIRMED/invalid, never zero-filled | completed | `E-FI-68`; missing/mask contracts |
| Formal consumers | S2 T-connectivity/T-isolation | storage, return, priority, grouped PPO, diagnostics, local and held-out evaluation consume only v004 identity | completed | `E-FI-68`; consumer/transaction/evaluation contracts |
| Warmup schedule | S1 T-boundary/T-scale | persisted iteration selects critic-only, actor-ramp, joint; actor weight is 0, monotonic ramp, 1; Critic remains enabled | completed | `E-FI-68`; v008 schedule contracts |
| Critic-only gradient | S2/S4 T-gradient/T-state | formal v015 v004 update changes scalar Critic but leaves actor/std exactly unchanged and takes exactly one step | completed | `E-FI-68`: Critic 10/10 changed; actor/std delta 0; step delta 1 |
| Warmup persistence | S3 T-version/T-resume/T-pre-mutation | cold HSL-v1 enters iteration 0 critic-only; v008/v004 resume preserves phase; v003/v007/unversioned resume rejects | completed | `E-FI-68`; checkpoint/resume contracts and S4 save |
| Frozen boundaries | S2 T-static/T-shape | one 6D actor, scalar Gain/Critic, 928/158/770, H/K, one-action-K, HSL, grouped reduction, value formula, and exact-one update do not change | completed | `E-FI-68`; regression suite and S4 telemetry |
| Bounded live | S4 official sentinel | one 8-env critic-only transaction records real Contact/phase/ZMP/admissibility/utility/Gain/credit, actor/std zero delta, Critic delta, exact-one update, and v008/v004 checkpoint | completed, efficacy unconfirmed | `E-FI-68`; all four Repairs Physics-inadmissible is retained as X1 quality evidence |
| X1 experiments | high-cost boundary | actor-ramp/long training, seeds, checkpoint trajectory, paired composition, and paper artifacts | ready, not authorized | requires a separate training-budget and quality-gate decision |

## Pass Rule

G5-P1 passes only when deterministic semantics, formal consumer and Warmup
isolation, persistence, and one bounded real-environment critic-only transaction
agree with FRS-GAIN-v004 and FRS-TRAIN-v008. Document activation alone is not
implementation evidence. Live actor-ramp/joint progression remains X1 evidence.

## Fail Rule

Ordinary code or contract-test failures stay inside the authorized one-shot
repair cycle. Stop for a method choice, unavailable authoritative sensor,
privileged-information leak, tier inversion, frozen-boundary change, unresolved
formal contradiction after one repair cycle, or harmful/no-op bounded metrics.
