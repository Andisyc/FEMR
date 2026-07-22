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
| Current code | source audit | foot-height/additive v003 and formal-v015 hardcoded joint/no-Warmup paths are explicitly contract-mismatch | confirmed mismatch | `E-FI-66--E-FI-67` |
| Expected support | S1 T-schema/T-provenance/T-hash | sealed Clean continuation deterministically yields `[K,2]` left/right `11/10/01/00`; all M attempts reuse one identity; actor cannot read it | pending G5-P1 | stop on resample, mixed identity, or actor leak |
| Actual Contact | S1 T-owner/T-value | actual left/right contact comes only from configured `contact_forces`; no robot-height fallback | pending G5-P1 | stop if sensor owner is unavailable or silently replaced |
| Contact alignment | S1 T-temporal/T-metamorphic | planned steps pass; early/late tolerance is bounded; extra/missed/dragging/out-of-window switches fail; foot/row permutation preserves identity | pending G5-P1 | v004 focused contracts |
| Phase ZMP | S1 T-phase/T-mask | single/double support uses matching domain; transitions permit planned transient then require recovery; flight masks ZMP as N/A | pending G5-P1 | v004 focused contracts |
| Physics ordering | S1 T-order/T-sign/T-noop | safe outranks unsafe; both unsafe compare deficit only; both safe compare Intent; repair cost cannot invert tier; no-op is zero before cost | pending G5-P1 | v004 golden fixtures |
| Missing evidence | S1 T-fail-closed | missing/non-finite Contact, phase, ZMP, survival, or identity is UNCONFIRMED/invalid, never zero-filled | pending G5-P1 | v004 negative contracts |
| Formal consumers | S2 T-connectivity/T-isolation | storage, return, priority, grouped PPO, diagnostics, local and held-out evaluation consume only v004 identity | pending G5-P1 | reject v002/v003 fallback and partial/mixed rows |
| Warmup schedule | S1 T-boundary/T-scale | persisted iteration selects critic-only, actor-ramp, joint; actor weight is 0, monotonic ramp, 1; Critic remains enabled | pending G5-P1 | formal v008 schedule contracts |
| Critic-only gradient | S2 T-gradient/T-state | formal v015 v004 update changes scalar Critic but leaves actor/std exactly unchanged and takes exactly one step | pending G5-P1 | legacy single-update owner is not evidence |
| Warmup persistence | S3 T-version/T-resume/T-pre-mutation | cold HSL-v1 enters iteration 0 critic-only; v008/v004 resume preserves phase; v003/v007/unversioned resume rejects | pending G5-P1 | checkpoint contracts |
| Frozen boundaries | S2 T-static/T-shape | one 6D actor, scalar Gain/Critic, 928/158/770, H/K, one-action-K, HSL, grouped reduction, value formula, and exact-one update do not change | pending G5-P1 | regression contracts |
| Bounded live | S4 official sentinel | one 8-env critic-only transaction records real Contact/phase/ZMP/admissibility/utility/Gain/credit, actor/std zero delta, Critic delta, exact-one update, and v008/v004 checkpoint | pending authorization | stop on missing identity, no-op Critic, actor drift, regression, harm, fallback, or update count != 1 |
| X1 experiments | high-cost boundary | long training, seeds, checkpoint trajectory, paired composition, and paper artifacts | blocked | requires complete G5-P1 S1/S2/S4 evidence |

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
