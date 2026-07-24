# FRS-v015 Physics-Constrained Intent Migration Checklist

Status: active, volatile acceptance surface. Updated: 2026-07-24.

| Step | Owner / tier | Acceptance assertion | Status | Evidence / stop |
| --- | --- | --- | --- | --- |
| P0 | governance | Noisy stays the shared zero-action baseline; Physics constraints and scalar Intent objective have separate authority | completed | E-FI-72; v005 proposal |
| P0 | owner audit | v004 scalar path is traced through Gain, storage return, scalar Critic/PPO, formal update, diagnostics and checkpoint | code-confirmed | CodeGraph 2026-07-23 |
| P0 | mismatch | Concept Figure Q-01 expressed constrained Intent while source still used v004/v003/v009 | resolved | E-FI-74 replaces the active formal source route; old identities reject |
| P1 | semantic | exact Contact/ZMP/survival residuals retain physical meaning and do not saturate severe states | completed | E-FI-73; GAIN-v005 physical-unit residuals |
| P1 | optimization | grouped joint constraint-gradient projection/recovery is exact, fail-closed and needs no second learned network | completed | E-FI-73; PPO-v004 KKT/recovery/no-common-descent rules |
| P1 | replay priority | constraint/frontier evidence may affect scenario selection but cannot become scalar Physics reward or actor-loss mass | completed | E-FI-73; selection-only authority |
| P1 | Critic curriculum | v004 Critic rejects; first v010 target entry is fresh critic-only; every global K increase recalibrates the same Critic before actor ramp/joint | completed | E-FI-73; TRAIN-v010 checkpoint-v5 identity |
| P1 | contracts | METHOD-v016, GAIN-v005, PPO-v004 and TRAIN-v010 agree on objective, constraints, Critic, optimizer and persistence identities | completed | E-FI-73; all four activated atomically |
| P2 | S1 evidence | raw K-step Contact/phase-ZMP/survival, N/A semantics, constraint residuals and projection pass value/collision/permutation/missing tests | completed | E-FI-74; no scalar Physics fallback |
| P2 | S2 connectivity | same sealed Noisy baseline and M Repair rows reach scalar objective plus vector constraints, grouped equal mass and exact-one update | completed | E-FI-74; v004/v003 formal consumers reject |
| P2 | S2 gradients | scalar Critic receives only Intent objective; actor receives Intent direction constrained by Physics; frozen GMT and actor inputs unchanged | completed | E-FI-74; KKT and parameter-authority contracts pass |
| P2 | S3 persistence | checkpoint-v5/receipt bind solver and contracts; v004/v009, tampered, or partial resume rejects before mutation | completed | E-FI-74; no persistent dual state or legacy scalar Gain payload |
| P3 | S4 official | one 8-env transaction logs raw evidence, constraints, projection, objective/value/advantage, parameter deltas and exactly one committed update | completed | E-FI-75; live log plus read-only checkpoint-v5 payload identity |
| P4-S0 | S0 admission audit | checkpoint-v5 phase, Critic budget, formal continuation route and held-out evidence coverage are read-only resolved before actor-ramp | completed | E-FI-76: K8 critic-only 1/200; formal resume contradiction and quality-report gaps confirmed |
| P4-S1 | S1 launcher/config | strict checkpoint-v5 resume is mutually exclusive with HSL initialization and preserves exact v010 schedule/iteration | completed | E-FI-77; explicit resume replaces HSL flag, missing/legacy/partial/mixed identity rejects |
| P4-S1 | S2 resume connectivity | actor/Critic/optimizer/sampler/normalizer/receipt restore exactly and one fake transaction advances exactly one update | completed | E-FI-77; strict semantic checkpoint and transaction contracts pass |
| P4-S1 | S1/S2 quality evidence | atomic report exposes raw expected/actual Contact, phase-ZMP trajectories/N-A masks, survival and evaluator-only sustained lean | completed | E-FI-77; missing evidence fails closed and no Clean actor route is added |
| P4-S1 | S3 persistence/isolation | resumed save remains coordinated checkpoint-v5 and evaluation mutates no optimizer/sampler/transaction/normalizer state | completed | E-FI-77; committed receipt survives idle re-save; save/reload isolation passes |
| P4 | quality | target distinguishes v004 plateau cases and actor update improves or preserves Physics without sustained lean/unplanned stepping | awaiting experiment decision | critic continuation and actor-ramp require separate authorization |

## Pass Rule

Engineering passes only when the formal route contains one scalar Intent
objective/Critic and explicit non-compensatory Physics constraints through
storage, actor update, diagnostics, and persistence. Retaining Noisy is required;
using it as the Physics threshold is forbidden.

## Fail Rule

Stop on a single scalar Physics utility, severe-state saturation, adverse-row
masking, Clean/Contact/ZMP leakage to actor input, second actor/Critic/optimizer,
mixed scenario/K identity, v004/v003 fallback, partial transaction update, or
any live optimizer count other than one.
