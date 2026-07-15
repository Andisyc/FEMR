# FrontRES Method Contract History

Read `../../README.md` first. These files are excluded from default recall.

## Version Sequence

| Version | Effective date | Design | Status | Superseded by |
| --- | --- | --- | --- | --- |
| `FRS-METHOD-v001` | before 2026-06-09, exact date unconfirmed | HSL proposal plus per-axis rho acceptance | superseded | `v002` |
| `FRS-METHOD-v002` | 2026-06-10 | Stable-to-Repair rho parameterization | superseded | `v003` |
| `FRS-METHOD-v003` | 2026-06-10 | Tri-Anchor projection | superseded | `v004` |
| `FRS-METHOD-v004` | 2026-06-11 | Structured joint Alpha-Rho policy gradient | rejected | `v005` |
| `FRS-METHOD-v005` | 2026-06-12 | Executable-Floor router plus repair retention | superseded | `v006` |
| `FRS-METHOD-v006` | 2026-06-19 | Conditional HRL repair authority | superseded | `v007` |
| `FRS-METHOD-v007` | 2026-06-23 | Proposal-conditioned acceptance | superseded | `v008` |
| `FRS-METHOD-v008` | 2026-06-23 | Proposal-conditioned authority actor-critic | stopped | `v009` |
| `FRS-METHOD-v009` | 2026-06-25 | HSL proposal plus binary/near-binary acceptance | superseded | `v010` |
| `FRS-METHOD-v010` | 2026-07-05 | Segment Replay HRL with dynamic reset and K-step curriculum | superseded | `v011` |
| `FRS-METHOD-v011` | 2026-07-13 | Segment Replay with paired Style/Physics Gain and Repair Cost | active | - |

The numbered files below preserve the durable method meaning and supersession
reason. `FRS-METHOD-v000-design-history-compendium.md` is the restricted raw
source bundle from which these records were extracted; `v000` is not a method
version.

GMT frontier search, DR mixtures, diagnostics cleanup, checkpoint rules,
modularization, and local loss fixes remain supporting engineering/training
history. They do not receive Method version numbers unless they changed the
learned variable or policy authority boundary.
