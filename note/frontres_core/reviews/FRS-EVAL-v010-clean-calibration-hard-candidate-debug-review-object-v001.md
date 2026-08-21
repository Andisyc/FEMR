# FRS-EVAL-v010 hard-candidate debug review object

- Observed failure: the official CUDA route reached raw Clean telemetry, then the selected source-0 window raised a hard Physics event and aborted the campaign.
- First invalid boundary: candidate-level orchestration treated one physically inadmissible Clean source as a campaign-fatal error.
- Preserved behavior: hard Survival/contact events remain invalid calibration evidence; FRS-GAIN-v009, training, Replay and optimizer routes remain untouched.
- Correction: publish typed hard-event provenance, reject only that candidate, try only explicitly declared fallback source indices, and atomically emit `TELEMETRY-GAP` when all configured candidates are invalid.
- Lifecycle: every prepared candidate is closed through the existing idempotent local-scenario owner; protected state and RNG checks remain in the typed raw gateway.
- Regression: official-entry pseudo transaction covers invalid primary to valid fallback and all-invalid to atomic `TELEMETRY-GAP`; telemetry alignment checks typed event fields.
- Evidence: focused calibration tests pass, Stage-3 suite passes 13/13, aggregate suite passes 61/61, syntax/JSON/diff checks pass.
- Unclaimed: real source-1 physical validity and real CUDA receipt remain one bounded live-only fact.
