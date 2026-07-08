# FrontRES Segment Sampler Evidence Ledger

Date: 2026-07-08

Purpose: replace the vague claim "boundary is clean" with auditable evidence.

## Claims And Evidence

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The mistaken `single_update_override` plumbing is absent from current source/note state. | static-confirmed | Search command: `ctx_search("single_update_override|_frontres_segment_live_last_storage_batch|_attach_sampler_ppo_update_summary|sampler_update_order|before_ppo_update|sampler-before-PPO|sampler update -> PPO|Level Replay-style sampler-before-PPO", source+rsl_rl+note)` returned 0 matches. | No searched literal remnants remain in `source/rsl_rl/rsl_rl` or `note`. | It does not prove semantic absence of every possible equivalent implementation. |
| The public runner wrapper no longer accepts or forwards `single_update_override`. | code-confirmed | `source/rsl_rl/rsl_rl/runners/on_policy_runner.py:687-694` has `run_frontres_segment_live_probe(self, init_at_random_ep_len=True)` forwarding only `init_at_random_ep_len`. | The wrapper interface no longer exposes the removed override. | It does not prove all runtime modes are healthy. |
| The live sampler step uses rollout summary to build sampler evidence, then calls `sampler.update_with_probe(evidence)`. | code-confirmed | `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py:213-232` calls live probe, builds `build_live_sampler_evidence(...)`, then calls `sampler.update_with_probe(evidence)`. | Sampler update input is the evidence object built from sample, rollout summary, horizon, and reset result. | It does not prove policy quality or physics correctness. |
| The retained regression test poisons PPO post-update diagnostics without changing sampler evidence. | contract-confirmed | `source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py:531-564` injects `ppo_post_update_distribution_kl_mean=1.0e9`, post-update ratios, and `ppo_param_delta_l2=1.0e9`, then asserts valid mask, gain, noisy/repaired scores, and update valid count from rollout evidence. | Sampler priority evidence is isolated from those post-update PPO diagnostic fields in this controlled S1/S2 contract. | It does not prove a full S4 training run has no unrelated sampler bug. |
| The test actually executed in the current worktree. | contract-confirmed | Command: `/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`; output included `[probe evidence-ppo-isolation] ... priority_after=0.043500` and `frontres_segment_live_sampler_contract: ok`. | The evidence-isolation test ran and passed now, not only existed in source. | It does not replace live IsaacLab training validation. |
| The concept note no longer claims literal Level Replay call-order alignment. | note-confirmed | `note/frontres_segment_replay/references/external_code_reuse_map.md:560-576` says live flow is `segment_reward -> PPO update -> sampler priority update`, then states Level Replay alignment is semantic: rollout-time evidence independent of post-update PPO diagnostics. | The durable design note distinguishes semantic evidence alignment from literal call-order changes. | It does not prove future edits will preserve this unless tests remain in CI/manual gate. |
| The test matrix records the same semantic boundary. | note-confirmed | `note/testing/test_control_board.md:186` records MAIN-37 as evidence-isolation / policy-update-independent sampler evidence, not sampler-before-PPO order. | The all-module-test inventory now points humans to the intended evidence boundary. | It does not enforce honesty by itself unless reports cite evidence rows. |

## Commands Re-run In This Audit

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py source/rsl_rl/rsl_rl/runners/on_policy_runner.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
```

Observed result:

```text
py_compile: exit 0
sampler contract: frontres_segment_live_sampler_contract: ok
required probe observed: [probe evidence-ppo-isolation]
```

## Anti-Black-Box Rule Extracted

Future `all-module-test` reports must not use ungrounded closing phrases such as
"clean", "covered", "OK", "safe", or "fixed" unless every such phrase is backed
by an evidence row with:

1. exact claim;
2. file lines or command;
3. observed output fact;
4. S tier and T kind;
5. explicit limitation.

If a claim lacks one of these fields, report it as `unconfirmed`, not as `clean`.
